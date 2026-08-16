#!/usr/bin/env python3
"""Match an N-example FineWeb-Edu batch gradient with one [1,S,vocab] vector.

The reference is the mean causal-LM gradient over N real samples.  Only one
continuous input/target sequence is optimized on pretrained GPT-2 small; that
sequence is then frozen and evaluated on other GPT-2 sizes and weight draws.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

from gradient_matching_gpt2 import continuous_gradient, get_blocks, metrics, reference_gradient


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-model", default="gpt2")
    p.add_argument("--models", nargs="+", default=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    p.add_argument("--weights", nargs="+", choices=["pretrained", "random"], default=["pretrained", "random"])
    p.add_argument("--n", type=int, default=3, help="Number of real FineWeb-Edu samples in the reference batch")
    p.add_argument("--s", "--seq-len", dest="seq_len", type=int, default=512)
    p.add_argument("--steps", type=int, default=30); p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--init", choices=["first-sample", "random"], default="first-sample")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--config", default="sample-10BT"); p.add_argument("--split", default="train")
    p.add_argument("--max-docs", type=int, default=1000); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="batch_to_single_results.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    return p.parse_args()


def load_model(name, weights, device, dtype):
    if weights == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(name), attn_implementation="eager"
        )
    return model.to(device=device, dtype=dtype).eval()


def optimize_single(model, batch_ids, a):
    vocab = model.config.vocab_size
    dtype = next(model.parameters()).dtype
    single_ids = batch_ids[:1]
    if a.init == "first-sample":
        x0 = F.one_hot(single_ids, vocab).to(dtype=dtype) * 8.0
        y0 = F.one_hot(single_ids, vocab).to(dtype=dtype) * 8.0
    else:
        x0 = torch.randn((1, a.seq_len, vocab), device=batch_ids.device, dtype=dtype)
        y0 = torch.randn_like(x0)
    x, y = torch.nn.Parameter(x0), torch.nn.Parameter(y0)
    reference, reference_loss = reference_gradient(model, batch_ids)
    optimizer = torch.optim.AdamW([x, y], lr=a.lr)
    history = []
    for step in range(a.steps + 1):
        continuous, continuous_loss = continuous_gradient(model, x, y, a.temperature, create_graph=True)
        row = metrics(continuous, reference)
        row.update(step=step, continuous_loss=float(continuous_loss.detach()))
        history.append(row)
        if step == a.steps:
            break
        objective = 1.0 - (continuous @ reference) / (continuous.norm() * reference.norm()).clamp_min(1e-12)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_([x, y], 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 5 == 0:
            print(f"SOURCE step={step:03d} cosine={row['cosine']:.5f} relL2={row['relative_l2']:.5f}", flush=True)
    return x.detach(), y.detach(), {"reference_loss": reference_loss, "history": history}


def evaluate(name, weights, batch_ids, x, y, a):
    device = torch.device(a.device)
    dtype = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32
    model = load_model(name, weights, device, dtype)
    batch_ids = batch_ids.to(device)
    x, y = x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype)
    reference, reference_loss = reference_gradient(model, batch_ids)
    continuous, continuous_loss = continuous_gradient(model, x, y, a.temperature, create_graph=False)
    row = metrics(continuous, reference)
    row.update(model=name, weights=weights, reference_loss=reference_loss,
               continuous_loss=float(continuous_loss.detach()))
    print(f"EVAL {name:16s} {weights:11s} cosine={row['cosine']:.5f} relL2={row['relative_l2']:.5f} norm_ratio={row['norm_ratio']:.5f}", flush=True)
    del model, reference, continuous, x, y
    if device.type == "cuda": torch.cuda.empty_cache()
    return row


def main():
    a = parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed); set_seed(a.seed)
    device = torch.device(a.device)
    dtype = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(a.source_model)
    batch_ids_cpu = get_blocks(tokenizer, a.dataset, a.config, a.split, a.n, a.seq_len, a.max_docs)
    batch_ids = batch_ids_cpu.to(device)
    print(f"Reference batch shape: {tuple(batch_ids.shape)}; optimized input/target shape: (1, {a.seq_len}, {tokenizer.vocab_size})", flush=True)

    source = load_model(a.source_model, "pretrained", device, dtype)
    start = time.time()
    x, y, source_log = optimize_single(source, batch_ids, a)
    optimization_seconds = time.time() - start
    del source
    if device.type == "cuda": torch.cuda.empty_cache()

    evaluations = []
    for name in a.models:
        for weights in a.weights:
            evaluations.append(evaluate(name, weights, batch_ids_cpu, x, y, a))
    payload = {
        "source_model": a.source_model, "source_weights": "pretrained",
        "reference_batch_shape": list(batch_ids.shape),
        "optimized_vector_shape": [1, a.seq_len, "vocab"],
        "optimization_seconds": optimization_seconds,
        "source_optimization": source_log, "evaluations": evaluations, "args": vars(a),
    }
    Path(a.out).write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {a.out}")


if __name__ == "__main__":
    main()

