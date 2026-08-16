#!/usr/bin/env python3
"""Transferability of continuous gradient-matching inputs across GPT-2 weights.

Only the source model (default: pretrained gpt2-small) is used to optimize the
continuous input/target tensors.  The optimized tensors are then frozen and
evaluated on every requested pretrained/random GPT-2 model.
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

from gradient_matching_gpt2 import (
    continuous_gradient,
    flat,
    get_blocks,
    metrics,
    reference_gradient,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-model", default="gpt2", help="Only this model is optimized; normally gpt2-small.")
    p.add_argument("--models", nargs="+", default=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    p.add_argument("--weights", nargs="+", choices=["pretrained", "random"], default=["pretrained", "random"])
    p.add_argument("--batch-size", "--n", dest="batch_size", type=int, default=3)
    p.add_argument("--seq-len", "--s", dest="seq_len", type=int, default=512)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--init", choices=["onehot", "random"], default="onehot")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--max-docs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="transfer_gradient_results.json")
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


def optimize_source(model, ids, a):
    vocab = model.config.vocab_size
    dtype = next(model.parameters()).dtype
    if a.init == "onehot":
        x0 = F.one_hot(ids, vocab).to(dtype=dtype) * 8.0
        y0 = F.one_hot(ids, vocab).to(dtype=dtype) * 8.0
    else:
        x0 = torch.randn((*ids.shape, vocab), device=ids.device, dtype=dtype)
        y0 = torch.randn_like(x0)
    x, y = torch.nn.Parameter(x0), torch.nn.Parameter(y0)
    ref, ref_loss = reference_gradient(model, ids)
    optimizer = torch.optim.AdamW([x, y], lr=a.lr)
    history = []
    for step in range(a.steps + 1):
        g, continuous_loss = continuous_gradient(model, x, y, a.temperature, create_graph=True)
        current = metrics(g, ref)
        current.update(step=step, continuous_loss=float(continuous_loss.detach()))
        history.append(current)
        if step == a.steps:
            break
        match_loss = 1.0 - (g @ ref) / (g.norm() * ref.norm()).clamp_min(1e-12)
        optimizer.zero_grad(set_to_none=True)
        match_loss.backward()
        torch.nn.utils.clip_grad_norm_([x, y], 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 5 == 0:
            print(f"SOURCE step={step:03d} cosine={current['cosine']:.5f} relL2={current['relative_l2']:.5f}", flush=True)
    return x.detach(), y.detach(), {"reference_loss": ref_loss, "history": history}


def evaluate_model(name, weights, ids, x, y, a):
    device = torch.device(a.device)
    dtype = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32
    model = load_model(name, weights, device, dtype)
    ids = ids.to(device)
    x, y = x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype)
    reference, reference_loss = reference_gradient(model, ids)
    optimized_gradient, continuous_loss = continuous_gradient(
        model, x, y, a.temperature, create_graph=False
    )
    result = metrics(optimized_gradient, reference)
    result.update(model=name, weights=weights, reference_loss=reference_loss,
                  continuous_loss=float(continuous_loss.detach()))
    del model, reference, optimized_gradient, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"EVAL {name:16s} {weights:11s} cosine={result['cosine']:.5f} relL2={result['relative_l2']:.5f} norm_ratio={result['norm_ratio']:.5f}", flush=True)
    return result


def main():
    a = parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    set_seed(a.seed)
    device = torch.device(a.device)
    dtype = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(a.source_model)
    ids_cpu = get_blocks(tokenizer, a.dataset, a.config, a.split,
                         a.batch_size, a.seq_len, a.max_docs)
    ids = ids_cpu.to(device)

    print(f"Loading source {a.source_model} pretrained; optimizing tensors with shape {tuple(ids.shape)} + vocab", flush=True)
    source = load_model(a.source_model, "pretrained", device, dtype)
    start = time.time()
    x, y, source_log = optimize_source(source, ids, a)
    source_time = time.time() - start
    del source
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = []
    for model_name in a.models:
        for weight_kind in a.weights:
            results.append(evaluate_model(model_name, weight_kind, ids_cpu, x, y, a))

    payload = {
        "source_model": a.source_model,
        "source_weights": "pretrained",
        "batch_size": a.batch_size,
        "seq_len": a.seq_len,
        "vector_shape": [a.batch_size, a.seq_len, "vocab"],
        "optimization_seconds": source_time,
        "source_optimization": source_log,
        "evaluations": results,
        "args": vars(a),
    }
    Path(a.out).write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {a.out}")


if __name__ == "__main__":
    main()
