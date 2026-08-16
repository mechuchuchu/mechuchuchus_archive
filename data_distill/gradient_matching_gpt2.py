#!/usr/bin/env python3
"""Full-parameter gradient matching with continuous GPT-2 inputs/targets."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["gpt2"])
    p.add_argument("--weights", nargs="+", choices=["pretrained", "random"], default=["pretrained", "random"])
    p.add_argument("--batch-size", type=int, default=3); p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=30); p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--init", choices=["onehot", "random"], default="onehot")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu"); p.add_argument("--config", default="sample-10BT")
    p.add_argument("--split", default="train"); p.add_argument("--max-docs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default="gradient_matching_results.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    return p.parse_args()

def get_blocks(tokenizer, name, config, split, batch, seq_len, max_docs):
    ds = load_dataset(name, name=config, split=split, streaming=True); blocks = []
    for i, row in enumerate(ds):
        text = row.get("text", "")
        tok_ids = tokenizer(text, add_special_tokens=False)["input_ids"] if text else []
        if len(tok_ids) >= seq_len: blocks.append(tok_ids[:seq_len])
        if i + 1 >= max_docs or len(blocks) >= batch: break
    if len(blocks) < batch: raise RuntimeError(f"Only collected {len(blocks)} documents; need {batch}")
    return torch.tensor(blocks, dtype=torch.long)

def flat(grads, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])

def reference_gradient(model, ids):
    params = tuple(p for p in model.parameters() if p.requires_grad); model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, labels=ids, use_cache=False)
    return flat(torch.autograd.grad(out.loss, params), params).detach(), float(out.loss.detach())

def continuous_gradient(model, x, y, temperature, create_graph=True):
    params = tuple(p for p in model.parameters() if p.requires_grad); wte = model.get_input_embeddings().weight
    xin = F.softmax(x / temperature, dim=-1) @ wte; out = model(inputs_embeds=xin, use_cache=False)
    logits = out.logits[:, :-1].float(); target = F.softmax(y[:, 1:] / temperature, dim=-1)
    loss = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    gs = torch.autograd.grad(loss, params, create_graph=create_graph, allow_unused=True)
    return flat(gs, params), loss

def metrics(g, ref):
    gn, rn = g.norm(), ref.norm()
    return {"cosine": float(F.cosine_similarity(g[None], ref[None]).detach()),
            "relative_l2": float((g - ref).norm().detach() / rn.clamp_min(1e-12)),
            "norm_ratio": float((gn / rn.clamp_min(1e-12)).detach()),
            "normalized_mse": float(((g / gn.clamp_min(1e-12) - ref / rn.clamp_min(1e-12)) ** 2).mean().detach())}

def run_one(model_name, weight_kind, ids, a):
    device = torch.device(a.device); dtype = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Higher-order derivatives through CUDA flash/efficient attention are not
    # implemented. Eager attention is required because input/target matching
    # differentiates through the parameter gradients.
    if weight_kind == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(model_name), attn_implementation="eager"
        )
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model.to(device=device, dtype=dtype).eval(); ids = ids.to(device)
    params = tuple(p for p in model.parameters() if p.requires_grad); ref, ref_loss = reference_gradient(model, ids)
    vocab = model.config.vocab_size
    if a.init == "onehot":
        x = F.one_hot(ids, vocab).to(dtype=dtype) * 8.0; y = F.one_hot(ids, vocab).to(dtype=dtype) * 8.0
    else:
        x = torch.randn((*ids.shape, vocab), device=device, dtype=dtype); y = torch.randn_like(x)
    x, y = torch.nn.Parameter(x), torch.nn.Parameter(y); opt = torch.optim.AdamW([x, y], lr=a.lr)
    history = []; t0 = time.time()
    for step in range(a.steps + 1):
        g, loss = continuous_gradient(model, x, y, a.temperature); m = metrics(g, ref)
        m.update(step=step, continuous_loss=float(loss.detach())); history.append(m)
        if step == a.steps: break
        match_loss = 1.0 - (g @ ref) / (g.norm() * ref.norm()).clamp_min(1e-12)
        opt.zero_grad(set_to_none=True); match_loss.backward(); torch.nn.utils.clip_grad_norm_([x, y], 1.0); opt.step()
        if step == 0 or (step + 1) % 5 == 0: print(f"[{model_name} {weight_kind}] step {step:03d} cosine={m['cosine']:.5f} relL2={m['relative_l2']:.5f}", flush=True)
    result = {"model": model_name, "weights": weight_kind, "batch": ids.shape[0], "seq_len": ids.shape[1], "vocab": vocab,
              "reference_loss": ref_loss, "seconds": time.time() - t0, "history": history, "final": history[-1]}
    del model, x, y, ref, g
    if device.type == "cuda": torch.cuda.empty_cache()
    return result

def main():
    a = args(); random.seed(a.seed); torch.manual_seed(a.seed); set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.models[0]); ids = get_blocks(tok, a.dataset, a.config, a.split, a.batch_size, a.seq_len, a.max_docs)
    results = []
    for model_name in a.models:
        for weight_kind in a.weights:
            set_seed(a.seed); results.append(run_one(model_name, weight_kind, ids, a))
            r = results[-1]["final"]
            print(f"SUMMARY {model_name:16s} {weight_kind:11s} cosine={r['cosine']:.5f} relL2={r['relative_l2']:.5f} norm_ratio={r['norm_ratio']:.5f}", flush=True)
            Path(a.out).write_text(json.dumps({"args": vars(a), "results": results}, indent=2))
    print(f"Saved {len(results)} runs to {a.out}")

if __name__ == "__main__": main()

