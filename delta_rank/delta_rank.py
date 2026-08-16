#!/usr/bin/env python3
"""
Layer-wise weight-delta effective rank analysis.

    Delta = W_finetuned - W_base   (e.g. allenai/SERA-8B vs Qwen/Qwen3-8B)

Memory strategy (targets <= 16GB RAM):
  * safetensors mmap; exactly ONE tensor pair is materialized at a time.
  * svdvals() only -- no U/V unless --save-factors.
  * tall/wide matrices use streamed Gram accumulation:
        G = sum_blocks (dB^T dB)  ->  sigma = sqrt(eigvalsh(G))
    so embed_tokens / lm_head (151936x4096) never fully enter RAM.

Peak RSS is roughly:  2 * (block bytes) + min_dim^2 * 8   (~0.6 GB default)

Usage
-----
    pip install torch safetensors huggingface_hub numpy

    python delta_rank.py run \
        --base Qwen/Qwen3-8B \
        --ft   allenai/SERA-8B \
        --out  runs/sera8b

    python delta_rank.py report --out runs/sera8b

Disk: downloads both models (~16GB each) via the HF cache unless you pass
      --base-dir / --ft-dir pointing at already-downloaded snapshots.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

torch.set_grad_enabled(False)

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
ENERGIES = (0.5, 0.9, 0.95, 0.99)

CSV_FIELDS = [
    "name", "layer", "kind", "m", "n", "min_dim",
    "base_fro", "ft_fro", "delta_fro", "rel_fro",
    "spectral", "nuclear", "stable_rank", "erank_entropy",
    "r50", "r90", "r95", "r99", "numerical_rank",
    "sigma_max_ratio", "method", "sec",
]


# ------------------------------------------------------------------ IO ----

def resolve_model(repo_or_dir: str, revision: str | None = None) -> dict[str, str]:
    """Return {tensor_name -> local .safetensors path} without loading anything."""
    p = Path(repo_or_dir)
    if p.is_dir():
        idx = p / "model.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
            return {k: str(p / v) for k, v in wm.items()}
        single = p / "model.safetensors"
        if single.exists():
            with safe_open(str(single), framework="pt") as f:
                return {k: str(single) for k in f.keys()}
        shards = sorted(p.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no safetensors in {p}")
        out = {}
        for s in shards:
            with safe_open(str(s), framework="pt") as f:
                for k in f.keys():
                    out[k] = str(s)
        return out

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        idx = hf_hub_download(repo_or_dir, "model.safetensors.index.json",
                              revision=revision)
        wm = json.loads(Path(idx).read_text())["weight_map"]
        shard_paths = {
            s: hf_hub_download(repo_or_dir, s, revision=revision)
            for s in sorted(set(wm.values()))
        }
        return {k: shard_paths[v] for k, v in wm.items()}
    except EntryNotFoundError:
        single = hf_hub_download(repo_or_dir, "model.safetensors", revision=revision)
        with safe_open(single, framework="pt") as f:
            return {k: single for k in f.keys()}


class Reader:
    """Lazy safetensors reader with cached (mmap-backed) file handles."""

    def __init__(self, name2path: dict[str, str]):
        self.name2path = name2path
        self._handles: dict[str, object] = {}

    def _h(self, name):
        path = self.name2path[name]
        h = self._handles.get(path)
        if h is None:
            h = safe_open(path, framework="pt")
            self._handles[path] = h
        return h

    def keys(self):
        return self.name2path.keys()

    def slice(self, name):
        return self._h(name).get_slice(name)

    def shape(self, name) -> tuple[int, ...]:
        return tuple(self.slice(name).get_shape())

    def tensor(self, name) -> torch.Tensor:
        return self._h(name).get_tensor(name)


# ------------------------------------------------------------- spectra ----

def sigmas_dense(rb: Reader, rf: Reader, name: str,
                 device: str, want_factors: int) -> tuple[torch.Tensor, dict]:
    """Load full delta, exact SVD. Cheap for <= ~64M elements."""
    Wb = rb.tensor(name).to(torch.float32)
    Wf = rf.tensor(name).to(torch.float32)
    base_fro = float(torch.linalg.norm(Wb))
    ft_fro = float(torch.linalg.norm(Wf))
    D = (Wf - Wb)
    del Wb, Wf
    gc.collect()

    if device != "cpu":
        D = D.to(device)

    extra = {}
    if want_factors:
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        k = min(want_factors, S.numel())
        extra["U"] = U[:, :k].cpu().to(torch.float16).numpy()
        extra["S"] = S[:k].cpu().float().numpy()
        extra["Vh"] = Vh[:k].cpu().to(torch.float16).numpy()
        s = S.double().cpu()
        del U, S, Vh
    else:
        s = torch.linalg.svdvals(D).double().cpu()

    del D
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()
    return s, {"base_fro": base_fro, "ft_fro": ft_fro, **extra}


def sigmas_gram(rb: Reader, rf: Reader, name: str,
                block: int, device: str) -> tuple[torch.Tensor, dict]:
    """
    Streamed Gram accumulation -- never holds the whole matrix.

    Slices along the LONG axis so the accumulator is min_dim x min_dim.
    sigma_i = sqrt(lambda_i(G)).  fp64 accumulation keeps sigma accurate to
    ~1e-8 * sigma_max, far below anything that moves an energy threshold.
    """
    sb, sf = rb.slice(name), rf.slice(name)
    m, n = rb.shape(name)
    k = min(m, n)
    G = torch.zeros(k, k, dtype=torch.float64,
                    device=device if device != "cpu" else "cpu")
    base_fro2 = ft_fro2 = 0.0

    long_axis_rows = m >= n  # slice rows -> G = D^T D (n x n)
    L = m if long_axis_rows else n

    for i in range(0, L, block):
        j = min(i + block, L)
        if long_axis_rows:
            b = sb[i:j].to(torch.float32)
            f = sf[i:j].to(torch.float32)
        else:
            b = sb[:, i:j].to(torch.float32)
            f = sf[:, i:j].to(torch.float32)
        base_fro2 += float((b.double() ** 2).sum())
        ft_fro2 += float((f.double() ** 2).sum())
        d = (f - b).double()
        del b, f
        if device != "cpu":
            d = d.to(device)
        G += d.T @ d if long_axis_rows else d @ d.T
        del d

    ev = torch.linalg.eigvalsh(G).flip(0).clamp_min(0.0)
    del G
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()
    return ev.sqrt().cpu(), {"base_fro": base_fro2 ** 0.5, "ft_fro": ft_fro2 ** 0.5}


# ------------------------------------------------------------- metrics ----

def rank_metrics(s: torch.Tensor, m: int, n: int) -> dict:
    s = s.double()
    s = s[s > 0] if (s > 0).any() else s
    if s.numel() == 0:
        return {k: 0.0 for k in
                ("spectral", "nuclear", "stable_rank", "erank_entropy",
                 "r50", "r90", "r95", "r99", "numerical_rank",
                 "sigma_max_ratio", "delta_fro")}

    s2 = s * s
    total = s2.sum()
    cum = torch.cumsum(s2, 0) / total

    out = {}
    for e in ENERGIES:
        out[f"r{int(e * 100)}"] = int(torch.searchsorted(cum, e).item()) + 1

    # Roy & Vetterli effective rank: exp(H(p)), p_i = sigma_i / sum(sigma)
    p = s / s.sum()
    p = p[p > 0]
    out["erank_entropy"] = float(torch.exp(-(p * p.log()).sum()))

    out["spectral"] = float(s[0])
    out["nuclear"] = float(s.sum())
    out["delta_fro"] = float(total.sqrt())
    out["stable_rank"] = float(total / s2[0])          # ||A||_F^2 / ||A||_2^2

    tol = max(m, n) * float(torch.finfo(torch.float32).eps) * float(s[0])
    out["numerical_rank"] = int((s > tol).sum())
    out["sigma_max_ratio"] = float(s[0] / s[min(len(s) - 1, 0)]) if len(s) else 0.0
    out["sigma_max_ratio"] = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    return out


def classify(name: str) -> str:
    for tag in ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "embed_tokens", "lm_head", "q_norm", "k_norm",
                "input_layernorm", "post_attention_layernorm", "norm"):
        if tag in name:
            return tag
    return "other"


def layer_of(name: str) -> int:
    mt = LAYER_RE.search(name)
    return int(mt.group(1)) if mt else -1


# ----------------------------------------------------------------- run ----

def cmd_run(a):
    out_dir = Path(a.out)
    (out_dir / "sigma").mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"

    print("[*] resolving base ...", flush=True)
    rb = Reader(resolve_model(a.base_dir or a.base, a.base_rev))
    print("[*] resolving finetune ...", flush=True)
    rf = Reader(resolve_model(a.ft_dir or a.ft, a.ft_rev))

    common = sorted(set(rb.keys()) & set(rf.keys()),
                    key=lambda x: (layer_of(x), x))
    only_b = set(rb.keys()) - set(rf.keys())
    only_f = set(rf.keys()) - set(rb.keys())
    if only_b or only_f:
        print(f"[!] base-only: {sorted(only_b)[:5]}{'...' if len(only_b)>5 else ''}")
        print(f"[!] ft-only  : {sorted(only_f)[:5]}{'...' if len(only_f)>5 else ''}")

    done = set()
    if csv_path.exists() and a.resume:
        with csv_path.open() as f:
            done = {r["name"] for r in csv.DictReader(f)}
        print(f"[*] resume: {len(done)} already done")

    fh = csv_path.open("a" if done else "w", newline="")
    w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if not done:
        w.writeheader()

    skip_pat = re.compile(a.skip) if a.skip else None

    for idx, name in enumerate(common):
        if name in done:
            continue
        if skip_pat and skip_pat.search(name):
            continue
        if a.only and not re.search(a.only, name):
            continue

        shp_b, shp_f = rb.shape(name), rf.shape(name)
        if shp_b != shp_f:
            print(f"[!] shape mismatch, skip: {name} {shp_b} vs {shp_f}")
            continue

        t0 = time.time()
        row = {k: "" for k in CSV_FIELDS}
        row.update(name=name, layer=layer_of(name), kind=classify(name))

        # --- 1D params (RMSNorm gains): no SVD, just norms
        if len(shp_b) == 1:
            b = rb.tensor(name).to(torch.float64)
            f = rf.tensor(name).to(torch.float64)
            d = f - b
            row.update(m=shp_b[0], n=1, min_dim=1,
                       base_fro=float(b.norm()), ft_fro=float(f.norm()),
                       delta_fro=float(d.norm()),
                       rel_fro=float(d.norm() / (b.norm() + 1e-12)),
                       method="1d", sec=round(time.time() - t0, 2))
            w.writerow(row); fh.flush()
            print(f"[{idx+1}/{len(common)}] {name:60s} 1d rel={row['rel_fro']:.4e}")
            del b, f, d
            continue

        m, n = shp_b
        numel = m * n
        ratio = max(m, n) / min(m, n)
        use_gram = a.force_gram or numel > a.dense_max_numel or ratio > a.gram_ratio

        if use_gram:
            s, info = sigmas_gram(rb, rf, name, a.block, a.device)
            method = "gram"
            extra = {}
        else:
            s, info = sigmas_dense(rb, rf, name, a.device, a.save_factors)
            method = "svd"
            extra = {k: v for k, v in info.items() if k in ("U", "S", "Vh")}

        mets = rank_metrics(s, m, n)
        row.update(m=m, n=n, min_dim=min(m, n), method=method,
                   base_fro=info["base_fro"], ft_fro=info["ft_fro"],
                   rel_fro=mets["delta_fro"] / (info["base_fro"] + 1e-12),
                   sec=round(time.time() - t0, 2))
        row.update({k: mets[k] for k in
                    ("delta_fro", "spectral", "nuclear", "stable_rank",
                     "erank_entropy", "r50", "r90", "r95", "r99",
                     "numerical_rank", "sigma_max_ratio")})
        w.writerow(row); fh.flush()

        np.save(out_dir / "sigma" / f"{name}.npy", s.float().numpy())
        if extra:
            np.savez_compressed(out_dir / "sigma" / f"{name}.factors.npz", **extra)

        print(f"[{idx+1}/{len(common)}] {name:55s} {m}x{n} {method:4s} "
              f"rel={row['rel_fro']:.3e} r90={mets['r90']:4d}/{min(m,n)} "
              f"erank={mets['erank_entropy']:7.1f} sr={mets['stable_rank']:6.1f} "
              f"({row['sec']}s)")

        del s
        gc.collect()

    fh.close()
    print(f"\n[+] wrote {csv_path}")


# -------------------------------------------------------------- report ----

def cmd_report(a):
    out_dir = Path(a.out)
    rows = list(csv.DictReader((out_dir / "results.csv").open()))
    rows = [r for r in rows if r["method"] != "1d"]

    def fl(r, k):
        try:
            return float(r[k])
        except (ValueError, KeyError):
            return float("nan")

    kinds = ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"]

    print("\n=== per-module-type summary (transformer blocks) ===")
    hdr = f"{'kind':<12}{'n':>4}{'min_dim':>9}{'rel_fro':>12}{'r90':>8}" \
          f"{'r90/d':>8}{'erank':>10}{'stable_r':>10}"
    print(hdr); print("-" * len(hdr))
    for k in kinds:
        sub = [r for r in rows if r["kind"] == k]
        if not sub:
            continue
        d = float(sub[0]["min_dim"])
        print(f"{k:<12}{len(sub):>4}{int(d):>9}"
              f"{np.mean([fl(r,'rel_fro') for r in sub]):>12.3e}"
              f"{np.mean([fl(r,'r90') for r in sub]):>8.1f}"
              f"{np.mean([fl(r,'r90') for r in sub])/d:>8.3f}"
              f"{np.mean([fl(r,'erank_entropy') for r in sub]):>10.1f}"
              f"{np.mean([fl(r,'stable_rank') for r in sub]):>10.2f}")

    print("\n=== per-layer (mean over the 7 projections) ===")
    hdr = f"{'layer':<7}{'rel_fro':>12}{'r90/d':>10}{'erank/d':>10}{'stable_r':>10}"
    print(hdr); print("-" * len(hdr))
    layers = sorted({int(r["layer"]) for r in rows if int(r["layer"]) >= 0})
    for L in layers:
        sub = [r for r in rows if int(r["layer"]) == L and r["kind"] in kinds]
        if not sub:
            continue
        print(f"{L:<7}"
              f"{np.mean([fl(r,'rel_fro') for r in sub]):>12.3e}"
              f"{np.mean([fl(r,'r90')/fl(r,'min_dim') for r in sub]):>10.3f}"
              f"{np.mean([fl(r,'erank_entropy')/fl(r,'min_dim') for r in sub]):>10.3f}"
              f"{np.mean([fl(r,'stable_rank') for r in sub]):>10.2f}")

    for r in rows:
        if r["kind"] in ("embed_tokens", "lm_head"):
            print(f"\n{r['kind']}: {r['m']}x{r['n']} rel_fro={fl(r,'rel_fro'):.3e} "
                  f"r90={r['r90']} erank={fl(r,'erank_entropy'):.1f} "
                  f"stable_rank={fl(r,'stable_rank'):.2f}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        for k in kinds:
            sub = sorted([r for r in rows if r["kind"] == k],
                         key=lambda r: int(r["layer"]))
            if not sub:
                continue
            xs = [int(r["layer"]) for r in sub]
            ax[0].plot(xs, [fl(r, "rel_fro") for r in sub], label=k, marker=".")
            ax[1].plot(xs, [fl(r, "r90") / fl(r, "min_dim") for r in sub],
                       label=k, marker=".")
            ax[2].plot(xs, [fl(r, "stable_rank") for r in sub], label=k, marker=".")
        ax[0].set(xlabel="layer", ylabel=r"$\|\Delta\|_F/\|W\|_F$", yscale="log")
        ax[1].set(xlabel="layer", ylabel="r@90% / min_dim")
        ax[2].set(xlabel="layer", ylabel="stable rank", yscale="log")
        for x in ax:
            x.grid(alpha=.3)
        ax[0].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = out_dir / "delta_rank.png"
        fig.savefig(p, dpi=140)
        print(f"\n[+] plot -> {p}")


# ------------------------------------------------------------------ cli ---

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--base", default="Qwen/Qwen3-8B")
    r.add_argument("--ft", default="allenai/SERA-8B")
    r.add_argument("--base-dir", default=None, help="local snapshot dir (skips download)")
    r.add_argument("--ft-dir", default=None)
    r.add_argument("--base-rev", default=None)
    r.add_argument("--ft-rev", default=None)
    r.add_argument("--out", default="runs/delta")
    r.add_argument("--device", default="cpu", help="cpu | cuda")
    r.add_argument("--block", type=int, default=8192,
                   help="rows/cols per streamed Gram block")
    r.add_argument("--dense-max-numel", type=int, default=64_000_000,
                   help="above this, use streamed Gram instead of dense SVD")
    r.add_argument("--gram-ratio", type=float, default=3.0,
                   help="max(m,n)/min(m,n) above this -> Gram path")
    r.add_argument("--force-gram", action="store_true")
    r.add_argument("--save-factors", type=int, default=0,
                   help="also dump top-K U,S,Vh (dense path only)")
    r.add_argument("--only", default=None, help="regex filter on tensor name")
    r.add_argument("--skip", default=None, help="regex to exclude")
    r.add_argument("--resume", action="store_true", default=True)
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("report")
    p.add_argument("--out", default="runs/delta")
    p.add_argument("--plot", action="store_true")
    p.set_defaults(func=cmd_report)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
