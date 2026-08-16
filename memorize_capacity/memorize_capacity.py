# memorize_capacity.py
#
# Random-vector memorization capacity:
#   FP32 vs BF16 vs INT8-STE vs INT4-STE
#
# All models have EXACTLY the same number of trainable parameters.
#
# pip install torch pandas tqdm
#
# Example:
#   python memorize_capacity.py
#
# GPU:
#   python memorize_capacity.py --device cuda
#
# Faster experiment:
#   python memorize_capacity.py --max_n 4096 --steps 3000 --seeds 3

import argparse
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# STE Quantization
# ============================================================

def ste_quantize(x, bits):
    """
    Symmetric per-tensor uniform quantization.

    Forward:
        x -> quantized(x)

    Backward:
        dx_quantized / dx = 1

    i.e. Straight-Through Estimator.
    """
    if bits == 32:
        return x

    if bits == 16:
        # BF16 simulation in forward pass.
        # Gradient is passed through the cast.
        return x.to(torch.bfloat16).to(x.dtype)

    assert bits in [4, 8]

    qmax = 2 ** (bits - 1) - 1

    # Detach scale so scale itself does not become a
    # learned differentiable quantity.
    scale = x.detach().abs().amax() / qmax

    scale = torch.clamp(scale, min=1e-8)

    q = torch.round(x / scale)
    q = torch.clamp(q, -qmax - 1, qmax)

    x_q = q * scale

    # STE:
    # forward  -> x_q
    # backward -> x
    return x + (x_q - x).detach()


# ============================================================
# Quantized Linear
# ============================================================

class QuantLinear(nn.Module):

    def __init__(self, in_features, out_features, bits):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        self.bias = nn.Parameter(
            torch.empty(out_features)
        )

        self.bits = bits

        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x):

        w = ste_quantize(self.weight, self.bits)

        # Bias is also quantized for INT4/INT8/BF16.
        b = ste_quantize(self.bias, self.bits)

        return F.linear(x, w, b)


# ============================================================
# Fixed-size MLP
# ============================================================

class MemorizationMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        bits,
        depth=3,
    ):
        super().__init__()

        layers = []

        dims = [
            input_dim
        ] + [
            hidden_dim
        ] * (depth - 1) + [
            output_dim
        ]

        for i in range(len(dims) - 1):

            layers.append(
                QuantLinear(
                    dims[i],
                    dims[i + 1],
                    bits,
                )
            )

            if i != len(dims) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Parameter count
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


# ============================================================
# Random memorization dataset
# ============================================================

def make_dataset(
    n,
    input_dim,
    output_dim,
    device,
):

    # Random keys
    x = torch.randn(
        n,
        input_dim,
        device=device,
    )

    # Random targets
    y = torch.randn(
        n,
        output_dim,
        device=device,
    )

    return x, y


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, x, y):

    pred = model(x)

    mse = F.mse_loss(
        pred,
        y,
    ).item()

    target_energy = (
        y.pow(2).mean().item()
        + 1e-12
    )

    normalized_mse = mse / target_energy

    return mse, normalized_mse


# ============================================================
# Train one memorization experiment
# ============================================================

def train_one(
    n,
    bits,
    args,
    seed,
):

    set_seed(seed)

    device = args.device

    x, y = make_dataset(
        n=n,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        device=device,
    )

    model = MemorizationMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        bits=bits,
        depth=args.depth,
    ).to(device)

    # IMPORTANT:
    # Every precision uses exactly the same architecture.
    #
    # Therefore parameter count is identical.
    param_count = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )

    best_nmse = float("inf")

    for step in range(args.steps):

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model(x)

        loss = F.mse_loss(
            pred,
            y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )

        optimizer.step()

        if step % args.eval_every == 0:

            _, nmse = evaluate(
                model,
                x,
                y,
            )

            best_nmse = min(
                best_nmse,
                nmse,
            )

            # Early stopping
            if nmse < args.threshold:
                break

    final_mse, final_nmse = evaluate(
        model,
        x,
        y,
    )

    success = (
        final_nmse < args.threshold
    )

    return {
        "N": n,
        "bits": bits,
        "seed": seed,
        "params": param_count,
        "mse": final_mse,
        "nmse": final_nmse,
        "best_nmse": best_nmse,
        "steps": step + 1,
        "success": success,
    }


# ============================================================
# Find capacity
# ============================================================

def find_capacity(
    bits,
    args,
):

    results = []

    # N sweep.
    #
    # Geometric sweep is much faster than
    # 1,2,3,...,N.
    n_values = []

    n = args.min_n

    while n <= args.max_n:

        n_values.append(n)

        n = int(
            math.ceil(
                n * args.growth
            )
        )

        if n == n_values[-1]:
            n += 1

    for n in n_values:

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"bits={bits}, N={n}"
        )

        seed_results = []

        for seed in args.seeds:

            result = train_one(
                n=n,
                bits=bits,
                args=args,
                seed=seed,
            )

            seed_results.append(
                result
            )

            print(
                f"  seed={seed:3d} "
                f"NMSE={result['nmse']:.4e} "
                f"steps={result['steps']:5d} "
                f"{'SUCCESS' if result['success'] else 'FAIL'}"
            )

            results.append(result)

        success_rate = np.mean(
            [
                r["success"]
                for r in seed_results
            ]
        )

        print(
            f"  success rate = "
            f"{success_rate:.2f}"
        )

        # We call capacity reached when
        # success rate falls below required rate.
        #
        # But continue the sweep so we can see
        # the entire curve.
    
    return results


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Random vector dimensions
    parser.add_argument(
        "--input_dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--output_dim",
        type=int,
        default=32,
    )

    # FIXED architecture
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=3,
    )

    # Training
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--eval_every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
    )

    # "memorized" definition
    #
    # normalized MSE:
    #
    #       MSE(pred, target)
    # ----------------------------
    #       mean(target^2)
    #
    # threshold = 1e-3 means
    # prediction error is ~0.1% of target energy.
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
    )

    # N sweep
    parser.add_argument(
        "--min_n",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max_n",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--growth",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )

    args = parser.parse_args()

    print("\nDevice:", args.device)

    print(
        "\nFixed architecture:"
    )

    dummy = MemorizationMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        bits=32,
        depth=args.depth,
    )

    print(
        dummy
    )

    params = count_parameters(
        dummy
    )

    print(
        f"\nTrainable parameters: "
        f"{params:,}"
    )

    print(
        f"Input dimension : {args.input_dim}"
    )

    print(
        f"Output dimension: {args.output_dim}"
    )

    print(
        f"Hidden dimension: {args.hidden_dim}"
    )

    print(
        f"Depth           : {args.depth}"
    )

    print(
        f"\nMemorization threshold "
        f"(normalized MSE): "
        f"{args.threshold}"
    )

    # ========================================================
    # Run experiments
    # ========================================================

    all_results = []

    # 4-bit
    all_results += find_capacity(
        bits=4,
        args=args,
    )

    # 8-bit
    all_results += find_capacity(
        bits=8,
        args=args,
    )

    # BF16
    all_results += find_capacity(
        bits=16,
        args=args,
    )

    # FP32
    all_results += find_capacity(
        bits=32,
        args=args,
    )

    df = pd.DataFrame(
        all_results
    )

    # ========================================================
    # Save raw results
    # ========================================================

    df.to_csv(
        "memorization_results.csv",
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = (
        df
        .groupby(
            ["bits", "N"]
        )
        .agg(
            success_rate=(
                "success",
                "mean",
            ),
            mean_nmse=(
                "nmse",
                "mean",
            ),
            std_nmse=(
                "nmse",
                "std",
            ),
            params=(
                "params",
                "first",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        "memorization_summary.csv",
        index=False,
    )

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # Capacity
    # ========================================================

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "MAX MEMORIZED N"
    )

    print(
        "=" * 80
    )

    required_success_rate = 0.67

    for bits in [
        4,
        8,
        16,
        32,
    ]:

        tmp = summary[
            summary["bits"] == bits
        ]

        successful = tmp[
            tmp["success_rate"]
            >= required_success_rate
        ]

        if len(successful) == 0:

            print(
                f"{bits:>2}-bit : "
                f"< {args.min_n}"
            )

        else:

            max_n = successful[
                "N"
            ].max()

            print(
                f"{bits:>2}-bit : "
                f"{max_n}"
            )

    print(
        "\nResults:"
    )

    print(
        "  memorization_results.csv"
    )

    print(
        "  memorization_summary.csv"
    )


if __name__ == "__main__":
    main()
