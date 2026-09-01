"""
probe_experiment.py가 남긴 results.jsonl을 집계하고 bar chart로 시각화.

사용:
  python analyze_results.py                          # results.jsonl 읽어서 요약 + 그림 저장
  python analyze_results.py --in results.jsonl --out figs/
  python analyze_results.py --metric probe_best_test_acc
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARM_ORDER = ["none", "noise", "real"]
ARM_LABEL = {
    "none": "A: random init\n(no pretrain)",
    "noise": "B: noise\nmemorized",
    "real": "C: real-data\npretrained",
}
ARM_COLOR = {"none": "#8c8c8c", "noise": "#d1495b", "real": "#3d5a80"}


def load_results(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  경고: {i}번째 줄 파싱 실패, 건너뜀")
    if not rows:
        raise SystemExit(f"{path}에 유효한 결과가 없습니다.")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """arch x pretrain x head 조합별로 seed 간 평균/표준편차/개수."""
    g = df.groupby(["arch", "pretrain", "head"])[metric]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    out["std"] = out["std"].fillna(0.0)  # seed 1개면 std=NaN
    out["pretrain"] = pd.Categorical(out["pretrain"], categories=ARM_ORDER, ordered=True)
    return out.sort_values(["arch", "head", "pretrain"])


def plot_main(summary: pd.DataFrame, metric: str, outdir: str, head_filter: str):
    """메인 그림: x축=arm, 그룹=arch, y축=probe test acc, error bar=seed std."""
    sub = summary[summary["head"] == head_filter]
    if sub.empty:
        print(f"  head={head_filter} 결과가 없어 메인 그림은 건너뜁니다.")
        return

    archs = sorted(sub["arch"].unique())
    arms = [a for a in ARM_ORDER if a in set(sub["pretrain"])]

    x = np.arange(len(arms))
    width = 0.8 / max(len(archs), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, arch in enumerate(archs):
        means, stds, ns = [], [], []
        for arm in arms:
            row = sub[(sub["arch"] == arch) & (sub["pretrain"] == arm)]
            means.append(row["mean"].iloc[0] if len(row) else np.nan)
            stds.append(row["std"].iloc[0] if len(row) else 0.0)
            ns.append(int(row["n"].iloc[0]) if len(row) else 0)

        pos = x + (i - (len(archs) - 1) / 2) * width
        bars = ax.bar(pos, means, width * 0.9, yerr=stds, capsize=4,
                      label=arch, alpha=0.9)
        for b, m, n in zip(bars, means, ns):
            if not np.isnan(m):
                ax.text(b.get_x() + b.get_width() / 2, m + 0.012,
                        f"{m:.3f}\n(n={n})", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.1, ls="--", lw=1, color="k", alpha=0.5)
    ax.text(ax.get_xlim()[1], 0.105, "chance (0.10)", ha="right", va="bottom",
            fontsize=8, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[a] for a in arms])
    ax.set_ylabel("CIFAR-10 test accuracy (frozen backbone)")
    ax.set_title(f"Frozen-backbone probe: {metric}  (head={head_filter})")
    ax.set_ylim(0, 1.0)
    ax.legend(title="arch")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(outdir, "main_arm_comparison.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  저장: {path}")


def plot_per_seed(df: pd.DataFrame, metric: str, outdir: str):
    """seed별 산점도 — bar chart의 평균이 몇 개 점에서 나온 건지 정직하게 보여줌."""
    archs = sorted(df["arch"].unique())
    fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 4), squeeze=False)

    for ax, arch in zip(axes[0], archs):
        sub = df[df["arch"] == arch]
        arms = [a for a in ARM_ORDER if a in set(sub["pretrain"])]
        for j, arm in enumerate(arms):
            vals = sub[sub["pretrain"] == arm][metric].values
            jitter = (np.random.rand(len(vals)) - 0.5) * 0.15
            ax.scatter(np.full(len(vals), j) + jitter, vals,
                       color=ARM_COLOR[arm], s=55, alpha=0.85, edgecolor="white")
            if len(vals):
                ax.hlines(vals.mean(), j - 0.25, j + 0.25,
                          color=ARM_COLOR[arm], lw=2)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([ARM_LABEL[a] for a in arms], fontsize=8)
        ax.set_title(arch)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-seed results (line = mean)")
    fig.tight_layout()
    path = os.path.join(outdir, "per_seed_scatter.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  저장: {path}")


def plot_noise_scaling(df: pd.DataFrame, metric: str, outdir: str):
    """noise arm에서 --noise-samples를 스윕했다면 그 효과를 따로 시각화."""
    sub = df[df["pretrain"] == "noise"]
    if sub.empty or sub["noise_samples"].nunique() < 2:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for arch in sorted(sub["arch"].unique()):
        s = sub[sub["arch"] == arch].groupby("noise_samples")[metric].agg(["mean", "std", "count"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0),
                    marker="o", capsize=4, label=arch)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("# memorized noise samples")
    ax.set_ylabel(metric)
    ax.set_title("Does memorizing more noise damage features more?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(outdir, "noise_scaling.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  저장: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", default="results.jsonl")
    p.add_argument("--out", dest="outdir", default="figs")
    p.add_argument("--metric", default="probe_final_test_acc",
                   choices=["probe_final_test_acc", "probe_best_test_acc"])
    p.add_argument("--head", default="linear", help="메인 그림에 쓸 head 종류")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_results(args.infile)
    print(f"{len(df)}개 런 로드됨\n")

    summary = summarize(df, args.metric)
    print("=== 요약 (seed 평균 ± std) ===")
    for _, r in summary.iterrows():
        print(f"  {r['arch']:9s} {r['pretrain']:6s} head={r['head']:6s} "
              f"{r['mean']:.4f} ± {r['std']:.4f}  (n={int(r['n'])})")

    csv_path = os.path.join(args.outdir, "summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\n  저장: {csv_path}")

    # seed 수가 적으면 경고 (random init 분산이 크므로)
    if summary["n"].min() < 3:
        print("\n  경고: seed 수가 3개 미만인 조합이 있습니다. "
              "random init 분산 때문에 결론 내리기 이릅니다.")

    print("\n=== 그림 생성 ===")
    plot_main(summary, args.metric, args.outdir, args.head)
    plot_per_seed(df, args.metric, args.outdir)
    plot_noise_scaling(df, args.metric, args.outdir)


if __name__ == "__main__":
    main()
