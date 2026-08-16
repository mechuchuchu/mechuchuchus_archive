"""
Scaling law smoke test — Dense vs Joint-ensemble capacity scaling
====================================================================

목적: "노드당 capacity(H)는 고정, k(노드 수)만 늘리는 joint training"이
"단일 dense 모델의 capacity를 늘리는 것"과 유사한 scaling exponent를
보이는지 확인. Bagging은 §5.5/토이실험에서 이미 bias floor에 갇힘이
확인됐으므로 이번 스윕에서는 제외(dense vs joint 두 축만).

Task: 32x32x3 random pixel image -> random label(1~1000) 순수 memorization.
구조 없는 input->label 매핑이라 "loss가 못 내려가는 이유 = 순전히 capacity
부족"으로 isolate됨 (LLM이 시퀀스를 마지막 토큰에 눌러담아 vocab으로
projection하는 것과 구조적으로 유사한 "무관한 고차원 입력을 저차원
결정으로 매�기"라는 점에서 이 toy가 그 압력을 최소 구성으로 재현).

- Dense family : 단일 CNN, width multiplier로 총 param 수 스윕
- Joint family : CNN member H개 폭 고정, k(멤버 수)를 스윕 -> 총 param
  수 = k * (per-member params). Ensemble logit = mean(logits), CE는
  ensemble logit에 직접 걸림 (joint training, bagging 아님).

결과는 JSON에 config별로 append 저장. FLOPs는 torch.profiler로 한 스텝
(forward+backward) 프로파일링 후 step 수를 곱해 총 training compute를
추정.

Usage:
    python scaling_sweep.py --family dense
    python scaling_sweep.py --family joint
    python scaling_sweep.py --plot   # JSON 읽어서 dense/joint 각각 별도 figure로 plot
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Config ----------------
IMG_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 1000

# 스모크 테스트 기본값 (로컬 3070에서 빠르게 도는 규모).
# 실제로 스케일 곡선을 제대로 뽑으려면 N_SAMPLES를 최대 1M까지,
# STEPS도 비례해서 키우는 걸 권장 (주석 참고).
N_SAMPLES = 100_000       # -> 1,000,000 까지 확장 가능 (메모리 되는 선에서)
BATCH_SIZE = 1024
STEPS = 2500             # -> 데이터/param 커지면 같이 늘릴 것
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
JSON_PATH = "scaling_results.json"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------- Data ----------------
SOFT_LABEL_TEMPERATURE = 1.0  # 낮을수록 target이 더 뾰족(=E가 작아짐), 높을수록 uniform에 가까움(=E가 커짐)


def make_dataset(n=N_SAMPLES, device=DEVICE):
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(n, CHANNELS, IMG_SIZE, IMG_SIZE, generator=g)
    # hard label 대신 soft target distribution: 샘플마다 고정된 random logit -> softmax
    # 이러면 loss가 0으로 안 내려가고, target distribution의 entropy만큼 irreducible floor(E)가 생김
    raw_logits = torch.randn(n, NUM_CLASSES, generator=g) / SOFT_LABEL_TEMPERATURE
    y_soft = F.softmax(raw_logits, dim=1)
    return x.to(device), y_soft.to(device)


def soft_cross_entropy(logits, target_probs):
    log_p = F.log_softmax(logits, dim=1)
    return -(target_probs * log_p).sum(dim=1).mean()


def irreducible_entropy_floor(target_probs):
    """target distribution 자체의 평균 entropy = 이론적 loss 하한(E). 모델이 target을
    완벽히 복제해도 CE는 0이 아니라 이 값까지만 내려갈 수 있음."""
    eps = 1e-12
    ent = -(target_probs * (target_probs + eps).log()).sum(dim=1).mean()
    return ent.item()


def iterate_batches(x, y, batch_size, steps, generator):
    n = x.shape[0]
    for _ in range(steps):
        # generator가 CPU 고정이므로 인덱스도 CPU에서 뽑고, x가 CUDA면 인덱싱 시 자동으로 맞춰짐
        idx = torch.randint(0, n, (batch_size,), generator=generator)
        yield x[idx.to(x.device)], y[idx.to(x.device)]


# ---------------- Model ----------------
class SmallCNN(nn.Module):
    """width multiplier로 capacity 조절되는 작은 CNN. per-member 모델로도, dense 단일 모델로도 씀."""

    def __init__(self, width=16, num_classes=NUM_CLASSES):
        super().__init__()
        w = width
        self.conv = nn.Sequential(
            nn.Conv2d(CHANNELS, w, 3, stride=2, padding=1), nn.ReLU(),   # 32 -> 16
            nn.Conv2d(w, w * 2, 3, stride=2, padding=1), nn.ReLU(),      # 16 -> 8
            nn.Conv2d(w * 2, w * 4, 3, stride=2, padding=1), nn.ReLU(),  # 8 -> 4
        )
        self.head = nn.Linear(w * 4 * 4 * 4, num_classes)

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(1)
        return self.head(h)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


class JointEnsemble(nn.Module):
    """k개 SmallCNN(각자 width=H)을 묶어서 logit 평균을 ensemble output으로."""

    def __init__(self, k, width, num_classes=NUM_CLASSES):
        super().__init__()
        self.members = nn.ModuleList([SmallCNN(width, num_classes) for _ in range(k)])

    def forward(self, x):
        logits = torch.stack([m(x) for m in self.members], dim=0)  # (k, B, C)
        return logits.mean(dim=0)


# ---------------- FLOPs profiling ----------------
def profile_step_flops(model, x_batch, y_batch, opt):
    """torch.profiler로 forward+backward 한 스텝의 FLOPs를 측정."""
    model.train()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
        + ([torch.profiler.ProfilerActivity.CUDA] if DEVICE == "cuda" else []),
        with_flops=True,
    ) as prof:
        opt.zero_grad()
        logits = model(x_batch)
        loss = soft_cross_entropy(logits, y_batch)
        loss.backward()
        opt.step()

    total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
    return total_flops


# ---------------- Training loop ----------------
def train_config(model, x, y, steps=STEPS, batch_size=BATCH_SIZE, lr=LR, tag=""):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # torch.randint(..., generator=...)가 CPU generator를 요구하므로 device와 무관하게 항상 cpu 고정
    gen = torch.Generator(device="cpu").manual_seed(SEED + 1)

    floor_E = irreducible_entropy_floor(y)  # target distribution의 entropy = 이론적 loss 하한

    loss_curve = []  # (step, loss) 서브샘플
    step_flops = None
    t0 = time.time()

    for step, (xb, yb) in enumerate(iterate_batches(x, y, batch_size, steps, gen)):
        if step == 10:  # 워밍업 후 한 스텝만 프로파일링 (early step은 캐시 미스 등으로 노이즈 큼)
            step_flops = profile_step_flops(model, xb, yb, opt)

        opt.zero_grad()
        logits = model(xb)
        loss = soft_cross_entropy(logits, yb)
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == steps - 1:
            loss_curve.append((step, loss.item()))
            print(f"[{tag}] step {step:5d}/{steps}  loss={loss.item():.5f}  (floor E={floor_E:.4f})", end="\r")

    elapsed = time.time() - t0
    total_params = count_params(model)
    total_flops_estimate = (step_flops * steps) if step_flops else None

    return {
        "final_loss": loss_curve[-1][1],
        "loss_curve": loss_curve,
        "total_params": total_params,
        "step_flops": step_flops,
        "total_flops_estimate": total_flops_estimate,
        "elapsed_sec": elapsed,
        "irreducible_floor_E": floor_E,
    }


def append_result(record):
    results = []
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    results.append(record)
    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {JSON_PATH} ({len(results)} records)")


# ---------------- Sweeps ----------------
# Dense: width multiplier 스윕 (총 param 수가 자연히 커짐)
DENSE_WIDTHS = [4, 8, 16, 32, 64, 96]

# Joint: per-member width는 고정(작게, "노드 하나의 capacity"에 해당),
# k(멤버 수)를 스윕해서 총 param 수를 키움 — 이게 핵심 비교축.
JOINT_MEMBER_WIDTH = 8
JOINT_K_LIST = [1, 2, 3, 5, 8, 13, 20]


def run_dense_sweep(x, y):
    for width in DENSE_WIDTHS:
        model = SmallCNN(width=width)
        record = train_config(model, x, y, tag=f"dense w={width}")
        record.update({"family": "dense", "width": width, "k": 1})
        append_result(record)


def run_joint_sweep(x, y):
    for k in JOINT_K_LIST:
        model = JointEnsemble(k=k, width=JOINT_MEMBER_WIDTH)
        record = train_config(model, x, y, tag=f"joint k={k} H={JOINT_MEMBER_WIDTH}")
        record.update({"family": "joint", "width": JOINT_MEMBER_WIDTH, "k": k})
        append_result(record)


# ---------------- Plotting (dense/joint 별개 figure) ----------------
def fit_power_law_with_floor(param_counts, losses, floor_E):
    """L(P) = A * P^(-alpha) + E  (E는 실측 irreducible entropy floor로 고정).
    scipy 있으면 nonlinear least squares, 없으면 (loss - E)에 대해 log-log 선형 fit으로 근사."""
    param_counts = np.asarray(param_counts, dtype=float)
    losses = np.asarray(losses, dtype=float)
    residual = np.clip(losses - floor_E, 1e-8, None)  # L - E > 0 이어야 power law로 fit 가능

    try:
        from scipy.optimize import curve_fit

        def model_fn(p, A, alpha):
            return A * p ** (-alpha) + floor_E

        (A, alpha), _ = curve_fit(
            model_fn, param_counts, losses, p0=[losses.max(), 0.3], maxfev=10000
        )
        return A, alpha
    except ImportError:
        # log-log 선형 근사: log(L - E) = log(A) - alpha * log(P)
        log_p = np.log(param_counts)
        log_r = np.log(residual)
        neg_alpha, log_A = np.polyfit(log_p, log_r, 1)
        return np.exp(log_A), -neg_alpha


def plot_family(records, family, out_path):
    import matplotlib.pyplot as plt

    fam_records = [r for r in records if r["family"] == family]
    fam_records.sort(key=lambda r: r["total_params"])
    params = np.array([r["total_params"] for r in fam_records])
    losses = np.array([r["final_loss"] for r in fam_records])
    # 모든 config가 같은 dataset(=같은 soft label)으로 생성됐으므로 floor_E는 공통값
    floor_E = fam_records[0]["irreducible_floor_E"]

    A, alpha = fit_power_law_with_floor(params, losses, floor_E)

    fig, ax = plt.subplots(figsize=(6, 5))
    color = "tab:blue" if family == "dense" else "tab:red"
    ax.scatter(params, losses, color=color, label=f"{family} (measured)")
    fit_x = np.linspace(params.min(), params.max(), 200)
    fit_y = A * fit_x ** (-alpha) + floor_E
    ax.plot(fit_x, fit_y, "k--", label=f"L = {A:.3g}·P^(-{alpha:.3f}) + E")
    ax.axhline(floor_E, color="gray", linestyle=":", alpha=0.7, label=f"E (irreducible floor) = {floor_E:.4f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total parameters")
    ax.set_ylabel("Final train loss (soft CE)")
    ax.set_title(f"{family} scaling +E: random-image soft-label memorization")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[saved] {out_path}  (A={A:.4g}, alpha={alpha:.4f}, E={floor_E:.4f})")


def make_plots():
    with open(JSON_PATH, "r") as f:
        records = json.load(f)
    plot_family(records, "dense", "scaling_dense.png")
    plot_family(records, "joint", "scaling_joint.png")


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["dense", "joint"], default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()

    if args.plot:
        make_plots()
    else:
        STEPS = args.steps
        x, y = make_dataset(n=args.n_samples)
        if args.family == "dense":
            run_dense_sweep(x, y)
        elif args.family == "joint":
            run_joint_sweep(x, y)
        else:
            print("Specify --family dense | joint, or --plot")
