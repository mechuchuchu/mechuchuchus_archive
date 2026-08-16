"""
저장된 synthetic data(M_vs_D_results.pt)에 대해, real gradient를 magnitude 기준으로
head(상위 X%)/tail(나머지)로 나누고, 각각에서 synthetic-vs-real cosine similarity를
따로 계산. "tail이 정보량은 작아도 정확도 기여는 크다"는 가설 검증용.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

RESULTS_PATH = "/mnt/user-data/outputs/M_vs_D_results.pt"

D = 28 * 28
HIDDEN = 128
NUM_CLASSES = 10
NUM_REGIONS = 16          # fresh independent random inits for this analysis
HEAD_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50]  # top-X% by |real gradient|

# ---------------- data ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).reshape(-1, D).to(DEVICE)
train_y = torch.tensor([train_ds[i][1] for i in range(len(train_ds))]).to(DEVICE)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, NUM_CLASSES)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def flat_grad(params):
    return torch.cat([g.reshape(-1) for g in params])


def real_gradient(model, x, y, batch_size=2000):
    model.zero_grad()
    accum = [torch.zeros_like(p) for p in model.parameters()]
    total = 0
    n = x.shape[0]
    for i in range(0, n, batch_size):
        xb, yb = x[i:i + batch_size], y[i:i + batch_size]
        out = model(xb)
        loss = F.cross_entropy(out, yb)
        grads = torch.autograd.grad(loss, model.parameters())
        w = xb.shape[0]
        for a, g in zip(accum, grads):
            a += g.detach() * w
        total += w
    return [a / total for a in accum]


def synthetic_gradient(model, synth_x, synth_y_logits):
    out = model(synth_x)
    log_p = F.log_softmax(out, dim=1)
    target_p = F.softmax(synth_y_logits, dim=1)
    loss = -(target_p * log_p).sum(dim=1).mean()
    grads = torch.autograd.grad(loss, model.parameters())
    return grads


def subset_cosine(g_synth_flat, g_real_flat, idx):
    a, b = g_synth_flat[idx], g_real_flat[idx]
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def subset_sign_agreement(g_synth_flat, g_real_flat, idx):
    """coordinate-wise: fraction where sign(synth) == sign(real). Magnitude-agnostic."""
    a, b = g_synth_flat[idx], g_real_flat[idx]
    agree = (torch.sign(a) == torch.sign(b)).float().mean().item()
    return agree


def analyze_M(M, synth_x, synth_y_logits):
    print(f"\n===== M = {M} =====")
    regions = [MLP().to(DEVICE) for _ in range(NUM_REGIONS)]

    # per-fraction cosine + sign-agreement accumulators
    head_cos = {f: [] for f in HEAD_FRACTIONS}
    tail_cos = {f: [] for f in HEAD_FRACTIONS}
    head_sign = {f: [] for f in HEAD_FRACTIONS}
    tail_sign = {f: [] for f in HEAD_FRACTIONS}
    full_cos = []
    full_sign = []

    for m in regions:
        g_real = real_gradient(m, train_x, train_y)
        g_real_flat = flat_grad(g_real)

        g_synth = synthetic_gradient(m, synth_x, synth_y_logits)
        g_synth_flat = flat_grad(g_synth)

        full_cos.append(F.cosine_similarity(
            g_synth_flat.unsqueeze(0), g_real_flat.unsqueeze(0)).item())
        full_sign.append((torch.sign(g_synth_flat) == torch.sign(g_real_flat)).float().mean().item())

        # sort indices by |real gradient| magnitude, descending
        order = torch.argsort(g_real_flat.abs(), descending=True)
        n_total = order.shape[0]

        for frac in HEAD_FRACTIONS:
            k = max(1, int(n_total * frac))
            head_idx = order[:k]
            tail_idx = order[k:]
            head_cos[frac].append(subset_cosine(g_synth_flat, g_real_flat, head_idx))
            tail_cos[frac].append(subset_cosine(g_synth_flat, g_real_flat, tail_idx))
            head_sign[frac].append(subset_sign_agreement(g_synth_flat, g_real_flat, head_idx))
            tail_sign[frac].append(subset_sign_agreement(g_synth_flat, g_real_flat, tail_idx))

    print(f"  full-vector cosine: mean={np.mean(full_cos):.4f} std={np.std(full_cos):.4f}")
    print(f"  full-vector sign agreement: mean={np.mean(full_sign):.4f} "
          f"(random baseline = 0.5000)")
    print(f"\n  {'head frac':>10} | {'head cosine':>14} | {'tail cosine':>14} | "
          f"{'head sign':>12} | {'tail sign':>12}")
    for frac in HEAD_FRACTIONS:
        h_c, t_c = np.mean(head_cos[frac]), np.mean(tail_cos[frac])
        h_s, t_s = np.mean(head_sign[frac]), np.mean(tail_sign[frac])
        print(f"  {frac*100:9.1f}% | {h_c:14.4f} | {t_c:14.4f} | "
              f"{h_s:12.4f} | {t_s:12.4f}")

    return {"full_cos": full_cos, "full_sign": full_sign,
            "head_cos": head_cos, "tail_cos": tail_cos,
            "head_sign": head_sign, "tail_sign": tail_sign}


if __name__ == "__main__":
    results = torch.load(RESULTS_PATH, map_location=DEVICE)

    all_out = {}
    for M, r in results.items():
        synth_x = r["synth_x"].to(DEVICE).requires_grad_(False)
        synth_y_logits = r["synth_y_logits"].to(DEVICE)
        all_out[M] = analyze_M(M, synth_x, synth_y_logits)

    print("\n===== head vs tail summary (top 5% shown) =====")
    print(f"{'M':>6} | {'head cos':>10} | {'tail cos':>10} | "
          f"{'head sign':>10} | {'tail sign':>10}")
    for M, out in all_out.items():
        hc = np.mean(out["head_cos"][0.05])
        tc = np.mean(out["tail_cos"][0.05])
        hs = np.mean(out["head_sign"][0.05])
        ts = np.mean(out["tail_sign"][0.05])
        print(f"{M:6d} | {hc:10.4f} | {tc:10.4f} | {hs:10.4f} | {ts:10.4f}")
