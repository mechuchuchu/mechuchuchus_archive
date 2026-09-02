"""
ResNet-152 identity-detach gradient probe
==========================================

목적: skip connection의 identity term이 gradient norm을 얼마나 "부풀리는지" 확인.
     8층 sigmoid 실험(direction-only SGD로 vanishing 우회)과 대칭되는 질문 —
     "ResNet은 정말 vanishing을 해결했나, 아니면 identity가 죽은 신호를 가리고 있을 뿐인가?"

y = x + F(x)  =>  dL/dx = dL/dy + dL/dy * dF/dx
                          ^^^^^^   ^^^^^^^^^^^^^
                          identity   branch (진짜 local signal)

3가지 모드로 같은 input/target에 대해 forward+backward:
  - full         : 정상 (identity + branch 합쳐진 채로 관측)
  - branch_only  : identity.detach() -> F(x) branch gradient만 순수하게 관측
  - identity_only: F(x).detach()     -> skip만 남은 gradient (identity가 얼마나 큰지)

각 Bottleneck block의 conv1.weight grad norm을 layer 순서대로 기록하고,
full vs branch_only 사이 gradient direction cosine similarity도 같이 본다.

Usage:
    pip install torch torchvision matplotlib --index-url https://download.pytorch.org/whl/cpu  # CPU면
    python resnet152_identity_detach_grad_probe.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152
from torchvision.models.resnet import Bottleneck
import csv
torch.set_default_dtype(torch.float64)
# ---------------------------------------------------------------------------
# 1. Bottleneck.forward monkey-patch: MODE에 따라 identity 혹은 branch를 detach
# ---------------------------------------------------------------------------
MODE = "full"  # "full" | "branch_only" | "identity_only"  (전역 스위치)


def patched_forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    if MODE == "branch_only":
        identity = identity.detach()
    elif MODE == "identity_only":
        out = out.detach()
    # MODE == "full" -> 아무것도 안 건드림

    out = out + identity
    out = self.relu(out)
    return out


Bottleneck.forward = patched_forward

# ---------------------------------------------------------------------------
# 2. 모델 준비 (random init — 8층 sigmoid 실험이 "init 직후" 상황이었으니 맞춰줌)
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

model = resnet152(weights=None).to(device)
model.train()

# Bottleneck block들을 순서대로 수집 (layer1 -> layer4, 얕은 층 -> 깊은 층 순서 아님에 주의:
# ResNet의 layer1이 입력에 가장 가까운(=8층 sigmoid의 "1층") 블록임)
blocks = []
for name, module in model.named_modules():
    if isinstance(module, Bottleneck):
        blocks.append((name, module))

print(f"총 Bottleneck block 수: {len(blocks)} (ResNet-152 기준 3+8+36+3=50)")

# 같은 input/target 고정 (세 모드 비교 공정성 위해)
torch.manual_seed(42)
x = torch.randn(4, 3, 224, 224, device=device)
target = torch.randint(0, 1000, (4,), device=device)

# ---------------------------------------------------------------------------
# 3. 세 모드로 forward+backward, 각 block conv1.weight grad norm 기록
# ---------------------------------------------------------------------------
results = {}  # mode -> list of (block_name, grad_norm, grad_flat_tensor)

for mode in ["full", "branch_only", "identity_only"]:
    MODE = mode
    model.zero_grad(set_to_none=True)

    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    per_block = []
    for name, module in blocks:
        g = module.conv1.weight.grad
        if g is None:
            norm = 0.0
            flat = torch.zeros(1)
        else:
            norm = g.norm().item()
            flat = g.flatten().detach().cpu().clone()
        per_block.append((name, norm, flat))

    results[mode] = per_block
    print(f"[{mode}] loss={loss.item():.4f}  done.")

# ---------------------------------------------------------------------------
# 4. 정리: grad norm 비교 + full vs branch_only direction cosine similarity
# ---------------------------------------------------------------------------
rows = []
for i, (name, _, _) in enumerate(results["full"]):
    n_full = results["full"][i][1]
    n_branch = results["branch_only"][i][1]
    n_ident = results["identity_only"][i][1]

    g_full = results["full"][i][2]
    g_branch = results["branch_only"][i][2]

    if g_full.norm() > 0 and g_branch.norm() > 0:
        cos = torch.dot(g_full, g_branch) / (g_full.norm() * g_branch.norm())
        cos = cos.item()
    else:
        cos = float("nan")

    ratio = n_branch / n_full if n_full > 0 else float("nan")

    rows.append({
        "block_idx": i,
        "block_name": name,
        "norm_full": n_full,
        "norm_branch_only": n_branch,
        "norm_identity_only": n_ident,
        "branch_over_full_ratio": ratio,
        "cos_full_vs_branch": cos,
    })

# CSV로 저장
out_csv = "resnet152_grad_probe_results.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n결과 저장: {out_csv}")

# 콘솔에 요약 출력 (앞/중간/뒤 블록 몇 개씩)
print(f"\n{'idx':>4} {'block':<20} {'full':>12} {'branch_only':>12} {'identity_only':>14} {'branch/full':>12} {'cos(full,branch)':>18}")
for r in rows:
    print(f"{r['block_idx']:>4} {r['block_name']:<20} "
          f"{r['norm_full']:>12.3e} {r['norm_branch_only']:>12.3e} {r['norm_identity_only']:>14.3e} "
          f"{r['branch_over_full_ratio']:>12.3f} {r['cos_full_vs_branch']:>18.3f}")

# ---------------------------------------------------------------------------
# 5. 시각화 (선택 — matplotlib 없으면 이 블록만 스킵됨)
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    idx = [r["block_idx"] for r in rows]
    nf = [r["norm_full"] for r in rows]
    nb = [r["norm_branch_only"] for r in rows]
    ni = [r["norm_identity_only"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].semilogy(idx, nf, label="full", marker="o", markersize=3)
    axes[0].semilogy(idx, nb, label="branch_only (identity detached)", marker="s", markersize=3)
    axes[0].semilogy(idx, ni, label="identity_only (branch detached)", marker="^", markersize=3)
    axes[0].set_ylabel("grad norm (log scale)")
    axes[0].set_title("conv1.weight grad norm per Bottleneck block (block 0 = closest to the input)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    cos_vals = [r["cos_full_vs_branch"] for r in rows]
    axes[1].plot(idx, cos_vals, color="darkred", marker="o", markersize=3)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("cos(full, branch_only)")
    axes[1].set_xlabel("block index")
    axes[1].set_title("Directional similarity between full gradient and branch-only gradient")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("resnet152_grad_probe.png", dpi=150)
    print("\n그래프 저장: resnet152_grad_probe.png")
except ImportError:
    print("\n(matplotlib 없어서 그래프는 스킵. CSV 결과로 확인 가능)")

# ---------------------------------------------------------------------------
# 해석 가이드
# ---------------------------------------------------------------------------
# - branch_over_full_ratio가 깊은 층(idx 작을수록 입력에 가까움)에서 1보다 훨씬 작으면:
#   -> identity term이 gradient norm 대부분을 차지 -> "8층 sigmoid의 vanishing이
#      ResNet 내부에도 존재하지만 identity가 가리고 있다"는 가설 지지.
# - cos_full_vs_branch가 idx 낮은(깊은) 층에서 낮으면:
#   -> "관측되는 full gradient의 방향조차 branch의 진짜 방향과 다르다" -> 거짓말이
#      magnitude뿐 아니라 direction 레벨까지 번짐 (더 강한 결과).
# - identity_only norm이 거의 full norm과 같으면: 그 층은 사실상 F(x)=0 근처에
#   머물면서 skip만 타고 있다는 뜻 (residual이 거의 안 배우고 있음).
