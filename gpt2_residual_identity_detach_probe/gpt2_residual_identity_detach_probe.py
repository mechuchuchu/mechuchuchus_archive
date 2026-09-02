"""
GPT-2 residual identity-detach gradient probe  (fixed)
======================================================

ResNet-152 실험의 GPT-2 버전.
GPT-2 block은 residual이 layer당 2개:

    h = h + Attn(LN1(h))     <- residual #1 (attn)
    h = h + MLP(LN2(h))      <- residual #2 (mlp)

모드:
  - full          : 정상
  - branch_only   : 모든 residual identity를 detach (gradient는 branch로만)
  - identity_only : 모든 branch 출력을 detach (gradient는 identity로만)

--- 원본 에러 원인 ---
TypeError: layer_norm(): argument 'input' must be Tensor, not tuple

최신 transformers의 GPT2Block.forward는 tuple이 아니라 **hidden_states 텐서 하나만**
리턴한다 (GPT2Model이 `hidden_states = block(...)` 로 받음).
원본 패치는 옛 API처럼 tuple을 리턴 -> 다음 block이 tuple을 받아 ln_1에서 터짐.
또 signature도 바뀌었다 (layer_past/head_mask/output_attentions 없음,
past_key_values/cache_position 사용, positional 호출).

--- 수정 방식 ---
버전에 하드코딩하지 않고,
  1) 패치 전에 forward hook으로 "block이 tuple을 리턴하는지" 실측
  2) 원본 forward의 signature에 bind해서 인자를 이름으로 복원
  3) self.attn 의 signature에 맞는 것만 골라서 전달
하도록 함. 구/신 API 양쪽에서 동작.
"""

import csv
import inspect

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

# ---------------------------------------------------------------------------
# 0. 모델 준비
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

config = GPT2Config()  # gpt2-small: 12 layer, 768 dim, 12 head
try:
    # fp64 + sdpa/flash 조합은 지원 안 되므로 eager로 고정
    config._attn_implementation = "eager"
except Exception:
    pass

model = GPT2LMHeadModel(config).to(device).double()
model.eval()  # dropout 등 stochastic 요소 제거

blocks = [(n, m) for n, m in model.named_modules() if isinstance(m, GPT2Block)]
print(f"총 GPT2Block 수: {len(blocks)}")

torch.manual_seed(42)
BATCH, SEQ = 4, 32
input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=device)
labels = input_ids.clone()

# ---------------------------------------------------------------------------
# 1. 이 버전의 block이 tuple을 리턴하는지 실측 (패치 전에)
# ---------------------------------------------------------------------------
_RETURNS_TUPLE = False


def _probe(module, args, output):
    global _RETURNS_TUPLE
    _RETURNS_TUPLE = isinstance(output, tuple)


_h = blocks[0][1].register_forward_hook(_probe)
with torch.no_grad():
    model(input_ids=input_ids, use_cache=False)
_h.remove()
print(f"block forward returns tuple: {_RETURNS_TUPLE}")

# ---------------------------------------------------------------------------
# 2. GPT2Block.forward monkey-patch
# ---------------------------------------------------------------------------
MODE = "full"  # "full" | "branch_only" | "identity_only"

_ORIG_BLOCK_FORWARD = GPT2Block.forward
_BLOCK_SIG = inspect.signature(_ORIG_BLOCK_FORWARD)


def _split_kwargs(sig, self_obj, hidden_states, args, kwargs):
    """호출 인자를 (named, var_kw) 로 복원. positional 호출도 이름으로 되살림."""
    bound = sig.bind(self_obj, hidden_states, *args, **kwargs)
    bound.apply_defaults()
    named, extra = {}, {}
    for k, v in bound.arguments.items():
        if k in ("self", "hidden_states"):
            continue
        p = sig.parameters[k]
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            extra.update(v)
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        else:
            named[k] = v
    return named, extra


def patched_forward(self, hidden_states, *args, **kwargs):
    named, extra = _split_kwargs(_BLOCK_SIG, self, hidden_states, args, kwargs)

    if named.get("encoder_hidden_states") is not None:
        raise NotImplementedError("이 probe는 cross-attention 미지원")

    attn_params = inspect.signature(self.attn.forward).parameters
    attn_kw = {k: v for k, v in named.items() if k in attn_params}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in attn_params.values()):
        attn_kw.update(extra)
    else:
        attn_kw.update({k: v for k, v in extra.items() if k in attn_params})

    # ---- residual #1 : attention ----
    residual = hidden_states
    attn_outputs = self.attn(self.ln_1(hidden_states), **attn_kw)
    if isinstance(attn_outputs, tuple):
        attn_output, rest = attn_outputs[0], attn_outputs[1:]
    else:
        attn_output, rest = attn_outputs, ()

    if MODE == "branch_only":
        hidden_states = attn_output + residual.detach()
    elif MODE == "identity_only":
        hidden_states = attn_output.detach() + residual
    else:
        hidden_states = attn_output + residual

    # ---- residual #2 : mlp ----
    residual = hidden_states
    mlp_output = self.mlp(self.ln_2(hidden_states))

    if MODE == "branch_only":
        hidden_states = residual.detach() + mlp_output
    elif MODE == "identity_only":
        hidden_states = residual + mlp_output.detach()
    else:
        hidden_states = residual + mlp_output

    # ---- 이 버전의 리턴 규약에 맞춰 돌려주기 ----
    if not _RETURNS_TUPLE:
        return hidden_states
    if named.get("use_cache", False):
        return (hidden_states,) + rest
    return (hidden_states,) + rest[1:]


GPT2Block.forward = patched_forward

# ---------------------------------------------------------------------------
# 3. 세 모드로 forward+backward
# ---------------------------------------------------------------------------
results = {}

for mode in ["full", "branch_only", "identity_only"]:
    MODE = mode
    model.zero_grad(set_to_none=True)

    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    loss = out.loss
    loss.backward()

    per_block = []
    for name, module in blocks:
        g_attn = module.attn.c_attn.weight.grad
        g_mlp = module.mlp.c_fc.weight.grad

        n_attn = g_attn.norm().item() if g_attn is not None else 0.0
        n_mlp = g_mlp.norm().item() if g_mlp is not None else 0.0

        zero = torch.zeros(1, dtype=torch.float64)
        f_attn = g_attn.flatten().detach().cpu().clone() if g_attn is not None else zero
        f_mlp = g_mlp.flatten().detach().cpu().clone() if g_mlp is not None else zero

        per_block.append((name, n_attn, n_mlp, f_attn, f_mlp))

    results[mode] = per_block
    print(f"[{mode}] loss={loss.item():.4f}  done.")

# ---------------------------------------------------------------------------
# 4. 정리
# ---------------------------------------------------------------------------
def cos(a, b):
    if a.norm() > 0 and b.norm() > 0:
        return (torch.dot(a, b) / (a.norm() * b.norm())).item()
    return float("nan")


rows = []
for i, (name, *_) in enumerate(results["full"]):
    _, nf_a, nf_m, gf_a, gf_m = results["full"][i]
    _, nb_a, nb_m, gb_a, gb_m = results["branch_only"][i]
    _, ni_a, ni_m, _, _ = results["identity_only"][i]

    rows.append({
        "layer_idx": i,
        "layer_name": name,
        "attn_norm_full": nf_a,
        "attn_norm_branch_only": nb_a,
        "attn_norm_identity_only": ni_a,
        "attn_branch_over_full": (nb_a / nf_a) if nf_a > 0 else float("nan"),
        "attn_cos_full_branch": cos(gf_a, gb_a),
        "mlp_norm_full": nf_m,
        "mlp_norm_branch_only": nb_m,
        "mlp_norm_identity_only": ni_m,
        "mlp_branch_over_full": (nb_m / nf_m) if nf_m > 0 else float("nan"),
        "mlp_cos_full_branch": cos(gf_m, gb_m),
    })

out_csv = "gpt2_residual_grad_probe_results.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n결과 저장: {out_csv}")

print(f"\n{'idx':>4} {'attn_full':>11} {'attn_branch':>12} {'attn_ratio':>10} {'attn_cos':>9} | "
      f"{'mlp_full':>11} {'mlp_branch':>12} {'mlp_ratio':>10} {'mlp_cos':>9}")
for r in rows:
    print(f"{r['layer_idx']:>4} {r['attn_norm_full']:>11.3e} {r['attn_norm_branch_only']:>12.3e} "
          f"{r['attn_branch_over_full']:>10.3e} {r['attn_cos_full_branch']:>9.3f} | "
          f"{r['mlp_norm_full']:>11.3e} {r['mlp_norm_branch_only']:>12.3e} "
          f"{r['mlp_branch_over_full']:>10.3e} {r['mlp_cos_full_branch']:>9.3f}")

# ---------------------------------------------------------------------------
# 5. 시각화
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    idx = [r["layer_idx"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].semilogy(idx, [r["attn_norm_full"] for r in rows], "o-", label="full", markersize=4)
    axes[0, 0].semilogy(idx, [r["attn_norm_branch_only"] for r in rows], "s-", label="branch_only", markersize=4)
    axes[0, 0].set_title("Attn branch (c_attn.weight) grad norm")
    axes[0, 0].set_ylabel("grad norm (log)")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].semilogy(idx, [r["mlp_norm_full"] for r in rows], "o-", label="full", markersize=4)
    axes[0, 1].semilogy(idx, [r["mlp_norm_branch_only"] for r in rows], "s-", label="branch_only", markersize=4)
    axes[0, 1].set_title("MLP branch (c_fc.weight) grad norm")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(idx, [r["attn_cos_full_branch"] for r in rows], "o-", color="darkred", markersize=4)
    axes[1, 0].axhline(0, color="gray", linewidth=0.8)
    axes[1, 0].set_title("cos(full, branch_only) — attn")
    axes[1, 0].set_xlabel("layer idx (0 = input side)")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(idx, [r["mlp_cos_full_branch"] for r in rows], "o-", color="darkred", markersize=4)
    axes[1, 1].axhline(0, color="gray", linewidth=0.8)
    axes[1, 1].set_title("cos(full, branch_only) — mlp")
    axes[1, 1].set_xlabel("layer idx (0 = input side)")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("gpt2_residual_grad_probe.png", dpi=150)
    print("\n그래프 저장: gpt2_residual_grad_probe.png")
except ImportError:
    print("\n(matplotlib 없어서 그래프 스킵)")
