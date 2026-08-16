# ============================================================
# MoE vs Dense logit-trajectory comparison
#
# Compares:
#   - OLMoE (sparse, discontinuous expert routing)
#   - Olmo-3-1025-7B (dense, no routing)
#
# Hypothesis being visualized:
#   MoE trajectories should show discrete "jumps" in logit
#   space that coincide with expert-set changes (a routing
#   discontinuity), while the dense model's trajectory should
#   vary smoothly with no equivalent discrete cause.
# ============================================================

import os
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ============================================================
# Config
# ============================================================

MODEL_NAME_MOE = "allenai/OLMoE-1B-7B-0924"
MODEL_NAME_DENSE = "allenai/Olmo-3-1025-7B"

N_STEPS = 100

SEED = 42

SAVE_DIR = "./olmoe_vs_dense_results"

os.makedirs(SAVE_DIR, exist_ok=True)

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

TEXT = (
    "The meaning of this sentence changes "
    "as the representation moves."
)


# ============================================================
# KL divergence helper
# ============================================================

def kl_divergence(logits_p, logits_q):
    """
    KL(P || Q), P = softmax(logits_p), Q = softmax(logits_q)
    """

    log_p = torch.log_softmax(logits_p, dim=-1)
    log_q = torch.log_softmax(logits_q, dim=-1)
    p = torch.softmax(logits_p, dim=-1)

    return torch.sum(p * (log_p - log_q), dim=-1)


# ============================================================
# Router hook factory (only used for MoE models)
# ============================================================

def make_router_hook(layer_idx, num_experts, top_k, token_position, routing_state):

    def hook(module, inputs, output):

        hidden_states = inputs[0]

        hidden_flat = hidden_states.reshape(
            -1, hidden_states.shape[-1]
        )

        router_logits = None
        candidate = output

        if isinstance(candidate, (tuple, list)):
            candidate = candidate[0]

        if (
            torch.is_tensor(candidate)
            and candidate.dim() >= 2
            and candidate.shape[-1] == num_experts
        ):
            router_logits = candidate.reshape(-1, num_experts)

        if router_logits is None:
            router_weight = module.weight
            router_logits = torch.nn.functional.linear(
                hidden_flat, router_weight
            )

        target_logits = router_logits[token_position]

        topk = torch.topk(target_logits, k=top_k, dim=-1)

        routing_state[layer_idx] = topk.indices.detach().cpu()

    return hook


def find_router_modules(model):
    """
    Returns [] for dense models (no MoE routers found).
    """

    router_modules = []

    for layer_idx, layer in enumerate(model.model.layers):

        router = None

        for name in ("gate", "router"):
            if hasattr(layer.mlp, name):
                router = getattr(layer.mlp, name)
                break

        if router is not None:
            router_modules.append((layer_idx, router))

    return router_modules


# ============================================================
# Core experiment runner
#
# Works for both MoE and dense models. Routing tracking is
# automatically skipped when the model exposes no gate/router
# modules (i.e. dense architectures like Olmo-3-1025-7B).
# ============================================================

def run_interpolation_experiment(model_name, n_steps=N_STEPS, seed=SEED):

    print("\n" + "=" * 60)
    print(f"Loading {model_name}")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        ),
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model.eval()

    if not torch.cuda.is_available():
        model = model.to(device)

    config = model.config

    num_experts = getattr(config, "num_experts", None)
    top_k = getattr(config, "num_experts_per_tok", None)
    num_layers = config.num_hidden_layers

    is_moe = num_experts is not None and top_k is not None

    print("Architecture:", "MoE" if is_moe else "Dense")
    print("Layers:", num_layers)
    if is_moe:
        print("Experts:", num_experts, "| Top-k:", top_k)

    inputs = tokenizer(TEXT, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    seq_len = input_ids.shape[-1]
    token_position = seq_len - 1

    embedding_layer = model.get_input_embeddings()
    embedding_dim = embedding_layer.embedding_dim

    original_embeddings = embedding_layer(input_ids).detach()

    a = torch.randn(embedding_dim, device=device, dtype=original_embeddings.dtype)
    b = torch.randn(embedding_dim, device=device, dtype=original_embeddings.dtype)
    a = a / a.norm()
    b = b / b.norm()

    router_modules = find_router_modules(model) if is_moe else []

    if is_moe and len(router_modules) == 0:
        print("WARNING: MoE config detected but no router modules found -- treating as dense.")
        is_moe = False

    routing_history = []
    logits_history = []

    hooks = []
    routing_state = {}

    if is_moe:
        for layer_idx, router in router_modules:
            hooks.append(
                router.register_forward_hook(
                    make_router_hook(
                        layer_idx, num_experts, top_k, token_position, routing_state
                    )
                )
            )

    def run_model(embedding_vector):

        routing_state.clear()

        embeddings = original_embeddings.clone()
        embeddings[0, token_position, :] = embedding_vector

        with torch.no_grad():
            outputs = model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
            )

        step_logits = outputs.logits[0, -1].float().detach()

        step_routing = None

        if is_moe:
            step_routing = []
            for layer_idx in range(num_layers):
                if layer_idx in routing_state:
                    step_routing.append(routing_state[layer_idx])
                else:
                    step_routing.append(torch.full((top_k,), -1, dtype=torch.long))

        return step_logits, step_routing

    print("Running interpolation...")

    for step in range(n_steps):

        t = step / (n_steps - 1)
        vector = (1.0 - t) * a + t * b

        step_logits, step_routing = run_model(vector)

        logits_history.append(step_logits.cpu())

        if is_moe:
            routing_history.append(torch.stack(step_routing))

        if step % 20 == 0:
            print(f"  step {step:03d}/{n_steps - 1}")

    for hook in hooks:
        hook.remove()

    logits = torch.stack(logits_history)

    routing = torch.stack(routing_history) if is_moe else None

    # ------------------------------------------------------------
    # Expert-change detection (MoE only)
    # ------------------------------------------------------------

    expert_change = None
    change_indices = np.array([], dtype=int)

    if is_moe:

        expert_change = torch.zeros(n_steps, num_layers, dtype=torch.bool)

        for step in range(1, n_steps):
            for layer in range(num_layers):
                previous_set = set(routing[step - 1, layer].tolist())
                current_set = set(routing[step, layer].tolist())
                if previous_set != current_set:
                    expert_change[step, layer] = True

        any_change = expert_change.sum(dim=1).numpy() > 0
        change_indices = np.where(any_change)[0]

    # ------------------------------------------------------------
    # KL divergences
    # ------------------------------------------------------------

    kl_step_to_step = np.array([
        kl_divergence(logits[step], logits[step - 1]).item()
        for step in range(1, n_steps)
    ])

    initial_logits = logits[0]
    kl_from_init = np.array([
        kl_divergence(logits[step], initial_logits).item()
        for step in range(n_steps)
    ])

    # ------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------

    pca = PCA(n_components=2)
    logits_2d = pca.fit_transform(logits.numpy())
    explained = pca.explained_variance_ratio_

    # Free VRAM before returning, so we can load the next model.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_name": model_name,
        "is_moe": is_moe,
        "num_layers": num_layers,
        "logits": logits,
        "routing": routing,
        "expert_change": expert_change,
        "change_indices": change_indices,
        "kl_step_to_step": kl_step_to_step,
        "kl_from_init": kl_from_init,
        "logits_2d": logits_2d,
        "explained_variance": explained,
    }


# ============================================================
# Run both models
# ============================================================

results_moe = run_interpolation_experiment(MODEL_NAME_MOE)
results_dense = run_interpolation_experiment(MODEL_NAME_DENSE)

torch.save(
    {"moe": results_moe, "dense": results_dense},
    os.path.join(SAVE_DIR, "moe_vs_dense_full_results.pt"),
)


# ============================================================
# Jump-size quantification
#
# For the MoE model: is step-to-step KL larger at steps where
# the expert set changed vs steps where it didn't?
# ============================================================

def summarize_jumps(results, label):

    kl = results["kl_step_to_step"]

    print(f"\n--- {label} ({results['model_name']}) ---")
    print(f"KL(step-to-step): mean={kl.mean():.4f}  std={kl.std():.4f}  max={kl.max():.4f}")

    if results["is_moe"]:

        change_idx = results["change_indices"]
        # kl_step_to_step[i] corresponds to step i+1 (comparing step i+1 vs i)
        change_mask = np.zeros_like(kl, dtype=bool)
        for c in change_idx:
            if c >= 1:
                change_mask[c - 1] = True

        if change_mask.any():
            print(
                f"  KL at expert-change steps   : mean={kl[change_mask].mean():.4f} "
                f"(n={change_mask.sum()})"
            )
        if (~change_mask).any():
            print(
                f"  KL at non-change steps       : mean={kl[~change_mask].mean():.4f} "
                f"(n={(~change_mask).sum()})"
            )


summarize_jumps(results_moe, "MoE")
summarize_jumps(results_dense, "Dense")


# ============================================================
# Figure 1
#
# Side-by-side PCA trajectories: MoE (with jump markers)
# vs Dense (smooth, no routing discontinuities)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, results, title in [
    (axes[0], results_moe, f"MoE: {results_moe['model_name']}"),
    (axes[1], results_dense, f"Dense: {results_dense['model_name']}"),
]:

    logits_2d = results["logits_2d"]
    explained = results["explained_variance"]

    ax.plot(logits_2d[:, 0], logits_2d[:, 1], linewidth=1.5, alpha=0.7, color="gray")

    scatter = ax.scatter(
        logits_2d[:, 0], logits_2d[:, 1],
        c=np.arange(N_STEPS), cmap="viridis", s=30,
    )

    if results["is_moe"] and len(results["change_indices"]) > 0:
        ci = results["change_indices"]
        ax.scatter(
            logits_2d[ci, 0], logits_2d[ci, 1],
            marker="x", s=110, linewidths=2, color="red",
            label="Expert routing changed",
        )
        ax.legend()

    ax.scatter(logits_2d[0, 0], logits_2d[0, 1], marker="o", s=150, edgecolor="black", label="a")
    ax.scatter(logits_2d[-1, 0], logits_2d[-1, 1], marker="X", s=150, edgecolor="black", label="b")

    ax.set_xlabel(f"PC1 ({explained[0]*100:.2f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.2f}%)")
    ax.set_title(title)
    ax.grid(alpha=0.25)

fig.colorbar(scatter, ax=axes, label="Interpolation step", fraction=0.03, pad=0.02)

fig.suptitle(
    "Logit-space trajectory: discontinuous MoE routing jumps vs smooth dense trajectory",
    fontsize=13,
)

fig.savefig(
    os.path.join(SAVE_DIR, "pca_trajectory_moe_vs_dense.png"),
    dpi=250, bbox_inches="tight",
)

plt.show()


# ============================================================
# Figure 2
#
# Overlaid step-to-step KL curves: MoE vs Dense, with
# MoE's expert-change points marked so it's visible whether
# large KL jumps line up with routing switches.
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

steps_x = np.arange(1, N_STEPS)

ax.plot(
    steps_x, results_moe["kl_step_to_step"],
    linewidth=1.8, marker="o", markersize=3,
    label=f"MoE ({results_moe['model_name']})", color="tab:red",
)

ax.plot(
    steps_x, results_dense["kl_step_to_step"],
    linewidth=1.8, marker="o", markersize=3,
    label=f"Dense ({results_dense['model_name']})", color="tab:blue",
)

for step in results_moe["change_indices"]:
    if step == 0:
        continue
    ax.axvline(x=step, linestyle="--", linewidth=0.8, alpha=0.35, color="tab:red")

ax.set_xlabel("Interpolation step")
ax.set_ylabel("KL(current || previous)")
ax.set_title(
    "Step-to-step KL divergence: MoE expert-routing jumps vs dense smooth drift\n"
    "(dashed red lines = MoE expert-set changes)"
)
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()
fig.savefig(
    os.path.join(SAVE_DIR, "kl_step_to_step_moe_vs_dense.png"),
    dpi=250,
)

plt.show()


# ============================================================
# Figure 3
#
# KL from initial logits, overlaid
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

steps_x_full = np.arange(N_STEPS)

ax.plot(
    steps_x_full, results_moe["kl_from_init"],
    linewidth=1.8, marker="o", markersize=3,
    label=f"MoE ({results_moe['model_name']})", color="tab:red",
)

ax.plot(
    steps_x_full, results_dense["kl_from_init"],
    linewidth=1.8, marker="o", markersize=3,
    label=f"Dense ({results_dense['model_name']})", color="tab:blue",
)

for step in results_moe["change_indices"]:
    if step == 0:
        continue
    ax.axvline(x=step, linestyle="--", linewidth=0.8, alpha=0.35, color="tab:red")

ax.set_xlabel("Interpolation step")
ax.set_ylabel("KL(P_t || P_0)")
ax.set_title("Cumulative divergence from initial logits: MoE vs Dense")
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()
fig.savefig(
    os.path.join(SAVE_DIR, "kl_from_initial_moe_vs_dense.png"),
    dpi=250,
)

plt.show()


# ============================================================
# Figure 4
#
# MoE expert-routing change heatmap (unique to MoE, no
# dense-model equivalent -- included for reference)
# ============================================================

if results_moe["is_moe"]:

    plt.figure(figsize=(12, 6))

    plt.imshow(
        results_moe["expert_change"].T.numpy(),
        aspect="auto",
        interpolation="nearest",
    )

    plt.xlabel("Interpolation step")
    plt.ylabel("MoE layer")
    plt.title(f"Expert-routing changes across layers ({results_moe['model_name']})")
    plt.colorbar(label="Expert set changed")
    plt.tight_layout()

    plt.savefig(
        os.path.join(SAVE_DIR, "expert_routing_change_heatmap.png"),
        dpi=250,
    )

    plt.show()


print("\n====================================")
print("Done")
print("====================================")
print("Saved to:", SAVE_DIR)
