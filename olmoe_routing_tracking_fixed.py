# ============================================================
# OLMoE expert routing tracking (fixed)
# ============================================================

import os
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

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"

N_STEPS = 100

SEED = 42

SAVE_DIR = "./olmoe_interpolation_results"

os.makedirs(SAVE_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Device
# ============================================================

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# Model
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=(
        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32
    ),
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)

model.eval()

if not torch.cuda.is_available():
    model = model.to(device)


# ============================================================
# Basic model information
# ============================================================

config = model.config

NUM_EXPERTS = config.num_experts
TOP_K = config.num_experts_per_tok
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size

print("====================================")
print("OLMoE configuration")
print("====================================")
print("Experts:", NUM_EXPERTS)
print("Top-k:", TOP_K)
print("Layers:", NUM_LAYERS)
print("Hidden size:", HIDDEN_SIZE)


# ============================================================
# Input
# ============================================================

text = (
    "The meaning of this sentence changes "
    "as the representation moves."
)

inputs = tokenizer(
    text,
    return_tensors="pt",
)

input_ids = inputs["input_ids"].to(device)

attention_mask = inputs["attention_mask"].to(device)

seq_len = input_ids.shape[-1]

token_position = seq_len - 1


# ============================================================
# Embedding
# ============================================================

embedding_layer = model.get_input_embeddings()

embedding_dim = embedding_layer.embedding_dim

original_embeddings = embedding_layer(
    input_ids
).detach()


# ============================================================
# Random a,b
# ============================================================

a = torch.randn(
    embedding_dim,
    device=device,
    dtype=original_embeddings.dtype,
)

b = torch.randn(
    embedding_dim,
    device=device,
    dtype=original_embeddings.dtype,
)

a = a / a.norm()
b = b / b.norm()


# ============================================================
# Routing storage
#
# routing_history[step][layer]
#
# = top-k expert IDs for token_position
# ============================================================

routing_history = []

logits_history = []


# ============================================================
# Hook factory
#
# FIX: reuse the router's own output logits when the module
# forward already returns them, instead of recomputing via
# module.weight. Falls back to the manual linear() computation
# if `output` doesn't look like the plain logits tensor (some
# router implementations return a tuple, or add bias / noise
# terms that a naive re-projection would miss).
# ============================================================

def make_router_hook(layer_idx):

    def hook(module, inputs, output):

        hidden_states = inputs[0]

        hidden_flat = hidden_states.reshape(
            -1,
            hidden_states.shape[-1],
        )

        target_index = token_position

        router_logits = None

        candidate = output

        if isinstance(candidate, (tuple, list)):
            candidate = candidate[0]

        if (
            torch.is_tensor(candidate)
            and candidate.dim() >= 2
            and candidate.shape[-1] == NUM_EXPERTS
        ):
            router_logits = candidate.reshape(
                -1,
                NUM_EXPERTS,
            )

        if router_logits is None:
            # Fallback: recompute manually.
            router_weight = module.weight

            router_logits = torch.nn.functional.linear(
                hidden_flat,
                router_weight,
            )

        target_logits = router_logits[target_index]

        topk = torch.topk(
            target_logits,
            k=TOP_K,
            dim=-1,
        )

        expert_ids = topk.indices.detach().cpu()

        current_routing[layer_idx] = expert_ids

    return hook


# ============================================================
# Find router modules
# ============================================================

router_modules = []

for layer_idx, layer in enumerate(model.model.layers):

    possible_names = [
        "gate",
        "router",
    ]

    router = None

    for name in possible_names:

        if hasattr(layer.mlp, name):
            router = getattr(
                layer.mlp,
                name,
            )
            break

    if router is None:

        print(
            f"WARNING: router not found "
            f"at layer {layer_idx}"
        )

    else:

        router_modules.append(
            (
                layer_idx,
                router,
            )
        )

        print(
            f"Layer {layer_idx}: "
            f"{router.__class__.__name__}"
        )


# ============================================================
# Register hooks
# ============================================================

hooks = []

for layer_idx, router in router_modules:

    hooks.append(
        router.register_forward_hook(
            make_router_hook(layer_idx)
        )
    )


# ============================================================
# Forward helper
# ============================================================

def run_model(embedding_vector):

    global current_routing

    current_routing = {}

    embeddings = original_embeddings.clone()

    embeddings[
        0,
        token_position,
        :
    ] = embedding_vector

    with torch.no_grad():

        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )

    logits = outputs.logits[
        0,
        -1,
    ].float().detach()

    routing = []

    for layer_idx in range(NUM_LAYERS):

        if layer_idx in current_routing:

            routing.append(
                current_routing[layer_idx]
            )

        else:

            routing.append(
                torch.full(
                    (TOP_K,),
                    -1,
                    dtype=torch.long,
                )
            )

    return logits, routing


# ============================================================
# Run interpolation
# ============================================================

print("\nRunning interpolation...")

for step in range(N_STEPS):

    t = step / (N_STEPS - 1)

    vector = (
        (1.0 - t) * a
        + t * b
    )

    logits, routing = run_model(
        vector
    )

    logits_history.append(
        logits.cpu()
    )

    routing_history.append(
        torch.stack(routing)
    )

    if step % 10 == 0:

        print(
            f"step {step:03d}/{N_STEPS - 1}"
        )


for hook in hooks:
    hook.remove()


# ============================================================
# Stack
# ============================================================

logits = torch.stack(
    logits_history
)

routing = torch.stack(
    routing_history
)

print("\nShapes")
print("logits :", logits.shape)
print("routing:", routing.shape)


# ============================================================
# Detect expert changes
# ============================================================

expert_change = torch.zeros(
    N_STEPS,
    NUM_LAYERS,
    dtype=torch.bool,
)


for step in range(1, N_STEPS):

    for layer in range(NUM_LAYERS):

        previous = routing[
            step - 1,
            layer
        ]

        current = routing[
            step,
            layer
        ]

        previous_set = set(
            previous.tolist()
        )

        current_set = set(
            current.tolist()
        )

        if previous_set != current_set:

            expert_change[
                step,
                layer
            ] = True


# ============================================================
# Aggregate change information
# ============================================================

change_count_per_step = (
    expert_change.sum(dim=1)
    .numpy()
)

any_expert_change = (
    change_count_per_step > 0
)

change_steps = np.where(
    any_expert_change
)[0]

change_indices = change_steps


print("\n====================================")
print("Expert routing changes")
print("====================================")

print(
    "Number of steps with changes:",
    len(change_steps)
)

print(
    "Change steps:",
    change_steps.tolist()
)


# ============================================================
# Detailed change information
# ============================================================

change_details = []

for step in change_steps:

    changed_layers = np.where(
        expert_change[step].numpy()
    )[0]

    change_details.append(
        {
            "step": int(step),
            "t": float(
                step / (N_STEPS - 1)
            ),
            "layers": changed_layers.tolist(),
        }
    )

    print(
        f"step={step:3d}, "
        f"t={step/(N_STEPS-1):.4f}, "
        f"layers={changed_layers.tolist()}"
    )


# ============================================================
# KL divergence helper
#
# FIX: this was defined twice with identical bodies. Kept once.
# ============================================================

def kl_divergence(logits_p, logits_q):
    """
    KL(P || Q)

    P = softmax(logits_p)
    Q = softmax(logits_q)
    """

    log_p = torch.log_softmax(
        logits_p,
        dim=-1,
    )

    log_q = torch.log_softmax(
        logits_q,
        dim=-1,
    )

    p = torch.softmax(
        logits_p,
        dim=-1,
    )

    return torch.sum(
        p * (log_p - log_q),
        dim=-1,
    )


# ------------------------------------------------------------
# KL(P_t || P_{t-1})  -- step-to-step divergence
# length N_STEPS - 1, indexed by step = 1..N_STEPS-1
# ------------------------------------------------------------

kl_values = []

for step in range(1, N_STEPS):

    kl = kl_divergence(
        logits[step],
        logits[step - 1],
    )

    kl_values.append(
        kl.item()
    )

kl_values = np.array(
    kl_values
)


# ------------------------------------------------------------
# KL(P_t || P_0)  -- divergence from initial distribution
# length N_STEPS, indexed by step = 0..N_STEPS-1
# ------------------------------------------------------------

initial_logits = logits[0]

kl_from_init = []

for step in range(N_STEPS):

    current_logits = logits[step]

    kl = kl_divergence(
        current_logits,
        initial_logits,
    )

    kl_from_init.append(
        kl.item()
    )

kl_from_init = np.array(
    kl_from_init
)

torch.save(
    torch.tensor(kl_from_init),
    os.path.join(
        SAVE_DIR,
        "kl_from_initial.pt"
    ),
)


# ============================================================
# PCA
# ============================================================

pca = PCA(
    n_components=2
)

logits_2d = pca.fit_transform(
    logits.numpy()
)

explained = (
    pca.explained_variance_ratio_
)


# ============================================================
# Figure 1
#
# PCA + expert change points
#
# FIX: added colorbar so the viridis coloring (step index)
# is actually legible.
# ============================================================

plt.figure(
    figsize=(11, 8)
)

plt.plot(
    logits_2d[:, 0],
    logits_2d[:, 1],
    linewidth=1.5,
    alpha=0.7,
)

scatter = plt.scatter(
    logits_2d[:, 0],
    logits_2d[:, 1],
    c=np.arange(N_STEPS),
    cmap="viridis",
    s=30,
)

plt.colorbar(
    scatter,
    label="Interpolation step",
)

if len(change_indices) > 0:

    plt.scatter(
        logits_2d[
            change_indices,
            0
        ],
        logits_2d[
            change_indices,
            1
        ],
        marker="x",
        s=100,
        linewidths=2,
        label="Expert routing changed",
    )

plt.scatter(
    logits_2d[0, 0],
    logits_2d[0, 1],
    marker="o",
    s=150,
    label="a",
)

plt.scatter(
    logits_2d[-1, 0],
    logits_2d[-1, 1],
    marker="X",
    s=150,
    label="b",
)

for idx in change_indices:

    plt.annotate(
        str(idx),
        (
            logits_2d[idx, 0],
            logits_2d[idx, 1],
        ),
        fontsize=8,
    )

plt.xlabel(
    f"PC1 ({explained[0]*100:.2f}%)"
)

plt.ylabel(
    f"PC2 ({explained[1]*100:.2f}%)"
)

plt.title(
    "OLMoE logits trajectory with expert-routing changes"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "logits_pca_with_expert_changes.png"
    ),
    dpi=250,
)

plt.show()


# ============================================================
# Figure 2
#
# KL(P_t || P_0) + expert change points
# ============================================================

plt.figure(
    figsize=(11, 6)
)

steps = np.arange(
    N_STEPS
)

plt.plot(
    steps,
    kl_from_init,
    linewidth=1.8,
    marker="o",
    markersize=3,
    label="KL(current || initial)",
)

for step in change_indices:

    # step 0 is the initialization itself,
    # so there is no previous routing state to compare.
    if step == 0:
        continue

    plt.axvline(
        x=step,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
    )

plt.xlabel(
    "Interpolation step"
)

plt.ylabel(
    "KL(P_t || P_0)"
)

plt.title(
    "KL divergence from initial logits with expert-routing changes"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "kl_from_initial_with_expert_changes.png"
    ),
    dpi=250,
)

plt.show()


# ============================================================
# Figure 3
#
# KL(P_t || P_{t-1}) + expert change points
#
# FIX: this used to plot nothing new -- it kept drawing on top
# of Figure 2's canvas (no plt.figure() call), so the saved PNG
# had the initial-KL curve but mislabeled axes/title. It also
# never used `kl_values` at all. Now it gets its own figure and
# actually plots kl_values, with a matching x-axis
# (len(kl_values) == N_STEPS - 1, indexed from step=1).
# ============================================================

plt.figure(
    figsize=(11, 6)
)

step_to_step_x = np.arange(1, N_STEPS)

plt.plot(
    step_to_step_x,
    kl_values,
    linewidth=1.8,
    marker="o",
    markersize=3,
    label="KL(current || previous)",
)

for step in change_indices:

    if step == 0:
        continue

    plt.axvline(
        x=step,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
    )

plt.xlabel(
    "Interpolation step"
)

plt.ylabel(
    "KL(current || previous)"
)

plt.title(
    "KL divergence with expert-routing changes"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "kl_with_expert_changes.png"
    ),
    dpi=250,
)

plt.show()


# ============================================================
# Figure 4
#
# Which layers changed?
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.imshow(
    expert_change.T.numpy(),
    aspect="auto",
    interpolation="nearest",
)

plt.xlabel(
    "Interpolation step"
)

plt.ylabel(
    "MoE layer"
)

plt.title(
    "Expert-routing changes across layers"
)

plt.colorbar(
    label="Expert set changed"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "expert_routing_change_heatmap.png"
    ),
    dpi=250,
)

plt.show()


# ============================================================
# Save everything
# ============================================================

torch.save(
    {
        "model_name": MODEL_NAME,

        "a": a.cpu(),
        "b": b.cpu(),

        "logits": logits,

        "routing": routing,

        "expert_change": expert_change,

        "kl_values": torch.tensor(
            kl_values
        ),

        "kl_from_init": torch.tensor(
            kl_from_init
        ),

        "pca_2d": torch.tensor(
            logits_2d
        ),

        "pca_explained_variance": torch.tensor(
            explained
        ),

        "change_details": change_details,
    },
    os.path.join(
        SAVE_DIR,
        "olmoe_interpolation_full_results.pt"
    ),
)


print("\n====================================")
print("Done")
print("====================================")

print(
    "Saved to:",
    SAVE_DIR
)
