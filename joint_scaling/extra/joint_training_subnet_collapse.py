import math
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# 0. Configuration
# ============================================================

SEED = 42

N = 512
INPUT_DIM = 10
OUTPUT_DIM = 10

HIDDEN_DIM = 128
EPOCHS = 5000
LR = 1e-3

# Same init + perturbation
NOISE_LEVELS = [
    0.0,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 2. Random memorization dataset
# ============================================================

X = torch.randn(
    N,
    INPUT_DIM,
    device=device
)

Y = torch.randn(
    N,
    OUTPUT_DIM,
    device=device
)


# ============================================================
# 3. MLP
# ============================================================

class MLP(nn.Module):

    def __init__(self, hidden_dim=128):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, OUTPUT_DIM)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. Initialization
# ============================================================

def init_weights(m):

    if isinstance(m, nn.Linear):

        nn.init.kaiming_normal_(
            m.weight,
            nonlinearity="relu"
        )

        nn.init.zeros_(m.bias)


# ============================================================
# 5. Joint model
# ============================================================

class JointTwoSubnet(nn.Module):

    def __init__(self):

        super().__init__()

        self.subnet1 = MLP(128)
        self.subnet2 = MLP(128)

    def forward(self, x):

        y1 = self.subnet1(x)
        y2 = self.subnet2(x)

        ensemble = 0.5 * (y1 + y2)

        return ensemble, y1, y2


# ============================================================
# 6. Initialization for joint model
# ============================================================

def initialize_joint_independent(model):

    model.subnet1.apply(init_weights)
    model.subnet2.apply(init_weights)


def initialize_joint_near_identical(
    model,
    noise_std
):

    # First subnet
    model.subnet1.apply(init_weights)

    # Second subnet = first subnet + noise
    with torch.no_grad():

        for p1, p2 in zip(
            model.subnet1.parameters(),
            model.subnet2.parameters()
        ):

            noise = (
                noise_std
                * torch.randn_like(p1)
            )

            p2.copy_(p1 + noise)


# ============================================================
# 7. Metrics
# ============================================================

@torch.no_grad()
def flatten_parameters(model):

    return torch.cat([
        p.detach().flatten()
        for p in model.parameters()
    ])


@torch.no_grad()
def parameter_cosine_similarity(
    model1,
    model2
):

    p1 = flatten_parameters(model1)
    p2 = flatten_parameters(model2)

    return torch.nn.functional.cosine_similarity(
        p1.unsqueeze(0),
        p2.unsqueeze(0)
    ).item()


@torch.no_grad()
def function_cosine_similarity(
    model1,
    model2,
    X
):

    y1 = model1(X).flatten()
    y2 = model2(X).flatten()

    return torch.nn.functional.cosine_similarity(
        y1.unsqueeze(0),
        y2.unsqueeze(0)
    ).item()


@torch.no_grad()
def relative_function_distance(
    model1,
    model2,
    X
):

    y1 = model1(X)
    y2 = model2(X)

    numerator = torch.norm(y1 - y2)
    denominator = torch.norm(y1) + 1e-12

    return (
        numerator / denominator
    ).item()


# ============================================================
# 8. Train single NN
# ============================================================

def train_single(
    hidden_dim,
    epochs=5000,
    lr=1e-3
):

    model = MLP(hidden_dim).to(device)

    model.apply(init_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    loss_fn = nn.MSELoss()

    history = []

    for epoch in range(epochs):

        optimizer.zero_grad()

        prediction = model(X)

        loss = loss_fn(
            prediction,
            Y
        )

        loss.backward()

        optimizer.step()

        history.append(
            loss.item()
        )

    return model, history


# ============================================================
# 9. Train independent 128 × 2
# ============================================================

def train_independent_two(
    epochs=5000,
    lr=1e-3
):

    model1 = MLP(128).to(device)
    model2 = MLP(128).to(device)

    model1.apply(init_weights)
    model2.apply(init_weights)

    optimizer1 = torch.optim.Adam(
        model1.parameters(),
        lr=lr
    )

    optimizer2 = torch.optim.Adam(
        model2.parameters(),
        lr=lr
    )

    loss_fn = nn.MSELoss()

    history = {
        "loss1": [],
        "loss2": [],
        "ensemble_loss": [],
        "param_cos": [],
        "func_cos": [],
        "func_dist": [],
    }

    for epoch in range(epochs):

        # ------------------------------
        # Subnet 1
        # ------------------------------

        optimizer1.zero_grad()

        pred1 = model1(X)

        loss1 = loss_fn(
            pred1,
            Y
        )

        loss1.backward()

        optimizer1.step()


        # ------------------------------
        # Subnet 2
        # ------------------------------

        optimizer2.zero_grad()

        pred2 = model2(X)

        loss2 = loss_fn(
            pred2,
            Y
        )

        loss2.backward()

        optimizer2.step()


        # ------------------------------
        # Metrics
        # ------------------------------

        with torch.no_grad():

            pred1 = model1(X)
            pred2 = model2(X)

            ensemble = (
                0.5 * pred1
                + 0.5 * pred2
            )

            ensemble_loss = loss_fn(
                ensemble,
                Y
            )

            pcos = parameter_cosine_similarity(
                model1,
                model2
            )

            fcos = function_cosine_similarity(
                model1,
                model2,
                X
            )

            fdist = relative_function_distance(
                model1,
                model2,
                X
            )


        history["loss1"].append(
            loss1.item()
        )

        history["loss2"].append(
            loss2.item()
        )

        history["ensemble_loss"].append(
            ensemble_loss.item()
        )

        history["param_cos"].append(
            pcos
        )

        history["func_cos"].append(
            fcos
        )

        history["func_dist"].append(
            fdist
        )


    return model1, model2, history


# ============================================================
# 10. Train joint model
# ============================================================

def train_joint(
    mode,
    noise_std=0.0,
    epochs=5000,
    lr=1e-3
):

    model = JointTwoSubnet().to(device)

    if mode == "independent":

        initialize_joint_independent(
            model
        )

    elif mode == "near_identical":

        initialize_joint_near_identical(
            model,
            noise_std
        )

    else:

        raise ValueError(
            "Unknown initialization mode"
        )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    loss_fn = nn.MSELoss()


    history = {

        "loss": [],

        "subnet1_loss": [],
        "subnet2_loss": [],

        "param_cos": [],
        "func_cos": [],
        "func_dist": [],
    }


    for epoch in range(epochs):

        optimizer.zero_grad()


        # ------------------------------
        # Forward
        # ------------------------------

        ensemble, y1, y2 = model(X)


        # IMPORTANT:
        #
        # Joint model optimizes ONLY
        #
        #       || (f1 + f2)/2 - Y ||²
        #
        # not individual subnet losses.
        #

        loss = loss_fn(
            ensemble,
            Y
        )


        loss.backward()

        optimizer.step()


        # ------------------------------
        # Metrics
        # ------------------------------

        with torch.no_grad():

            ensemble, y1, y2 = model(X)

            subnet1_loss = loss_fn(
                y1,
                Y
            ).item()

            subnet2_loss = loss_fn(
                y2,
                Y
            ).item()

            pcos = parameter_cosine_similarity(
                model.subnet1,
                model.subnet2
            )

            fcos = function_cosine_similarity(
                model.subnet1,
                model.subnet2,
                X
            )

            fdist = relative_function_distance(
                model.subnet1,
                model.subnet2,
                X
            )


        history["loss"].append(
            loss.item()
        )

        history["subnet1_loss"].append(
            subnet1_loss
        )

        history["subnet2_loss"].append(
            subnet2_loss
        )

        history["param_cos"].append(
            pcos
        )

        history["func_cos"].append(
            fcos
        )

        history["func_dist"].append(
            fdist
        )


    return model, history


# ============================================================
# 11. Train everything
# ============================================================

print("\n==============================")
print("1. Single 128")
print("==============================")

single128, h_single128 = train_single(
    128,
    EPOCHS,
    LR
)


print("\n==============================")
print("2. Single 256")
print("==============================")

single256, h_single256 = train_single(
    256,
    EPOCHS,
    LR
)


print("\n==============================")
print("3. Independent 128 × 2")
print("==============================")

ind1, ind2, h_independent = (
    train_independent_two(
        EPOCHS,
        LR
    )
)


print("\n==============================")
print("4. Joint 128 × 2 / independent init")
print("==============================")

joint_ind, h_joint_ind = train_joint(
    mode="independent",
    epochs=EPOCHS,
    lr=LR
)


# ============================================================
# 12. Noise-level experiments
# ============================================================

noise_results = {}

for noise in NOISE_LEVELS:

    print("\n==============================")
    print(
        f"5. Joint 128 × 2 / "
        f"noise = {noise}"
    )
    print("==============================")


    model, history = train_joint(
        mode="near_identical",
        noise_std=noise,
        epochs=EPOCHS,
        lr=LR
    )

    noise_results[noise] = {
        "model": model,
        "history": history
    }


# ============================================================
# 13. Final summary
# ============================================================

print("\n\n========================================")
print("FINAL RESULTS")
print("========================================")


print(
    f"Single 128 final loss: "
    f"{h_single128[-1]:.8e}"
)

print(
    f"Single 256 final loss: "
    f"{h_single256[-1]:.8e}"
)

print(
    f"Independent 128×2 ensemble loss: "
    f"{h_independent['ensemble_loss'][-1]:.8e}"
)

print(
    f"Joint 128×2 independent-init loss: "
    f"{h_joint_ind['loss'][-1]:.8e}"
)


print("\nNoise experiments")
print("----------------------------------------")

for noise in NOISE_LEVELS:

    h = noise_results[noise]["history"]

    print(
        f"noise={noise:<8g} "
        f"loss={h['loss'][-1]:.4e} "
        f"param_cos={h['param_cos'][-1]:.6f} "
        f"func_cos={h['func_cos'][-1]:.6f} "
        f"func_dist={h['func_dist'][-1]:.6f}"
    )


# ============================================================
# 14. Plot 1:
#     Main loss comparison
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    h_single128,
    label="Single 128"
)

plt.plot(
    h_single256,
    label="Single 256"
)

plt.plot(
    h_independent["ensemble_loss"],
    label="Independent 128×2 ensemble"
)

plt.plot(
    h_joint_ind["loss"],
    label="Joint 128×2 / independent init"
)

for noise in NOISE_LEVELS:

    h = noise_results[noise]["history"]

    plt.plot(
        h["loss"],
        label=f"Joint / noise={noise:g}"
    )


plt.yscale("log")

plt.xlabel("Epoch")
plt.ylabel("Training MSE")

plt.title(
    "Random-vector memorization: "
    "loss comparison"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 15. Plot 2:
#     Independent 128×2 individual vs ensemble
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    h_independent["loss1"],
    label="Independent subnet 1"
)

plt.plot(
    h_independent["loss2"],
    label="Independent subnet 2"
)

plt.plot(
    h_independent["ensemble_loss"],
    label="Independent ensemble"
)

plt.yscale("log")

plt.xlabel("Epoch")
plt.ylabel("MSE")

plt.title(
    "Independent 128×2"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 16. Plot 3:
#     Function similarity vs training
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    h_joint_ind["func_cos"],
    label="Independent init"
)

for noise in NOISE_LEVELS:

    h = noise_results[noise]["history"]

    plt.plot(
        h["func_cos"],
        label=f"Noise={noise:g}"
    )


plt.xlabel("Epoch")

plt.ylabel(
    "Function cosine similarity"
)

plt.title(
    "Do the two subnetworks become similar?"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 17. Plot 4:
#     Functional distance
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    h_joint_ind["func_dist"],
    label="Independent init"
)

for noise in NOISE_LEVELS:

    h = noise_results[noise]["history"]

    plt.plot(
        h["func_dist"],
        label=f"Noise={noise:g}"
    )


plt.xlabel("Epoch")

plt.ylabel(
    "||f1-f2|| / ||f1||"
)

plt.title(
    "Functional diversity during joint training"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 18. Plot 5:
#     Parameter similarity
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    h_joint_ind["param_cos"],
    label="Independent init"
)

for noise in NOISE_LEVELS:

    h = noise_results[noise]["history"]

    plt.plot(
        h["param_cos"],
        label=f"Noise={noise:g}"
    )


plt.xlabel("Epoch")

plt.ylabel(
    "Parameter cosine similarity"
)

plt.title(
    "Parameter convergence between subnetworks"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 19. Plot 6:
#     Final noise sweep
# ============================================================

final_losses = []
final_func_cos = []
final_func_dist = []
final_param_cos = []


for noise in NOISE_LEVELS:

    h = noise_results[noise]["history"]

    final_losses.append(
        h["loss"][-1]
    )

    final_func_cos.append(
        h["func_cos"][-1]
    )

    final_func_dist.append(
        h["func_dist"][-1]
    )

    final_param_cos.append(
        h["param_cos"][-1]
    )


# ---- loss ----

plt.figure(figsize=(9, 6))

plt.semilogx(
    np.maximum(NOISE_LEVELS, 1e-12),
    final_losses,
    marker="o"
)

plt.xlabel(
    "Initialization noise"
)

plt.ylabel(
    "Final joint training MSE"
)

plt.title(
    "Final loss vs initialization noise"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ---- function similarity ----

plt.figure(figsize=(9, 6))

plt.semilogx(
    np.maximum(NOISE_LEVELS, 1e-12),
    final_func_cos,
    marker="o"
)

plt.xlabel(
    "Initialization noise"
)

plt.ylabel(
    "Final function cosine similarity"
)

plt.title(
    "Final function similarity vs initialization noise"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ---- function distance ----

plt.figure(figsize=(9, 6))

plt.semilogx(
    np.maximum(NOISE_LEVELS, 1e-12),
    final_func_dist,
    marker="o"
)

plt.xlabel(
    "Initialization noise"
)

plt.ylabel(
    "Final functional distance"
)

plt.title(
    "Final functional diversity vs initialization noise"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ---- parameter similarity ----

plt.figure(figsize=(9, 6))

plt.semilogx(
    np.maximum(NOISE_LEVELS, 1e-12),
    final_param_cos,
    marker="o"
)

plt.xlabel(
    "Initialization noise"
)

plt.ylabel(
    "Final parameter cosine similarity"
)

plt.title(
    "Final parameter similarity vs initialization noise"
)

plt.grid(True)

plt.tight_layout()

plt.show()
