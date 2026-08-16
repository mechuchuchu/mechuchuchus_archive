# gradient_matching_mnist.py

import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MNIST
REAL_BATCH_SIZE = 50

# Synthetic dataset
NUM_SYNTH = 50

# Number of parameter regions used during synthetic optimization
NUM_TRAIN_REGIONS = 8

# Number of unseen parameter regions for evaluation
NUM_TEST_REGIONS = 4

# Hidden layer
HIDDEN = 128

# Optimization
STEPS = 1500
LR = 0.1

# Gradient matching
COSINE_WEIGHT = 1.0
MSE_WEIGHT = 0.1

# How strongly parameters are perturbed around the anchor model
REGION_STD = 0.10

# Print every N iterations
PRINT_EVERY = 100


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("Device:", DEVICE)


# ============================================================
# 3. Model
# ============================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(784, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 10)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. Loss supporting soft labels
# ============================================================

def soft_cross_entropy(logits, targets):
    """
    logits:
        [B, 10]

    targets:
        [B, 10]
        soft labels whose rows approximately sum to 1.
    """

    log_probs = F.log_softmax(logits, dim=1)

    loss = -(targets * log_probs).sum(dim=1)

    return loss.mean()


# ============================================================
# 5. Flatten gradients
# ============================================================

def flatten_grads(grads):
    return torch.cat([
        g.reshape(-1)
        for g in grads
    ])


# ============================================================
# 6. Gradient of a dataset batch
# ============================================================

def compute_gradient(
    model,
    x,
    y,
    create_graph=False
):
    """
    Computes gradient of loss w.r.t. model parameters.

    create_graph=True is required when we want to
    differentiate the gradient-matching loss w.r.t.
    the synthetic data.
    """

    logits = model(x)

    loss = soft_cross_entropy(logits, y)

    grads = torch.autograd.grad(
        loss,
        tuple(model.parameters()),
        create_graph=create_graph,
        retain_graph=create_graph
    )

    return flatten_grads(grads)


# ============================================================
# 7. Gradient matching distance
# ============================================================

def gradient_distance(g_syn, g_real):
    """
    Compare two gradients.

    Cosine term:
        1 - cos(g_syn, g_real)

    Normalized MSE:
        ||g_syn - g_real||^2 / ||g_real||^2
    """

    g_syn_norm = g_syn / (g_syn.norm() + 1e-8)
    g_real_norm = g_real / (g_real.norm() + 1e-8)

    cosine_loss = 1.0 - torch.sum(
        g_syn_norm * g_real_norm
    )

    mse_loss = (
        (g_syn_norm - g_real_norm) ** 2
    ).mean()

    return (
        COSINE_WEIGHT * cosine_loss
        + MSE_WEIGHT * mse_loss
    )


# ============================================================
# 8. Create parameter regions
# ============================================================

def make_parameter_regions(
    anchor_model,
    num_regions,
    std
):
    """
    Create several nearby parameter regions.

    Every region is a different model initialization,
    centered around the same anchor parameters.
    """

    regions = []

    anchor_state = copy.deepcopy(
        anchor_model.state_dict()
    )

    for _ in range(num_regions):

        model = MLP().to(DEVICE)

        state = {}

        for name, tensor in anchor_state.items():

            state[name] = (
                tensor
                + std * torch.randn_like(tensor)
            )

        model.load_state_dict(state)

        regions.append(model)

    return regions


# ============================================================
# 9. Load MNIST
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=REAL_BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

loader_iter = iter(train_loader)


# ============================================================
# 10. Initial anchor model
# ============================================================

anchor_model = MLP().to(DEVICE)


# ============================================================
# 11. Create train parameter regions
# ============================================================

train_regions = make_parameter_regions(
    anchor_model,
    NUM_TRAIN_REGIONS,
    REGION_STD
)


# ============================================================
# 12. Create unseen test parameter regions
# ============================================================

test_regions = make_parameter_regions(
    anchor_model,
    NUM_TEST_REGIONS,
    REGION_STD
)


# ============================================================
# 13. Pick fixed real batches
# ============================================================

real_batches = []

for k in range(NUM_TRAIN_REGIONS):

    try:
        x_real, y_real = next(loader_iter)

    except StopIteration:
        loader_iter = iter(train_loader)
        x_real, y_real = next(loader_iter)

    x_real = x_real.view(
        x_real.size(0),
        -1
    ).to(DEVICE)

    y_real = F.one_hot(
        y_real,
        num_classes=10
    ).float().to(DEVICE)

    real_batches.append(
        (x_real, y_real)
    )


# ============================================================
# 14. Initialize synthetic data
# ============================================================

# Random synthetic images.
#
# Shape:
#   [50, 784]

synthetic_x = torch.randn(
    NUM_SYNTH,
    784,
    device=DEVICE,
    requires_grad=True
)

# Learnable label logits.
#
# We optimize logits rather than probabilities.
# Softmax converts them into 10-dimensional labels.

synthetic_y_logits = torch.randn(
    NUM_SYNTH,
    10,
    device=DEVICE,
    requires_grad=True
)


# ============================================================
# 15. Synthetic optimizer
# ============================================================

optimizer = torch.optim.Adam(
    [
        synthetic_x,
        synthetic_y_logits
    ],
    lr=LR
)


# ============================================================
# 16. Precompute real gradients
# ============================================================

print()
print("Precomputing real gradients...")

real_gradients = []

for k, model in enumerate(train_regions):

    x_real, y_real = real_batches[k]

    g_real = compute_gradient(
        model,
        x_real,
        y_real,
        create_graph=False
    )

    # No need to keep graph.
    g_real = g_real.detach()

    real_gradients.append(g_real)


print("Done.")


# ============================================================
# 17. Synthetic optimization
# ============================================================

print()
print("Starting gradient matching...")
print()

for step in range(STEPS):

    optimizer.zero_grad()

    # Convert label logits to probability distributions
    synthetic_y = F.softmax(
        synthetic_y_logits,
        dim=1
    )

    total_loss = 0.0

    # --------------------------------------------------------
    # Match gradients in EVERY parameter region
    # --------------------------------------------------------

    for k, model in enumerate(train_regions):

        g_real = real_gradients[k]

        g_syn = compute_gradient(
            model,
            synthetic_x,
            synthetic_y,
            create_graph=True
        )

        loss_k = gradient_distance(
            g_syn,
            g_real
        )

        total_loss = total_loss + loss_k

    # Average across parameter regions
    total_loss = (
        total_loss / NUM_TRAIN_REGIONS
    )

    # --------------------------------------------------------
    # Optimize synthetic tensor
    # --------------------------------------------------------

    total_loss.backward()

    optimizer.step()

    # Optional image-value constraint
    with torch.no_grad():
        synthetic_x.clamp_(
            -3.0,
            3.0
        )

    if step % PRINT_EVERY == 0:

        print(
            f"step={step:4d} "
            f"loss={total_loss.item():.6f}"
        )


# ============================================================
# 18. Evaluate gradient similarity
# ============================================================

def cosine_similarity(a, b):

    a = a / (a.norm() + 1e-8)
    b = b / (b.norm() + 1e-8)

    return torch.sum(a * b).item()


@torch.no_grad()
def get_hard_labels(y_logits):

    return y_logits.argmax(dim=1)


print()
print("=" * 70)
print("Evaluation")
print("=" * 70)


# Synthetic labels
synthetic_y = F.softmax(
    synthetic_y_logits,
    dim=1
)

# ------------------------------------------------------------
# Train regions
# ------------------------------------------------------------

train_scores = []

for k, model in enumerate(train_regions):

    x_real, y_real = real_batches[k]

    # Real gradient
    g_real = real_gradients[k]

    # Synthetic gradient
    #
    # We don't need higher-order gradients during evaluation.
    g_syn = compute_gradient(
        model,
        synthetic_x,
        synthetic_y,
        create_graph=False
    )

    score = cosine_similarity(
        g_syn.detach(),
        g_real
    )

    train_scores.append(score)

    print(
        f"Train region {k:2d}: "
        f"cosine similarity = {score:.6f}"
    )


# ------------------------------------------------------------
# Unseen parameter regions
# ------------------------------------------------------------

print()
print("Unseen parameter regions:")

# Use one fixed real batch for evaluating
# the unseen parameter regions.

x_test_real, y_test_real = real_batches[0]

test_scores = []

for k, model in enumerate(test_regions):

    # Real gradient at this NEW parameter region
    g_real = compute_gradient(
        model,
        x_test_real,
        y_test_real,
        create_graph=False
    ).detach()

    # Synthetic gradient at same NEW parameter region
    g_syn = compute_gradient(
        model,
        synthetic_x,
        synthetic_y,
        create_graph=False
    )

    score = cosine_similarity(
        g_syn.detach(),
        g_real
    )

    test_scores.append(score)

    print(
        f"Test region {k:2d}: "
        f"cosine similarity = {score:.6f}"
    )


# ============================================================
# 19. Summary
# ============================================================

print()
print("=" * 70)

print(
    "Mean train-region similarity:",
    np.mean(train_scores)
)

print(
    "Mean unseen-region similarity:",
    np.mean(test_scores)
)

print("=" * 70)


# ============================================================
# 20. Inspect synthetic labels
# ============================================================

hard_labels = get_hard_labels(
    synthetic_y_logits
)

print()
print("Synthetic hard-label distribution:")

for digit in range(10):

    count = (
        hard_labels == digit
    ).sum().item()

    print(
        f"digit {digit}: {count}"
    )
