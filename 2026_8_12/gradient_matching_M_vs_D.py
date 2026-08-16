"""
Dataset Distillation (Wang 2018) Section 3.3 lower bound 검증:
Linear case에서 M >= D (D=784, MNIST pixel dim)일 때 arbitrary random init에
대한 일반화가 이론적으로 보장됨. 이걸 실제 (nonlinear) MLP + gradient matching
파이프라인에서 M=50 vs M=784로 비교.

Region 생성 = 완전 독립 random init (기존 basin-diversity 버전과 동일 방식)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ---------------- config ----------------
NUM_TRAIN_REGIONS = 16      # gradient matching 대상 (독립 random init)
NUM_TEST_REGIONS = 16       # held-out, 학습에 안 쓴 independent init
HIDDEN = 128
D = 28 * 28                 # feature dimension
NUM_CLASSES = 10

DISTILL_STEPS = 1500
DISTILL_LR = 0.05

EVAL_STEPS = 100
EVAL_LR = 0.1

# M 후보: 이론적 하한(D=784)과 기존 baseline(50) 비교
M_CANDIDATES = [50, 784]

# ---------------- data ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_x = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).reshape(-1, D).to(DEVICE)
train_y = torch.tensor([train_ds[i][1] for i in range(len(train_ds))]).to(DEVICE)
test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).reshape(-1, D).to(DEVICE)
test_y = torch.tensor([test_ds[i][1] for i in range(len(test_ds))]).to(DEVICE)


# ---------------- model ----------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, NUM_CLASSES)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def model_parameters(model):
    return [p for p in model.parameters()]


def flat_grad(params):
    return torch.cat([g.reshape(-1) for g in params])


def real_gradient(model, x, y, batch_size=2000):
    """Full-dataset average gradient via minibatch accumulation (exact)."""
    model.zero_grad()
    total = 0
    accum = [torch.zeros_like(p) for p in model.parameters()]
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
    """Gradient of CE loss (soft labels) w.r.t. model params, differentiable in synth_x/logits."""
    out = model(synth_x)
    log_p = F.log_softmax(out, dim=1)
    target_p = F.softmax(synth_y_logits, dim=1)
    loss = -(target_p * log_p).sum(dim=1).mean()
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    return grads


def grad_matching_loss(g_synth, g_real):
    # cosine + MSE on flattened concatenated gradient (as in prior notebooks)
    fs = flat_grad(g_synth)
    fr = flat_grad([g.detach() for g in g_real])
    cos = 1 - F.cosine_similarity(fs.unsqueeze(0), fr.unsqueeze(0)).squeeze()
    mse = F.mse_loss(fs, fr)
    return cos + mse, cos.detach().item()


def build_regions(num_regions):
    """Independent random inits (kaiming, fresh MLP each time)."""
    regions = []
    for _ in range(num_regions):
        m = MLP().to(DEVICE)
        regions.append(m)
    return regions


def eval_accuracy(synth_x, synth_y_logits, init_model_fn, n_probe=20):
    """Train fresh probes (from random init) on synthetic set, report mean test accuracy."""
    accs = []
    target_p = F.softmax(synth_y_logits.detach(), dim=1)
    for _ in range(n_probe):
        probe = init_model_fn().to(DEVICE)
        opt = torch.optim.SGD(probe.parameters(), lr=EVAL_LR)
        for _ in range(EVAL_STEPS):
            opt.zero_grad()
            out = probe(synth_x.detach())
            log_p = F.log_softmax(out, dim=1)
            loss = -(target_p * log_p).sum(dim=1).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = probe(test_x).argmax(dim=1)
            acc = (pred == test_y).float().mean().item()
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


def run_experiment(M):
    print(f"\n===== M = {M} (D = {D}) =====")

    train_regions = build_regions(NUM_TRAIN_REGIONS)
    test_regions = build_regions(NUM_TEST_REGIONS)  # held-out, independent from train

    # precompute real target gradients (fixed, non-differentiable) for each region
    real_grads_train = [real_gradient(m, train_x, train_y) for m in train_regions]
    real_grads_test = [real_gradient(m, train_x, train_y) for m in test_regions]

    # init synthetic data: random real images as starting point (common trick)
    idx = torch.randperm(train_x.shape[0])[:M]
    synth_x = train_x[idx].clone().detach().requires_grad_(True)
    synth_y_logits = torch.randn(M, NUM_CLASSES, device=DEVICE, requires_grad=True)

    opt = torch.optim.Adam([synth_x, synth_y_logits], lr=DISTILL_LR)

    for step in range(DISTILL_STEPS):
        opt.zero_grad()
        total_loss = 0.0
        cos_vals = []
        for m, g_real in zip(train_regions, real_grads_train):
            g_synth = synthetic_gradient(m, synth_x, synth_y_logits)
            loss, cos = grad_matching_loss(g_synth, g_real)
            total_loss = total_loss + loss
            cos_vals.append(cos)
        total_loss.backward()
        opt.step()

        if step % 300 == 0 or step == DISTILL_STEPS - 1:
            print(f"  step {step:4d} | loss {total_loss.item():.4f} | "
                  f"train cosine {1 - np.mean(cos_vals):.4f}")

    # gradient cosine on unseen regions (sanity check, no grad needed for opt but need graph)
    unseen_cos = []
    for m, g_real in zip(test_regions, real_grads_test):
        g_synth = synthetic_gradient(m, synth_x, synth_y_logits)
        _, cos = grad_matching_loss(g_synth, g_real)
        unseen_cos.append(1 - cos)
    print(f"  unseen-region gradient cosine: mean={np.mean(unseen_cos):.4f} "
          f"std={np.std(unseen_cos):.4f}")

    # downstream accuracy: train fresh probes from scratch on synth_x/synth_y_logits
    acc_mean, acc_std = eval_accuracy(synth_x, synth_y_logits, MLP, n_probe=20)
    print(f"  downstream accuracy (fresh random-init probes): "
          f"{acc_mean*100:.2f}% +/- {acc_std*100:.2f}%")

    return {
        "M": M,
        "unseen_cosine_mean": float(np.mean(unseen_cos)),
        "unseen_cosine_std": float(np.std(unseen_cos)),
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "synth_x": synth_x.detach().cpu(),
        "synth_y_logits": synth_y_logits.detach().cpu(),
    }


if __name__ == "__main__":
    results = {}
    for M in M_CANDIDATES:
        results[M] = run_experiment(M)

    print("\n===== SUMMARY (M >= D lower bound check, D=784) =====")
    for M, r in results.items():
        print(f"M={M:4d} | unseen cosine={r['unseen_cosine_mean']:.4f} | "
              f"accuracy={r['acc_mean']*100:.2f}% +/- {r['acc_std']*100:.2f}%")

    # save synthetic sets for later inspection (e.g. visualize like synthetic_full.pt)
    torch.save(results, "/mnt/user-data/outputs/M_vs_D_results.pt")
    print("\nsaved -> /mnt/user-data/outputs/M_vs_D_results.pt")
