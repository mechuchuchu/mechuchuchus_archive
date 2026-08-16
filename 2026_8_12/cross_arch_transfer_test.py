"""
gradient_matching_M_vs_D.py 에서 저장한 M_vs_D_results.pt 를 로드해서,
MLP용으로 distill된 synthetic data가 CNN(전혀 다른 architecture)에도
통하는지 테스트. Cross-architecture generalization 확인용.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

RESULTS_PATH = "/mnt/user-data/outputs/M_vs_D_results.pt"  # 로컬 경로에 맞게 수정

EVAL_STEPS = 100
EVAL_LR = 0.1
N_PROBE = 20

# ---------------- data (test set only, for eval) ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(DEVICE)  # (N,1,28,28)
test_y = torch.tensor([test_ds[i][1] for i in range(len(test_ds))]).to(DEVICE)


# ---------------- architectures ----------------
class MLP(nn.Module):
    def __init__(self, hidden=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.fc2(F.relu(self.fc1(x)))


class SmallCNN(nn.Module):
    """LeNet 스타일, MNIST용 표준 소형 CNN."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)  # 14x14 -> 14x14
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # -> 16x14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # -> 32x7x7
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


ARCHS = {
    "MLP (same as distillation source)": MLP,
    "SmallCNN (unseen architecture)": SmallCNN,
}


def eval_accuracy(synth_x_img, synth_y_logits, model_ctor, n_probe=N_PROBE):
    """synth_x_img: (M,1,28,28) tensor. Train fresh probes from random init, report test acc."""
    accs = []
    target_p = F.softmax(synth_y_logits.detach(), dim=1).to(DEVICE)
    for _ in range(n_probe):
        probe = model_ctor().to(DEVICE)
        opt = torch.optim.SGD(probe.parameters(), lr=EVAL_LR)
        for _ in range(EVAL_STEPS):
            opt.zero_grad()
            out = probe(synth_x_img)
            log_p = F.log_softmax(out, dim=1)
            loss = -(target_p * log_p).sum(dim=1).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = probe(test_x).argmax(dim=1)
            acc = (pred == test_y).float().mean().item()
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


if __name__ == "__main__":
    results = torch.load(RESULTS_PATH, map_location=DEVICE)

    print("===== Cross-architecture transfer test =====")
    print(f"(synthetic data was distilled using an MLP as the gradient-matching model)\n")

    summary = []
    for M, r in results.items():
        synth_x_flat = r["synth_x"].to(DEVICE)          # (M, 784)
        synth_x_img = synth_x_flat.reshape(-1, 1, 28, 28)  # for CNN
        synth_y_logits = r["synth_y_logits"].to(DEVICE)

        print(f"--- M = {M} ---")
        for name, ctor in ARCHS.items():
            acc_mean, acc_std = eval_accuracy(synth_x_img, synth_y_logits, ctor)
            print(f"  {name:35s}: {acc_mean*100:.2f}% +/- {acc_std*100:.2f}%")
            summary.append((M, name, acc_mean, acc_std))
        print()

    print("===== SUMMARY =====")
    print(f"{'M':>5} | {'architecture':35s} | accuracy")
    for M, name, acc_mean, acc_std in summary:
        print(f"{M:5d} | {name:35s} | {acc_mean*100:.2f}% +/- {acc_std*100:.2f}%")
