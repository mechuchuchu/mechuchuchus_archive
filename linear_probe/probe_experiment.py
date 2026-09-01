"""
Frozen-backbone linear/MLP probe 실험 스크립트

실험 축:
  --arch     {resnet18, mlp}   백본 아키텍처
  --pretrain {none, noise, real}
                 none  : 랜덤 초기화 그대로 freeze (arm A)
                 noise : 무작위 노이즈(입력/라벨 모두 랜덤)를 memorize 후 freeze (arm B)
                 real  : 실제 CIFAR-10으로 전체 학습 후 freeze (arm C, upper bound)
  --head     {linear, mlp}     probe head 형태
  --seed                       random init 분산이 크므로 여러 seed 필수

데이터: HuggingFace uoft-cs/cifar10 (torchvision 서버 이슈 우회)

예시:
  python probe_experiment.py --arch resnet18 --pretrain none  --seed 0
  python probe_experiment.py --arch resnet18 --pretrain noise --seed 0
  python probe_experiment.py --arch resnet18 --pretrain real  --seed 0
  python probe_experiment.py --arch mlp      --pretrain noise --head mlp --seed 0
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from datasets import load_dataset


# =========================================================
# 유틸
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 백본 정의
# =========================================================
class LayerNorm2d(nn.GroupNorm):
    """BatchNorm2d 대체용. GroupNorm(1, C) == channel-wise LayerNorm."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__(1, num_channels, eps=eps)


def replace_bn_with_ln(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, LayerNorm2d(child.num_features))
        else:
            replace_bn_with_ln(child)


def cifar_stem(model: nn.Module) -> nn.Module:
    """ImageNet stem(7x7/s2 + maxpool)은 32x32 입력에 과도한 다운샘플.
    conv1을 3x3/s1로 축소하고 maxpool 제거."""
    old = model.conv1
    model.conv1 = nn.Conv2d(old.in_channels, old.out_channels,
                            kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class ResNetBackbone(nn.Module):
    """ResNet-18 (CIFAR stem + LayerNorm), FC 직전까지. 출력 512-d."""
    feat_dim = 512

    def __init__(self):
        super().__init__()
        m = resnet18(weights=None, num_classes=10)
        m = cifar_stem(m)
        replace_bn_with_ln(m)
        m.fc = nn.Identity()
        self.net = m

    def forward(self, x):
        return self.net(x)


class MLPBackbone(nn.Module):
    """32*32*3 -> 4096 -> 1024. head 직전까지. 출력 1024-d."""
    feat_dim = 1024

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def build_backbone(arch: str) -> nn.Module:
    if arch == "resnet18":
        return ResNetBackbone()
    if arch == "mlp":
        return MLPBackbone()
    raise ValueError(f"unknown arch: {arch}")


def build_head(kind: str, feat_dim: int, num_classes: int = 10) -> nn.Module:
    if kind == "linear":
        return nn.Linear(feat_dim, num_classes)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
    raise ValueError(f"unknown head: {kind}")


class Model(nn.Module):
    def __init__(self, arch: str, head: str, num_classes: int = 10):
        super().__init__()
        self.backbone = build_backbone(arch)
        self.head = build_head(head, self.backbone.feat_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))

    def reset_head(self, head: str, num_classes: int = 10):
        device = next(self.parameters()).device
        self.head = build_head(head, self.backbone.feat_dim, num_classes).to(device)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


# =========================================================
# 데이터
# =========================================================
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class HFCifar10(Dataset):
    def __init__(self, split: str, transform=None):
        self.ds = load_dataset("uoft-cs/cifar10", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["img"].convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, item["label"]


class RandomNoiseDataset(Dataset):
    """입력/라벨 모두 랜덤. memorize 대상이므로 생성 시점에 고정."""
    def __init__(self, n_samples: int, num_classes: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n_samples, 3, 32, 32, generator=g)
        self.labels = torch.randint(0, num_classes, (n_samples,), generator=g)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_cifar_loaders(batch_size=128, workers=4, augment=True):
    train_tf_list = []
    if augment:
        train_tf_list += [transforms.RandomCrop(32, padding=4),
                          transforms.RandomHorizontalFlip()]
    train_tf_list += [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    train_tf = transforms.Compose(train_tf_list)
    test_tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

    train_ds = HFCifar10("train", train_tf)
    test_ds = HFCifar10("test", test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=workers, pin_memory=True)
    return train_loader, test_loader


# =========================================================
# 학습 / 평가 루프
# =========================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return correct / total


def run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def pretrain_noise(model, device, args):
    """arm B: 전체 파라미터로 무작위 노이즈를 memorize."""
    ds = RandomNoiseDataset(args.noise_samples, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model.unfreeze_all()
    # weight decay는 memorization을 방해하므로 0
    optimizer = optim.SGD(model.parameters(), lr=args.pretrain_lr,
                          momentum=0.9, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    reached = False
    for epoch in range(args.pretrain_epochs):
        loss, acc = run_epoch(model, loader, optimizer, criterion, device)
        print(f"[noise-pretrain {epoch+1:03d}/{args.pretrain_epochs}] "
              f"loss={loss:.4f} acc={acc:.4f}")
        if acc >= args.target_acc:
            print(f"-> memorize 완료 (epoch {epoch+1}, acc={acc:.4f})")
            reached = True
            break
    if not reached:
        print(f"경고: target_acc={args.target_acc} 미달성 (final acc={acc:.4f}). "
              f"--pretrain-epochs를 늘리거나 --noise-samples를 줄여보세요.")
    return {"noise_final_acc": acc, "noise_reached_target": reached}


def pretrain_real(model, device, args):
    """arm C: 실제 CIFAR-10으로 전체 학습 (upper bound)."""
    train_loader, test_loader = get_cifar_loaders(args.batch_size, args.workers, augment=True)
    model.unfreeze_all()
    optimizer = optim.SGD(model.parameters(), lr=args.pretrain_lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.pretrain_epochs):
        loss, acc = run_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        test_acc = evaluate(model, test_loader, device)
        print(f"[real-pretrain {epoch+1:03d}/{args.pretrain_epochs}] "
              f"loss={loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}")
    return {"real_pretrain_test_acc": test_acc}


def probe(model, device, args):
    """backbone freeze 후 head만 CIFAR-10으로 학습."""
    train_loader, test_loader = get_cifar_loaders(args.batch_size, args.workers,
                                                  augment=not args.no_augment)
    model.reset_head(args.head)
    model.freeze_backbone()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_tr:,} / Total: {n_all:,} ({100*n_tr/n_all:.3f}%)")

    optimizer = optim.SGD(trainable, lr=args.probe_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.probe_epochs)
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(args.probe_epochs):
        loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        test_acc = evaluate(model, test_loader, device)
        best = max(best, test_acc)
        print(f"[probe {epoch+1:02d}/{args.probe_epochs}] "
              f"loss={loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
    return {"probe_final_test_acc": test_acc, "probe_best_test_acc": best}


# =========================================================
# main
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Frozen-backbone probe experiment")
    p.add_argument("--arch", choices=["resnet18", "mlp"], default="resnet18",
                   help="백본 아키텍처")
    p.add_argument("--pretrain", choices=["none", "noise", "real"], default="none",
                   help="freeze 전 백본을 무엇으로 학습할지 (실험 arm)")
    p.add_argument("--head", choices=["linear", "mlp"], default="linear",
                   help="probe head 형태")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--noise-samples", type=int, default=4096,
                   help="memorize 대상 노이즈 샘플 수")
    p.add_argument("--target-acc", type=float, default=0.999,
                   help="memorize 완료 판정 기준 train accuracy")
    p.add_argument("--pretrain-epochs", type=int, default=300)
    p.add_argument("--pretrain-lr", type=float, default=0.01)

    p.add_argument("--probe-epochs", type=int, default=50)
    p.add_argument("--probe-lr", type=float, default=0.1)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-augment", action="store_true",
                   help="probe 단계에서 data augmentation 끄기")
    p.add_argument("--out", type=str, default="results.jsonl",
                   help="결과 append할 jsonl 경로")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"config: arch={args.arch} pretrain={args.pretrain} "
          f"head={args.head} seed={args.seed}")

    model = Model(args.arch, args.head).to(device)
    result = {k: v for k, v in vars(args).items()}

    if args.pretrain == "noise":
        print("\n=== Phase 1: 무작위 노이즈 memorize ===")
        result.update(pretrain_noise(model, device, args))
    elif args.pretrain == "real":
        print("\n=== Phase 1: 실제 CIFAR-10 full training ===")
        result.update(pretrain_real(model, device, args))
    else:
        print("\n=== Phase 1: skip (랜덤 초기화 그대로 사용) ===")

    print("\n=== Phase 2: backbone freeze + head probe on CIFAR-10 ===")
    result.update(probe(model, device, args))

    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"\n결과 저장: {os.path.abspath(args.out)}")
    print(json.dumps({k: result[k] for k in result if "acc" in k}, indent=2))


if __name__ == "__main__":
    main()
