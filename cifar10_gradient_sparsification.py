import os
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets import load_dataset


# ============================================================
# 0. Configuration
# ============================================================

SEED = 42

EPOCHS = 40
BATCH_SIZE = 2048

NUM_WORKERS = 8

# ------------------------------------------------------------
# bf16 mixed precision
#
# - forward / loss 계산은 bfloat16 autocast로 수행
# - weight와 gradient는 fp32로 유지됨
#   -> gradient sparsification 로직은 수정 불필요
# - bf16은 fp32와 exponent 범위가 같아서
#   GradScaler가 필요 없음
# ------------------------------------------------------------
USE_BF16 = (
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
)

AMP_DTYPE = torch.bfloat16

# ------------------------------------------------------------
# 7 gradient ratios
# ------------------------------------------------------------
KEEP_RATIOS = [
    1.0,      # 100%
    0.50,     # 50%
    0.25,     # 25%
    0.10,     # 10%
    0.05,     # 5%
    0.01,     # 1%
    0.005,    # 0.5%
    0.001     # 0.1%
]

# ------------------------------------------------------------
# 2 optimizers
# ------------------------------------------------------------
OPTIMIZERS = [
    #"SGD",
    "Adam"
]

# Total:
# 8 ratios x 2 optimizers = 16 runs
# ------------------------------------------------------------

OUTPUT_DIR = "./results_cifar10_gradient_bf16"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("bf16 autocast:", "ON" if USE_BF16 else "OFF (bf16 unsupported)")

print("=" * 70)


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 완전 재현성을 원할 때 사용
    # 성능은 조금 떨어질 수 있음
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. Load CIFAR-10 from Hugging Face
# ============================================================

print("\nLoading CIFAR-10 from Hugging Face...")

hf_dataset = load_dataset(
    "uoft-cs/cifar10"
)

print(hf_dataset)

print("Train size:", len(hf_dataset["train"]))
print("Test size :", len(hf_dataset["test"]))


# ============================================================
# 4. Dataset wrapper
# ============================================================

class CIFAR10HFDataset(Dataset):

    def __init__(
        self,
        hf_split,
        transform=None
    ):

        self.dataset = hf_split
        self.transform = transform

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]

        # HF uoft-cs/cifar10 image column = "img"
        image = item["img"]

        label = item["label"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# 5. Transforms
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


train_dataset = CIFAR10HFDataset(
    hf_dataset["train"],
    transform=train_transform
)


test_dataset = CIFAR10HFDataset(
    hf_dataset["test"],
    transform=test_transform
)


# ============================================================
# 6. ResNet
# ============================================================

class BasicBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm2d(
                    out_channels
                )
            )

        else:

            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(
            inplace=True
        )


    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

        out = self.relu(out)

        return out


class SmallResNet(nn.Module):

    def __init__(
        self,
        num_classes=10
    ):

        super().__init__()

        self.conv = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(
            inplace=True
        )


        self.layer1 = nn.Sequential(

            BasicBlock(
                64,
                64
            ),

            BasicBlock(
                64,
                64
            )
        )


        self.layer2 = nn.Sequential(

            BasicBlock(
                64,
                128,
                stride=2
            ),

            BasicBlock(
                128,
                128
            )
        )


        self.layer3 = nn.Sequential(

            BasicBlock(
                128,
                256,
                stride=2
            ),

            BasicBlock(
                256,
                256
            )
        )


        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.fc = nn.Linear(
            256,
            num_classes
        )


    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.fc(x)

        return x


# ============================================================
# 7. Random Gradient Sparsification
# ============================================================

@torch.no_grad()
def random_gradient_sparsification(
    model,
    keep_ratio
):
    """
    모든 parameter의 gradient를 하나의 global vector처럼 보고
    keep_ratio 만큼을 무작위로 선택한다.

    예:
        1.0   -> 100%
        0.5   -> 50%
        0.25  -> 25%
        0.1   -> 10%
        0.05  -> 5%
        0.01  -> 1%
        0.001 -> 0.1%

    중요:
    gradient를 실제 parameter.grad에 반영한 후
    optimizer.step()이 실행된다.

    bf16 참고:
    autocast를 써도 backward 후 p.grad는 fp32이므로
    이 함수는 수정 없이 그대로 동작한다.
    """

    if keep_ratio >= 1.0:
        return


    # --------------------------------------------------------
    # parameter 목록
    # --------------------------------------------------------

    params = []

    total_elements = 0

    for p in model.parameters():

        if p.grad is not None:

            params.append(p)

            total_elements += p.grad.numel()


    if total_elements == 0:
        return


    # --------------------------------------------------------
    # 0%
    # --------------------------------------------------------

    if keep_ratio <= 0:

        for p in params:
            p.grad.zero_()

        return


    # --------------------------------------------------------
    # 유지할 gradient 개수
    # --------------------------------------------------------

    num_keep = int(
        total_elements * keep_ratio
    )

    num_keep = max(
        1,
        num_keep
    )


    # --------------------------------------------------------
    # global random selection
    #
    # random permutation을 사용해서
    # 정확하게 num_keep개의 gradient만 선택
    # --------------------------------------------------------

    random_scores = torch.rand(
        total_elements,
        device=DEVICE
    )

    selected_indices = torch.topk(
        random_scores,
        k=num_keep,
        largest=True,
        sorted=False
    ).indices

    mask = torch.zeros(
        total_elements,
        dtype=torch.bool,
        device=DEVICE
    )

    mask[selected_indices] = True


    # --------------------------------------------------------
    # 각 parameter gradient에 mask 적용
    # --------------------------------------------------------

    offset = 0

    for p in params:

        n = p.grad.numel()

        local_mask = mask[
            offset:offset + n
        ]

        grad_flat = p.grad.view(-1)

        grad_flat[~local_mask] = 0.0

        offset += n


# ============================================================
# 8. Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    test_loader,
    criterion
):

    model.eval()

    total = 0

    correct = 0

    loss_sum = 0.0


    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # bf16 autocast forward
        # ----------------------------------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
            enabled=USE_BF16
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        batch_size = images.size(0)

        loss_sum += (
            loss.item()
            * batch_size
        )


        _, predicted = torch.max(
            outputs,
            dim=1
        )


        total += batch_size

        correct += (
            predicted == labels
        ).sum().item()


    avg_loss = (
        loss_sum / total
    )

    accuracy = (
        100.0 * correct / total
    )


    return avg_loss, accuracy


# ============================================================
# 9. Create optimizer
# ============================================================

def create_optimizer(
    model,
    optimizer_name
):

    if optimizer_name == "SGD":

        optimizer = optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )


    elif optimizer_name == "Adam":

        optimizer = optim.Adam(
            model.parameters(),
            lr=4e-3,
            weight_decay=5e-4
        )


    else:

        raise ValueError(
            f"Unknown optimizer: {optimizer_name}"
        )


    return optimizer


# ============================================================
# 10. One experiment
# ============================================================

def run_experiment(
    optimizer_name,
    keep_ratio,
    run_id
):

    # --------------------------------------------------------
    # Same initialization for every experiment
    # --------------------------------------------------------

    set_seed(SEED)


    print("\n")
    print("=" * 70)

    print(
        f"RUN {run_id:02d}/{len(OPTIMIZERS) * len(KEEP_RATIOS)}"
    )

    print(
        f"Optimizer      : {optimizer_name}"
    )

    print(
        f"Gradient kept  : "
        f"{keep_ratio * 100:g}%"
    )

    print(
        f"Precision      : "
        f"{'bf16 autocast' if USE_BF16 else 'fp32'}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SmallResNet().to(DEVICE)


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = create_optimizer(
        model,
        optimizer_name
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )


    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = []


    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        model.train()


        train_loss_sum = 0.0

        train_correct = 0

        train_total = 0


        # ====================================================
        # Training
        # ====================================================

        for images, labels in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            # -----------------------------------------------
            # Forward (bf16 autocast)
            #
            # bf16은 fp32와 exponent가 동일해서
            # GradScaler 없이 그냥 backward하면 된다.
            # -----------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )


            with torch.autocast(
                device_type="cuda",
                dtype=AMP_DTYPE,
                enabled=USE_BF16
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            # -----------------------------------------------
            # Backward
            #
            # autocast 밖에서 실행
            # gradient는 fp32로 생성됨
            # -----------------------------------------------

            loss.backward()


            # -----------------------------------------------
            # Random gradient sparsification
            # (fp32 gradient에 그대로 적용)
            # -----------------------------------------------

            random_gradient_sparsification(
                model,
                keep_ratio
            )


            # -----------------------------------------------
            # Optimizer update
            # -----------------------------------------------

            optimizer.step()


            # -----------------------------------------------
            # Statistics
            # -----------------------------------------------

            batch_size = images.size(0)

            train_loss_sum += (
                loss.item()
                * batch_size
            )


            _, predicted = torch.max(
                outputs,
                dim=1
            )


            train_total += batch_size

            train_correct += (
                predicted == labels
            ).sum().item()


        scheduler.step()


        # ====================================================
        # Train metrics
        # ====================================================

        train_loss = (
            train_loss_sum
            / train_total
        )

        train_accuracy = (
            100.0
            * train_correct
            / train_total
        )


        # ====================================================
        # Test
        # ====================================================

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion
        )


        current_lr = (
            optimizer.param_groups[0]["lr"]
        )


        # ====================================================
        # Save history
        # ====================================================

        history.append({

            "run_id": run_id,

            "optimizer": optimizer_name,

            "precision": (
                "bf16" if USE_BF16 else "fp32"
            ),

            "keep_ratio": keep_ratio,

            "keep_percent": keep_ratio * 100,

            "epoch": epoch,

            "learning_rate": current_lr,

            "train_loss": train_loss,

            "train_accuracy": train_accuracy,

            "test_loss": test_loss,

            "test_accuracy": test_accuracy
        })


        # ====================================================
        # Console
        # ====================================================

        print(

            f"[{epoch:02d}/{EPOCHS}] "

            f"LR={current_lr:.6f} | "

            f"Train Loss={train_loss:.4f} | "

            f"Train Acc={train_accuracy:.2f}% | "

            f"Test Loss={test_loss:.4f} | "

            f"Test Acc={test_accuracy:.2f}%"

        )


    # ========================================================
    # Save individual run CSV
    # ========================================================

    ratio_string = (
        f"{keep_ratio:g}"
        .replace(".", "p")
    )

    run_name = (
        f"run_{run_id:02d}_"
        f"{optimizer_name}_"
        f"{ratio_string}_bf16"
    )


    run_csv_path = os.path.join(
        OUTPUT_DIR,
        f"{run_name}.csv"
    )


    run_df = pd.DataFrame(
        history
    )


    run_df.to_csv(
        run_csv_path,
        index=False
    )


    # ========================================================
    # Final result
    # ========================================================

    final_row = history[-1]

    best_accuracy = max(
        x["test_accuracy"]
        for x in history
    )

    best_epoch = (
        max(
            history,
            key=lambda x:
            x["test_accuracy"]
        )["epoch"]
    )


    final_result = {

        "run_id": run_id,

        "optimizer": optimizer_name,

        "precision": (
            "bf16" if USE_BF16 else "fp32"
        ),

        "keep_ratio": keep_ratio,

        "keep_percent": keep_ratio * 100,

        "final_train_loss":
            final_row["train_loss"],

        "final_train_accuracy":
            final_row["train_accuracy"],

        "final_test_loss":
            final_row["test_loss"],

        "final_test_accuracy":
            final_row["test_accuracy"],

        "best_test_accuracy":
            best_accuracy,

        "best_epoch":
            best_epoch
    }


    print("\nFinal result:")

    print(
        f"Test Accuracy: "
        f"{final_row['test_accuracy']:.2f}%"
    )

    print(
        f"Best Accuracy: "
        f"{best_accuracy:.2f}% "
        f"(epoch {best_epoch})"
    )


    return history, final_result


# ============================================================
# 11. Run all experiments
# ============================================================

all_history = []

all_final_results = []


run_id = 0


for optimizer_name in OPTIMIZERS:

    for keep_ratio in KEEP_RATIOS:

        run_id += 1


        history, final_result = run_experiment(

            optimizer_name=optimizer_name,

            keep_ratio=keep_ratio,

            run_id=run_id
        )


        all_history.extend(
            history
        )

        all_final_results.append(
            final_result
        )


# ============================================================
# 12. Save complete epoch CSV
# ============================================================

all_history_df = pd.DataFrame(
    all_history
)


all_history_path = os.path.join(
    OUTPUT_DIR,
    "all_runs_epoch_metrics.csv"
)


all_history_df.to_csv(
    all_history_path,
    index=False
)


# ============================================================
# 13. Save final result CSV
# ============================================================

final_df = pd.DataFrame(
    all_final_results
)


# Accuracy 기준으로 정렬
final_df = final_df.sort_values(
    by="final_test_accuracy",
    ascending=False
)


final_csv_path = os.path.join(
    OUTPUT_DIR,
    "final_results.csv"
)


final_df.to_csv(
    final_csv_path,
    index=False
)


# ============================================================
# 14. Print final result table
# ============================================================

print("\n")
print("=" * 90)
print("FINAL RESULTS (bf16)")
print("=" * 90)


display_columns = [

    "run_id",

    "optimizer",

    "keep_percent",

    "final_test_loss",

    "final_test_accuracy",

    "best_test_accuracy",

    "best_epoch"
]


print(
    final_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================
# 15. Plot 1
#     SGD Train Loss
# ============================================================

plt.figure(
    figsize=(11, 7)
)


sgd_df = all_history_df[
    all_history_df["optimizer"] == "SGD"
]


for ratio in KEEP_RATIOS:

    subset = sgd_df[
        sgd_df["keep_ratio"] == ratio
    ]

    plt.plot(
        subset["epoch"],
        subset["train_loss"],
        label=f"{ratio * 100:g}%"
    )


plt.xlabel("Epoch")

plt.ylabel("Train Loss")

plt.title(
    "CIFAR-10 (bf16) - SGD Train Loss\n"
    "Random Gradient Sparsification"
)

plt.grid(True, alpha=0.3)

plt.legend(
    title="Gradient kept"
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_SGD_train_loss.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 16. Plot 2
#     Adam Train Loss
# ============================================================

plt.figure(
    figsize=(11, 7)
)


adam_df = all_history_df[
    all_history_df["optimizer"] == "Adam"
]


for ratio in KEEP_RATIOS:

    subset = adam_df[
        adam_df["keep_ratio"] == ratio
    ]

    plt.plot(
        subset["epoch"],
        subset["train_loss"],
        label=f"{ratio * 100:g}%"
    )


plt.xlabel("Epoch")

plt.ylabel("Train Loss")

plt.title(
    "CIFAR-10 (bf16) - Adam Train Loss\n"
    "Random Gradient Sparsification"
)

plt.grid(True, alpha=0.3)

plt.legend(
    title="Gradient kept"
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_Adam_train_loss.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 17. Plot 3
#     SGD Test Accuracy
# ============================================================

plt.figure(
    figsize=(11, 7)
)


for ratio in KEEP_RATIOS:

    subset = sgd_df[
        sgd_df["keep_ratio"] == ratio
    ]

    plt.plot(
        subset["epoch"],
        subset["test_accuracy"],
        label=f"{ratio * 100:g}%"
    )


plt.xlabel("Epoch")

plt.ylabel("Test Accuracy (%)")

plt.title(
    "CIFAR-10 (bf16) - SGD Test Accuracy\n"
    "Random Gradient Sparsification"
)

plt.grid(True, alpha=0.3)

plt.legend(
    title="Gradient kept"
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_SGD_test_accuracy.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 18. Plot 4
#     Adam Test Accuracy
# ============================================================

plt.figure(
    figsize=(11, 7)
)


for ratio in KEEP_RATIOS:

    subset = adam_df[
        adam_df["keep_ratio"] == ratio
    ]

    plt.plot(
        subset["epoch"],
        subset["test_accuracy"],
        label=f"{ratio * 100:g}%"
    )


plt.xlabel("Epoch")

plt.ylabel("Test Accuracy (%)")

plt.title(
    "CIFAR-10 (bf16) - Adam Test Accuracy\n"
    "Random Gradient Sparsification"
)

plt.grid(True, alpha=0.3)

plt.legend(
    title="Gradient kept"
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_Adam_test_accuracy.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 19. Plot 5
#     Final accuracy comparison - all runs
# ============================================================

plot_df = pd.DataFrame(
    all_final_results
)


plot_df["label"] = (
    plot_df["optimizer"]
    + " "
    + plot_df["keep_percent"]
        .map(lambda x: f"{x:g}%")
)


plot_df = plot_df.sort_values(
    "final_test_accuracy",
    ascending=False
)


plt.figure(
    figsize=(14, 7)
)


plt.bar(
    plot_df["label"],
    plot_df["final_test_accuracy"]
)


plt.xlabel(
    "Optimizer / Gradient kept"
)

plt.ylabel(
    "Final Test Accuracy (%)"
)

plt.title(
    "CIFAR-10 (bf16) - Final Test Accuracy\n"
    "All Experiments"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_final_accuracy_comparison.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 20. Plot 6
#     Final accuracy vs gradient ratio
# ============================================================

plt.figure(
    figsize=(10, 7)
)


sgd_final = (
    pd.DataFrame(all_final_results)
    .query("optimizer == 'SGD'")
    .sort_values("keep_ratio")
)


adam_final = (
    pd.DataFrame(all_final_results)
    .query("optimizer == 'Adam'")
    .sort_values("keep_ratio")
)


plt.plot(
    sgd_final["keep_percent"],
    sgd_final["final_test_accuracy"],
    marker="o",
    label="SGD"
)


plt.plot(
    adam_final["keep_percent"],
    adam_final["final_test_accuracy"],
    marker="o",
    label="Adam"
)


plt.xscale("log")


plt.xlabel(
    "Gradient kept (%)"
)

plt.ylabel(
    "Final Test Accuracy (%)"
)

plt.title(
    "CIFAR-10 (bf16) - Gradient Ratio vs Final Accuracy"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_accuracy_vs_gradient_ratio.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 21. Save a human-readable TXT summary
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "summary.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CIFAR-10 Random Gradient Sparsification (bf16)\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Device: {DEVICE}\n"
    )

    f.write(
        f"Precision: "
        f"{'bf16 autocast' if USE_BF16 else 'fp32'}\n"
    )

    f.write(
        f"Epochs: {EPOCHS}\n"
    )

    f.write(
        f"Batch size: {BATCH_SIZE}\n"
    )

    f.write(
        f"Runs: {len(all_final_results)}\n\n"
    )


    for _, row in final_df.iterrows():

        f.write(

            f"Run {int(row['run_id']):02d} | "

            f"{row['optimizer']:5s} | "

            f"{row['keep_percent']:6g}% | "

            f"Final Acc: "
            f"{row['final_test_accuracy']:.2f}% | "

            f"Best Acc: "
            f"{row['best_test_accuracy']:.2f}% | "

            f"Best Epoch: "
            f"{int(row['best_epoch'])}\n"
        )


# ============================================================
# 22. Done
# ============================================================

print("\n")
print("=" * 70)
print("ALL EXPERIMENTS FINISHED (bf16)")
print("=" * 70)

print(
    "Results saved to:"
)

print(
    os.path.abspath(OUTPUT_DIR)
)

print("\nFiles:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        "  ",
        filename
    )
