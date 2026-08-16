import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# HuggingFace datasets
# pip install datasets
from datasets import load_dataset
num_workers = max(1, os.cpu_count() - 1)

# ============================================================
# Configuration
# ============================================================

DATA_ROOT = "./data"
SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 서로 다른 seed 3개
SEEDS = [0, 1, 2]

NUM_CLASSES = 10
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 5e-4

# DataLoader worker 프로세스 개수
# CPU 코어 수, 디스크/네트워크 I/O 상황에 맞게 조절
NUM_WORKERS = num_workers

# interpolation point 개수
NUM_INTERPOLATIONS = 100

# HuggingFace Hub의 CIFAR-10 dataset repo id
# torchvision 공식 서버(www.cs.toronto.edu)가 느리거나
# 다운로드가 안 될 때가 많아서 HF hub에서 받아온다.
HF_DATASET_ID = "uoft-cs/cifar10"

# ============================================================
# CIFAR-10 class
#
# uoft-cs/cifar10의 label 순서는 torchvision CIFAR10과 동일하다.
# (airplane=0, automobile=1, ..., truck=9)
# ============================================================

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# 원하는 두 클래스
CLASS_A = "cat"
CLASS_B = "dog"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# STE Sign
# ============================================================

class SignSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator
        return grad_output


class SignActivation(nn.Module):

    def forward(self, x):
        return SignSTE.apply(x)


# ============================================================
# Sign CNN
#
# Sign activation은 네트워크 전체에서 정확히 1번
# ============================================================

class SignCNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(

            # 32 x 32
            nn.Conv2d(
                3, 64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 16 x 16
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128, 128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),

            # =================================================
            # SIGN
            # 딱 한 번
            # =================================================
            SignActivation(),

            nn.MaxPool2d(2),

            # 8 x 8
            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256, 256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 4 x 4
            nn.Conv2d(
                256, 512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                512, 512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(
            512,
            num_classes,
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


# ============================================================
# Dropout CNN
#
# Sign + Dropout 함께 사용
# (SignCNN과 동일한 위치에 Sign을 넣고, 추가로 Dropout2d를 사용)
# ============================================================

class DropoutCNN(nn.Module):

    def __init__(
        self,
        num_classes=10,
        dropout=0.3,
    ):
        super().__init__()

        self.features = nn.Sequential(

            # =================================================
            # 32 x 32
            # =================================================

            nn.Conv2d(
                3, 64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(
                64, 64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # =================================================
            # 16 x 16
            # =================================================

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(
                128, 128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),

            # =================================================
            # SIGN
            #
            # DropoutCNN에서도 정확히 1번
            # SignCNN과 동일한 위치
            # =================================================

            SignActivation(),

            nn.Dropout2d(dropout),

            nn.MaxPool2d(2),

            # =================================================
            # 8 x 8
            # =================================================

            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(
                256, 256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # =================================================
            # 4 x 4
            # =================================================

            nn.Conv2d(
                256, 512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(
                512, 512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(dropout),

            nn.Linear(
                512,
                num_classes,
            ),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# HuggingFace CIFAR-10 wrapper
#
# torchvision.datasets.CIFAR10과 동일한 인터페이스로 맞춘다:
#   - dataset[idx] -> (transformed_image, label)
#   - dataset.targets -> label 전체 리스트
#     (find_image_by_class에서 이미지를 디코딩하지 않고
#      클래스별 인덱스만 빠르게 찾기 위해 필요)
# ============================================================

class HFCIFAR10(Dataset):

    def __init__(self, split="train", transform=None):
        super().__init__()

        # 최초 1회만 실제로 다운로드하고,
        # 이후에는 HF hub 캐시(~/.cache/huggingface)에서 읽는다.
        self.hf_dataset = load_dataset(
            HF_DATASET_ID,
            split=split,
        )

        self.transform = transform

        # torchvision CIFAR10의 .targets와 동일한 역할.
        # 이미지를 디코딩하지 않고 label 컬럼만 가져오므로 빠르다.
        self.targets = list(self.hf_dataset["label"])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]

        image = example["img"]  # PIL.Image (RGB)
        label = example["label"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# CIFAR-10 Dataset
# ============================================================

def get_datasets():

    train_transform = transforms.Compose([
        transforms.RandomCrop(
            32,
            padding=4,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=(
                0.4914,
                0.4822,
                0.4465,
            ),
            std=(
                0.2470,
                0.2435,
                0.2616,
            ),
        ),
    ])

    # interpolation용
    #
    # augmentation을 사용하지 않는다.
    # 그래야 같은 CIFAR 이미지가 항상 동일하게 나온다.
    eval_transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=(
                0.4914,
                0.4822,
                0.4465,
            ),
            std=(
                0.2470,
                0.2435,
                0.2616,
            ),
        ),
    ])

    train_dataset = HFCIFAR10(
        split="train",
        transform=train_transform,
    )

    eval_dataset = HFCIFAR10(
        split="train",
        transform=eval_transform,
    )

    return train_dataset, eval_dataset


# ============================================================
# Find a specific CIFAR-10 image
# ============================================================

def find_image_by_class(
    dataset,
    class_name,
    occurrence=0,
):
    """
    class_name에 해당하는 CIFAR-10 이미지 중
    occurrence번째 이미지를 반환한다.

    예:
        occurrence=0 -> 해당 클래스의 첫 번째 이미지
        occurrence=1 -> 해당 클래스의 두 번째 이미지
    """

    if class_name not in CIFAR10_CLASSES:
        raise ValueError(
            f"Unknown class: {class_name}\n"
            f"Available classes: {CIFAR10_CLASSES}"
        )

    target_label = CIFAR10_CLASSES.index(
        class_name
    )

    count = 0

    for idx in range(len(dataset)):

        # dataset.targets는 transform과 무관하게
        # 원래 CIFAR label을 가지고 있다.
        label = dataset.targets[idx]

        if label == target_label:

            if count == occurrence:
                image, label = dataset[idx]

                return (
                    image,
                    label,
                    idx,
                )

            count += 1

    raise RuntimeError(
        f"Could not find image for class={class_name}"
    )


# ============================================================
# Select two different classes
# ============================================================

def get_two_cifar_images(
    dataset,
    class_a,
    class_b,
    occurrence_a=0,
    occurrence_b=0,
):
    """
    class_a의 이미지 한 장,
    class_b의 이미지 한 장을 선택한다.
    """

    if class_a == class_b:
        raise ValueError(
            "class_a와 class_b는 서로 달라야 합니다."
        )

    x0, y0, idx0 = find_image_by_class(
        dataset,
        class_a,
        occurrence_a,
    )

    x1, y1, idx1 = find_image_by_class(
        dataset,
        class_b,
        occurrence_b,
    )

    print()
    print("=" * 80)
    print("Interpolation images")
    print("=" * 80)

    print(
        f"Image A : "
        f"class={class_a}, "
        f"label={y0}, "
        f"dataset index={idx0}"
    )

    print(
        f"Image B : "
        f"class={class_b}, "
        f"label={y1}, "
        f"dataset index={idx1}"
    )

    return (
        x0,
        x1,
        y0,
        y1,
        idx0,
        idx1,
    )


# ============================================================
# Linear interpolation
# ============================================================

def make_interpolations(
    x0,
    x1,
    num_points=100,
):
    """
    x0 -> x1 사이를 num_points개로 선형보간한다.

    alpha=0:
        x0

    alpha=1:
        x1
    """

    alphas = torch.linspace(
        0.0,
        1.0,
        num_points,
    )

    interpolation = (
        (1.0 - alphas[:, None, None, None])
        * x0[None]
        +
        alphas[:, None, None, None]
        * x1[None]
    )

    return interpolation, alphas


# ============================================================
# Training
# ============================================================

def train_model(
    model,
    train_loader,
    seed,
):
    set_seed(seed)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    # epoch별 loss/acc/lr을 기록해서
    # 나중에 json으로 저장한다.
    history = []

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        scheduler.step()

        epoch_loss = (
            running_loss / total
        )

        epoch_acc = (
            correct / total
        )

        epoch_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch "
            f"{epoch + 1:03d}/{EPOCHS} | "
            f"loss={epoch_loss:.4f} | "
            f"acc={epoch_acc:.4f} | "
            f"lr={epoch_lr:.6f}"
        )

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "acc": epoch_acc,
            "lr": epoch_lr,
        })

    return model, history


# ============================================================
# Logits extraction
# ============================================================

@torch.no_grad()
def get_logits(
    model,
    inputs,
):
    """
    inputs:
        [N, 3, 32, 32]

    return:
        [N, 10]
    """

    # 중요:
    # DropoutCNN도 inference에서는 dropout을 꺼야 한다.
    model.eval()

    logits_list = []

    for start in range(
        0,
        len(inputs),
        BATCH_SIZE,
    ):
        batch = inputs[
            start:start + BATCH_SIZE
        ]

        batch = batch.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(batch)

        logits_list.append(
            logits.cpu()
        )

    return torch.cat(
        logits_list,
        dim=0,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("CIFAR-10 Sign STE vs Dropout")
    print("=" * 80)

    print(f"Device      : {DEVICE}")
    print(f"Class A     : {CLASS_A}")
    print(f"Class B     : {CLASS_B}")
    print(f"Seeds       : {SEEDS}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Interpolation points: {NUM_INTERPOLATIONS}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset, interpolation_dataset = (
        get_datasets()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Select CIFAR-10 images
    # --------------------------------------------------------

    (
        x0,
        x1,
        y0,
        y1,
        idx0,
        idx1,
    ) = get_two_cifar_images(
        interpolation_dataset,
        CLASS_A,
        CLASS_B,
    )

    # --------------------------------------------------------
    # Linear interpolation
    # --------------------------------------------------------

    interpolation, alphas = (
        make_interpolations(
            x0,
            x1,
            NUM_INTERPOLATIONS,
        )
    )

    print()
    print(
        "Interpolation shape:",
        interpolation.shape,
    )

    # [100, 3, 32, 32]

    # --------------------------------------------------------
    # Save interpolation inputs
    # --------------------------------------------------------

    torch.save(
        {
            "x0": x0,
            "x1": x1,

            "y0": y0,
            "y1": y1,

            "class_a": CLASS_A,
            "class_b": CLASS_B,

            "image_index_a": idx0,
            "image_index_b": idx1,

            "alphas": alphas,

            "interpolation": interpolation,
        },
        os.path.join(
            SAVE_DIR,
            "cifar10_cat_dog_interpolation_inputs.pt",
        ),
    )

    # --------------------------------------------------------
    # Train 6 models
    # --------------------------------------------------------

    models = []
    model_names = []

    # {model_name: [ {epoch, loss, acc, lr}, ... ]}
    all_history = {}

    for seed in SEEDS:

        # ====================================================
        # Sign CNN
        # ====================================================

        print()
        print("=" * 80)
        print(
            f"Training SignCNN | seed={seed}"
        )
        print("=" * 80)

        set_seed(seed)

        sign_model = SignCNN(
            num_classes=NUM_CLASSES
        )

        sign_model, sign_history = train_model(
            sign_model,
            train_loader,
            seed,
        )

        model_name = (
            f"sign_ste_seed{seed}"
        )

        models.append(sign_model)
        model_names.append(model_name)
        all_history[model_name] = sign_history

        torch.save(
            sign_model.state_dict(),
            os.path.join(
                SAVE_DIR,
                f"{model_name}.pt",
            ),
        )

        with open(
            os.path.join(
                SAVE_DIR,
                f"{model_name}_loss.json",
            ),
            "w",
        ) as f:
            json.dump(
                sign_history,
                f,
                indent=2,
            )

        # ====================================================
        # Dropout CNN
        # ====================================================

        print()
        print("=" * 80)
        print(
            f"Training DropoutCNN | seed={seed}"
        )
        print("=" * 80)

        set_seed(seed)

        dropout_model = DropoutCNN(
            num_classes=NUM_CLASSES,
            dropout=0.3,
        )

        dropout_model, dropout_history = train_model(
            dropout_model,
            train_loader,
            seed,
        )

        model_name = (
            f"dropout_seed{seed}"
        )

        models.append(dropout_model)
        model_names.append(model_name)
        all_history[model_name] = dropout_history

        torch.save(
            dropout_model.state_dict(),
            os.path.join(
                SAVE_DIR,
                f"{model_name}.pt",
            ),
        )

        with open(
            os.path.join(
                SAVE_DIR,
                f"{model_name}_loss.json",
            ),
            "w",
        ) as f:
            json.dump(
                dropout_history,
                f,
                indent=2,
            )

    # --------------------------------------------------------
    # Save combined loss history (all 6 models)
    # --------------------------------------------------------

    all_history_path = os.path.join(
        SAVE_DIR,
        "all_models_loss_history.json",
    )

    with open(
        all_history_path,
        "w",
    ) as f:
        json.dump(
            all_history,
            f,
            indent=2,
        )

    print()
    print(
        "Saved combined loss history:",
        all_history_path,
    )

    # --------------------------------------------------------
    # Extract logits from all 6 models
    # --------------------------------------------------------

    all_logits = []

    for model_name, model in zip(
        model_names,
        models,
    ):

        print()
        print(
            f"Extracting logits: {model_name}"
        )

        logits = get_logits(
            model,
            interpolation,
        )

        print(
            "logits shape:",
            logits.shape,
        )

        # [100, 10]

        all_logits.append(logits)

    # --------------------------------------------------------
    # Stack
    # --------------------------------------------------------

    # [6, 100, 10]
    all_logits = torch.stack(
        all_logits,
        dim=0,
    )

    # [100, 6, 10]
    all_logits = all_logits.permute(
        1,
        0,
        2,
    ).contiguous()

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    output = {

        # ----------------------------------------------------
        # Original images
        # ----------------------------------------------------

        "x0": x0,
        "x1": x1,

        "y0": y0,
        "y1": y1,

        "class_a": CLASS_A,
        "class_b": CLASS_B,

        "image_index_a": idx0,
        "image_index_b": idx1,

        # ----------------------------------------------------
        # Interpolation
        # ----------------------------------------------------

        # [100]
        "alphas": alphas,

        # [100, 3, 32, 32]
        "interpolation": interpolation,

        # ----------------------------------------------------
        # Logits
        # ----------------------------------------------------

        # [100, 6, 10]
        "logits": all_logits,

        # ----------------------------------------------------
        # Model information
        # ----------------------------------------------------

        "model_names": model_names,
        "seeds": SEEDS,
    }

    output_path = os.path.join(
        SAVE_DIR,
        (
            "cifar10_"
            f"{CLASS_A}_to_{CLASS_B}_"
            "sign_ste_dropout_logits.pt"
        ),
    )

    torch.save(
        output,
        output_path,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Finished")
    print("=" * 80)

    print(
        "Interpolation:",
        interpolation.shape,
    )

    print(
        "Logits:",
        all_logits.shape,
    )

    print(
        "Models:",
        model_names,
    )

    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()