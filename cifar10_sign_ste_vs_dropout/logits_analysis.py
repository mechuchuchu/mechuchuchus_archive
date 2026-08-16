# ============================================================
# CIFAR-10 Logit KL / PCA Analysis
#
# Input:
#   cifar10_cat_to_dog_sign_ste_dropout_logits.pt
#
# Output:
#   14 PNG graphs
#   analysis_results.pt
#
# Sign:
#   1. 3 seeds adjacent KL
#   2. mean adjacent KL
#   3. 4 trajectories adjacent KL
#   4. 3 seeds from-start KL
#   5. mean from-start KL
#   6. 4 trajectories from-start KL
#   7. PCA trajectory
#
# Dropout:
#   same 7 graphs
# ============================================================

import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ============================================================
# Configuration
# ============================================================

RESULT_DIR = "./results"

INPUT_FILE = (
    "cifar10_cat_to_dog_sign_ste_dropout_logits.pt"
)

OUTPUT_DIR = os.path.join(
    RESULT_DIR,
    "kl_pca_analysis",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Plot settings
# ============================================================

FIGSIZE = (10, 6)
DPI = 200


# ============================================================
# Load
# ============================================================

def load_data():

    path = os.path.join(
        RESULT_DIR,
        INPUT_FILE,
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find:\n{path}"
        )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    return data


# ============================================================
# KL divergence
# ============================================================

def kl_divergence(
    logits_p,
    logits_q,
):
    """
    KL(P || Q)

    logits -> probability distribution

    P = softmax(logits_p)
    Q = softmax(logits_q)

    반환:
        마지막 dimension이 class dimension이라고 가정.
    """

    log_p = F.log_softmax(
        logits_p,
        dim=-1,
    )

    q = F.softmax(
        logits_q,
        dim=-1,
    )

    return F.kl_div(
        log_p,
        q,
        reduction="none",
    ).sum(dim=-1)


# ============================================================
# Adjacent KL
# ============================================================

def compute_adjacent_kl(logits):
    """
    logits:
        [100, 10]

    계산:
        KL(logit[0] || logit[1])
        KL(logit[1] || logit[2])
        ...
        KL(logit[98] || logit[99])

    return:
        [99]
    """

    current = logits[:-1]
    next_logits = logits[1:]

    kl = kl_divergence(
        current,
        next_logits,
    )

    return kl


# ============================================================
# From-start KL
# ============================================================

def compute_from_start_kl(logits):
    """
    항상 첫 번째 logits를 reference로 사용.

        KL(logit[0] || logit[0])
        KL(logit[0] || logit[1])
        ...
        KL(logit[0] || logit[99])

    return:
        [100]
    """

    start = logits[0:1]

    kl = kl_divergence(
        start.expand_as(logits),
        logits,
    )

    return kl


# ============================================================
# Mean logits
# ============================================================

def compute_mean_logits(
    logits_3_seeds,
):
    """
    logits_3_seeds:
        [3, 100, 10]

    return:
        [100, 10]
    """

    return logits_3_seeds.mean(
        dim=0
    )


# ============================================================
# Plot: 3 seeds adjacent KL
# ============================================================

def plot_three_seed_adjacent(
    seed_logits,
    model_type,
):
    """
    seed_logits:
        [3, 100, 10]
    """

    kl_values = []

    for seed in range(3):

        kl = compute_adjacent_kl(
            seed_logits[seed]
        )

        kl_values.append(
            kl.numpy()
        )

    plt.figure(
        figsize=FIGSIZE
    )

    x = range(1, 100)

    for seed in range(3):

        plt.plot(
            x,
            kl_values[seed],
            label=f"seed {seed}",
        )

    plt.xlabel(
        "Interpolation step (i → i+1)"
    )

    plt.ylabel(
        "KL(logit[i] || logit[i+1])"
    )

    plt.title(
        f"{model_type}: Adjacent KL - 3 Seeds"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        f"{model_type}_3seeds_adjacent_kl.png",
    )

    plt.savefig(
        path,
        dpi=DPI,
    )

    plt.close()

    return torch.tensor(
        kl_values
    )


# ============================================================
# Plot: mean adjacent KL
# ============================================================

def plot_mean_adjacent(
    seed_logits,
    model_type,
):
    mean_logits = compute_mean_logits(
        seed_logits
    )

    kl = compute_adjacent_kl(
        mean_logits
    )

    plt.figure(
        figsize=FIGSIZE
    )

    x = range(1, 100)

    plt.plot(
        x,
        kl.numpy(),
        label="mean logits",
    )

    plt.xlabel(
        "Interpolation step (i → i+1)"
    )

    plt.ylabel(
        "KL(mean[i] || mean[i+1])"
    )

    plt.title(
        f"{model_type}: Mean Logits Adjacent KL"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        f"{model_type}_mean_adjacent_kl.png",
    )

    plt.savefig(
        path,
        dpi=DPI,
    )

    plt.close()

    return kl


# ============================================================
# Plot: 4 trajectories adjacent KL
#
# seed0
# seed1
# seed2
# mean
# ============================================================

def plot_four_adjacent(
    seed_logits,
    model_type,
):
    mean_logits = compute_mean_logits(
        seed_logits
    )

    trajectories = [
        seed_logits[0],
        seed_logits[1],
        seed_logits[2],
        mean_logits,
    ]

    labels = [
        "seed 0",
        "seed 1",
        "seed 2",
        "mean",
    ]

    all_kl = []

    for logits in trajectories:

        kl = compute_adjacent_kl(
            logits
        )

        all_kl.append(
            kl.numpy()
        )

    plt.figure(
        figsize=FIGSIZE
    )

    x = range(1, 100)

    for label, kl in zip(
        labels,
        all_kl,
    ):

        plt.plot(
            x,
            kl,
            label=label,
        )

    plt.xlabel(
        "Interpolation step (i → i+1)"
    )

    plt.ylabel(
        "KL(logit[i] || logit[i+1])"
    )

    plt.title(
        f"{model_type}: Adjacent KL - 4 Trajectories"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        f"{model_type}_4trajectories_adjacent_kl.png",
    )

    plt.savefig(
        path,
        dpi=DPI,
    )

    plt.close()

    return torch.tensor(
        all_kl
    )


# ============================================================
# Plot: 3 seeds from-start KL
# ============================================================

def plot_three_seed_from_start(
    seed_logits,
    model_type,
):
    kl_values = []

    for seed in range(3):

        kl = compute_from_start_kl(
            seed_logits[seed]
        )

        kl_values.append(
            kl.numpy()
        )

    plt.figure(
        figsize=FIGSIZE
    )

    x = range(100)

    for seed in range(3):

        plt.plot(
            x,
            kl_values[seed],
            label=f"seed {seed}",
        )

    plt.xlabel(
        "Interpolation index"
    )

    plt.ylabel(
        "KL(logit[0] || logit[i])"
    )

    plt.title(
        f"{model_type}: KL From Start - 3 Seeds"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        f"{model_type}_3seeds_from_start_kl.png",
    )

    plt.savefig(
        path,
        dpi=DPI,
    )

    plt.close()

    return torch.tensor(
        kl_values
    )


# ============================================================
# Plot: mean from-start KL
# ============================================================

def plot_mean_from_start(
    seed_logits,
    model_type,
):
    mean_logits = compute_mean_logits(
        seed_logits
    )

    kl = compute_from_start_kl(
        mean_logits
    )

    plt.figure(
        figsize=FIGSIZE
    )

    x = range(100)

    plt.plot(
        x,
        kl.numpy(),
        label="mean logits",
    )

    plt.xlabel(
        "Interpolation index"
    )

    plt.ylabel(
        "KL(mean[0] || mean[i])"
    )

    plt.title(
        f"{model_type}: Mean Logits KL From Start"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        f"{model_type}_mean_from_start_kl.png",
    )

    plt.savefig(
        path,
        dpi=DPI,
    )

    plt.close()

    return kl


# ============================================================
# Plot: 4 trajectories from-start KL
# ============================================================

def plot_four_from_start(
    seed_logits,
    model_type,
):
    mean_logits = compute_mean_logits(
        seed_logits
    )

    trajectories = [
        seed_logits[0],
        seed_logits[1],
        seed_logits[2],
        mean_logits,
    ]

    labels = [
        "seed 0",
        "seed 1",
        "seed 2",
        "mean",
    ]

    all_kl = []

    for logits in trajectories:

        kl = compute_from_start_kl(
            logits
        )

        all_kl.append(
            kl.numpy()
        )

    plt.figure(
        figsize=FIGSIZE
    )

    x = range(100)

    for label, kl in zip(
        labels,
        all_kl,
    ):

        plt.plot(
            x,
            kl,
            label=label,
        )

    plt.xlabel(
        "Interpolation index"
    )

    plt.ylabel(
        "KL(logit[0] || logit[i])"
    )

    plt.title(
        f"{model_type}: KL From Start - 4 Trajectories"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        f"{model_type}_4trajectories_from_start_kl.png",
    )

    plt.savefig(
        path,
        dpi=DPI,
    )

    plt.close()

    return torch.tensor(
        all_kl
    )


# ============================================================
# PCA
# ============================================================

def compute_pca_trajectories(
    seed_logits,
):
    """
    seed_logits:
        [3, 100, 10]

    4 trajectories:
        seed0
        seed1
        seed2
        mean

    각 trajectory를 합쳐서 PCA를 fit한 뒤
    다시 4개 trajectory로 분리.

    return:
        [4, 100, 2]
    """

    mean_logits = compute_mean_logits(
        seed_logits
    )

    trajectories = torch.stack(
        [
            seed_logits[0],
            seed_logits[1],
            seed_logits[2],
            mean_logits,
        ],
        dim=0,
    )

    # [4, 100, 10]
    n_trajectories = trajectories.shape[0]
    n_points = trajectories.shape[1]

    flattened = trajectories.reshape(
        -1,
        trajectories.shape[-1],
    )

    pca = PCA(
        n_components=2
    )

    projected = pca.fit_transform(
        flattened.numpy()
    )

    projected = torch.from_numpy(
        projected
    ).float()

    projected = projected.reshape(
        n_trajectories,
        n_points,
        2,
    )

    explained_variance = (
        pca.explained_variance_ratio_
    )

    return (
        projected,
        torch.from_numpy(
            explained_variance
        ).float(),
    )


# ============================================================
# Plot PCA trajectory
# ============================================================

def plot_pca(
    seed_logits,
    model_type,
):
    projected, explained_variance = (
        compute_pca_trajectories(
            seed_logits
        )
    )

    labels = [
        "seed 0",
        "seed 1",
        "seed 2",
        "mean",
    ]

    plt.figure(
        figsize=FIGSIZE
    )

    for i, label in enumerate(labels):

        trajectory = projected[i]

        plt.plot(
            trajectory[:, 0].numpy(),
            trajectory[:, 1].numpy(),
            marker="o",
            markersize=2,
            linewidth=1.2,
            label=label,
        )

        # 시작점
        plt.scatter(
            trajectory[0, 0].item(),
            trajectory[0, 1].item(),
            marker="o",
            s=60,
        )

        # 끝점
        plt.scatter(
            trajectory[-1, 0].item(),
            trajectory[-1, 1].item(),
            marker="x",
            s=70,
        )

    pc1 = explained_variance[0].item()
    pc2 = explained_variance[1].item()

    plt.xlabel(
        f"PC1 ({pc1 * 100:.2f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({pc2 * 100:.2f}% variance)"
    )

    plt.title(
        f"{model_type}: Logit PCA Trajectory"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        f"{model_type}_pca_trajectory.png",
    )

    plt.savefig(
        path,
        dpi=DPI,
    )

    plt.close()

    return projected, explained_variance


# ============================================================
# Analyze one model family
# ============================================================

def analyze_model_family(
    all_logits,
    model_type,
):
    """
    all_logits:
        [3, 100, 10]
    """

    print()
    print("=" * 80)
    print(f"Analyzing: {model_type}")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Three seeds adjacent KL
    # --------------------------------------------------------

    three_adjacent = (
        plot_three_seed_adjacent(
            all_logits,
            model_type,
        )
    )

    # --------------------------------------------------------
    # 2. Mean adjacent KL
    # --------------------------------------------------------

    mean_adjacent = (
        plot_mean_adjacent(
            all_logits,
            model_type,
        )
    )

    # --------------------------------------------------------
    # 3. Four trajectories adjacent KL
    # --------------------------------------------------------

    four_adjacent = (
        plot_four_adjacent(
            all_logits,
            model_type,
        )
    )

    # --------------------------------------------------------
    # 4. Three seeds from-start KL
    # --------------------------------------------------------

    three_from_start = (
        plot_three_seed_from_start(
            all_logits,
            model_type,
        )
    )

    # --------------------------------------------------------
    # 5. Mean from-start KL
    # --------------------------------------------------------

    mean_from_start = (
        plot_mean_from_start(
            all_logits,
            model_type,
        )
    )

    # --------------------------------------------------------
    # 6. Four trajectories from-start KL
    # --------------------------------------------------------

    four_from_start = (
        plot_four_from_start(
            all_logits,
            model_type,
        )
    )

    # --------------------------------------------------------
    # 7. PCA trajectory
    # --------------------------------------------------------

    pca_trajectory, explained_variance = (
        plot_pca(
            all_logits,
            model_type,
        )
    )

    # --------------------------------------------------------
    # Return numerical results
    # --------------------------------------------------------

    return {
        "three_seeds_adjacent_kl":
            three_adjacent,

        "mean_adjacent_kl":
            mean_adjacent,

        "four_trajectories_adjacent_kl":
            four_adjacent,

        "three_seeds_from_start_kl":
            three_from_start,

        "mean_from_start_kl":
            mean_from_start,

        "four_trajectories_from_start_kl":
            four_from_start,

        "pca_trajectory":
            pca_trajectory,

        "pca_explained_variance":
            explained_variance,
    }


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load original result
    # --------------------------------------------------------

    data = load_data()

    logits = data["logits"]

    print("=" * 80)
    print("Loaded data")
    print("=" * 80)

    print(
        "logits shape:",
        logits.shape,
    )

    print(
        "model names:",
        data["model_names"],
    )

    print(
        "class A:",
        data["class_a"],
    )

    print(
        "class B:",
        data["class_b"],
    )

    # --------------------------------------------------------
    # Expected:
    #
    # [100, 6, 10]
    #
    # model order:
    #   sign_ste_seed0
    #   dropout_seed0
    #   sign_ste_seed1
    #   dropout_seed1
    #   sign_ste_seed2
    #   dropout_seed2
    # --------------------------------------------------------

    assert logits.ndim == 3

    assert logits.shape[0] == 100
    assert logits.shape[1] == 6
    assert logits.shape[2] == 10

    # --------------------------------------------------------
    # Extract Sign models
    # --------------------------------------------------------

    #
    # [100, 6, 10]
    #
    # model order:
    # 0 = sign seed0
    # 1 = dropout seed0
    # 2 = sign seed1
    # 3 = dropout seed1
    # 4 = sign seed2
    # 5 = dropout seed2
    #

    sign_logits = torch.stack(
        [
            logits[:, 0, :],
            logits[:, 2, :],
            logits[:, 4, :],
        ],
        dim=0,
    )

    # [3, 100, 10]

    # --------------------------------------------------------
    # Extract Dropout models
    # --------------------------------------------------------

    dropout_logits = torch.stack(
        [
            logits[:, 1, :],
            logits[:, 3, :],
            logits[:, 5, :],
        ],
        dim=0,
    )

    # [3, 100, 10]

    # --------------------------------------------------------
    # Analyze Sign
    # --------------------------------------------------------

    sign_results = analyze_model_family(
        sign_logits,
        "sign_ste",
    )

    # --------------------------------------------------------
    # Analyze Dropout
    # --------------------------------------------------------

    dropout_results = analyze_model_family(
        dropout_logits,
        "dropout",
    )

    # --------------------------------------------------------
    # Save numerical analysis
    # --------------------------------------------------------

    analysis_output = {

        "class_a":
            data["class_a"],

        "class_b":
            data["class_b"],

        "alphas":
            data["alphas"],

        "model_names":
            data["model_names"],

        "sign": sign_results,

        "dropout": dropout_results,
    }

    output_path = os.path.join(
        OUTPUT_DIR,
        "analysis_results.pt",
    )

    torch.save(
        analysis_output,
        output_path,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Analysis finished")
    print("=" * 80)

    print(
        "Output directory:",
        OUTPUT_DIR,
    )

    print(
        "Numerical results:",
        output_path,
    )

    print()
    print("Generated plots:")

    for filename in sorted(
        os.listdir(OUTPUT_DIR)
    ):
        if filename.endswith(".png"):
            print(
                "  ",
                filename,
            )


if __name__ == "__main__":
    main()