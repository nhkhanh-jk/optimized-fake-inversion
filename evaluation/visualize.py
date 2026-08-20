"""
FakeInversion - Visualization
===============================
Plotting tools: ROC curves, inversion feature visualization,
t-SNE embeddings, and heatmaps.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab
import seaborn as sns

import config
from utils import setup_logger


logger = setup_logger("visualize")

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_inversion_features(
    original,
    noise_map,
    reconstruction,
    caption: str = "",
    save_path: str = None,
    title: str = "",
):
    """Visualize the 3 components of FakeInversion features.

    Args:
        original: PIL Image or numpy array.
        noise_map: PIL Image or numpy array.
        reconstruction: PIL Image or numpy array.
        caption: BLIP-generated caption.
        save_path: Path to save the figure.
        title: Figure title.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(np.array(original))
    axes[0].set_title("Original Image (x)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(np.array(noise_map))
    axes[1].set_title("Decoded Noise Map D(ẑ_T)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(np.array(reconstruction))
    axes[2].set_title("Reconstruction D(ẑ₀)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    if caption:
        fig.suptitle(f"{title}\nCaption: \"{caption}\"", fontsize=11, y=1.02)
    elif title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_real_vs_fake_comparison(
    real_components: dict,
    fake_components: dict,
    save_path: str = None,
):
    """Side-by-side comparison of inversion features for real vs fake.

    Args:
        real_components: Dict from DDIMInverter.extract_components() for a real image.
        fake_components: Dict from DDIMInverter.extract_components() for a fake image.
        save_path: Path to save the figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Real image row
    axes[0, 0].imshow(np.array(real_components["original"]))
    axes[0, 0].set_title("REAL: Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.array(real_components["noise_map"]))
    axes[0, 1].set_title("REAL: Noise Map")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(np.array(real_components["reconstruction"]))
    axes[0, 2].set_title("REAL: Reconstruction")
    axes[0, 2].axis("off")

    # Fake image row
    axes[1, 0].imshow(np.array(fake_components["original"]))
    axes[1, 0].set_title("FAKE: Original")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.array(fake_components["noise_map"]))
    axes[1, 1].set_title("FAKE: Noise Map")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(np.array(fake_components["reconstruction"]))
    axes[1, 2].set_title("FAKE: Reconstruction")
    axes[1, 2].axis("off")

    fig.suptitle(
        "FakeInversion: Real vs Fake Comparison",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_roc_curves(
    roc_data: dict = None,
    roc_json_path: str = None,
    save_path: str = None,
):
    """Plot ROC curves for each generator model.

    Args:
        roc_data: Dict mapping model_name → {fpr, tpr, auc}.
        roc_json_path: Path to saved ROC JSON data.
        save_path: Path to save the figure.
    """
    if roc_data is None and roc_json_path:
        with open(roc_json_path, "r") as f:
            roc_data = json.load(f)

    if not roc_data:
        logger.warning("No ROC data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, len(roc_data)))

    for i, (model_name, data) in enumerate(sorted(roc_data.items())):
        fpr = np.array(data["fpr"])
        tpr = np.array(data["tpr"])
        auc = data["auc"]

        # Highlight training model differently
        if model_name == config.TRAIN_SOURCE_MODEL:
            ax.plot(fpr, tpr, color=colors[i], linewidth=2.5, linestyle="--",
                    label=f"{model_name} (train) AUC={auc:.3f}")
        else:
            ax.plot(fpr, tpr, color=colors[i], linewidth=1.5,
                    label=f"{model_name} AUC={auc:.3f}")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves per Generator Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_accuracy_heatmap(
    results: dict = None,
    results_json_path: str = None,
    save_path: str = None,
):
    """Plot heatmap of per-model detection accuracy.

    Args:
        results: Dict mapping model_name → metrics.
        results_json_path: Path to JSON results file.
        save_path: Path to save the figure.
    """
    if results is None and results_json_path:
        with open(results_json_path, "r") as f:
            results = json.load(f)

    if not results:
        logger.warning("No results data available.")
        return

    # Extract metrics
    models = []
    metrics_names = ["accuracy", "auc", "ap"]
    data_matrix = []

    for model_name in sorted(results.keys()):
        if model_name.startswith("_"):
            continue
        metrics = results[model_name]
        models.append(model_name)
        row = [metrics.get(m, 0) for m in metrics_names]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    fig, ax = plt.subplots(figsize=(8, max(6, len(models) * 0.5)))

    sns.heatmap(
        data_matrix,
        annot=True,
        fmt=".3f",
        xticklabels=["Accuracy", "AUC", "AP"],
        yticklabels=models,
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title(
        "Detection Performance per Generator Model",
        fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("Generator Model", fontsize=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_training_curves(
    log_dir: str = None,
    save_path: str = None,
):
    """Plot training loss and metrics curves from TensorBoard logs.

    Args:
        log_dir: Path to TensorBoard log directory.
        save_path: Path to save the figure.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.warning("TensorBoard not installed. Cannot plot training curves.")
        return

    if log_dir is None:
        log_dir = config.LOGS_DIR

    # Find event files
    event_files = []
    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.startswith("events.out"):
                event_files.append(os.path.join(root, f))

    if not event_files:
        logger.warning(f"No TensorBoard event files found in {log_dir}")
        return

    ea = EventAccumulator(event_files[0])
    ea.Reload()

    # Extract scalars
    tags = ea.Tags().get("scalars", [])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    if "train/loss" in tags and "val/loss" in tags:
        train_loss = [(s.step, s.value) for s in ea.Scalars("train/loss")]
        val_loss = [(s.step, s.value) for s in ea.Scalars("val/loss")]

        axes[0].plot(*zip(*train_loss), label="Train", linewidth=2)
        axes[0].plot(*zip(*val_loss), label="Val", linewidth=2)
        axes[0].set_title("Loss", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    if "train/accuracy" in tags and "val/accuracy" in tags:
        train_acc = [(s.step, s.value) for s in ea.Scalars("train/accuracy")]
        val_acc = [(s.step, s.value) for s in ea.Scalars("val/accuracy")]

        axes[1].plot(*zip(*train_acc), label="Train", linewidth=2)
        axes[1].plot(*zip(*val_acc), label="Val", linewidth=2)
        axes[1].set_title("Accuracy", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # AUC curves
    if "train/auc" in tags and "val/auc" in tags:
        train_auc = [(s.step, s.value) for s in ea.Scalars("train/auc")]
        val_auc = [(s.step, s.value) for s in ea.Scalars("val/auc")]

        axes[2].plot(*zip(*train_auc), label="Train", linewidth=2)
        axes[2].plot(*zip(*val_auc), label="Val", linewidth=2)
        axes[2].set_title("AUC", fontsize=13, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    fig.suptitle("Training Curves", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def generate_all_plots(results_dir: str = None):
    """Generate all visualization plots from saved results.

    Args:
        results_dir: Directory with JSON result files.
    """
    if results_dir is None:
        results_dir = config.RESULTS_DIR

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ROC curves
    roc_path = os.path.join(results_dir, "roc_data.json")
    if os.path.exists(roc_path):
        plot_roc_curves(
            roc_json_path=roc_path,
            save_path=os.path.join(plots_dir, "roc_curves.png"),
        )

    # Accuracy heatmap
    per_model_path = os.path.join(results_dir, "per_model_results.json")
    if os.path.exists(per_model_path):
        plot_accuracy_heatmap(
            results_json_path=per_model_path,
            save_path=os.path.join(plots_dir, "accuracy_heatmap.png"),
        )

    # Training curves
    plot_training_curves(
        save_path=os.path.join(plots_dir, "training_curves.png"),
    )

    logger.info(f"✅ All plots saved to {plots_dir}")


if __name__ == "__main__":
    generate_all_plots()
