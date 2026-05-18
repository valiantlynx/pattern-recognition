"""Visualization utilities for the project."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .config import OUT_DIR, CLASS_NAMES


def save_fig(fig, filename):
    """Save figure to output directory."""
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_samples(X_raw, y, filename="task1_samples.png"):
    """Plot one random sample from each class."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for cls_idx, ax in enumerate(axes.flat):
        indices = np.where(y == cls_idx)[0]
        sample_idx = np.random.choice(indices)
        ax.imshow(X_raw[sample_idx], cmap="gray")
        ax.set_title(CLASS_NAMES[cls_idx], fontsize=11)
        ax.axis("off")
    fig.suptitle("Random Sample from Each Class", fontsize=14)
    fig.tight_layout()
    save_fig(fig, filename)


def plot_class_distribution(y_train, y_test, filename="task1_class_distribution.png"):
    """Plot class frequency bar charts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, labels, title in zip(
        axes, [y_train, y_test], ["Training Set (60,000)", "Test Set (10,000)"]
    ):
        unique, counts = np.unique(labels, return_counts=True)
        ax.bar(unique, counts, color="steelblue", edgecolor="black")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.set_xticks(range(10))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
        for i, c in enumerate(counts):
            ax.text(i, c + 50, str(c), ha="center", fontsize=8)
    fig.suptitle("Class Distribution", fontsize=14)
    fig.tight_layout()
    save_fig(fig, filename)


def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Plot a heatmap confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    save_fig(fig, filename)


def plot_training_curves(
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    title,
    filename="task4_training_curves.png",
):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label="Training Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()

    ax2.plot(train_accs, label="Training Accuracy")
    ax2.plot(val_accs, label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    save_fig(fig, filename)
