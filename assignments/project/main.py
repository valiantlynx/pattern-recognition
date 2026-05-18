"""
IKT215 Final Project: Fashion MNIST Classification
Main runner script - executes all 5 tasks.

Usage: uv run python assignments/project/main.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DEVICE, CLASS_NAMES, OUT_DIR
from src.data import FashionMNISTData
from src.visualization import (
    plot_samples,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_training_curves,
    save_fig,
)
from src.knn_classifier import KNNBenchmark
from src.pca_analysis import PCAAnalysis
from src.neural_network import ModelSelector, NNTrainer

from sklearn.metrics import accuracy_score


def task1_eda(data):
    """Task 1: Exploratory Analysis (15%)."""
    print("\n" + "=" * 70)
    print("TASK 1: EXPLORATORY ANALYSIS")
    print("=" * 70)

    data.summary()
    print("\nPreprocessing applied:")
    print("  1. Flatten 28x28 images to 784-dimensional vectors")
    print("  2. Normalize pixel values from [0, 255] to [0, 1]")
    print("     Justification: Normalization ensures features are on same scale,")
    print("     important for distance-based methods (kNN) and NN convergence.")

    plot_samples(data.X_train_raw, data.y_train)
    plot_class_distribution(data.y_train, data.y_test)

    print("\nClass distribution (perfectly balanced):")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(data.y_train == i)
        print(f"  {name:15s}: {count} ({count / len(data.y_train) * 100:.1f}%)")


def task2_knn(data):
    """Task 2: kNN Benchmark (20%)."""
    print("\n" + "=" * 70)
    print("TASK 2: kNN BENCHMARK (5-fold CV)")
    print("=" * 70)

    knn = KNNBenchmark()
    knn.cross_validate(data.X_train, data.y_train, n_jobs=8)
    knn.print_top_results()

    knn.fit(data.X_train, data.y_train)
    y_pred = knn.predict(data.X_test)
    metrics = knn.evaluate(data.y_test, y_pred)

    plot_confusion_matrix(
        data.y_test,
        y_pred,
        title=f"kNN Confusion Matrix (k={knn.best_params['k']}, "
        f"{knn.best_params['metric']}, {knn.best_params['weights']})",
        filename="task2_knn_confusion_matrix.png",
    )

    # Per-class analysis
    print("\n  Per-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        mask = data.y_test == i
        class_acc = accuracy_score(data.y_test[mask], y_pred[mask])
        print(f"    {name:15s}: {class_acc:.4f}")

    return knn, y_pred, metrics


def task3_pca(data, knn):
    """Task 3: PCA + kNN (15%)."""
    print("\n" + "=" * 70)
    print("TASK 3: PCA + kNN")
    print("=" * 70)

    pca = PCAAnalysis(knn.best_params)
    pca.run(data.X_train, data.y_train, data.X_test, data.y_test)
    pca.plot_accuracy_vs_components()
    pca.plot_runtime_vs_accuracy()

    return pca


def task4_nn(data):
    """Task 4: Dense Neural Network (30%)."""
    print("\n" + "=" * 70)
    print(f"TASK 4: DENSE NEURAL NETWORK (device={DEVICE})")
    print("=" * 70)

    selector = ModelSelector()
    selector.select(data.X_train, data.y_train)
    selector.train_final(data.X_train, data.y_train)
    nn_preds, nn_metrics = selector.evaluate(data.X_test, data.y_test)

    # Plot training curves for final model
    h = selector.final_history
    plot_training_curves(
        h["train_loss"],
        h["val_loss"],
        h["train_acc"],
        h["val_acc"],
        title=f"Final Model: {selector.best_name}",
        filename="task4_training_curves.png",
    )

    # Confusion matrix
    plot_confusion_matrix(
        data.y_test,
        nn_preds,
        title=f"NN Confusion Matrix ({selector.best_name})",
        filename="task4_nn_confusion_matrix.png",
    )

    # Model selection comparison figure
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(selector.results.keys())
    val_accs = [selector.results[n]["val_acc"] for n in names]
    params = [selector.results[n]["n_params"] for n in names]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    bars = ax.bar(range(len(names)), val_accs, color=colors, edgecolor="black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [f"{n}\n({p:,} params)" for n, p in zip(names, params)], fontsize=9
    )
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Model Selection: Validation Accuracy Comparison")
    ax.set_ylim(0.85, 0.95)
    for bar, acc in zip(bars, val_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    save_fig(fig, "task4_model_selection.png")

    # Per-class accuracy
    print("\n  Per-class accuracy (NN):")
    for i, name in enumerate(CLASS_NAMES):
        mask = data.y_test == i
        class_acc = accuracy_score(data.y_test[mask], nn_preds[mask])
        print(f"    {name:15s}: {class_acc:.4f}")

    return selector, nn_preds, nn_metrics


def task5_comparison(data, knn_preds, knn_metrics, nn_preds, nn_metrics, knn):
    """Task 5: Comparative Analysis (10%)."""
    print("\n" + "=" * 70)
    print("TASK 5: COMPARATIVE ANALYSIS")
    print("=" * 70)

    print(f"\n  {'Metric':<15} {'kNN':<15} {'Dense NN':<15}")
    print("  " + "-" * 45)
    for metric in ["accuracy", "precision", "recall"]:
        print(
            f"  {metric.capitalize():<15} {knn_metrics[metric]:<15.4f} {nn_metrics[metric]:<15.4f}"
        )

    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_names = ["Accuracy", "Precision", "Recall"]
    knn_scores = [
        knn_metrics["accuracy"],
        knn_metrics["precision"],
        knn_metrics["recall"],
    ]
    nn_scores = [nn_metrics["accuracy"], nn_metrics["precision"], nn_metrics["recall"]]
    x = np.arange(len(metrics_names))
    width = 0.35
    bars1 = ax.bar(
        x - width / 2,
        knn_scores,
        width,
        label=f"kNN (k={knn.best_params['k']})",
        color="#2196F3",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        nn_scores,
        width,
        label="Dense NN",
        color="#F44336",
        edgecolor="black",
    )
    ax.set_ylabel("Score")
    ax.set_title("kNN vs. Neural Network on Fashion MNIST")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0.8, 1.0)
    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.4f}",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    save_fig(fig, "task5_comparison.png")

    # Per-class comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    knn_per_class = [
        accuracy_score(data.y_test[data.y_test == i], knn_preds[data.y_test == i])
        for i in range(10)
    ]
    nn_per_class = [
        accuracy_score(data.y_test[data.y_test == i], nn_preds[data.y_test == i])
        for i in range(10)
    ]
    x = np.arange(10)
    ax.bar(
        x - width / 2,
        knn_per_class,
        width,
        label="kNN",
        color="#2196F3",
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        nn_per_class,
        width,
        label="Dense NN",
        color="#F44336",
        edgecolor="black",
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy: kNN vs. Neural Network")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0.6, 1.05)
    fig.tight_layout()
    save_fig(fig, "task5_per_class_comparison.png")

    # Final summary
    winner = (
        "Neural Network" if nn_metrics["accuracy"] > knn_metrics["accuracy"] else "kNN"
    )
    print(f"\n  Winner: {winner}")
    print(f"\n  Discussion:")
    print(
        f"    - NN learns hierarchical feature representations suitable for image data"
    )
    print(
        f"    - kNN relies on raw pixel distances which struggle with translation/rotation"
    )
    print(
        f"    - Shirt vs T-shirt/Pullover/Coat confusion is common (visually similar)"
    )
    print(f"\n  Improvement suggestion:")
    print(f"    - Use Convolutional Neural Network (CNN) to exploit spatial structure")
    print(f"    - Data augmentation (random crops, flips) to improve generalization")


def main():
    """Run all tasks."""
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")

    # Load data
    data = FashionMNISTData()

    # Task 1
    task1_eda(data)

    # Task 2
    knn, knn_preds, knn_metrics = task2_knn(data)

    # Task 3
    pca = task3_pca(data, knn)

    # Task 4
    selector, nn_preds, nn_metrics = task4_nn(data)

    # Task 5
    task5_comparison(data, knn_preds, knn_metrics, nn_preds, nn_metrics, knn)

    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETE")
    print(f"Figures saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
