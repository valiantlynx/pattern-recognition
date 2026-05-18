"""Run Task 4: Neural Network. Saves results to pickle."""

import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import DEVICE, CLASS_NAMES
from src.data import FashionMNISTData
from src.neural_network import ModelSelector
from src.visualization import plot_confusion_matrix, plot_training_curves, save_fig
from sklearn.metrics import accuracy_score

print(f"Device: {DEVICE}")
data = FashionMNISTData()

selector = ModelSelector()
selector.select(data.X_train, data.y_train)
selector.train_final(data.X_train, data.y_train)
nn_preds, nn_metrics = selector.evaluate(data.X_test, data.y_test)

# Plot training curves
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

# Model selection comparison
fig, ax = plt.subplots(figsize=(9, 5))
names = list(selector.results.keys())
val_accs = [selector.results[n]["val_acc"] for n in names]
params = [selector.results[n]["n_params"] for n in names]
colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
bars = ax.bar(range(len(names)), val_accs, color=colors, edgecolor="black")
ax.set_xticks(range(len(names)))
ax.set_xticklabels([f"{n}\n({p:,} params)" for n, p in zip(names, params)], fontsize=9)
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

# Save for task5
results = {"nn_preds": nn_preds, "nn_metrics": nn_metrics}
with open(os.path.join(os.path.dirname(__file__), "results_task4.pkl"), "wb") as f:
    pickle.dump(results, f)
print("\nTask 4 complete! Results saved to results_task4.pkl")
