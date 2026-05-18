"""Task 2 GPU-accelerated kNN using torch.cdist on CUDA."""

import os, sys, time, pickle
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import SEED, DEVICE, CLASS_NAMES
from src.data import FashionMNISTData
from src.visualization import plot_confusion_matrix

print(f"Device: {DEVICE}")
data = FashionMNISTData()

X_train, y_train = data.X_train, data.y_train
X_test, y_test = data.X_test, data.y_test


def gpu_knn_predict(X_train_t, y_train_t, X_test_t, k, metric_p, weights):
    """GPU kNN prediction using torch.cdist."""
    # Compute distance matrix on GPU
    dists = torch.cdist(X_test_t, X_train_t, p=metric_p)  # (n_test, n_train)
    # Get k nearest neighbors
    topk_dists, topk_indices = torch.topk(dists, k=k, largest=False, dim=1)
    # Get labels of nearest neighbors
    topk_labels = y_train_t[topk_indices]  # (n_test, k)

    if weights == "uniform":
        # Majority vote
        preds = torch.mode(topk_labels, dim=1).values
    else:
        # Distance-weighted vote
        # Avoid division by zero
        w = 1.0 / (topk_dists + 1e-8)
        # Weighted vote per class
        n_classes = 10
        votes = torch.zeros(X_test_t.shape[0], n_classes, device=X_test_t.device)
        for c in range(n_classes):
            mask = (topk_labels == c).float()
            votes[:, c] = (w * mask).sum(dim=1)
        preds = votes.argmax(dim=1)

    return preds


def gpu_knn_cv(X_train, y_train, k_values, metrics, weight_options, n_splits=5):
    """5-fold CV for kNN entirely on GPU."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    combos = [(k, m, w) for k in k_values for m in metrics for w in weight_options]
    results = []

    # Move full training data to GPU
    X_gpu = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_gpu = torch.tensor(y_train, dtype=torch.long, device=DEVICE)

    for k, metric_name, weight in tqdm(combos, desc="GPU kNN CV"):
        metric_p = 2.0 if metric_name == "euclidean" else 1.0
        fold_accs = []

        for train_idx, val_idx in kfold.split(X_train):
            X_tr = X_gpu[train_idx]
            y_tr = y_gpu[train_idx]
            X_val = X_gpu[val_idx]
            y_val = y_gpu[val_idx]

            preds = gpu_knn_predict(X_tr, y_tr, X_val, k, metric_p, weight)
            acc = (preds == y_val).float().mean().item()
            fold_accs.append(acc)

        results.append(
            {
                "k": k,
                "metric": metric_name,
                "weights": weight,
                "mean_acc": np.mean(fold_accs),
                "std_acc": np.std(fold_accs),
            }
        )

    results = sorted(results, key=lambda x: x["mean_acc"], reverse=True)
    return results


# Run CV
print("\n" + "=" * 70)
print("TASK 2 (GPU): kNN BENCHMARK (5-fold CV on CUDA)")
print("=" * 70)

k_values = [1, 3, 5, 7, 9, 15, 21]
metrics = ["euclidean", "manhattan"]
weight_options = ["uniform", "distance"]

t0 = time.time()
cv_results = gpu_knn_cv(X_train, y_train, k_values, metrics, weight_options)
cv_time = time.time() - t0
print(f"\n  CV completed in {cv_time:.1f}s")

# Print top results
print(f"\n  Top 10 CV Results:")
print(f"  {'k':>3} {'metric':<12} {'weights':<10} {'mean_acc':<10} {'std_acc':<10}")
for r in cv_results[:10]:
    print(
        f"  {r['k']:>3} {r['metric']:<12} {r['weights']:<10} {r['mean_acc']:<10.4f} {r['std_acc']:<10.4f}"
    )

best = cv_results[0]
print(
    f"\n  Best: k={best['k']}, metric={best['metric']}, weights={best['weights']}, acc={best['mean_acc']:.4f}"
)

# Final evaluation on test set
print("\n  Evaluating on test set...")
X_train_gpu = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
y_train_gpu = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
X_test_gpu = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
y_test_gpu = torch.tensor(y_test, dtype=torch.long, device=DEVICE)

metric_p = 2.0 if best["metric"] == "euclidean" else 1.0
t0 = time.time()
preds = gpu_knn_predict(
    X_train_gpu, y_train_gpu, X_test_gpu, best["k"], metric_p, best["weights"]
)
pred_time = time.time() - t0
preds_np = preds.cpu().numpy()

acc = accuracy_score(y_test, preds_np)
prec = precision_score(y_test, preds_np, average="weighted")
rec = recall_score(y_test, preds_np, average="weighted")

print(f"\n  kNN Test Results (GPU):")
print(f"    Accuracy:  {acc:.4f}")
print(f"    Precision: {prec:.4f}")
print(f"    Recall:    {rec:.4f}")
print(f"    Pred time: {pred_time:.2f}s")

# Per-class accuracy
print("\n  Per-class accuracy:")
for i, name in enumerate(CLASS_NAMES):
    mask = y_test == i
    class_acc = accuracy_score(y_test[mask], preds_np[mask])
    print(f"    {name:15s}: {class_acc:.4f}")

# Confusion matrix
plot_confusion_matrix(
    y_test,
    preds_np,
    title=f"kNN Confusion Matrix (k={best['k']}, {best['metric']}, {best['weights']}) [GPU]",
    filename="task2_knn_confusion_matrix.png",
)

# Save results
results = {
    "best_params": {
        "k": best["k"],
        "metric": best["metric"],
        "weights": best["weights"],
    },
    "cv_results": cv_results,
    "knn_preds": preds_np,
    "knn_metrics": {"accuracy": acc, "precision": prec, "recall": rec},
}
with open(os.path.join(os.path.dirname(__file__), "results_task2.pkl"), "wb") as f:
    pickle.dump(results, f)

print(f"\nTask 2 (GPU) complete! Total CV time: {cv_time:.1f}s")
print("Results saved to results_task2.pkl")
