"""Task 2 using sklearn GridSearchCV with n_jobs=-1."""

import os, sys, time, pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import SEED, CLASS_NAMES
from src.data import FashionMNISTData
from src.visualization import plot_confusion_matrix

data = FashionMNISTData()
X_train, y_train = data.X_train, data.y_train
X_test, y_test = data.X_test, data.y_test

print("\n" + "=" * 70)
print("TASK 2 (GridSearchCV): kNN BENCHMARK")
print("=" * 70)

param_grid = {
    "n_neighbors": [1, 3, 5, 7, 9, 15, 21],
    "metric": ["euclidean", "manhattan"],
    "weights": ["uniform", "distance"],
}

print(f"  Grid: {7 * 2 * 2} combinations, 5-fold CV")
print(f"  Using all CPU cores (n_jobs=-1)...")

t0 = time.time()
grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(algorithm="brute"),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="accuracy",
)
grid_search.fit(X_train, y_train)
cv_time = time.time() - t0

print(f"\n  CV completed in {cv_time:.1f}s")
print(f"  Best params: {grid_search.best_params_}")
print(f"  Best CV accuracy: {grid_search.best_score_:.4f}")

# Predict on test set
t0 = time.time()
preds = grid_search.predict(X_test)
pred_time = time.time() - t0

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, average="weighted")
rec = recall_score(y_test, preds, average="weighted")

print(f"\n  kNN Test Results (GridSearchCV):")
print(f"    Accuracy:  {acc:.4f}")
print(f"    Precision: {prec:.4f}")
print(f"    Recall:    {rec:.4f}")
print(f"    Pred time: {pred_time:.2f}s")

# Per-class accuracy
print("\n  Per-class accuracy:")
for i, name in enumerate(CLASS_NAMES):
    mask = y_test == i
    class_acc = accuracy_score(y_test[mask], preds[mask])
    print(f"    {name:15s}: {class_acc:.4f}")

# Save results
bp = grid_search.best_params_
cv_results = []
results_df = grid_search.cv_results_
for i in range(len(results_df["params"])):
    cv_results.append(
        {
            "k": results_df["params"][i]["n_neighbors"],
            "metric": results_df["params"][i]["metric"],
            "weights": results_df["params"][i]["weights"],
            "mean_acc": results_df["mean_test_score"][i],
            "std_acc": results_df["std_test_score"][i],
        }
    )
cv_results = sorted(cv_results, key=lambda x: x["mean_acc"], reverse=True)

results = {
    "best_params": {
        "k": bp["n_neighbors"],
        "metric": bp["metric"],
        "weights": bp["weights"],
    },
    "cv_results": cv_results,
    "knn_preds": preds,
    "knn_metrics": {"accuracy": acc, "precision": prec, "recall": rec},
}
with open(
    os.path.join(os.path.dirname(__file__), "results_task2_gridsearch.pkl"), "wb"
) as f:
    pickle.dump(results, f)

print(f"\nTask 2 (GridSearchCV) complete! Total time: {cv_time:.1f}s")
