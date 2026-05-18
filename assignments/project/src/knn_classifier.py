"""kNN benchmark classifier with parallel cross-validation."""

import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from joblib import Parallel, delayed
from tqdm import tqdm

from .config import SEED


class KNNBenchmark:
    """
    kNN classifier with 5-fold CV for hyperparameter selection.
    Uses joblib for parallel evaluation of hyperparameter combinations.
    """

    def __init__(self):
        self.k_values = [1, 3, 5, 7, 9, 15, 21]
        self.metrics = ["euclidean", "manhattan"]
        self.weights = ["uniform", "distance"]
        self.best_params = None
        self.best_cv_acc = 0.0
        self.cv_results = []
        self.model = None

    def _evaluate_single_combo(self, k, metric, weight, X, y, kfold):
        """Evaluate a single hyperparameter combo with 5-fold CV."""
        fold_accs = []
        for train_idx, val_idx in kfold.split(X):
            knn = KNeighborsClassifier(
                n_neighbors=k, metric=metric, weights=weight, n_jobs=-1
            )
            knn.fit(X[train_idx], y[train_idx])
            y_pred = knn.predict(X[val_idx])
            fold_accs.append(accuracy_score(y[val_idx], y_pred))
        return {
            "k": k,
            "metric": metric,
            "weights": weight,
            "mean_acc": np.mean(fold_accs),
            "std_acc": np.std(fold_accs),
        }

    def cross_validate(self, X_train, y_train, n_jobs=8):
        """
        Run 5-fold CV for all hyperparameter combinations in parallel.
        Each combo runs sequentially (folds), but combos run in parallel.
        """
        kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
        combos = [
            (k, metric, weight)
            for k in self.k_values
            for metric in self.metrics
            for weight in self.weights
        ]

        print(f"  Testing {len(combos)} hyperparameter combinations (5-fold CV)...")
        print(f"  Parallelizing with {n_jobs} workers...")

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._evaluate_single_combo)(
                k, metric, weight, X_train, y_train, kfold
            )
            for k, metric, weight in tqdm(combos, desc="  kNN CV", unit="combo")
        )

        self.cv_results = sorted(results, key=lambda x: x["mean_acc"], reverse=True)
        self.best_params = {
            "k": self.cv_results[0]["k"],
            "metric": self.cv_results[0]["metric"],
            "weights": self.cv_results[0]["weights"],
        }
        self.best_cv_acc = self.cv_results[0]["mean_acc"]

        print(f"\n  Best CV Accuracy: {self.best_cv_acc:.4f}")
        print(
            f"  Best Params: k={self.best_params['k']}, "
            f"metric={self.best_params['metric']}, weights={self.best_params['weights']}"
        )
        return self.cv_results

    def fit(self, X_train, y_train):
        """Train final model with best params on full training set."""
        self.model = KNeighborsClassifier(
            n_neighbors=self.best_params["k"],
            metric=self.best_params["metric"],
            weights=self.best_params["weights"],
            n_jobs=-1,
        )
        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.fit_time = time.time() - t0
        return self

    def predict(self, X_test):
        """Predict on test set."""
        t0 = time.time()
        preds = self.model.predict(X_test)
        self.predict_time = time.time() - t0
        return preds

    def evaluate(self, y_true, y_pred):
        """Compute and print test metrics."""
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted")
        rec = recall_score(y_true, y_pred, average="weighted")
        print(f"\n  kNN Test Results:")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    Fit time:  {self.fit_time:.2f}s")
        print(f"    Pred time: {self.predict_time:.2f}s")
        return {"accuracy": acc, "precision": prec, "recall": rec}

    def print_top_results(self, n=10):
        """Print top N CV results."""
        print(f"\n  Top {n} CV Results:")
        print(
            f"  {'k':>3} {'metric':<12} {'weights':<10} {'mean_acc':<10} {'std_acc':<10}"
        )
        for r in self.cv_results[:n]:
            print(
                f"  {r['k']:>3} {r['metric']:<12} {r['weights']:<10} "
                f"{r['mean_acc']:<10.4f} {r['std_acc']:<10.4f}"
            )
