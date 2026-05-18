"""PCA dimensionality reduction + kNN evaluation."""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from .config import OUT_DIR
from .visualization import save_fig


class PCAAnalysis:
    """Evaluate kNN with PCA at different component counts."""

    def __init__(self, knn_params: dict):
        """
        Args:
            knn_params: Best params from Task 2 (k, metric, weights).
        """
        self.knn_params = knn_params
        self.components_list = [5, 50, 150, 784]
        self.results = []

    def run(self, X_train, y_train, X_test, y_test):
        """Run PCA + kNN for each component count."""
        print(f"  Evaluating {len(self.components_list)} configurations...")

        for n_comp in tqdm(self.components_list, desc="  PCA + kNN"):
            t_start = time.time()

            if n_comp < 784:
                pca = PCA(n_components=n_comp, random_state=42)
                X_tr = pca.fit_transform(X_train)
                X_te = pca.transform(X_test)
                var_explained = np.sum(pca.explained_variance_ratio_)
            else:
                X_tr, X_te = X_train, X_test
                var_explained = 1.0

            knn = KNeighborsClassifier(
                n_neighbors=self.knn_params["k"],
                metric=self.knn_params["metric"],
                weights=self.knn_params["weights"],
                n_jobs=-1,
            )
            knn.fit(X_tr, y_train)
            y_pred = knn.predict(X_te)
            t_total = time.time() - t_start

            acc = accuracy_score(y_test, y_pred)
            self.results.append(
                {
                    "n_components": n_comp,
                    "accuracy": acc,
                    "time": t_total,
                    "variance_explained": var_explained,
                }
            )

        # Print results
        print(
            f"\n  {'Components':<12} {'Accuracy':<10} {'Time (s)':<10} {'Var Explained':<15}"
        )
        for r in self.results:
            print(
                f"  {r['n_components']:<12} {r['accuracy']:<10.4f} "
                f"{r['time']:<10.1f} {r['variance_explained']:<15.4f}"
            )

        return self.results

    def plot_accuracy_vs_components(self):
        """Plot accuracy vs number of PCA components."""
        fig, ax = plt.subplots(figsize=(8, 5))
        comps = [r["n_components"] for r in self.results]
        accs = [r["accuracy"] for r in self.results]
        ax.plot(comps, accs, "o-", color="steelblue", linewidth=2, markersize=8)
        for c, a in zip(comps, accs):
            ax.annotate(
                f"{a:.4f}",
                (c, a),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=10,
            )
        ax.set_xlabel("Number of Principal Components")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("kNN Accuracy vs. Number of PCA Components")
        ax.set_xscale("log")
        ax.set_xticks(comps)
        ax.set_xticklabels([str(c) for c in comps])
        fig.tight_layout()
        save_fig(fig, "task3_accuracy_vs_components.png")

    def plot_runtime_vs_accuracy(self):
        """Plot runtime vs accuracy trade-off."""
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in self.results:
            ax.scatter(r["time"], r["accuracy"], s=120, zorder=5)
            label = (
                f"{r['n_components']} comp"
                if r["n_components"] < 784
                else "784 (no PCA)"
            )
            ax.annotate(
                label,
                (r["time"], r["accuracy"]),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=10,
            )
        ax.set_xlabel("Total Runtime (PCA + kNN fit + predict) [seconds]")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Runtime vs. Accuracy Trade-off")
        fig.tight_layout()
        save_fig(fig, "task3_runtime_vs_accuracy.png")
