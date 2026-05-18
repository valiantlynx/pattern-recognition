"""Run Task 3: PCA + kNN. Requires results_task2.pkl."""

import os, sys, pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import FashionMNISTData
from src.pca_analysis import PCAAnalysis

data = FashionMNISTData()

pkl_path = os.path.join(os.path.dirname(__file__), "results_task2.pkl")
with open(pkl_path, "rb") as f:
    t2 = pickle.load(f)

pca = PCAAnalysis(t2["best_params"])
pca.run(data.X_train, data.y_train, data.X_test, data.y_test)
pca.plot_accuracy_vs_components()
pca.plot_runtime_vs_accuracy()
print("\nTask 3 complete!")
