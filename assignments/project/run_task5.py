"""Run Task 5: Comparative Analysis. Requires results_task2.pkl and results_task4.pkl."""

import os, sys, pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import FashionMNISTData
from src.knn_classifier import KNNBenchmark
from main import task5_comparison

data = FashionMNISTData()

base = os.path.dirname(__file__)
with open(os.path.join(base, "results_task2.pkl"), "rb") as f:
    t2 = pickle.load(f)
with open(os.path.join(base, "results_task4.pkl"), "rb") as f:
    t4 = pickle.load(f)

# Create a minimal knn object for task5
knn = KNNBenchmark()
knn.best_params = t2["best_params"]

task5_comparison(
    data, t2["knn_preds"], t2["knn_metrics"], t4["nn_preds"], t4["nn_metrics"], knn
)
print("\nTask 5 complete!")
