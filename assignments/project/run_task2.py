"""Run Task 2: kNN Benchmark only. Saves results to pickle for later tasks."""

import os, sys, pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import FashionMNISTData
from main import task2_knn

data = FashionMNISTData()
knn, knn_preds, knn_metrics = task2_knn(data)

# Save results for task3 and task5
results = {
    "best_params": knn.best_params,
    "cv_results": knn.cv_results,
    "knn_preds": knn_preds,
    "knn_metrics": knn_metrics,
}
with open(os.path.join(os.path.dirname(__file__), "results_task2.pkl"), "wb") as f:
    pickle.dump(results, f)
print("\nTask 2 complete! Results saved to results_task2.pkl")
