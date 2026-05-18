"""Project configuration and constants."""

import os
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Plot styling
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 12,
        "font.family": "serif",
        "axes.grid": True,
        "figure.figsize": (10, 6),
    }
)
