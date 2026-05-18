"""Data loading and preprocessing for Fashion MNIST."""

import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor

from .config import PROJECT_DIR


class FashionMNISTData:
    """Handles loading and preprocessing of Fashion MNIST dataset."""

    def __init__(self):
        data_root = PROJECT_DIR  # downloads to assignments/project/data/
        self.train_dataset = datasets.FashionMNIST(
            root=data_root, train=True, download=True, transform=ToTensor()
        )
        self.test_dataset = datasets.FashionMNIST(
            root=data_root, train=False, download=True, transform=ToTensor()
        )

        # Raw numpy arrays
        self.X_train_raw = self.train_dataset.data.numpy().astype(np.float32)
        self.y_train = self.train_dataset.targets.numpy()
        self.X_test_raw = self.test_dataset.data.numpy().astype(np.float32)
        self.y_test = self.test_dataset.targets.numpy()

        # Preprocessing: flatten + normalize to [0, 1]
        self.X_train = self.X_train_raw.reshape(-1, 784) / 255.0
        self.X_test = self.X_test_raw.reshape(-1, 784) / 255.0

    def summary(self):
        """Print dataset summary."""
        print(
            f"Training set: {self.X_train_raw.shape} -> flattened {self.X_train.shape}"
        )
        print(f"Test set:     {self.X_test_raw.shape} -> flattened {self.X_test.shape}")
        print(
            f"Pixel range (raw): [{self.X_train_raw.min():.0f}, {self.X_train_raw.max():.0f}]"
        )
        print(
            f"Pixel range (norm): [{self.X_train.min():.2f}, {self.X_train.max():.2f}]"
        )
        print(f"Classes: 10 (balanced, 6000 per class in training)")
