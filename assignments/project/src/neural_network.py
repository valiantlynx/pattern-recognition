"""Dense neural network with model selection for Fashion MNIST."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm

from .config import SEED, DEVICE


class DenseNN(nn.Module):
    """
    Fully connected neural network for multi-class classification.
    Configurable number of hidden layers, dropout, and activation.
    """

    def __init__(
        self, hidden_layers: list, dropout: float = 0.3, activation: str = "relu"
    ):
        super().__init__()
        layers = []
        input_dim = 784
        act_fn = nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.1)

        for h in hidden_layers:
            layers.extend(
                [
                    nn.Linear(input_dim, h),
                    nn.BatchNorm1d(h),
                    act_fn,
                    nn.Dropout(dropout),
                ]
            )
            input_dim = h
        layers.append(nn.Linear(input_dim, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class NNTrainer:
    """Trains a DenseNN with early stopping and learning rate scheduling."""

    def __init__(
        self, epochs=150, lr=0.001, batch_size=256, patience=20, weight_decay=1e-4
    ):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.weight_decay = weight_decay

    def train(self, model, X_train, y_train, X_val, y_val, verbose=False):
        """
        Train model with early stopping on validation loss.
        Returns training history dict.
        """
        model = model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=7
        )

        # Move data to device
        X_train_t = torch.FloatTensor(X_train).to(DEVICE)
        y_train_t = torch.LongTensor(y_train).to(DEVICE)
        X_val_t = torch.FloatTensor(X_val).to(DEVICE)
        y_val_t = torch.LongTensor(y_val).to(DEVICE)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        epoch_iter = (
            tqdm(range(self.epochs), desc="    Training", leave=False)
            if verbose
            else range(self.epochs)
        )

        for epoch in epoch_iter:
            # --- Training ---
            model.train()
            epoch_loss, correct, total = 0.0, 0, 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
                correct += (outputs.argmax(1) == y_batch).sum().item()
                total += y_batch.size(0)

            history["train_loss"].append(epoch_loss / total)
            history["train_acc"].append(correct / total)

            # --- Validation ---
            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t)
                val_loss = criterion(val_out, y_val_t).item()
                val_acc = (val_out.argmax(1) == y_val_t).float().mean().item()
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            scheduler.step(val_loss)

            # --- Early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        tqdm.write(f"    Early stopping at epoch {epoch + 1}")
                    break

            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(
                    t_loss=f"{history['train_loss'][-1]:.4f}",
                    v_loss=f"{val_loss:.4f}",
                    v_acc=f"{val_acc:.4f}",
                )

        if best_state:
            model.load_state_dict(best_state)
        return history

    def predict(self, model, X):
        """Get predictions from model."""
        model = model.to(DEVICE)
        model.eval()
        X_t = torch.FloatTensor(X).to(DEVICE)
        with torch.no_grad():
            preds = model(X_t).argmax(1).cpu().numpy()
        return preds


class ModelSelector:
    """
    Implements validation error minimization for model selection.
    Compares multiple architectures and selects the best.
    """

    MODEL_CONFIGS = {
        "A: [256, 128]": {"hidden_layers": [256, 128], "dropout": 0.3, "lr": 0.001},
        "B: [512, 256]": {"hidden_layers": [512, 256], "dropout": 0.3, "lr": 0.001},
        "C: [512, 256, 128]": {
            "hidden_layers": [512, 256, 128],
            "dropout": 0.4,
            "lr": 0.0005,
        },
        "D: [1024, 512, 256]": {
            "hidden_layers": [1024, 512, 256],
            "dropout": 0.4,
            "lr": 0.0005,
        },
    }

    def __init__(self):
        self.results = {}
        self.best_name = None
        self.final_model = None
        self.final_history = None

    def select(self, X_train, y_train):
        """
        Split training data 80/20, train each config, pick best by val accuracy.
        """
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
        )
        print(f"  Model selection: {len(X_tr)} train, {len(X_val)} validation")

        for name, cfg in self.MODEL_CONFIGS.items():
            print(f"\n  Training {name}...")
            torch.manual_seed(SEED)
            model = DenseNN(
                hidden_layers=cfg["hidden_layers"],
                dropout=cfg["dropout"],
            ).to(DEVICE)
            print(f"    Parameters: {model.num_params:,}")

            trainer = NNTrainer(epochs=150, lr=cfg["lr"], patience=20)
            history = trainer.train(model, X_tr, y_tr, X_val, y_val, verbose=True)

            val_preds = trainer.predict(model, X_val)
            val_acc = accuracy_score(y_val, val_preds)

            self.results[name] = {
                "model": model,
                "val_acc": val_acc,
                "history": history,
                "n_params": model.num_params,
                "cfg": cfg,
            }
            print(f"    Val Accuracy: {val_acc:.4f}")

        # Select best
        self.best_name = max(self.results, key=lambda k: self.results[k]["val_acc"])
        print(
            f"\n  >>> Selected: {self.best_name} "
            f"(Val Acc: {self.results[self.best_name]['val_acc']:.4f})"
        )
        return self.best_name

    def train_final(self, X_train, y_train):
        """Retrain best model on full training data (with 10% internal val for early stopping)."""
        best_cfg = self.results[self.best_name]["cfg"]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
        )

        print(
            f"\n  Retraining {self.best_name} on full data ({len(X_tr)} train, {len(X_val)} val)..."
        )
        torch.manual_seed(SEED)
        self.final_model = DenseNN(
            hidden_layers=best_cfg["hidden_layers"],
            dropout=best_cfg["dropout"],
        ).to(DEVICE)

        trainer = NNTrainer(epochs=200, lr=best_cfg["lr"], patience=25)
        self.final_history = trainer.train(
            self.final_model, X_tr, y_tr, X_val, y_val, verbose=True
        )
        return self.final_model

    def evaluate(self, X_test, y_test):
        """Evaluate final model on test set."""
        trainer = NNTrainer()
        preds = trainer.predict(self.final_model, X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        print(f"\n  Final NN Test Results:")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        return preds, {"accuracy": acc, "precision": prec, "recall": rec}
