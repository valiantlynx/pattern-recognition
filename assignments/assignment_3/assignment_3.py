"""
IKT215 Assignment 3: Function Approximation Classification,
Error Estimation and Model Selection

Dataset: mfeat-factors (handwritten digit recognition, 10 classes, 216 features)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import os
import warnings

warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Plotting defaults
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "font.family": "serif",
        "axes.grid": True,
    }
)

OUT_DIR = "/home/valiantlynx/projects/pattern-recognition/assignments/assignment_3"
DATA_DIR = "/home/valiantlynx/projects/pattern-recognition/data/raw"

# ============================================================================
# PART 1: SETUP AND EXPLORATION
# ============================================================================
print("=" * 70)
print("PART 1: SETUP AND EXPLORATION")
print("=" * 70)

train_data = np.load(os.path.join(DATA_DIR, "mfeat_factors_train.npz"))
test_data = np.load(os.path.join(DATA_DIR, "mfeat_factors_test.npz"))

X_train_raw = train_data["data"].astype(np.float64)
y_train_full = train_data["labels"].astype(np.int64)
X_test_raw = test_data["data"].astype(np.float64)
y_test = test_data["labels"].astype(np.int64)

n_train, n_features = X_train_raw.shape
n_test = X_test_raw.shape[0]
classes = np.unique(y_train_full)
n_classes = len(classes)

print(f"Training samples: {n_train}")
print(f"Test samples:     {n_test}")
print(f"Features:         {n_features}")
print(f"Classes:          {n_classes} ({classes.tolist()})")
print(f"Samples per class (train): {n_train // n_classes}")
print(f"Samples per class (test):  {n_test // n_classes}")
print(f"Feature range: [{X_train_raw.min():.0f}, {X_train_raw.max():.0f}]")
print(f"Feature mean:  {X_train_raw.mean():.2f}")
print(f"Feature std:   {X_train_raw.std():.2f}")

# Normalize using StandardScaler (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

print(f"\nAfter normalization:")
print(f"Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
print(f"Test  mean: {X_test_scaled.mean():.4f}, std: {X_test_scaled.std():.4f}")

# Figure: Class distribution
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
for ax, labels, title in zip(
    axes, [y_train_full, y_test], ["Training Set", "Test Set"]
):
    unique, counts = np.unique(labels, return_counts=True)
    ax.bar(unique, counts, color="steelblue", edgecolor="black")
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xticks(unique)
plt.suptitle("Class Distribution", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), bbox_inches="tight")
plt.close()

# Figure: Feature statistics (boxplot of a subset of features)
fig, ax = plt.subplots(figsize=(8, 3))
ax.boxplot(
    [X_train_scaled[:, i] for i in range(0, 216, 24)],
    labels=[f"F{i}" for i in range(0, 216, 24)],
)
ax.set_title("Feature Distribution After Standardization (Every 24th Feature)")
ax.set_xlabel("Feature Index")
ax.set_ylabel("Standardized Value")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_boxplot.png"), bbox_inches="tight")
plt.close()


# ============================================================================
# PART 2: NEURAL NETWORK IMPLEMENTATION
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: NEURAL NETWORK IMPLEMENTATION")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class FullyConnectedNN(nn.Module):
    """
    Fully connected neural network for multi-class classification.
    Architecture: Input(216) -> Hidden1 -> Hidden2 -> Output(10)
    Uses ReLU activations and softmax output.
    """

    def __init__(
        self, input_dim=216, hidden1=128, hidden2=64, output_dim=10, dropout=0.3
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=150,
    lr=0.001,
    weight_decay=1e-4,
    patience=20,
    verbose=False,
):
    """Train the model with early stopping based on validation loss."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_losses.append(epoch_loss / total)
        train_accs.append(correct / total)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_losses.append(val_loss / val_total)
        val_accs.append(val_correct / val_total)

        scheduler.step(val_losses[-1])

        # Early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        if verbose and (epoch + 1) % 25 == 0:
            print(
                f"  Epoch {epoch + 1}: train_loss={train_losses[-1]:.4f}, "
                f"val_loss={val_losses[-1]:.4f}, "
                f"train_acc={train_accs[-1]:.4f}, val_acc={val_accs[-1]:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, X, y):
    """Evaluate model and return predictions and accuracy."""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    acc = accuracy_score(y, preds)
    return preds, acc


# --- Train initial model for Part 2 demo ---
# Use a hold-out split from training data for validation during training
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled,
    y_train_full,
    test_size=0.2,
    random_state=SEED,
    stratify=y_train_full,
)

train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model_main = FullyConnectedNN(
    input_dim=216, hidden1=128, hidden2=64, output_dim=10, dropout=0.3
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model_main.parameters())
trainable_params = sum(p.numel() for p in model_main.parameters() if p.requires_grad)
print(f"\nNetwork Architecture:")
print(model_main)
print(f"\nTotal parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

print("\nTraining primary model...")
train_losses, val_losses, train_accs, val_accs = train_model(
    model_main,
    train_loader,
    val_loader,
    epochs=200,
    lr=0.001,
    weight_decay=1e-4,
    patience=25,
    verbose=True,
)

# Evaluate on the held-out test set
test_preds, test_acc = evaluate_model(model_main, X_test_scaled, y_test)
print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
print(f"Test Error Rate: {1 - test_acc:.4f} ({(1 - test_acc) * 100:.2f}%)")

# Figure: Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(train_losses, label="Train Loss")
ax1.plot(val_losses, label="Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("Training and Validation Loss")
ax1.legend()

ax2.plot(train_accs, label="Train Accuracy")
ax2.plot(val_accs, label="Validation Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training and Validation Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), bbox_inches="tight")
plt.close()

# Figure: Confusion matrix on test set
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), bbox_inches="tight")
plt.close()

print("\nClassification Report (Test Set):")
print(classification_report(y_test, test_preds, digits=4))


# ============================================================================
# PART 3: ERROR ESTIMATION
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: ERROR ESTIMATION")
print("=" * 70)

# --- 3.1: Resubstitution Error ---
print("\n--- 3.1: Resubstitution Error ---")
# True resubstitution: train on ALL training data, evaluate on the SAME data.
# For early stopping, we use the training data itself as validation -- this is
# intentional for resubstitution since we explicitly evaluate on training data.
resub_full_ds = TensorDataset(
    torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train_full)
)
resub_full_loader = DataLoader(resub_full_ds, batch_size=64, shuffle=True)
# Use the same data for validation (monitoring convergence only)
resub_full_val_loader = DataLoader(resub_full_ds, batch_size=64, shuffle=False)

model_resub = FullyConnectedNN(
    input_dim=216, hidden1=128, hidden2=64, output_dim=10, dropout=0.3
).to(device)
torch.manual_seed(SEED)
train_model(
    model_resub,
    resub_full_loader,
    resub_full_val_loader,
    epochs=200,
    lr=0.001,
    patience=25,
)

# Resubstitution: evaluate on all training data (same data used for training)
resub_preds, resub_acc = evaluate_model(model_resub, X_train_scaled, y_train_full)
resub_error = 1.0 - resub_acc
print(f"Resubstitution Accuracy: {resub_acc:.4f}")
print(f"Resubstitution Error:    {resub_error:.4f} ({resub_error * 100:.2f}%)")

# Also get test error for comparison
_, resub_test_acc = evaluate_model(model_resub, X_test_scaled, y_test)
resub_test_error = 1.0 - resub_test_acc
print(
    f"True Test Error:         {resub_test_error:.4f} ({resub_test_error * 100:.2f}%)"
)


# --- 3.2: Hold-Out Error ---
print("\n--- 3.2: Hold-Out Error ---")
# Split training data 80/20 for hold-out
X_ho_train, X_ho_test, y_ho_train, y_ho_test = train_test_split(
    X_train_scaled,
    y_train_full,
    test_size=0.2,
    random_state=SEED,
    stratify=y_train_full,
)

# Further split training portion for internal validation during training
X_ho_tr, X_ho_val, y_ho_tr, y_ho_val = train_test_split(
    X_ho_train, y_ho_train, test_size=0.15, random_state=SEED, stratify=y_ho_train
)

ho_tr_ds = TensorDataset(torch.FloatTensor(X_ho_tr), torch.LongTensor(y_ho_tr))
ho_val_ds = TensorDataset(torch.FloatTensor(X_ho_val), torch.LongTensor(y_ho_val))
ho_tr_loader = DataLoader(ho_tr_ds, batch_size=64, shuffle=True)
ho_val_loader = DataLoader(ho_val_ds, batch_size=64, shuffle=False)

model_ho = FullyConnectedNN(
    input_dim=216, hidden1=128, hidden2=64, output_dim=10, dropout=0.3
).to(device)
torch.manual_seed(SEED)
train_model(model_ho, ho_tr_loader, ho_val_loader, epochs=200, lr=0.001, patience=25)

ho_preds, ho_acc = evaluate_model(model_ho, X_ho_test, y_ho_test)
ho_error = 1.0 - ho_acc
print(f"Hold-Out Accuracy: {ho_acc:.4f}")
print(f"Hold-Out Error:    {ho_error:.4f} ({ho_error * 100:.2f}%)")
print(f"Training size: {len(X_ho_train)}, Hold-out size: {len(X_ho_test)}")


# --- 3.3: 5-Fold Cross-Validation ---
print("\n--- 3.3: 5-Fold Cross-Validation ---")
kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_errors = []
cv_accs_list = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
    X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_cv_train, y_cv_val = y_train_full[train_idx], y_train_full[val_idx]

    # Internal train/val split for early stopping
    X_cv_tr, X_cv_es, y_cv_tr, y_cv_es = train_test_split(
        X_cv_train, y_cv_train, test_size=0.1, random_state=SEED, stratify=y_cv_train
    )

    cv_tr_ds = TensorDataset(torch.FloatTensor(X_cv_tr), torch.LongTensor(y_cv_tr))
    cv_es_ds = TensorDataset(torch.FloatTensor(X_cv_es), torch.LongTensor(y_cv_es))
    cv_tr_loader = DataLoader(cv_tr_ds, batch_size=64, shuffle=True)
    cv_es_loader = DataLoader(cv_es_ds, batch_size=64, shuffle=False)

    model_cv = FullyConnectedNN(
        input_dim=216, hidden1=128, hidden2=64, output_dim=10, dropout=0.3
    ).to(device)
    torch.manual_seed(SEED + fold)
    train_model(model_cv, cv_tr_loader, cv_es_loader, epochs=200, lr=0.001, patience=25)

    preds_cv, acc_cv = evaluate_model(model_cv, X_cv_val, y_cv_val)
    err_cv = 1.0 - acc_cv
    cv_errors.append(err_cv)
    cv_accs_list.append(acc_cv)
    print(f"  Fold {fold + 1}: Accuracy={acc_cv:.4f}, Error={err_cv:.4f}")

cv_mean_error = np.mean(cv_errors)
cv_std_error = np.std(cv_errors)
cv_mean_acc = np.mean(cv_accs_list)
cv_std_acc = np.std(cv_accs_list)
print(f"\n5-Fold CV Mean Accuracy: {cv_mean_acc:.4f} +/- {cv_std_acc:.4f}")
print(f"5-Fold CV Mean Error:    {cv_mean_error:.4f} +/- {cv_std_error:.4f}")
print("\nPer-fold breakdown (for report table):")
for i, (acc, err) in enumerate(zip(cv_accs_list, cv_errors)):
    print(f"  Fold {i + 1}: Accuracy={acc:.4f}, Error={err:.4f}, Samples={320}")

# Summary comparison figure
print("\n--- Error Estimation Summary ---")
print(f"Resubstitution Error: {resub_error:.4f}")
print(f"Hold-Out Error:       {ho_error:.4f}")
print(f"5-Fold CV Error:      {cv_mean_error:.4f} +/- {cv_std_error:.4f}")
print(f"True Test Error:      {resub_test_error:.4f}")

# Figure: Error estimation comparison
fig, ax = plt.subplots(figsize=(7, 4))
methods = ["Resubstitution", "Hold-Out", "5-Fold CV", "True Test"]
errors = [resub_error, ho_error, cv_mean_error, resub_test_error]
error_bars = [0, 0, cv_std_error, 0]
colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]
bars = ax.bar(
    methods,
    errors,
    yerr=error_bars,
    capsize=5,
    color=colors,
    edgecolor="black",
    alpha=0.85,
)
ax.set_ylabel("Error Rate")
ax.set_title("Comparison of Error Estimation Methods")
for bar, err in zip(bars, errors):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{err:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "error_estimation_comparison.png"), bbox_inches="tight"
)
plt.close()


# ============================================================================
# PART 4: MODEL SELECTION
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: MODEL SELECTION")
print("=" * 70)

# Define network configurations to compare
configs = {
    "Config A (Small)": {"hidden1": 64, "hidden2": 32, "dropout": 0.2, "lr": 0.001},
    "Config B (Medium)": {"hidden1": 128, "hidden2": 64, "dropout": 0.3, "lr": 0.001},
    "Config C (Large)": {"hidden1": 256, "hidden2": 128, "dropout": 0.4, "lr": 0.0005},
}

# --- 4.1: Train-Validation-Test Approach ---
print("\n--- 4.1: Train-Validation-Test Approach ---")
# Split training data: 70% train, 15% validation, 15% test-like
# The actual test set is data/raw/mfeat_factors_test.npz
X_tvt_train, X_tvt_rest, y_tvt_train, y_tvt_rest = train_test_split(
    X_train_scaled,
    y_train_full,
    test_size=0.3,
    random_state=SEED,
    stratify=y_train_full,
)
X_tvt_val, X_tvt_internal_test, y_tvt_val, y_tvt_internal_test = train_test_split(
    X_tvt_rest, y_tvt_rest, test_size=0.5, random_state=SEED, stratify=y_tvt_rest
)

print(
    f"Train: {len(X_tvt_train)}, Validation: {len(X_tvt_val)}, Internal Test: {len(X_tvt_internal_test)}"
)

tvt_results = {}
for name, cfg in configs.items():
    print(f"\n  Training {name}...")
    model_tvt = FullyConnectedNN(
        input_dim=216,
        hidden1=cfg["hidden1"],
        hidden2=cfg["hidden2"],
        output_dim=10,
        dropout=cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model_tvt.parameters())

    tr_ds = TensorDataset(torch.FloatTensor(X_tvt_train), torch.LongTensor(y_tvt_train))
    vl_ds = TensorDataset(torch.FloatTensor(X_tvt_val), torch.LongTensor(y_tvt_val))
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    vl_loader = DataLoader(vl_ds, batch_size=64, shuffle=False)

    torch.manual_seed(SEED)
    t_losses, v_losses, t_accs, v_accs = train_model(
        model_tvt, tr_loader, vl_loader, epochs=200, lr=cfg["lr"], patience=25
    )

    _, val_acc = evaluate_model(model_tvt, X_tvt_val, y_tvt_val)
    _, internal_test_acc = evaluate_model(
        model_tvt, X_tvt_internal_test, y_tvt_internal_test
    )
    _, final_test_acc = evaluate_model(model_tvt, X_test_scaled, y_test)

    tvt_results[name] = {
        "val_acc": val_acc,
        "internal_test_acc": internal_test_acc,
        "final_test_acc": final_test_acc,
        "model": model_tvt,
        "train_losses": t_losses,
        "val_losses": v_losses,
        "n_params": n_params,
    }
    print(
        f"    Params: {n_params}, Val Acc: {val_acc:.4f}, "
        f"Internal Test Acc: {internal_test_acc:.4f}, Final Test Acc: {final_test_acc:.4f}"
    )

# Select best model by validation accuracy
best_tvt_name = max(tvt_results, key=lambda k: tvt_results[k]["val_acc"])
print(f"\n  Selected model (TVT): {best_tvt_name}")
print(f"  Validation Acc: {tvt_results[best_tvt_name]['val_acc']:.4f}")
print(f"  Final Test Acc: {tvt_results[best_tvt_name]['final_test_acc']:.4f}")


# --- 4.2: Cross-Validation Approach ---
print("\n--- 4.2: Cross-Validation Approach ---")
cv_results = {}
for name, cfg in configs.items():
    print(f"\n  Cross-validating {name}...")
    fold_accs = []
    kfold_ms = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(kfold_ms.split(X_train_scaled)):
        X_f_train, X_f_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_f_train, y_f_val = y_train_full[train_idx], y_train_full[val_idx]

        # Internal split for early stopping
        X_f_tr, X_f_es, y_f_tr, y_f_es = train_test_split(
            X_f_train, y_f_train, test_size=0.1, random_state=SEED, stratify=y_f_train
        )

        f_tr_ds = TensorDataset(torch.FloatTensor(X_f_tr), torch.LongTensor(y_f_tr))
        f_es_ds = TensorDataset(torch.FloatTensor(X_f_es), torch.LongTensor(y_f_es))
        f_tr_loader = DataLoader(f_tr_ds, batch_size=64, shuffle=True)
        f_es_loader = DataLoader(f_es_ds, batch_size=64, shuffle=False)

        model_cv_ms = FullyConnectedNN(
            input_dim=216,
            hidden1=cfg["hidden1"],
            hidden2=cfg["hidden2"],
            output_dim=10,
            dropout=cfg["dropout"],
        ).to(device)
        torch.manual_seed(SEED + fold)
        train_model(
            model_cv_ms, f_tr_loader, f_es_loader, epochs=200, lr=cfg["lr"], patience=25
        )

        _, fold_acc = evaluate_model(model_cv_ms, X_f_val, y_f_val)
        fold_accs.append(fold_acc)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    cv_results[name] = {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "fold_accs": fold_accs,
        "cfg": cfg,
    }
    print(f"    CV Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"    Per-fold: {[f'{a:.4f}' for a in fold_accs]}")

# Select best model by CV accuracy
best_cv_name = max(cv_results, key=lambda k: cv_results[k]["mean_acc"])
print(f"\n  Selected model (CV): {best_cv_name}")
print(
    f"  CV Accuracy: {cv_results[best_cv_name]['mean_acc']:.4f} +/- {cv_results[best_cv_name]['std_acc']:.4f}"
)

# Retrain best CV model on full training data and evaluate on test
best_cfg = cv_results[best_cv_name]["cfg"]
X_final_tr, X_final_es, y_final_tr, y_final_es = train_test_split(
    X_train_scaled,
    y_train_full,
    test_size=0.1,
    random_state=SEED,
    stratify=y_train_full,
)
final_tr_ds = TensorDataset(torch.FloatTensor(X_final_tr), torch.LongTensor(y_final_tr))
final_es_ds = TensorDataset(torch.FloatTensor(X_final_es), torch.LongTensor(y_final_es))
final_tr_loader = DataLoader(final_tr_ds, batch_size=64, shuffle=True)
final_es_loader = DataLoader(final_es_ds, batch_size=64, shuffle=False)

model_final_cv = FullyConnectedNN(
    input_dim=216,
    hidden1=best_cfg["hidden1"],
    hidden2=best_cfg["hidden2"],
    output_dim=10,
    dropout=best_cfg["dropout"],
).to(device)
torch.manual_seed(SEED)
train_model(
    model_final_cv,
    final_tr_loader,
    final_es_loader,
    epochs=200,
    lr=best_cfg["lr"],
    patience=25,
)

final_cv_preds, final_cv_acc = evaluate_model(model_final_cv, X_test_scaled, y_test)
print(f"  Final Test Acc (CV-selected model): {final_cv_acc:.4f}")


# --- Model Selection Figures ---

# Figure: TVT comparison bar chart
fig, ax = plt.subplots(figsize=(8, 4))
names = list(tvt_results.keys())
val_accs_bar = [tvt_results[n]["val_acc"] for n in names]
test_accs_bar = [tvt_results[n]["final_test_acc"] for n in names]
x = np.arange(len(names))
width = 0.35
bars1 = ax.bar(
    x - width / 2,
    val_accs_bar,
    width,
    label="Validation Acc",
    color="#2196F3",
    edgecolor="black",
)
bars2 = ax.bar(
    x + width / 2,
    test_accs_bar,
    width,
    label="Test Acc",
    color="#F44336",
    edgecolor="black",
)
ax.set_xlabel("Configuration")
ax.set_ylabel("Accuracy")
ax.set_title("Train-Validation-Test Model Selection")
ax.set_xticks(x)
ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=9)
ax.legend()
ax.set_ylim(0.85, 1.0)
for bar in bars1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
for bar in bars2:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tvt_comparison.png"), bbox_inches="tight")
plt.close()

# Figure: CV comparison bar chart
fig, ax = plt.subplots(figsize=(8, 4))
names_cv = list(cv_results.keys())
cv_mean_accs = [cv_results[n]["mean_acc"] for n in names_cv]
cv_std_accs = [cv_results[n]["std_acc"] for n in names_cv]
bars = ax.bar(
    names_cv,
    cv_mean_accs,
    yerr=cv_std_accs,
    capsize=5,
    color=["#2196F3", "#FF9800", "#4CAF50"],
    edgecolor="black",
    alpha=0.85,
)
ax.set_ylabel("Accuracy")
ax.set_title("5-Fold Cross-Validation Model Selection")
ax.set_ylim(0.85, 1.0)
for bar, acc in zip(bars, cv_mean_accs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{acc:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cv_model_selection.png"), bbox_inches="tight")
plt.close()

# Figure: Confusion matrix for final selected model
fig, ax = plt.subplots(figsize=(6, 5))
cm_final = confusion_matrix(y_test, final_cv_preds)
disp_final = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=classes)
disp_final.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Final Model Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "final_confusion_matrix.png"), bbox_inches="tight")
plt.close()

# Figure: Training loss curves for TVT configs
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, name in zip(axes, names):
    r = tvt_results[name]
    ax.plot(r["train_losses"], label="Train")
    ax.plot(r["val_losses"], label="Validation")
    ax.set_title(name.replace(" (", "\n("), fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
plt.suptitle("Training Curves for Different Configurations", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "config_training_curves.png"), bbox_inches="tight")
plt.close()


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(
    f"\nDataset: mfeat-factors, {n_train} train / {n_test} test, {n_features} features, {n_classes} classes"
)
print(f"\nPart 2 - Primary NN Test Accuracy: {test_acc:.4f}")
print(f"\nPart 3 - Error Estimation:")
print(f"  Resubstitution Error: {resub_error:.4f}")
print(f"  Hold-Out Error:       {ho_error:.4f}")
print(f"  5-Fold CV Error:      {cv_mean_error:.4f} +/- {cv_std_error:.4f}")
print(f"\nPart 4 - Model Selection:")
print(
    f"  TVT Selected:  {best_tvt_name} -> Test Acc: {tvt_results[best_tvt_name]['final_test_acc']:.4f}"
)
print(f"  CV Selected:   {best_cv_name} -> Test Acc: {final_cv_acc:.4f}")

print("\n--- Config Parameters Summary ---")
for name, cfg in configs.items():
    n_p = tvt_results[name]["n_params"]
    print(
        f"  {name}: H1={cfg['hidden1']}, H2={cfg['hidden2']}, "
        f"dropout={cfg['dropout']}, lr={cfg['lr']}, params={n_p}"
    )

print("\nAll figures saved. Done!")
