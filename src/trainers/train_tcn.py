#!/usr/bin/env python3
"""
train_tcn.py (3-class) - Temporal CNN / TCN trainer (does NOT overwrite LSTM scripts)

Purpose
  Train a **Temporal Convolutional Network (TCN / 1D-CNN)** on the same
  pose-derived CSV features (left/right elbow angles + shoulder-hip distance)
  using sliding windows of length T (default 16).

Why TCN?
  - It's a CNN-family model for time series.
  - Same input as your LSTM: (T, F=3), but uses temporal convolutions.
  - Often trains faster and is stable, good for "wider" model comparison in a thesis.

CSV expected columns:
  left_elbow_angle, right_elbow_angle, shoulder_hip_dist, view_type, label

Labels in CSV:
  0: DOWN
  1: UP
 -1: UNCERTAIN  -> mapped to class id 2

Example
  python src/trainers/train_tcn.py --data ../../data/pushup_dataset.csv --seq_len 16

Saves under --models_dir (default: src/models):
  - tcn.pt
  - scaler_tcn.joblib
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data/pushup_dataset.csv")
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--models_dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience (val_loss)")

    # TCN config
    p.add_argument("--channels", type=str, default="32,64,64",
                   help="Comma-separated channel sizes, e.g. 32,64,64")
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)

    # labeling
    p.add_argument("--label_mode", type=str, default="middle", choices=["middle", "majority"],
                   help="Window label rule: middle-frame label or majority label")
    return p.parse_args()


def make_sequences_global(df, seq_len=16, feature_cols=None, label_mode="middle"):
    """Sliding window sequences across the whole dataframe."""
    if feature_cols is None:
        feature_cols = ["left_elbow_angle", "right_elbow_angle", "shoulder_hip_dist"]

    feats = df[feature_cols].values.astype(np.float32)
    labels = df["label"].values
    N = len(df)

    X_list, y_list = [], []
    for s in range(0, N - seq_len + 1):
        e = s + seq_len
        if label_mode == "middle":
            mid = s + seq_len // 2
            lbl = int(labels[mid])
        else:
            vals, counts = np.unique(labels[s:e], return_counts=True)
            lbl = int(vals[np.argmax(counts)])
        X_list.append(feats[s:e])
        y_list.append(lbl)

    if len(X_list) == 0:
        return (
            np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int64)


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), int(self.y[idx])


class TemporalBlock(nn.Module):
    """A minimal TCN residual block (Conv1d -> ReLU -> Dropout -> Conv1d -> ReLU -> Dropout) + residual."""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2  # keep length with odd kernel
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # x: (B,C,T)
        y = self.conv1(x)
        y = self.relu(y)
        y = self.drop(y)

        y = self.conv2(y)
        y = self.relu(y)
        y = self.drop(y)

        res = x if self.downsample is None else self.downsample(x)
        return y + res


class PushupTCN(nn.Module):
    def __init__(self, input_size=3, channels=(32, 64, 64), kernel_size=3, dropout=0.1, num_classes=3):
        super().__init__()
        layers = []
        in_ch = input_size
        dilation = 1
        for out_ch in channels:
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            in_ch = out_ch
            dilation *= 2  # exponential receptive field
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        # x: (B,T,F) -> (B,F,T)
        x = x.permute(0, 2, 1)
        h = self.net(x)         # (B,C,T)
        h = h.mean(dim=2)       # global average pooling over time -> (B,C)
        return self.head(h)     # (B,3)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * xb.size(0)
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        trues.extend(yb.detach().cpu().numpy().tolist())

    return total_loss / max(1, len(loader.dataset)), np.array(trues), np.array(preds)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += float(loss.item()) * xb.size(0)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            trues.extend(yb.cpu().numpy().tolist())

    return total_loss / max(1, len(loader.dataset)), np.array(trues), np.array(preds)


def main():
    args = parse_args()

    # models_dir default: <project>/src/models
    if args.models_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        args.models_dir = os.path.join(base_dir, "src", "models")
    os.makedirs(args.models_dir, exist_ok=True)

    print(f"Loading CSV: {args.data}")
    df = pd.read_csv(args.data)

    # keep only valid labels in {0,1,-1}
    df = df[df["label"].isin([0, 1, -1])].copy()

    # map -1 -> 2
    df["label"] = df["label"].replace({-1: 2}).astype(int)

    feature_cols = ["left_elbow_angle", "right_elbow_angle", "shoulder_hip_dist"]
    X, y = make_sequences_global(df, seq_len=args.seq_len, feature_cols=feature_cols, label_mode=args.label_mode)
    print(f"All windows: {X.shape} labels dist: {dict(zip(*np.unique(y, return_counts=True)))}")

    # split train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(args.test_size + args.val_size), random_state=args.random_seed, stratify=y
    )
    val_ratio = args.val_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=args.random_seed, stratify=y_temp
    )

    print(f"Split sizes -> Train: {X_train.shape} Val: {X_val.shape} Test: {X_test.shape}")
    print(f"Train dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Val dist: {dict(zip(*np.unique(y_val, return_counts=True)))}")
    print(f"Test dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # scaler fit on train (flatten T dimension)
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_2d)

    def scale_X(X_in):
        X2 = X_in.reshape(-1, X_in.shape[-1])
        Xs = scaler.transform(X2).astype(np.float32)
        return Xs.reshape(X_in.shape)

    X_train_s = scale_X(X_train)
    X_val_s = scale_X(X_val)
    X_test_s = scale_X(X_test)

    scaler_path = os.path.join(args.models_dir, "scaler_tcn.joblib")
    joblib.dump(scaler, scaler_path)

    # class weights
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw_map = {int(c): float(w) for c, w in zip(classes, cw)}
    weights = np.ones(3, dtype=np.float32)
    for c, w in cw_map.items():
        weights[c] = w
    print(f"Class weights: {weights.tolist()}")

    device = torch.device(args.device)
    if str(device).startswith("cuda") and (not torch.cuda.is_available()):
        print("[WARN] cuda requested but not available. Falling back to cpu.")
        device = torch.device("cpu")

    # parse channels
    ch = tuple(int(x.strip()) for x in args.channels.split(",") if x.strip())
    if len(ch) == 0:
        raise ValueError("--channels must contain at least one int, e.g. 32,64,64")

    model = PushupTCN(input_size=3, channels=ch, kernel_size=args.kernel_size,
                      dropout=args.dropout, num_classes=3).to(device)

    train_ds = SequenceDataset(X_train_s, y_train)
    val_ds = SequenceDataset(X_val_s, y_val)
    test_ds = SequenceDataset(X_test_s, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    best_val = float("inf")
    best_path = os.path.join(args.models_dir, "tcn.pt")
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_y, tr_p = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_y, va_p = eval_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")

        # early stop on val loss
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            patience_left = args.patience
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[INFO] Early stopping.")
                break

        # small val report per epoch (optional)
        print("Val confusion matrix:")
        print(confusion_matrix(va_y, va_p, labels=[0, 1, 2]))

    # load best and test
    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_y, te_p = eval_epoch(model, test_loader, criterion, device)

    print("=" * 80)
    print(f"Best model saved to: {best_path}")
    print(f"Scaler saved to   : {scaler_path}")
    print(f"Test loss         : {te_loss:.4f}")
    print("Test confusion matrix (labels=[DOWN(0),UP(1),UNC(2)]):")
    print(confusion_matrix(te_y, te_p, labels=[0, 1, 2]))
    print("Test report:")
    print(classification_report(te_y, te_p, target_names=["DOWN", "UP", "UNCERTAIN"], digits=4))
    print("=" * 80)


if __name__ == "__main__":
    main()
