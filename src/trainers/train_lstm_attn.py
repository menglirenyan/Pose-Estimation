#!/usr/bin/env python3
"""train_lstm_attn.py (3-class) - separate Attention trainer

Purpose
  Train **LSTM + Temporal Attention** on your push-up CSV without touching
  (or overwriting) the original `train_lstm.py` script.

CSV expected columns:
  left_elbow_angle, right_elbow_angle, shoulder_hip_dist, view_type, label

Labels in CSV:
  0: DOWN
  1: UP
 -1: UNCERTAIN  -> mapped to class id 2

Example
  python src/trainers/train_lstm_attn.py --data ../../data/pushup_dataset.csv --seq_len 16

Saves under --models_dir (default: src/models):
  - lstm_attn.pt
  - scaler_lstm_attn.joblib

Notes
  - This is the same dataset / scaling / early-stopping logic as your LSTM trainer,
    only the architecture is fixed to Attention-LSTM.
  - main_realtime.py already supports loading this model via --models lstm_attn (or attn).
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

    # model config (Attention-LSTM)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--attn_dim", type=int, default=64)

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


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) temporal attention over LSTM outputs."""

    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h):
        # h: (B,T,H)
        u = torch.tanh(self.proj(h))          # (B,T,A)
        scores = self.v(u).squeeze(-1)        # (B,T)
        alpha = torch.softmax(scores, dim=1)  # (B,T)
        context = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # (B,H)
        return context, alpha


class PushupAttnLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=1,
                 bidirectional=False, dropout=0.1, num_classes=3, attn_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.hidden_dim = hidden_size * (2 if bidirectional else 1)
        self.attn = TemporalAttention(self.hidden_dim, attn_dim=attn_dim)
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)           # (B,T,H)
        context, _alpha = self.attn(out)
        return self.fc(context)


def train_loop(model, optimizer, criterion, loader, device):
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

        total_loss += loss.item() * xb.size(0)
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())

    if len(loader.dataset) == 0:
        return 0.0, np.array([]), np.array([])
    return total_loss / len(loader.dataset), np.concatenate(preds), np.concatenate(trues)


def eval_loop(model, criterion, loader, device):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(yb.cpu().numpy())
    if len(loader.dataset) == 0:
        return 0.0, np.array([]), np.array([])
    return total_loss / len(loader.dataset), np.concatenate(preds), np.concatenate(trues)


def _dist(y):
    u, c = np.unique(y, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))


def main():
    args = parse_args()

    # default models_dir -> src/models
    if args.models_dir is None:
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)      # src/trainers
        src_dir = os.path.dirname(script_dir)          # src
        args.models_dir = os.path.join(src_dir, "models")

    os.makedirs(args.models_dir, exist_ok=True)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    print("Loading CSV:", args.data)
    df = pd.read_csv(args.data)
    feature_cols = ["left_elbow_angle", "right_elbow_angle", "shoulder_hip_dist"]

    # 1) windows
    X_all, y_all_raw = make_sequences_global(df, seq_len=args.seq_len, feature_cols=feature_cols, label_mode="middle")
    if X_all.shape[0] == 0:
        raise SystemExit("No windows generated. Check your CSV and seq_len.")

    # 2) map labels: -1 -> 2 (UNCERTAIN), 0 -> 0 (DOWN), 1 -> 1 (UP)
    label_map = {-1: 2, 0: 0, 1: 1}
    try:
        y_all = np.array([label_map[int(v)] for v in y_all_raw], dtype=np.int64)
    except KeyError as e:
        raise SystemExit(f"Found unexpected label value in CSV: {e}. Expected only -1/0/1.")

    print("All windows:", X_all.shape, "labels dist:", _dist(y_all))

    # 3) split
    test_ratio = args.test_size
    val_ratio = args.val_size
    try:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_all, y_all, test_size=test_ratio, random_state=args.random_seed, stratify=y_all
        )
        relative_val = val_ratio / (1.0 - test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=relative_val, random_state=args.random_seed, stratify=y_trainval
        )
    except ValueError as e:
        print("[WARN] Stratified split failed:", e)
        print("[WARN] Falling back to non-stratified random split.")
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_all, y_all, test_size=test_ratio, random_state=args.random_seed, shuffle=True
        )
        relative_val = val_ratio / (1.0 - test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=relative_val, random_state=args.random_seed, shuffle=True
        )

    print("Split sizes -> Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
    print("Train dist:", _dist(y_train))
    print("Val dist:", _dist(y_val))
    print("Test dist:", _dist(y_test))

    # 4) scaler fit on train frames
    B, T, F = X_train.shape
    scaler = StandardScaler().fit(X_train.reshape(-1, F))

    def transform_windows(X):
        B2, T2, F2 = X.shape
        return scaler.transform(X.reshape(-1, F2)).reshape(B2, T2, F2)

    X_train = transform_windows(X_train)
    X_val = transform_windows(X_val)
    X_test = transform_windows(X_test)

    # 5) class weights for 3 classes [0,1,2]
    present = np.unique(y_train)
    if len(present) < 2:
        print("[WARN] Training set has <2 classes. Using uniform weights.")
        weight_tensor = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    else:
        w = np.ones(3, dtype=np.float32)
        w_present = compute_class_weight("balanced", classes=present, y=y_train)
        for cls, ww in zip(present.tolist(), w_present.tolist()):
            w[int(cls)] = float(ww)
        weight_tensor = torch.tensor(w, dtype=torch.float32)
    print("Class weights:", weight_tensor.tolist())

    # 6) loaders
    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    # 7) device safety
    if args.device.startswith("cuda") and (not torch.cuda.is_available()):
        print("[WARN] You requested CUDA but torch.cuda.is_available() is False. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # 8) build model (fixed to Attention-LSTM)
    model = PushupAttnLSTM(
        input_size=3,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        num_classes=3,
        attn_dim=args.attn_dim,
    ).to(device)

    model_filename = "lstm_attn.pt"
    scaler_filename = "scaler_lstm_attn.joblib"
    model_path = os.path.join(args.models_dir, model_filename)
    scaler_path = os.path.join(args.models_dir, scaler_filename)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))

    # 9) train with early stopping
    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    print(f"Training Attention-LSTM -> will save: {model_filename}, {scaler_filename}")

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = train_loop(model, optimizer, criterion, train_loader, device)
        val_loss, val_preds, val_trues = eval_loop(model, criterion, val_loader, device)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if len(val_trues) > 0:
            print("Val report:")
            print(classification_report(
                val_trues,
                val_preds,
                target_names=["DOWN", "UP", "UNCERTAIN"],
                digits=4,
            ))

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            ckpt = {
                "state_dict": model.state_dict(),
                "hparams": {
                    "model_name": "lstm_attn",
                    "input_size": 3,
                    "hidden_size": int(args.hidden_size),
                    "num_layers": int(args.num_layers),
                    "bidirectional": bool(args.bidirectional),
                    "dropout": float(args.dropout),
                    "attn_dim": int(args.attn_dim),
                    "num_classes": 3,
                    "seq_len": int(args.seq_len),
                    "lr": float(args.lr),
                    "batch_size": int(args.batch_size),
                    "epochs": int(args.epochs),
                    "patience": int(args.patience),
                    "random_seed": int(args.random_seed),
                    "feature_cols": ["left_elbow_angle", "right_elbow_angle", "shoulder_hip_dist"],
                    "label_map": {"-1": 2, "0": 0, "1": 1},
                },
            }
            torch.save(ckpt, model_path)

            joblib.dump(scaler, scaler_path)
            print(f"Saved best model + scaler (epoch {epoch}).")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[EarlyStop] No improvement for {args.patience} epochs. Stop at epoch {epoch}.")
                break

    # 10) test with best
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        _, test_preds, test_trues = eval_loop(model, criterion, test_loader, device)
        if len(test_trues) > 0:
            print("Test report:")
            print(classification_report(
                test_trues,
                test_preds,
                target_names=["DOWN", "UP", "UNCERTAIN"],
                digits=4,
            ))
            print("Confusion matrix (rows=true, cols=pred):")
            print(confusion_matrix(test_trues, test_preds, labels=[0, 1, 2]))
    else:
        print("[WARN] No saved model found; skip test eval.")

    print("Done. Best epoch:", best_epoch)


if __name__ == "__main__":
    main()
