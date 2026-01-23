"""
train_classical.py

Train classical sklearn models for push-up detection.
Default: train ALL models (logistic + mlp)

Outputs:
    src/models/{model}.joblib
    src/models/scaler_{model}.joblib
    src/models/meta_{model}.json
"""

import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================
# argparse
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    # 单模型（兼容旧用法）
    p.add_argument("--model", choices=["logistic", "mlp"], default=None,
                   help="Train a single model (legacy).")

    # 多模型（推荐）
    p.add_argument("--models", nargs="*", choices=["logistic", "mlp"], default=None,
                   help="Train one or more models. If omitted, train ALL by default.")

    p.add_argument("--data", default="../data/pushup_dataset.csv",
                   help="CSV file containing features and label")
    p.add_argument("--outdir", default=None,
                   help="Directory to save trained model and scaler")

    # 新增：随机种子
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")

    return p.parse_args()


# =========================
# data
# =========================
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["label"].isin([0, 1])]
    features = ["left_elbow_angle", "right_elbow_angle", "shoulder_hip_dist"]
    X = df[features].values
    y = df["label"].values.astype(int)
    return X, y, df


# =========================
# model factory
# =========================
def build_model(name, seed):
    if name == "logistic":
        return LogisticRegression(max_iter=1000, random_state=seed)
    elif name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=seed
        )
    else:
        raise ValueError("Unknown model: " + name)


# =========================
# train one model
# =========================
def train_one(model_name, X, y, args):
    print("=" * 80)
    print(f"Training model: {model_name} | seed={args.seed}")

    # seed 控制（核心）
    np.random.seed(args.seed)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = build_model(model_name, args.seed)
    model.fit(Xs, y)

    # eval on train
    y_pred = model.predict(Xs)
    acc = accuracy_score(y, y_pred)
    print(f"[{model_name}] Train accuracy: {acc:.4f}")
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))

    # save model
    os.makedirs(args.outdir, exist_ok=True)
    model_path = os.path.join(args.outdir, f"{model_name}.joblib")
    scaler_path = os.path.join(args.outdir, f"scaler_{model_name}.joblib")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # save meta
    meta = {
        "model": model_name,
        "random_seed": args.seed,
        "model_path": os.path.abspath(model_path),
        "scaler_path": os.path.abspath(scaler_path),
        "train_data": os.path.abspath(args.data),
        "features": ["left_elbow_angle", "right_elbow_angle", "shoulder_hip_dist"],
        "sklearn_estimator": type(model).__name__,
        "sklearn_params": model.get_params(deep=True),
        "label_definition": {
            "0": "DOWN",
            "1": "UP"
        }
    }

    meta_path = os.path.join(args.outdir, f"meta_{model_name}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[{model_name}] Saved model  -> {model_path}")
    print(f"[{model_name}] Saved scaler -> {scaler_path}")
    print(f"[{model_name}] Saved meta   -> {meta_path}")


# =========================
# main
# =========================
def main():
    args = parse_args()
    if args.outdir is None:
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)  # .../src/trainers
        src_dir = os.path.dirname(script_dir)  # .../src
        args.outdir = os.path.join(src_dir, "models")  # .../src/models
    X, y, _ = load_data(args.data)

    print(f"Loaded {len(y)} samples | UP={(y==1).sum()} DOWN={(y==0).sum()}")
    print(f"Global random seed = {args.seed}")

    # 决定训练哪些模型
    if args.models:
        models_to_train = args.models
    elif args.model:
        models_to_train = [args.model]
    else:
        models_to_train = ["logistic", "mlp"]  # 默认全部

    print("Models to train:", models_to_train)

    for m in models_to_train:
        train_one(m, X, y, args)

    print("=" * 80)
    print("All training done.")


if __name__ == "__main__":
    main()
