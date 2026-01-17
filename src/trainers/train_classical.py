"""
train_classical.py

合并训练脚本：支持训练 logistic 或 mlp（sklearn）。
用法示例：
    python trainers/train_classical.py --model logistic
    python trainers/train_classical.py --model mlp --data data/pushup_dataset.csv

输出：
    models/{model_name}.joblib
    models/scaler_{model_name}.joblib
"""
import os
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["logistic", "mlp"], required=True,
                   help="Which model to train")
    p.add_argument("--data", default="pushup_dataset.csv",
                   help="CSV file containing features and label")
    p.add_argument("--outdir", default="src/models",
                   help="Directory to save trained model and scaler")
    return p.parse_args()

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # 只保留标签为 0/1 的样本（UP/DOWN）
    df = df[df["label"].isin([0, 1])]
    features = ["left_elbow_angle", "right_elbow_angle", "shoulder_hip_dist"]
    X = df[features].values
    y = df["label"].values.astype(int)
    return X, y, df

def build_model(name):
    if name == "logistic":
        # logistic 对于线性分界面效果可解释
        return LogisticRegression(max_iter=1000)
    elif name == "mlp":
        # 一个简单的 MLP baseline
        return MLPClassifier(hidden_layer_sizes=(16, 8),
                             activation="relu",
                             solver="adam",
                             max_iter=1000,
                             random_state=42)
    else:
        raise ValueError("Unknown model: " + name)

def main():
    args = parse_args()
    X, y, df = load_data(args.data)
    print(f"Loaded {len(y)} samples. UP={int((y==1).sum())}, DOWN={int((y==0).sum())}")

    # 标准化
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 训练
    model = build_model(args.model)
    print("Training", args.model)
    model.fit(Xs, y)

    # 评估（训练集）
    y_pred = model.predict(Xs)
    acc = accuracy_score(y, y_pred)
    print(f"Train accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))
    print("Classification report:")
    print(classification_report(y, y_pred))

    # 打印模型信息（可解释性）
    if args.model == "logistic":
        print("Model classes_: ", model.classes_)
        print("Coefficients:")
        for feat, coef in zip(["left_elbow_angle", "right_elbow_angle", "shoulder_hip_dist"], model.coef_[0]):
            print(f"  {feat}: {coef:.4f}")

    # 保存 model + scaler（统一命名）
    os.makedirs(args.outdir, exist_ok=True)
    model_path = os.path.join(args.outdir, f"{args.model}.joblib")
    scaler_path = os.path.join(args.outdir, f"scaler_{args.model}.joblib")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Saved model ->", model_path)
    print("Saved scaler ->", scaler_path)

if __name__ == "__main__":
    main()
