# train_and_save_logistic.py
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# 1. 读取并清洗数据
# =========================

df = pd.read_csv("pushup_dataset.csv")

# 只保留 UP / DOWN（丢弃 -1）
df = df[df["label"].isin([0, 1])]

features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "shoulder_hip_dist"
]

X = df[features].values
y = df["label"].values

print("样本数:", len(df))
print("UP:", sum(y == 1), "DOWN:", sum(y == 0))

# =========================
# 2. 特征标准化
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 3. 训练逻辑回归模型
# =========================

model = LogisticRegression()
model.fit(X_scaled, y)

# =========================
# 4. 训练集评估
# =========================

y_pred = model.predict(X_scaled)

print("Accuracy:", accuracy_score(y, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

print("模型类别顺序 model.classes_ =", model.classes_)

for name, coef in zip(features, model.coef_[0]):
    print(f"{name}: {coef:.3f}")

# =========================
# 5. 保存模型与 scaler
# =========================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/logistic.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("模型已保存到 models/logistic.joblib")
print("Scaler 已保存到 models/scaler.joblib")
