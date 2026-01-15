import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =========================
# 1. 读取数据
# =========================

df = pd.read_csv("data/pushup_dataset.csv")
df = df[df["label"].isin([0, 1])]

features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "shoulder_hip_dist"
]

X = df[features].values
y = df["label"].values

# =========================
# 2. 标准化 + 训练
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# =========================
# 3. 概率输出
# =========================

proba = model.predict_proba(X_scaled)
p_up = proba[:, 0]
p_down = proba[:, 1]

# =========================
# 4. 画概率空间
# =========================

plt.figure()
plt.scatter(
    p_down[y == 0],
    p_up[y == 0],
    label="UP (0)",
    alpha=0.6
)

plt.scatter(
    p_down[y == 1],
    p_up[y == 1],
    label="DOWN (1)",
    alpha=0.6
)

plt.xlabel("P(DOWN)")
plt.ylabel("P(UP)")
plt.title("Probability Space of Push-up States")
plt.legend()
plt.grid(True)
plt.show()
