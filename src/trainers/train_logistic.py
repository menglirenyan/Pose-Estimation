import pandas as pd
import numpy as np

df = pd.read_csv("../../data/pushup_dataset.csv")

# 只保留 UP / DOWN
df = df[df["label"].isin([0, 1])]

features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "shoulder_hip_dist"
]

X = df[features].values
y = df["label"].values

#特征标准化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#训练
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_scaled, y)

#训练集效果
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = model.predict(X_scaled)

print("Accuracy:", accuracy_score(y, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

for name, coef in zip(features, model.coef_[0]):
    print(f"{name}: {coef:.3f}")

# =========================
# 单帧推理函数（核心）
# =========================

def predict_pushup_state(left_elbow_angle,
                          right_elbow_angle,
                          shoulder_hip_dist,
                          threshold=0.7):
    """
    返回:
        0  -> UP
        1  -> DOWN
        -1 -> 不确定
    """

    x = np.array([[left_elbow_angle,
                   right_elbow_angle,
                   shoulder_hip_dist]])

    x_scaled = scaler.transform(x)
    p_up, p_down = model.predict_proba(x_scaled)[0]

    if p_up > threshold:
        return 0
    elif p_down > threshold:
        return 1
    else:
        return -1

