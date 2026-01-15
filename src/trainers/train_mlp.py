import pandas as pd
import joblib

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv("../../data/pushup_dataset.csv")
df = df[df["label"].isin([0, 1])]

features = [
    "left_elbow_angle",
    "right_elbow_angle",
    "shoulder_hip_dist"
]

X = df[features].values
y = df["label"].values

# =========================
# 2. 标准化
# =========================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 3. MLP 模型
# =========================
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

mlp.fit(X_scaled, y)

# =========================
# 4. 保存模型
# =========================
joblib.dump(scaler, "models/scaler_mlp.joblib")
joblib.dump(mlp, "models/mlp.joblib")

print("MLP model saved.")
