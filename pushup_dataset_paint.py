import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("pushup_dataset.csv")
df = df[df["label"].isin([0, 1])]

up = df[df["label"] == 1]
down = df[df["label"] == 0]

plt.figure(figsize=(7, 5))
plt.scatter(
    up["right_elbow_angle"],
    up["shoulder_hip_dist"],
    c="green",
    label="UP",
    alpha=0.6
)

plt.scatter(
    down["right_elbow_angle"],
    down["shoulder_hip_dist"],
    c="red",
    label="DOWN",
    alpha=0.6
)

plt.xlabel("Right Elbow Angle (deg)")
plt.ylabel("Shoulder–Hip Distance (normalized)")
plt.legend()
plt.title("Feature Space Visualization (2D)")
plt.show()

###画出边界
X = df[["right_elbow_angle", "shoulder_hip_dist"]].values
y = df["label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# 网格
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min(), X[:, 0].max(), 200),
    np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)
zz = model.predict(grid_scaled).reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, zz, alpha=0.2)

plt.scatter(up["right_elbow_angle"], up["shoulder_hip_dist"], c="green", label="UP")
plt.scatter(down["right_elbow_angle"], down["shoulder_hip_dist"], c="red", label="DOWN")

plt.xlabel("Right Elbow Angle")
plt.ylabel("Shoulder–Hip Distance")
plt.legend()
plt.title("Logistic Regression Decision Boundary")
plt.show()
