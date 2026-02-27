# 基于姿态识别的俯卧撑动作检测系统
本毕业设计旨在串联机器学习到深度学习的知识体系，实现基于单摄像头+MediaPipe的俯卧撑动作检测、状态识别与计数，并对比不同算法的性能表现。

## 项目简介
### 核心目标
通过人体姿态关键点提取、特征工程与多模型对比，完成俯卧撑动作的状态识别（如：标准/不标准/准备/完成）与计数逻辑验证，构建从机器学习到深度学习的完整技术链路。
### 技术栈
- 关键点提取：MediaPipe Pose
- 数据处理：Python、OpenCV、Pandas、NumPy（数据清洗、标注、特征计算）
- 模型训练：Scikit-learn（Logistic Regression）、PyTorch/TensorFlow（MLP、LSTM、Attention-LSTM）
- 评估指标：混淆矩阵、Precision/Recall/F1-Score、准确率

## 项目结构
```
├── data/                # 俯卧撑数据集（标注文件、原始视频/图片）
├── feature/             # 特征工程代码（肘角、肩髋距离等特征计算）
├── models/              # 模型定义（LR、MLP、LSTM、Attention-LSTM）
├── evaluate/            # 离线评估代码（指标计算、混淆矩阵绘制）
├── utils/               # 工具函数（关键点提取、数据标注、计数逻辑）
├── main.py              # 主运行脚本（数据处理→训练→评估）
├── requirements.txt     # 依赖包清单
└── results/             # 评估结果（混淆矩阵图、指标报表）
```

## 环境配置
1. 克隆本仓库：
```bash
git clone [你的仓库地址]
cd 俯卧撑动作检测系统
```
2. 安装依赖：
```bash
pip install -r requirements.txt
# 补充：MediaPipe、OpenCV 若需单独安装
pip install mediapipe opencv-python
```

## 快速运行
### 1. 数据集准备
将标注好的俯卧撑动作视频/图片放入 `data/` 目录，运行特征提取脚本：
```bash
python feature/extract_features.py
```
### 2. 模型训练与对比
```bash
# 运行所有模型训练与评估
python main.py
```
### 3. 查看评估结果
训练完成后，评估指标与混淆矩阵会保存至 `results/` 目录，可直接查看可视化结果。

## 核心模块说明
### 1. 关键点提取
基于 MediaPipe Pose 提取人体 33 个关键点，聚焦俯卧撑相关关节（肘部、肩部、髋部），输出关键点坐标与置信度。
### 2. 特征工程
- 肘角：计算肘部关节的角度（判断手臂弯曲程度）；
- 肩髋距离：计算肩部与髋部关键点的欧式距离（判断身体是否保持平直）；
- 时序特征：对连续帧关键点序列进行归一化、差分处理（适配LSTM/Attention-LSTM）。
### 3. 模型对比
| 模型          | 适用场景                | 核心优势                  |
|---------------|-------------------------|---------------------------|
| Logistic Regression | 静态动作二分类          | 轻量、易解释              |
| MLP           | 静态特征多分类          | 非线性拟合能力            |
| LSTM          | 时序动作识别            | 捕捉长短期时序依赖        |
| Attention-LSTM| 时序动作关键帧聚焦      | 强化重要帧的特征权重      |
### 4. 离线评估
输出各模型的混淆矩阵（可视化）、Precision/Recall/F1-Score 表格，对比不同模型的动作识别精度与计数准确率。

## 结果展示
- 混淆矩阵示例：`results/confusion_matrix_attention_lstm.png`
- 指标汇总表：`results/metrics_summary.csv`
- 俯卧撑计数可视化：`results/count_demo.mp4`（可选）

## 致谢
感谢指导老师的建议，以及 MediaPipe、Scikit-learn、PyTorch 等开源库的支持。
```
