import numpy as np
import joblib
from abc import ABC, abstractmethod

# 状态常量（全局统一）
STATE_DOWN = 0
STATE_UP = 1
STATE_UNCERTAIN = -1

class PushupModel(ABC):
    """统一模型接口：任何模型（sklearn / Keras / CNN）都应实现这一接口。"""

    @abstractmethod
    def predict_proba_from_features(self, feat):
        """
        输入:
            feat: 可被模型接受的特征向量，形状 (n_features,) 或 (1, n_features)
        返回:
            numpy array [p(label0), p(label1)] 对应于类 0 和类 1 的概率（保证长度为2）
        """
        pass

    def predict_state_from_features(self, feat, thresh=0.75):
        """
        在统一阈值下把概率映射成三态： DOWN(0) / UP(1) / UNCERTAIN(-1)
        返回: (state, p_down, p_up) ，p_down/p_up 分别为类0/类1 的概率
        """
        proba = self.predict_proba_from_features(feat)
        # proba expected shape (2,)
        # index 0 -> class 0 (DOWN), index 1 -> class 1 (UP)
        p_down = float(proba[0])
        p_up = float(proba[1])

        if p_up >= thresh and p_up > p_down:
            return STATE_UP, p_down, p_up
        elif p_down >= thresh and p_down > p_up:
            return STATE_DOWN, p_down, p_up
        else:
            return STATE_UNCERTAIN, p_down, p_up


class LogisticPushupModel(PushupModel):
    """
    Wrapper for sklearn logistic regression (or other sklearn classifiers providing predict_proba).
    Requires a fitted model and a fitted scaler (both saved via joblib).
    """

    def __init__(self, model_path, scaler_path, classes=None):
        """
        model_path: path to joblib saved sklearn model
        scaler_path: path to joblib saved StandardScaler
        classes: optional, explicit class ordering (list like [0,1]) ; if None we'll use model.classes_
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        # Ensure we know where class 0/1 are in predict_proba output:
        self.classes_ = getattr(self.model, "classes_", None)
        if self.classes_ is None:
            raise ValueError("Loaded model has no attribute classes_. Was it trained with sklearn?")
        # map indices: idx_of_0 = where classes_ == 0
        self.index_of_0 = int(list(self.classes_).index(0))
        self.index_of_1 = int(list(self.classes_).index(1))

    def predict_proba_from_features(self, feat):
        """
        feat: array-like (3,) or (1,3)
        return: np.array([p_class0, p_class1])
        """
        x = np.array(feat).reshape(1, -1)
        x_s = self.scaler.transform(x)
        proba_all = self.model.predict_proba(x_s)[0]  # order correspond to model.classes_
        # Reorder explicitly to [p(class0), p(class1)]
        p0 = proba_all[self.index_of_0]
        p1 = proba_all[self.index_of_1]
        return np.array([p0, p1])
