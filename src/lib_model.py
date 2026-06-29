# -*- coding: utf-8 -*-
"""
lib_model.py  —  머신러닝 모델(직접 구현판)
------------------------------------------------------------
scikit-learn / xgboost 를 설치하지 않아도 돌아가도록, '인공지능 수학'에
나오는 핵심 알고리즘을 numpy 로 직접 구현했다. 두 모델을 제공한다.

 (1) LogisticRegression  — 경사하강법으로 학습. 계수(coef_)가 곧
     '어떤 변수가 위험을 키우는지' 설명이 된다.  ⇒ 원인 설명에 사용.
 (2) RandomForest        — 의사결정나무(지니 불순도) 여러 그루의 평균.
     변수 중요도를 주고, 비선형 패턴을 더 잘 잡는다.  ⇒ 주력 예측.

※ 이 파일을 scikit-learn 으로 바꾸고 싶다면 함수 호출부만 교체하면 된다.
   (사용법은 README 참고)
"""
import numpy as np


# ===========================================================
# 표준화: 각 변수를 (값-평균)/표준편차 로 바꿔 스케일을 통일
#   - 경사하강법이 빠르고 안정적으로 수렴하게 해준다.
# ===========================================================
class Standardizer:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ===========================================================
# (1) 로지스틱 회귀  (Logistic Regression)
#   z = w·x + b,  p = 1/(1+e^-z),  손실 = 로그손실(클래스 가중치 적용)
#   가중치 갱신: w ← w - η·∂L/∂w   (경사하강법)
# ===========================================================
class LogisticRegression:
    def __init__(self, lr=0.1, epochs=400, l2=0.001, class_weight=True):
        self.lr = lr            # 학습률 η
        self.epochs = epochs    # 반복 횟수
        self.l2 = l2            # 과적합 방지(L2 규제)
        self.class_weight = class_weight

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        # 불균형 보정: 사고(1) 표본에 더 큰 가중치
        if self.class_weight:
            pos = max(y.sum(), 1); neg = max((1 - y).sum(), 1)
            w_pos = n / (2 * pos); w_neg = n / (2 * neg)
        else:
            w_pos = w_neg = 1.0
        sw = np.where(y == 1, w_pos, w_neg)
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            err = (p - y) * sw
            grad_w = X.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def contributions(self, x_row):
        """표준화된 한 행에 대해 변수별 기여도(coef × 값). 원인 설명용."""
        return self.w * x_row


# ===========================================================
# (2) 의사결정나무 + 랜덤 포레스트
#   - 지니 불순도가 가장 많이 줄어드는 곳에서 데이터를 가른다.
# ===========================================================
def _gini(y):
    if len(y) == 0:
        return 0.0
    p = y.mean()
    return 1 - (p**2 + (1 - p)**2)


class DecisionTree:
    def __init__(self, max_depth=6, min_leaf=8, n_feat=None, rng=None):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.n_feat = n_feat          # 각 분기에서 후보로 볼 변수 수(랜덤)
        self.rng = rng or np.random.default_rng(0)
        self.importance = None

    def fit(self, X, y):
        self.d = X.shape[1]
        self.importance = np.zeros(self.d)
        self.root = self._build(X, y, 0)
        s = self.importance.sum()
        if s > 0:
            self.importance /= s
        return self

    def _build(self, X, y, depth):
        node = {"leaf": True, "val": float(y.mean()) if len(y) else 0.0}
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or y.min() == y.max():
            return node
        n, d = X.shape
        feats = np.arange(d)
        if self.n_feat:
            feats = self.rng.choice(d, size=min(self.n_feat, d), replace=False)
        best = None
        parent_imp = _gini(y) * n
        for f in feats:
            vals = np.unique(X[:, f])
            if len(vals) > 12:  # 후보 임계값을 12개 분위로 제한(속도)
                vals = np.quantile(X[:, f], np.linspace(0.1, 0.9, 9))
            for thr in vals:
                left = X[:, f] <= thr
                if left.sum() < self.min_leaf or (~left).sum() < self.min_leaf:
                    continue
                imp = _gini(y[left]) * left.sum() + _gini(y[~left]) * (~left).sum()
                gain = parent_imp - imp
                if best is None or gain > best[0]:
                    best = (gain, f, thr, left)
        if best is None or best[0] <= 0:
            return node
        gain, f, thr, left = best
        self.importance[f] += gain
        return {"leaf": False, "f": int(f), "thr": float(thr),
                "L": self._build(X[left], y[left], depth + 1),
                "R": self._build(X[~left], y[~left], depth + 1)}

    def predict_proba(self, X):
        # 벡터화된 예측: 노드별로 행 묶음을 한 번에 내려보냄(빠름)
        out = np.zeros(len(X))
        stack = [(self.root, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if len(idx) == 0:
                continue
            if node["leaf"]:
                out[idx] = node["val"]
            else:
                m = X[idx, node["f"]] <= node["thr"]
                stack.append((node["L"], idx[m]))
                stack.append((node["R"], idx[~m]))
        return out


class RandomForest:
    def __init__(self, n_trees=60, max_depth=7, min_leaf=8, n_feat=None, seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.n_feat = n_feat
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.n_feat = self.n_feat or max(1, int(round(np.sqrt(d))))
        self.trees = []
        self.feature_importances_ = np.zeros(d)
        for t in range(self.n_trees):
            idx = rng.integers(0, n, n)             # 부트스트랩 표본
            tr = DecisionTree(self.max_depth, self.min_leaf, self.n_feat,
                              np.random.default_rng(self.seed + t))
            tr.fit(X[idx], y[idx])
            self.trees.append(tr)
            self.feature_importances_ += tr.importance
        self.feature_importances_ /= self.n_trees
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)


# ===========================================================
# (3) 그래디언트 부스팅  (XGBoost 의 핵심 아이디어)
#   - 얕은 '회귀 나무' 를 여러 개 쌓되, 각 나무는 '직전까지의 오차'를
#     보완하도록 학습한다. 보통 랜덤포레스트보다 성능이 좋다.
# ===========================================================
class _RegTree:
    """회귀용 의사결정나무: 분산이 가장 많이 줄도록 분할(잎=평균)."""
    def __init__(self, max_depth=3, min_leaf=15, n_feat=None, rng=None):
        self.max_depth = max_depth; self.min_leaf = min_leaf
        self.n_feat = n_feat; self.rng = rng or np.random.default_rng(0)

    def fit(self, X, g):
        self.root = self._build(X, g, 0); return self

    def _build(self, X, g, depth):
        node = {"leaf": True, "val": float(g.mean()) if len(g) else 0.0}
        if depth >= self.max_depth or len(g) < 2 * self.min_leaf:
            return node
        n, d = X.shape
        feats = np.arange(d)
        if self.n_feat:
            feats = self.rng.choice(d, size=min(self.n_feat, d), replace=False)
        parent = g.var() * n
        best = None
        for f in feats:
            vals = np.quantile(X[:, f], np.linspace(0.1, 0.9, 9))
            for thr in np.unique(vals):
                L = X[:, f] <= thr
                if L.sum() < self.min_leaf or (~L).sum() < self.min_leaf:
                    continue
                imp = g[L].var() * L.sum() + g[~L].var() * (~L).sum()
                gain = parent - imp
                if best is None or gain > best[0]:
                    best = (gain, f, thr, L)
        if best is None or best[0] <= 0:
            return node
        _, f, thr, L = best
        return {"leaf": False, "f": int(f), "thr": float(thr),
                "L": self._build(X[L], g[L], depth+1),
                "R": self._build(X[~L], g[~L], depth+1)}

    def predict(self, X):
        out = np.zeros(len(X))
        stack = [(self.root, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if len(idx) == 0:
                continue
            if node["leaf"]:
                out[idx] = node["val"]
            else:
                m = X[idx, node["f"]] <= node["thr"]
                stack.append((node["L"], idx[m]))
                stack.append((node["R"], idx[~m]))
        return out


class GradientBoosting:
    """로지스틱 손실 기반 그래디언트 부스팅(분류)."""
    def __init__(self, n_trees=120, lr=0.1, max_depth=3, min_leaf=15, subsample=0.8, seed=1):
        self.n_trees = n_trees; self.lr = lr; self.max_depth = max_depth
        self.min_leaf = min_leaf; self.subsample = subsample; self.seed = seed

    @staticmethod
    def _sig(z): return 1/(1+np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(y)
        p0 = np.clip(y.mean(), 1e-3, 1-1e-3)
        self.F0 = np.log(p0/(1-p0))      # 초기 로그-오즈
        F = np.full(n, self.F0)
        self.trees = []
        for t in range(self.n_trees):
            grad = y - self._sig(F)       # 음의 그래디언트(=잔차)
            idx = rng.choice(n, int(n*self.subsample), replace=False)
            tr = _RegTree(self.max_depth, self.min_leaf,
                          n_feat=max(1, int(np.sqrt(X.shape[1]))),
                          rng=np.random.default_rng(self.seed+t))
            tr.fit(X[idx], grad[idx])
            F += self.lr * tr.predict(X)
            self.trees.append(tr)
        return self

    def predict_proba(self, X):
        F = np.full(len(X), self.F0)
        for tr in self.trees:
            F += self.lr * tr.predict(X)
        return self._sig(F)


# ===========================================================
# 평가지표 (불균형 데이터에 맞는 지표들)
# ===========================================================
def confusion(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    TP = int(((yhat == 1) & (y == 1)).sum())
    FP = int(((yhat == 1) & (y == 0)).sum())
    FN = int(((yhat == 0) & (y == 1)).sum())
    TN = int(((yhat == 0) & (y == 0)).sum())
    return TP, FP, FN, TN


def metrics(y, p, thr=0.5):
    TP, FP, FN, TN = confusion(y, p, thr)
    rec = TP / (TP + FN) if TP + FN else 0.0
    prec = TP / (TP + FP) if TP + FP else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (TP + TN) / max(len(y), 1)
    return {"recall": rec, "precision": prec, "f1": f1, "accuracy": acc, "auc": roc_auc(y, p)}


def roc_auc(y, p):
    """ROC-AUC 를 순위 기반(Mann–Whitney U)으로 계산."""
    y = np.asarray(y); p = np.asarray(p)
    pos = p[y == 1]; neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    # 동점 보정
    rp = ranks[y == 1].sum()
    auc = (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)
