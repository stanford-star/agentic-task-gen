"""Predictors without a scikit-learn interface, wrapped in one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sdm
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class KumoTabular(BaseEstimator):
    """NVIDIA's Kumo Tabular foundation model: fit caches the rows it predicts from in context."""

    task: str  # sdm's task
    target_stype: str

    def __init__(self, size="small", num_estimators=8, random_state=0, device="cpu"):
        self.size = size
        self.num_estimators = num_estimators
        self.random_state = random_state
        self.device = device

    def fit(self, X: pd.DataFrame, y) -> KumoTabular:
        self.stypes_ = sdm.infer_stypes(X)
        self.model_ = sdm.models.KumoTabular(task=self.task, size=self.size, device=self.device).eval()
        target = pd.DataFrame({"target": np.asarray(y)})
        self.model_.fit(
            self.table(X),
            sdm.TableTensor.from_pandas(target, {"target": self.target_stype}, device=self.device),
            num_estimators=self.num_estimators,
            generator=torch.Generator(self.device).manual_seed(self.random_state),
        )
        return self

    def table(self, X: pd.DataFrame):
        return sdm.TableTensor.from_pandas(X, self.stypes_, device=self.device)

    def query(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model_.predict(self.table(X)).to_pandas()


class KumoTabularClassifier(ClassifierMixin, KumoTabular):
    task, target_stype = "classification", "categorical"

    def predict_proba(self, X):
        return self.query(X).to_numpy(float)  # a column per class, in sorted order


class KumoTabularRegressor(RegressorMixin, KumoTabular):
    task, target_stype = "regression", "numerical"

    def predict(self, X):
        return self.query(X)["q500"].to_numpy(float)  # the median of its 999 quantiles
