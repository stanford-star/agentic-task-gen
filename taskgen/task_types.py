"""Everything that differs between binary classification and regression tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from tabicl import TabICLClassifier, TabICLRegressor
from xgboost import XGBClassifier, XGBRegressor

from taskgen import config
from taskgen.predictors import KumoTabularClassifier, KumoTabularRegressor


class TaskType(ABC):
    name: str  # RelBench's task_type
    label_format: str  # what the target column must hold
    metric: str
    higher_is_better: bool
    baseline_statistic: str  # label statistic the per-entity and history baselines predict
    leakage_metric: str  # key of config.LEAKAGE_SCORE
    predictors: dict  # the estimator class for each config.PREDICTOR

    @abstractmethod
    def is_valid(self, labels: pd.Series) -> bool:
        """Whether a target column holds label_format."""

    def predictor(self):
        """An unfitted config.PREDICTOR."""
        return self.predictors[config.PREDICTOR](**config.PREDICTOR_PARAMS[config.PREDICTOR])

    def sample(self, rows: pd.DataFrame, target: str) -> pd.DataFrame:
        """At most LEARNABILITY_SAMPLE_SIZE rows, drawn uniformly."""
        return rows.sample(min(len(rows), config.LEARNABILITY_SAMPLE_SIZE), random_state=config.SEED)

    @abstractmethod
    def predict(self, predictor, features: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def score(self, y, pred, y_train) -> float:
        """The metric."""

    @abstractmethod
    def leakage_score(self, y, pred) -> float:
        """The leakage_metric."""


class BinaryClassification(TaskType):
    name = "binary_classification"
    label_format = "BIGINT 0/1"
    metric = "auroc"
    higher_is_better = True
    baseline_statistic = "mean"
    leakage_metric = "auroc"
    predictors = {
        "lightgbm": LGBMClassifier,
        "xgboost": XGBClassifier,
        "tabicl": TabICLClassifier,
        "kumo-tabular": KumoTabularClassifier,
    }

    def is_valid(self, labels):
        return is_integer_dtype(labels) and set(pd.unique(labels)) <= {0, 1}

    def sample(self, rows, target):
        """Drawn uniformly, plus a row of a class the draw missed: fitting and AUROC need both."""
        drawn = super().sample(rows, target)
        return pd.concat([drawn, rows[~rows[target].isin(drawn[target])].drop_duplicates(target)])

    def predict(self, predictor, features):
        return predictor.predict_proba(features)[:, 1]

    def score(self, y, pred, y_train):
        return roc_auc_score(y, pred)

    def leakage_score(self, y, pred):
        return roc_auc_score(y, pred)


class Regression(TaskType):
    name = "regression"
    label_format = "finite numbers"
    metric = "nmae"
    higher_is_better = False
    baseline_statistic = "median"
    leakage_metric = "r2"
    predictors = {
        "lightgbm": LGBMRegressor,
        "xgboost": XGBRegressor,
        "tabicl": TabICLRegressor,
        "kumo-tabular": KumoTabularRegressor,
    }

    def is_valid(self, labels):
        numeric = is_numeric_dtype(labels) and not is_bool_dtype(labels)
        return numeric and bool(np.isfinite(labels.to_numpy(float)).all())

    def predict(self, predictor, features):
        return predictor.predict(features)

    def score(self, y, pred, y_train):
        return mean_absolute_error(y, pred) / (np.std(y_train, ddof=1) or 1.0)

    def leakage_score(self, y, pred):
        return r2_score(y, pred)


TASK_TYPES = {task_type.name: task_type for task_type in (BinaryClassification(), Regression())}
