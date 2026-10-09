"""Learnability: a predictor on the featurizer's features against featureless baselines, scored on val + test."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from taskgen import config
from taskgen.candidate import Candidate, label_rows
from taskgen.checks.base import Check, Outcome, violations
from taskgen.featurizer import Featurizer


class Learnability(Check):
    """The model, a predictor on the featurizer's features, reliably beats the best baseline, without being
    accurate enough to suggest leakage."""

    name = "learnability"
    task_faults = ()  # the task already passed every other check, so a failure here is the harness's

    def run(self, candidate: Candidate) -> Outcome:
        result = self.measure(candidate)
        rules = {
            f"suspiciously accurate ({result['suspected_leak']}); a column probably leaks the label: find it "
            "(top_features may name it) and hide it with remove_columns": result["suspected_leak"],
            f"no reliable lift over the {result['best_baseline']} baseline "
            f"({config.LIFT_CONFIDENCE:.0%} CI {result['lift_ci']})": result["lift_ci"][0] <= 0,
        }
        return Outcome(violations(rules), result)

    def measure(self, candidate: Candidate) -> dict:
        start = time.time()
        task_type, target = candidate.task_type, candidate.spec.target_col
        train = candidate.splits["train"]
        held_out = pd.concat([candidate.splits["val"], candidate.splits["test"]], ignore_index=True)
        held_out = task_type.sample(held_out, target)

        predictor, model_predictions = self.fit_and_predict(
            candidate, task_type.sample(train, target), held_out
        )
        baselines = self.baseline_predictions(candidate, train, held_out)
        predictions = {"model": model_predictions, **baselines}
        y, y_train = held_out[target].to_numpy(float), train[target]
        scores = {name: task_type.score(y, pred, y_train) for name, pred in predictions.items()}
        direction = 1 if task_type.higher_is_better else -1
        best = max(baselines, key=lambda name: direction * scores[name])

        def lift(rows) -> float:
            """How much better the model scores than the best baseline on `rows`."""
            model_score = task_type.score(y[rows], predictions["model"][rows], y_train)
            return direction * (model_score - task_type.score(y[rows], predictions[best][rows], y_train))

        rng = np.random.default_rng(config.SEED)
        resamples = (rng.integers(0, len(y), len(y)) for _ in range(config.BOOTSTRAP_RESAMPLES))
        lifts = [lift(rows) for rows in resamples if np.unique(y[rows]).size > 1]
        tail = (1 - config.LIFT_CONFIDENCE) / 2
        accuracy = task_type.leakage_score(y, predictions["model"])
        suspected_leak = accuracy >= config.LEAKAGE_SCORE[task_type.leakage_metric]
        return {
            "predictor": config.PREDICTOR,
            "metric": task_type.metric,
            "scores": {name: round(float(score), 4) for name, score in scores.items()},
            "best_baseline": best,
            "lift": round(lift(np.arange(len(y))), 4),
            "lift_ci": [round(float(q), 4) for q in np.quantile(lifts, [tail, 1 - tail])],
            "suspected_leak": f"{task_type.leakage_metric} {accuracy:.3f}" if suspected_leak else None,
            "top_features": self.top_features(predictor),
            "seconds": round(time.time() - start, 1),
        }

    @staticmethod
    def fit_and_predict(candidate: Candidate, train: pd.DataFrame, held_out: pd.DataFrame):
        """The predictor fitted on the train rows' features, and its predictions for the held-out rows."""
        featurizer, predictor = Featurizer(candidate), candidate.task_type.predictor()
        predictor.fit(featurizer.fit_transform(train), train[candidate.spec.target_col])
        predictions = candidate.task_type.predict(predictor, featurizer.transform(held_out))
        return predictor, np.asarray(predictions, dtype=float)

    @staticmethod
    def top_features(predictor) -> list[str]:
        """The features with the largest shares of the predictor's importances, if it reports any."""
        if not hasattr(predictor, "feature_importances_"):
            return []
        importance = pd.Series(predictor.feature_importances_, index=predictor.feature_names_in_)
        share = importance / importance.sum()
        return [f"{name} ({v:.0%})" for name, v in share.nlargest(5).items() if v > 0]

    @staticmethod
    def baseline_predictions(candidate: Candidate, train: pd.DataFrame, held_out: pd.DataFrame) -> dict:
        """Featureless predictions: global and per-entity constants, and the entity's labels known strictly
        before the row's timestamp, as in the featurizer."""
        spec, statistic = candidate.spec, candidate.task_type.baseline_statistic
        global_value = train[spec.target_col].agg(statistic)
        per_entity = train.groupby(spec.entity_col)[spec.target_col].agg(statistic)
        known = candidate.known_labels
        known = known.assign(
            history=known.groupby("entity")["label"].transform(lambda s: s.expanding().agg(statistic))
        )
        rows = label_rows(held_out, spec).reset_index(drop=True).rename_axis("row").reset_index()
        history = pd.merge_asof(
            rows.sort_values("timestamp"),
            known,
            on="timestamp",
            by="entity",
            allow_exact_matches=False,
            suffixes=("", "_known"),
        ).sort_values("row")
        return {
            "global": np.full(len(held_out), global_value),
            "per_entity": held_out[spec.entity_col].map(per_entity).fillna(global_value).to_numpy(float),
            "history": history["history"].fillna(global_value).to_numpy(float),
            "last": history["label_known"].fillna(global_value).to_numpy(float),
        }
