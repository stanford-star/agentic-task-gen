"""Learnability: RDBLearn + LightGBM against featureless baselines, scored on val + test."""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from fastdfs import DFSConfig
from fastdfs.api import create_rdb
from fastdfs.dfs.duckdb_database import DuckDBBuilder
from rdblearn.config import RDBLearnConfig

from taskgen import config
from taskgen.candidate import Candidate, label_rows
from taskgen.checks.base import Check, Outcome, violations


@contextmanager
def single_threaded_dfs():
    """fastdfs's features vary with DuckDB's thread count, which it has no setting for."""
    original = DuckDBBuilder.__init__

    def init(self, path):
        original(self, path)
        self.db.execute("SET threads = 1")

    DuckDBBuilder.__init__ = init
    try:
        yield
    finally:
        DuckDBBuilder.__init__ = original


def known_labels(candidate: Candidate) -> pd.DataFrame:
    """Every label of the task, dated by the end of its window."""
    known = label_rows(pd.concat(candidate.splits.values()), candidate.spec)
    known["timestamp"] += candidate.horizon
    return known.sort_values("timestamp", ignore_index=True)


def to_fastdfs(candidate: Candidate):
    """The database plus a label_history table, the history the baselines see."""
    tables, spec = candidate.db.table_dict, candidate.spec
    history = known_labels(candidate).astype({"entity": "int64"})
    return create_rdb(
        {name: table.df.copy(deep=False) for name, table in tables.items()} | {"label_history": history},
        name="db",
        primary_keys={name: table.pkey_col for name, table in tables.items() if table.pkey_col},
        foreign_keys=[
            (name, col, parent, tables[parent].pkey_col)
            for name, table in tables.items()
            for col, parent in table.fkey_col_to_pkey_table.items()
        ]
        + [("label_history", "entity", spec.entity_table, tables[spec.entity_table].pkey_col)],
        time_columns={name: table.time_col for name, table in tables.items() if table.time_col}
        | {"label_history": "timestamp"},
    )


class Learnability(Check):
    """The model reliably beats the best baseline, without being accurate enough to suggest leakage."""

    name = "learnability"
    task_faults = ()  # the task already passed every other check, so a failure here is the harness's

    def run(self, candidate: Candidate) -> Outcome:
        result = self.measure(candidate)
        rules = {
            f"suspiciously accurate ({result['suspected_leak']}); a column probably leaks the label: find it "
            "in top_features and hide it with remove_columns": result["suspected_leak"],
            f"no reliable lift over the {result['best_baseline']} baseline "
            f"({config.LIFT_CONFIDENCE:.0%} CI {result['lift_ci']})": result["lift_ci"][0] <= 0,
        }
        return Outcome(violations(rules), result)

    def measure(self, candidate: Candidate) -> dict:
        start = time.time()
        task_type, target = candidate.task_type, candidate.spec.target_col
        train = candidate.splits["train"]
        held_out = pd.concat([candidate.splits["val"], candidate.splits["test"]], ignore_index=True)
        sample_size = min(len(held_out), config.LEARNABILITY_SAMPLE_SIZE)
        held_out = held_out.sample(sample_size, random_state=config.SEED)

        model, model_predictions = self.fit_and_predict(candidate, train, held_out)
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
        importance = pd.Series(
            model.base_estimator.feature_importances_, index=model.downstream_feature_columns_
        )
        return {
            "metric": task_type.metric,
            "scores": {name: round(float(score), 4) for name, score in scores.items()},
            "best_baseline": best,
            "lift": round(lift(np.arange(len(y))), 4),
            "lift_ci": [round(float(q), 4) for q in np.quantile(lifts, [tail, 1 - tail])],
            "suspected_leak": f"{task_type.leakage_metric} {accuracy:.3f}" if suspected_leak else None,
            "top_features": [f"{name} ({int(v)})" for name, v in importance.nlargest(5).items() if v > 0],
            "seconds": round(time.time() - start, 1),
        }

    @staticmethod
    def fit_and_predict(candidate: Candidate, train: pd.DataFrame, held_out: pd.DataFrame):
        """The model fitted on train, and its predictions for the held-out rows."""
        spec, db = candidate.spec, candidate.db
        inputs = [spec.entity_col, spec.time_col]
        model = candidate.task_type.model(
            RDBLearnConfig(
                dfs=DFSConfig(max_depth=config.DFS_MAX_DEPTH),
                max_train_samples=config.LEARNABILITY_SAMPLE_SIZE,
                random_seed=config.SEED,
            )
        )
        entity_key = f"{spec.entity_table}.{db.table_dict[spec.entity_table].pkey_col}"
        with single_threaded_dfs():
            model.fit(
                X=train[inputs],
                y=train[spec.target_col],
                rdb=to_fastdfs(candidate),
                key_mappings={spec.entity_col: entity_key},
                cutoff_time_column=spec.time_col,
            )
            return model, np.asarray(candidate.task_type.predict(model, held_out[inputs]), dtype=float)

    @staticmethod
    def baseline_predictions(candidate: Candidate, train: pd.DataFrame, held_out: pd.DataFrame) -> dict:
        """Featureless predictions: global and per-entity constants, and the entity's known labels."""
        spec, statistic = candidate.spec, candidate.task_type.baseline_statistic
        global_value = train[spec.target_col].agg(statistic)
        per_entity = train.groupby(spec.entity_col)[spec.target_col].agg(statistic)
        known = known_labels(candidate)
        known["history"] = known.groupby("entity")["label"].transform(lambda s: s.expanding().agg(statistic))
        rows = label_rows(held_out, spec).reset_index(drop=True).rename_axis("row").reset_index()
        history = pd.merge_asof(
            rows.sort_values("timestamp"), known, on="timestamp", by="entity", suffixes=("", "_known")
        ).sort_values("row")
        return {
            "global": np.full(len(held_out), global_value),
            "per_entity": held_out[spec.entity_col].map(per_entity).fillna(global_value).to_numpy(float),
            "history": history["history"].fillna(global_value).to_numpy(float),
            "last": history["label_known"].fillna(global_value).to_numpy(float),
        }
