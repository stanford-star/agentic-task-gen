"""The train, val and test splits RelBench materializes from the task's SQL, and their rows."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_dtype

from taskgen import config
from taskgen.candidate import Candidate
from taskgen.checks.base import Check, Outcome, violations


def label_stats(labels: pd.Series) -> dict:
    counts = labels.value_counts()
    mode, top = next(zip(counts.index.tolist(), counts.tolist()), (None, 0))
    return {
        "rows": len(labels),
        "mean": round(float(labels.mean()), 4),
        "mode": mode,
        "mode_share": round(top / max(len(labels), 1), 4),
        "minority": len(labels) - top,
    }


class Splits(Check):
    """RelBench materializes the train, val and test splits; its errors become the reasons."""

    name = "splits"

    def run(self, candidate: Candidate) -> Outcome:
        horizon, (val, _) = candidate.horizon, candidate.cutoffs
        train_timestamps = max(0, (val - horizon - candidate.db.min_timestamp) // horizon + 1)
        if train_timestamps > config.MAX_TRAIN_TIMESTAMPS:  # RelBench would take hours or run out of memory
            limit = config.MAX_TRAIN_TIMESTAMPS
            return Outcome([f"{train_timestamps} train timestamps, more than {limit}: lengthen timedelta"])
        rows = {split: len(df) for split, df in candidate.splits.items()}
        return Outcome([], {"train_timestamps": train_timestamps, "rows": rows})


class Output(Check):
    """Every split row is well-formed, and its time column holds the timestamp it was queried for."""

    name = "output"

    def run(self, candidate: Candidate) -> Outcome:
        where: dict[str, list[str]] = {}
        for split, df in candidate.splits.items():
            for violation in self.split_violations(df, candidate):
                where.setdefault(violation, []).append(split)
        if where:
            return Outcome([f"{violation} (in {', '.join(splits)})" for violation, splits in where.items()])
        wrong = sum(int((rows["timestamp"] != t).sum()) for t, rows in candidate.labels.items())
        message = f"{wrong} rows have a {candidate.spec.time_col} other than the timestamp queried"
        return Outcome(violations({message: wrong}))

    @staticmethod
    def split_violations(df: pd.DataFrame, candidate: Candidate) -> list[str]:
        spec, task_type = candidate.spec, candidate.task_type
        columns = [spec.time_col, spec.entity_col, spec.target_col]
        if sorted(df.columns) != sorted(columns):
            return [f"output columns {list(df.columns)} != {columns}"]
        if not is_datetime64_dtype(df[spec.time_col]):
            return [f"{spec.time_col} must hold the timestamp itself, a TIMESTAMP without time zone"]
        entities = candidate.db.table_dict[spec.entity_table]
        created = (
            entities.df.set_index(entities.pkey_col)[entities.time_col]
            if entities.time_col
            else pd.Series(dtype="datetime64[ns]")
        )
        unborn = df[spec.entity_col].map(created).astype("datetime64[ns]") > df[spec.time_col]
        rules = {
            "NULLs in the output": df.isna().to_numpy().any(),
            f"{spec.entity_col} must hold {spec.entity_table} primary keys": (
                not df[spec.entity_col].isin(entities.df[entities.pkey_col]).all()
            ),
            f"{unborn.sum()} rows label entities created after their timestamp; keep "
            f"{spec.entity_table}.{entities.time_col} <= the timestamp": unborn.any(),
            f"duplicate ({spec.time_col}, {spec.entity_col}) rows": df.duplicated(columns[:2]).any(),
            f"{spec.target_col} must hold {task_type.label_format}": (
                not task_type.is_valid(df[spec.target_col])
            ),
        }
        return violations(rules)


class SplitSizes(Check):
    """Every split has enough rows, and enough labels other than the most common one."""

    name = "split sizes"

    def run(self, candidate: Candidate) -> Outcome:
        stats = {split: label_stats(df[candidate.spec.target_col]) for split, df in candidate.splits.items()}
        floors = {"rows": config.MIN_ROWS, "minority": config.MIN_MINORITY}
        rules = {
            f"{split}: {key} {s[key]} < {floor}": s[key] < floor
            for split, s in stats.items()
            for key, floor in floors.items()
        }
        return Outcome(violations(rules), stats)
