"""Leakage: reading outside the strict past and the label window, or timestamps in one query affecting each
other."""

from __future__ import annotations

import numpy as np
import pandas as pd
from relbench.base import Database, Table

from taskgen import config
from taskgen.candidate import KEY, Candidate
from taskgen.checks.base import Check, Outcome, violations


def changed_labels(before: pd.DataFrame, after: pd.DataFrame) -> np.ndarray:
    """Per (timestamp, entity) row of either frame: whether the other lacks it or labels it differently."""
    both = before.merge(after, on=KEY, how="outer", suffixes=("", "_after"))
    return ~np.isclose(both["label"], both["label_after"])


def without_rows_at(db: Database, timestamp: pd.Timestamp) -> Database:
    """The database without its rows timestamped exactly `timestamp`."""
    return Database(
        {
            name: Table(
                df=table.df[table.df[table.time_col] != timestamp] if table.time_col else table.df,
                fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                pkey_col=table.pkey_col,
                time_col=table.time_col,
            )
            for name, table in db.table_dict.items()
        }
    )


class LabelWindow(Check):
    """A timestamp's rows use the database strictly before the timestamp and the label window
    (timestamp, timestamp + timedelta], nothing at the timestamp itself or after the window."""

    name = "label window"

    def run(self, candidate: Candidate) -> Outcome:
        db, horizon = candidate.db, candidate.horizon
        after = sum(
            changed_labels(rows, candidate.query(db.upto(t + horizon), [t])).sum()
            for t, rows in candidate.labels.items()
        )
        at = sum(
            changed_labels(candidate.query(db, [t]), candidate.query(without_rows_at(db, t), [t])).sum()
            for t in self.timestamps_with_rows(candidate)
        )
        rules = {
            f"{after} labels change when rows after the timestamp + timedelta are removed; "
            "the label window is (timestamp, timestamp + timedelta]": after,
            f"{at} rows change when rows at exactly the timestamp are removed; use rows strictly "
            "before the timestamp (< timestamp), and the label window (timestamp, timestamp + timedelta]": at,
        }
        return Outcome(violations(rules))

    @staticmethod
    def timestamps_with_rows(candidate: Candidate) -> list[pd.Timestamp]:
        """The latest task timestamps at which the database has rows: the only ones where reading rows at
        the timestamp changes anything."""
        tables = candidate.db.table_dict.values()
        row_times = pd.concat([table.df[table.time_col] for table in tables if table.time_col])
        task_times = pd.DatetimeIndex(pd.concat(candidate.splits.values())[candidate.spec.time_col].unique())
        return list(task_times[task_times.isin(row_times)].sort_values()[-config.STRICT_PAST_TIMESTAMPS :])


class TimestampIndependence(Check):
    """A timestamp's rows do not depend on the other timestamps in the query."""

    name = "timestamp independence"

    def run(self, candidate: Candidate) -> Outcome:
        together = candidate.query(candidate.db, candidate.checked_timestamps)
        changed = changed_labels(pd.concat(candidate.labels.values()), together).sum()
        message = (
            f"{changed} rows change when the timestamps are queried together; partition "
            "window functions and aggregates by the timestamp"
        )
        return Outcome(violations({message: changed}))
