"""Leakage: reading past the label window, or timestamps in one query affecting each other."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taskgen.candidate import KEY, Candidate
from taskgen.checks.base import Check, Outcome, violations


def changed_labels(before: pd.DataFrame, after: pd.DataFrame) -> np.ndarray:
    """Per (timestamp, entity) row of either frame: whether the other lacks it or labels it differently."""
    both = before.merge(after, on=KEY, how="outer", suffixes=("", "_after"))
    return ~np.isclose(both["label"], both["label_after"])


class LabelWindow(Check):
    """Labels use rows up to the timestamp + timedelta, and nothing later."""

    name = "label window"

    def run(self, candidate: Candidate) -> Outcome:
        db, horizon = candidate.db, candidate.horizon
        changed = sum(
            changed_labels(rows, candidate.query(db.upto(t + horizon), [t])).sum()
            for t, rows in candidate.labels.items()
        )
        message = (
            f"{changed} labels change when rows after the timestamp + timedelta are removed; "
            "the label window is (timestamp, timestamp + timedelta]"
        )
        return Outcome(violations({message: changed}))


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
