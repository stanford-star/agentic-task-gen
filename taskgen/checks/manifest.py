"""The task manifest, checked before any query runs."""

from __future__ import annotations

import pandas as pd
from relbench.manifest import KIND_FORECAST

from taskgen.candidate import Candidate
from taskgen.checks.base import Check, Outcome, violations
from taskgen.task_types import TASK_TYPES

# the keys of generate.md's example
MANIFEST_KEYS = {
    "name",
    "kind",
    "task_type",
    "description",
    "entity_table",
    "entity_col",
    "target_col",
    "time_col",
    "timedelta",
    "num_eval_timestamps",
    "remove_columns",
    "sql",
    "manifest_version",
}


class Manifest(Check):
    """A RelBench 3 forecast manifest for an entity table of this database."""

    name = "manifest"

    def run(self, candidate: Candidate) -> Outcome:
        spec, tables = candidate.spec, candidate.dataset.tables
        unknown = sorted(set(candidate.raw_manifest) - MANIFEST_KEYS)
        sql, timedelta, eval_timestamps = spec.sql or "", spec.timedelta, spec.num_eval_timestamps
        requested = spec.remove_columns or []
        removed = [tuple(pair) for pair in requested if isinstance(pair, (list, tuple)) and len(pair) == 2]
        missing = [f"{t}.{c}" for t, c in removed if c not in candidate.columns.get(t, [])]
        keys = {(t, col) for t, table in tables.items() for col in (table.pkey, table.time_col, *table.fkeys)}
        hidden_keys = [f"{t}.{c}" for t, c in removed if (t, c) in keys]
        rules = {
            f"unknown manifest keys {unknown}": unknown,
            f"name must be the directory name '{candidate.name}'": spec.name != candidate.name,
            "kind must be 'forecast'": spec.kind != KIND_FORECAST,
            f"task_type must be one of {list(TASK_TYPES)}": spec.task_type not in TASK_TYPES,
            "entity_table must be a table with a primary key": (
                spec.entity_table not in tables or not tables[spec.entity_table].pkey
            ),
            "time_col, entity_col and target_col must differ": (
                len({spec.time_col, spec.entity_col, spec.target_col}) < 3
            ),
            "sql must read `timestamps` and use INTERVAL '{timedelta}'": (
                "timestamps" not in sql or "{timedelta}" not in sql
            ),
            "timedelta must be a positive duration with a unit, such as '30 days'": (
                not (isinstance(timedelta, str) and any(c.isalpha() for c in timedelta))
                or candidate.horizon <= pd.Timedelta(0)
            ),
            "num_eval_timestamps must be a positive integer": (
                not (isinstance(eval_timestamps, int) and eval_timestamps > 0)
            ),
            "remove_columns must be a list of [table, column] pairs": len(removed) != len(requested),
            f"remove_columns names columns that do not exist: {missing}": missing,
            f"remove_columns cannot hide keys or time columns: {hidden_keys}": hidden_keys,
        }
        findings = {"task_type": spec.task_type, "entity_table": spec.entity_table, "timedelta": timedelta}
        return Outcome(violations(rules), findings)
