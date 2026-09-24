"""Novelty: a task must not restate one already kept."""

from __future__ import annotations

import pandas as pd
from relbench.manifest import TaskManifest

from taskgen import config
from taskgen.candidate import KEY, Candidate, label_rows
from taskgen.checks.base import Check, Outcome


class Novelty(Check):
    """The train labels are not a near-duplicate of a kept task's on the same entity table."""

    name = "novelty"

    def run(self, candidate: Candidate) -> Outcome:
        mine = label_rows(candidate.splits["train"], candidate.spec)
        for name, theirs in self.kept_labels(candidate):
            both = mine.merge(theirs, on=KEY)
            overlaps = len(both) >= config.DUPLICATE_MIN_OVERLAP * min(len(mine), len(theirs))
            rho = abs(both["label_x"].corr(both["label_y"], method="spearman"))
            if overlaps and rho >= config.DUPLICATE_CORR:
                return Outcome([f"duplicate of kept task '{name}' (|spearman| {rho:.4f})"])
        return Outcome()

    @staticmethod
    def kept_labels(candidate: Candidate):
        """The train labels of the other kept tasks on the candidate's entity table."""
        for manifest in sorted(candidate.ws.tasks_dir.glob("*/manifest.yaml")):
            spec = TaskManifest.load(manifest)
            if spec.entity_table == candidate.spec.entity_table and spec.name != candidate.name:
                yield spec.name, label_rows(pd.read_parquet(manifest.with_name("train.parquet")), spec)
