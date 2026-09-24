"""``taskgen init``: a workspace with a RelBench 3 dataset and the agent's folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import relbench
import yaml
from relbench.manifest import DatasetManifest, validate_dataset_manifest

from taskgen import REPO_ROOT

WORKSPACE_ROOT = REPO_ROOT / "workspace"


@dataclass(frozen=True)
class Workspace:
    name: str

    @property
    def root(self) -> Path:
        return WORKSPACE_ROOT / self.name

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def db_dir(self) -> Path:
        return self.dataset_dir / "db"

    @property
    def tasks_dir(self) -> Path:
        return self.dataset_dir / "tasks"

    @property
    def results_tsv(self) -> Path:
        return self.root / "results.tsv"

    def candidate(self, task: str) -> Path:
        return self.root / "candidates" / task

    def manifest(self) -> DatasetManifest:
        return DatasetManifest.load(self.dataset_dir / "manifest.yaml")

    @classmethod
    def create(
        cls,
        dataset: str,
        name: str | None = None,
        val_cutoff: str | None = None,
        test_cutoff: str | None = None,
    ) -> Workspace:
        """A new workspace for a RelBench 3 dataset (Hub spec or local folder)."""
        local = Path(dataset)  # read local folders directly: RelBench cannot load one without cutoffs
        source = local if (local / "manifest.yaml").exists() else relbench.load_dataset(dataset).dataset_dir
        source = Path(source).resolve()
        raw = yaml.safe_load((source / "manifest.yaml").read_text())
        raw["val_timestamp"] = val_cutoff or raw.get("val_timestamp")
        raw["test_timestamp"] = test_cutoff or raw.get("test_timestamp")
        if not (raw["val_timestamp"] and raw["test_timestamp"]):
            raise ValueError("the dataset has no val/test cutoffs: pass --val and --test")
        manifest = DatasetManifest.from_dict(raw)
        validate_dataset_manifest(manifest, source / "db")
        tz_aware = [
            f"{table}.{table_spec.time_col}"
            for table, table_spec in manifest.tables.items()
            if table_spec.time_col and is_tz_aware(source / "db" / f"{table}.parquet", table_spec.time_col)
        ]
        if tz_aware:
            raise ValueError(
                f"{', '.join(tz_aware)}: timezone-aware; RelBench compares time columns "
                "with naive cutoffs, so convert them to naive UTC in the source data"
            )

        if pd.Timestamp(manifest.test_timestamp) <= pd.Timestamp(manifest.val_timestamp):
            raise ValueError("the test cutoff must be after the val cutoff")

        ws = cls(name or manifest.name)
        if ws.root.exists():
            raise FileExistsError(f"{ws.root} already exists")
        ws.tasks_dir.mkdir(parents=True)
        for folder in ("candidates", "notes", "scratch", "logs", "rejected", "reviews"):
            (ws.root / folder).mkdir()
        manifest.save(ws.dataset_dir / "manifest.yaml")
        ws.db_dir.symlink_to(source / "db", target_is_directory=True)
        return ws

    @classmethod
    def open(cls, name: str) -> Workspace:
        ws = cls(name)
        if not (ws.dataset_dir / "manifest.yaml").exists():
            raise FileNotFoundError(f"no workspace {ws.root}: run `taskgen init` first")
        return ws


def is_tz_aware(parquet: Path, column: str) -> bool:
    column_type = pq.read_schema(parquet).field(column).type
    return getattr(column_type, "tz", None) is not None  # only timestamp types have a tz
