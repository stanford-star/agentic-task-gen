"""A candidate task and everything the checks derive from it."""

from __future__ import annotations

import json
import shutil
from functools import cached_property
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import relbench
import yaml
from relbench.base import Database, EntityTask
from relbench.load import _ForecastEntityTask
from relbench.manifest import DatasetManifest, TaskManifest

from taskgen import config
from taskgen.task_types import TASK_TYPES, TaskType
from taskgen.workspace import Workspace

SPLITS = ("train", "val", "test")
KEY = ["timestamp", "entity"]


def read_yaml(path: Path) -> dict:
    data = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must be a YAML mapping")
    return data


def to_json(data: dict) -> str:
    return json.dumps(data, indent=2, default=str)


def label_rows(df: pd.DataFrame, spec) -> pd.DataFrame:
    """A task table as (timestamp, entity, label) columns with comparable dtypes."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[spec.time_col]).astype("datetime64[ns]"),
            "entity": df[spec.entity_col].astype(float),
            "label": df[spec.target_col].astype(float),
        }
    )


class Candidate:
    def __init__(self, ws: Workspace, name: str):
        self.ws, self.name = ws, name
        self.dir = ws.candidate(name)
        self.stored_dir = ws.tasks_dir / name

    @cached_property
    def raw_manifest(self) -> dict:
        return read_yaml(self.dir / "manifest.yaml")

    @cached_property
    def spec(self) -> TaskManifest:
        spec = TaskManifest.from_dict(self.raw_manifest)
        spec.validate()
        return spec

    @cached_property
    def dataset(self) -> DatasetManifest:
        return self.ws.manifest()

    @cached_property
    def cutoffs(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """The dataset's val and test cutoffs."""
        return pd.Timestamp(self.dataset.val_timestamp), pd.Timestamp(self.dataset.test_timestamp)

    @cached_property
    def columns(self) -> dict[str, list[str]]:
        """Each table's columns, from its parquet schema."""
        return {
            table: pq.read_schema(self.ws.db_dir / f"{table}.parquet").names for table in self.dataset.tables
        }

    @cached_property
    def task_type(self) -> TaskType:
        return TASK_TYPES[self.spec.task_type]

    @cached_property
    def horizon(self) -> pd.Timedelta:
        return pd.Timedelta(self.spec.timedelta)

    @cached_property
    def task(self) -> EntityTask:
        # load_task(regenerate=True) without its eager train split
        return _ForecastEntityTask(
            relbench.load_dataset(self.ws.dataset_dir), self.spec, task_dir=self.dir, regenerate=True
        )

    @cached_property
    def db(self) -> Database:
        return self.task.get_db(upto_test_timestamp=False)

    @cached_property
    def checked_timestamps(self) -> list[pd.Timestamp]:
        """Where leakage is checked: the latest train timestamps and the val and test cutoffs."""
        val, test = self.cutoffs
        train = [val - k * self.horizon for k in range(config.LEAKAGE_TRAIN_TIMESTAMPS, 0, -1)]
        return [*train, val, test]

    @cached_property
    def labels(self) -> dict[pd.Timestamp, pd.DataFrame]:
        """The task's SQL at each checked timestamp, queried alone."""
        return {t: self.query(self.db, [t]) for t in self.checked_timestamps}

    @cached_property
    def splits(self) -> dict[str, pd.DataFrame]:
        return {split: self.task.get_table(split, mask_input_cols=False, db=self.db).df for split in SPLITS}

    def query(self, db: Database, timestamps) -> pd.DataFrame:
        """The task's SQL at `timestamps` on `db`, run as RelBench does."""
        return label_rows(self.task.make_table(db, pd.DatetimeIndex(timestamps)).df, self.spec)

    def store(self, report: dict) -> None:
        """Keeps the task in the workspace's RelBench dataset, replacing a stored version."""
        self.remove_stored()
        self.stored_dir.mkdir()
        shutil.copy2(self.dir / "manifest.yaml", self.stored_dir / "manifest.yaml")
        for split, df in self.splits.items():
            df.to_parquet(self.stored_dir / f"{split}.parquet", index=False)
        (self.stored_dir / "taskgen.json").write_text(to_json(report))

    def remove_stored(self) -> None:
        shutil.rmtree(self.stored_dir, ignore_errors=True)
