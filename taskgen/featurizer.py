"""The featurizer: RDBLearn's features for (entity, timestamp) rows, from the database before each timestamp."""

from __future__ import annotations

from contextlib import contextmanager

import pandas as pd
from fastdfs import DFSConfig, compute_dfs_features, create_rdb
from fastdfs.dfs.duckdb_database import DuckDBBuilder
from fastdfs.transform import (
    CanonicalizeTypes,
    FeaturizeDatetime,
    FillMissingPrimaryKey,
    FilterColumn,
    HandleDummyTable,
    RDBTransformPipeline,
    RDBTransformWrapper,
)
from fastdfs.utils.type_utils import safe_convert_to_string
from rdblearn.config import TemporalDiffConfig
from rdblearn.preprocessing import TabularPreprocessor

from taskgen import config
from taskgen.candidate import Candidate


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


def to_fastdfs(candidate: Candidate):
    """The database plus a label_history table, the history the baselines see."""
    tables, spec = candidate.db.table_dict, candidate.spec
    history = candidate.known_labels.astype({"entity": "int64"})
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


class Featurizer:
    """RDBLearn's features: fastdfs aggregates of the database strictly before each row's timestamp,
    preprocessed as RDBLearn does."""

    def __init__(self, candidate: Candidate):
        spec, tables = candidate.spec, candidate.db.table_dict
        self.entity_col, self.time_col = spec.entity_col, spec.time_col
        self.key_mappings = {spec.entity_col: f"{spec.entity_table}.{tables[spec.entity_table].pkey_col}"}
        prepare = RDBTransformPipeline(  # RDBLearn's preparation of the database
            [
                HandleDummyTable(),
                FillMissingPrimaryKey(),
                RDBTransformWrapper(FeaturizeDatetime(features=["epochtime"])),
                RDBTransformWrapper(FilterColumn(drop_dtypes=["text"])),
                RDBTransformWrapper(CanonicalizeTypes()),
            ]
        )
        self.rdb = prepare(to_fastdfs(candidate))
        self.preprocessor = TabularPreprocessor(
            temporal_diff_config=TemporalDiffConfig(), cutoff_time=self.time_col
        )

    def fit_transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        aggregates = self.aggregates(rows)
        return self.preprocessor.fit(aggregates).transform(aggregates)

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor.transform(self.aggregates(rows))

    def aggregates(self, rows: pd.DataFrame) -> pd.DataFrame:
        """The rows' entity and timestamp, with fastdfs's aggregates."""
        inputs = rows[[self.entity_col, self.time_col]].copy()
        inputs[self.entity_col] = safe_convert_to_string(inputs[self.entity_col])
        with single_threaded_dfs():
            return compute_dfs_features(
                self.rdb,
                inputs,
                key_mappings=self.key_mappings,
                cutoff_time_column=self.time_col,
                config=DFSConfig(max_depth=config.DFS_MAX_DEPTH),
            )
