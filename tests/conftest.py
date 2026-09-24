"""A small synthetic shop database in RelBench 3 layout, and a fresh workspace per test."""

import os
import warnings

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402
from relbench.manifest import DatasetManifest, TableSpec  # noqa: E402

import taskgen.workspace as W  # noqa: E402

VAL, TEST, END = "2020-10-01", "2021-01-01", pd.Timestamp("2021-06-30")


def make_shop(root):
    rng = np.random.default_rng(0)
    n_users, n_items = 300, 60
    start = pd.Timestamp("2019-01-01")
    signup_days = np.sort(rng.uniform(0, (pd.Timestamp(VAL) - start).days, n_users))
    users = pd.DataFrame(
        {
            "user_id": np.arange(n_users),
            "signup_date": (start + pd.to_timedelta(signup_days, unit="D")).floor("D"),
            "country": rng.choice(["A", "B", "C"], n_users),
            "age": rng.integers(18, 70, n_users),
        }
    )
    items = pd.DataFrame(
        {
            "item_id": np.arange(n_items),
            "category": rng.choice(["x", "y", "z"], n_items),
            "price": np.round(rng.lognormal(3, 0.6, n_items), 2),
        }
    )
    rows = []
    for u, signup in zip(users["user_id"], users["signup_date"]):
        rate = rng.lognormal(np.log(0.15), 0.8)
        stop = min(signup + pd.Timedelta(days=float(rng.exponential(300))), END)
        days = max((stop - signup).days, 0)
        n = rng.poisson(rate * days)
        ts = signup + pd.to_timedelta(np.sort(rng.uniform(0, max(days, 1), n)), unit="D")
        rows.append(
            pd.DataFrame(
                {
                    "ts": ts.floor("h"),
                    "user_id": u,
                    "item_id": rng.integers(0, n_items, n),
                    "qty": rng.integers(1, 4, n),
                }
            )
        )
    events = pd.concat(rows).sort_values("ts", kind="stable").reset_index(drop=True)
    events["amount"] = np.round(items["price"].to_numpy()[events["item_id"]] * events["qty"], 2)
    events.insert(0, "event_id", np.arange(len(events)))

    (root / "db").mkdir(parents=True)
    for name, df in [("users", users), ("items", items), ("events", events)]:
        df.to_parquet(root / "db" / f"{name}.parquet", index=False)
    DatasetManifest(
        name="synth-shop",
        val_timestamp=VAL,
        test_timestamp=TEST,
        tables={
            "users": TableSpec(pkey="user_id", time_col="signup_date"),
            "items": TableSpec(pkey="item_id"),
            "events": TableSpec(
                pkey="event_id", time_col="ts", fkeys={"user_id": "users", "item_id": "items"}
            ),
        },
    ).save(root / "manifest.yaml")
    return root


@pytest.fixture(scope="session")
def shop(tmp_path_factory):
    return make_shop(tmp_path_factory.mktemp("src") / "synth-shop")


@pytest.fixture
def ws(shop, tmp_path, monkeypatch):
    monkeypatch.setattr(W, "WORKSPACE_ROOT", tmp_path / "workspace")
    return W.Workspace.create(str(shop))


def write_candidate(
    ws,
    name,
    sql,
    task_type="binary_classification",
    target_col="label",
    timedelta="30 days",
    **extra,
):
    d = ws.candidate(name)
    d.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "kind": "forecast",
        "task_type": task_type,
        "description": f"test task {name}",
        "entity_table": "users",
        "entity_col": "user_id",
        "target_col": target_col,
        "time_col": "date",
        "timedelta": timedelta,
        "sql": sql,
        "manifest_version": 1,
        **extra,
    }
    (d / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return d
