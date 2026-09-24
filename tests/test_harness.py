"""Well-formed tasks are kept and load in RelBench; leaky or malformed ones are discarded."""

import re

import pytest
import relbench
import yaml

from taskgen import REPO_ROOT
from taskgen.evaluate import evaluate
from tests.conftest import write_candidate

SPEND = """
SELECT t.timestamp AS date, u.user_id, COALESCE(SUM(e.amount), 0) AS label
FROM timestamps t
JOIN users u ON u.signup_date <= t.timestamp
LEFT JOIN events e ON e.user_id = u.user_id
    AND e.ts > t.timestamp AND e.ts <= t.timestamp + INTERVAL '{timedelta}'
GROUP BY t.timestamp, u.user_id
"""

AVG_BASKET = """
SELECT t.timestamp AS date, e.user_id, AVG(e.amount) AS label
FROM timestamps t
JOIN users u ON u.signup_date <= t.timestamp
JOIN events e ON e.user_id = u.user_id
    AND e.ts > t.timestamp AND e.ts <= t.timestamp + INTERVAL '{timedelta}'
GROUP BY t.timestamp, e.user_id
"""

RANK_ACROSS_TIMES = f"SELECT date, user_id, percent_rank() OVER (ORDER BY label) AS label FROM ({SPEND})"


def reasons(report):
    return " | ".join(report["reasons"])


def test_generate_md_example_is_kept_and_loads(ws):
    section = (REPO_ROOT / "generate.md").read_text().split("## Writing a task", 1)[1]
    example = re.search(r"```yaml\n(.*?)```", section, flags=re.DOTALL).group(1)
    name = yaml.safe_load(example)["name"]
    ws.candidate(name).mkdir(parents=True)
    (ws.candidate(name) / "manifest.yaml").write_text(example)
    r = evaluate(ws, name)
    assert r["verdict"] == "keep", reasons(r)
    task = relbench.load_dataset(ws.dataset_dir).load_task(name)
    assert list(task.get_table("train").df.columns) == ["date", "user_id", "churn"]
    assert list(task.get_table("test").df.columns) == ["date", "user_id"]  # target masked


def test_regression_is_kept_and_duplicates_are_caught(ws):
    write_candidate(ws, "users-spend", SPEND, task_type="regression")
    assert evaluate(ws, "users-spend")["verdict"] == "keep"
    write_candidate(ws, "users-spend-x2", SPEND.replace("COALESCE", "2 * COALESCE"), task_type="regression")
    r = evaluate(ws, "users-spend-x2")
    assert r["verdict"] == "discard" and "duplicate of kept task 'users-spend'" in reasons(r)


@pytest.mark.parametrize(
    "sql, expected",
    [
        (
            SPEND.replace("<= t.timestamp + INTERVAL", "<= t.timestamp + 2 * INTERVAL"),
            "rows after the timestamp + timedelta are removed",
        ),
        (RANK_ACROSS_TIMES, "queried together"),
        (
            SPEND.replace("JOIN users u ON u.signup_date <= t.timestamp", "CROSS JOIN users u"),
            "created after their timestamp",
        ),
        (f"SELECT * FROM ({SPEND}) UNION ALL SELECT * FROM ({SPEND})", "duplicate (date, user_id)"),
    ],
    ids=["label-window", "cross-time", "future-entities", "duplicate-rows"],
)
def test_broken_task_is_discarded(ws, sql, expected):
    write_candidate(ws, "broken", sql, task_type="regression")
    r = evaluate(ws, "broken")
    assert r["verdict"] == "discard" and expected in reasons(r), reasons(r)
    assert not (ws.tasks_dir / "broken").exists()


def test_cohort_from_the_label_window_is_allowed(ws):
    # the cohort comes from the window itself; only learnability may reject it
    write_candidate(ws, "users-avg-basket", AVG_BASKET, task_type="regression")
    r = evaluate(ws, "users-avg-basket")
    assert r["verdict"] == "keep" or reasons(r).startswith("learnability:"), reasons(r)
