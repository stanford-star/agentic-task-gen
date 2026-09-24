"""``taskgen evaluate``: runs the checks in order, stopping at the first that finds issues."""

from __future__ import annotations

import csv
import datetime as dt
import re
import time
import traceback

from taskgen.candidate import Candidate, to_json
from taskgen.checks import CHECKS
from taskgen.checks.base import first_line
from taskgen.checks.learnability import Learnability
from taskgen.checks.manifest import Manifest
from taskgen.checks.splits import Splits, SplitSizes
from taskgen.workspace import Workspace

# an error says nothing about the task, so it leaves a stored version alone
ON_VERDICT = {
    "keep": lambda candidate, report: candidate.store(report),
    "discard": lambda candidate, report: candidate.remove_stored(),
    "error": lambda candidate, report: None,
}


def run_checks(candidate: Candidate, report: dict) -> tuple[str, list[str]]:
    """The verdict and its reasons; each check's findings go into the report."""
    for check in CHECKS:
        try:
            outcome = check(candidate)
        except Exception as error:  # the harness failed, not the task
            report["traceback"] = traceback.format_exc()
            return "error", [f"{check.name}: {type(error).__name__}: {first_line(error)}"]
        if outcome.findings:
            report["findings"][check.name] = outcome.findings
        if outcome.issues:
            return "discard", [f"{check.name}: {issue}" for issue in outcome.issues]
    return "keep", []


def evaluate(ws: Workspace, name: str) -> dict:
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name):
        raise ValueError(f"task names are kebab-case, such as users-churn-30d, not {name!r}")
    candidate = Candidate(ws, name)
    if not (candidate.dir / "manifest.yaml").exists():
        raise FileNotFoundError(f"no {candidate.dir / 'manifest.yaml'}")
    now = dt.datetime.now().isoformat(timespec="seconds")
    report = {"task": name, "verdict": None, "reasons": [], "time": now, "findings": {}}
    start = time.time()
    report["verdict"], report["reasons"] = run_checks(candidate, report)
    report["seconds"] = round(time.time() - start, 1)
    ON_VERDICT[report["verdict"]](candidate, report)
    (candidate.dir / "report.json").write_text(to_json(report))
    append_result_row(ws, report)
    return report


def result_row(report: dict) -> dict:
    """The report's main fields: its row in results.tsv, and what `taskgen evaluate` prints."""
    findings = report["findings"]
    manifest, sizes = findings.get(Manifest.name, {}), findings.get(SplitSizes.name, {})
    split_rows = findings.get(Splits.name, {}).get("rows", {})
    learnability = findings.get(Learnability.name, {})
    train, scores = sizes.get("train"), learnability.get("scores", {})
    return {
        "time": report["time"],
        "task": report["task"],
        "verdict": report["verdict"],
        "task_type": manifest.get("task_type"),
        "timedelta": manifest.get("timedelta"),
        "rows": "/".join(str(rows) for rows in split_rows.values()) or None,
        "label": f"mean={train['mean']} mode={train['mode']} ({train['mode_share']:.0%})" if train else None,
        "metric": learnability.get("metric"),
        "model": scores.get("model"),
        "best_baseline": scores.get(learnability.get("best_baseline")),
        "lift": learnability.get("lift"),
        "lift_ci": learnability.get("lift_ci"),
        "top_features": ", ".join(learnability.get("top_features", [])) or None,
        "reasons": " | ".join(report["reasons"]) or None,
        "seconds": report["seconds"],
    }


def append_result_row(ws: Workspace, report: dict) -> None:
    row = {
        key: "" if value is None else str(value).replace("\t", " ").replace("\n", " ")
        for key, value in result_row(report).items()
    }
    is_new = not ws.results_tsv.exists()
    with open(ws.results_tsv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def summary(report: dict) -> str:
    fields = result_row(report).items()
    return "\n".join(f"{key + ':':<16}{value}" for key, value in fields if value is not None)
