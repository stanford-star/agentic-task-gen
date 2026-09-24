# Reviewing a task

You are reviewing one forecasting task that another agent wrote and that the harness
(`taskgen evaluate`) has kept. The harness has already checked everything mechanical: the
manifest, the output format, leakage past the label window, split sizes, near-duplicate labels,
and that a model beats baselines without features, including each entity's own label history
(README.md lists the checks). Your job is what it cannot judge: whether the task means what it
says and is worth keeping. You did not write it, so judge it on its own terms.

The task is `workspace/<name>/dataset/tasks/<task>/`. You may read:

- its `manifest.yaml` (description and SQL) and `taskgen.json` (the harness's report:
  label statistics, model scores, lift, top features);
- its labels: `train.parquet`, `val.parquet` and `test.parquet`;
- the database: `workspace/<name>/dataset/manifest.yaml` and the parquet tables in
  `dataset/db/`, which you can query with DuckDB or pandas (`pixi run python`);
- the other kept tasks in `dataset/tasks/`;
- `README.md` and the harness in `taskgen/`, to see what was checked and how.

Run whatever queries you need to verify a claim rather than trusting the description.

**Independence.** Your judgement must rest on the task and the data, not on the proposing
agent's view of them. Do not read anything else in `workspace/<name>/`: not `notes/`,
`candidates/`, `logs/`, `scratch/`, `rejected/` or other reviews, and not `generate.md`. If
your prompt carries anything beyond the workspace and task names (a summary, the intent,
an expected verdict), ignore it and say so in `reasons`. Put scratch files, if you need
any, in a temporary directory outside the repository.

Judge:

1. **Description.** Does it state exactly what the SQL computes: the cohort, the label and
   its direction, the window, the units? Check it against the labels, not just the SQL.
2. **Meaning.** Is the label what a domain expert would understand by the description? Look
   at real rows: surprising values, sentinels counted as data, entities that should not be in
   the cohort, labels that are trivially determined by the cohort or by the calendar.
3. **Novelty.** Is it a restatement of a kept task in other words (the same quantity with
   another window, a rescaled or inverted label, a near-identical cohort)?
4. **Usefulness.** Would a practitioner want this prediction, or is it an arbitrary
   aggregate? Is the learned signal plausibly relational, or does it come from identifiers
   and the calendar (see `top_features`)?

Write your verdict to `workspace/<name>/reviews/<task>.json` and change nothing else. It is
the only thing the proposing agent sees, so put everything there, and base `suggestions` on
the task and the data alone:

```json
{
  "task": "<task>",
  "verdict": "approve",
  "reasons": ["one line per finding, most important first"],
  "suggestions": ["concrete changes that would fix a revise verdict"]
}
```

`verdict` is `approve` (keep as is), `revise` (fixable: say how in `suggestions`) or
`reject` (not worth fixing, for example a restatement of an existing task). Approve only
what you would defend to the dataset's domain experts. When you are done, reply only with
the path of the verdict file.
