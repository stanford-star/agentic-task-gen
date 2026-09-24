# Generating tasks

You are generating **forecasting tasks** for a relational database in RelBench 3 format.
A task asks: *at a timestamp t, for each entity of some table, what happens in the
window (t, t + timedelta]?* Models may use only the data up to t. Its labels are defined by a
DuckDB query and stored as a RelBench task (binary classification or regression). Your goal
is a **diverse** set of valid, learnable, non-redundant tasks that a domain expert would find
meaningful.

You design and write the tasks. The harness (`taskgen/`) decides whether they are valid:
it checks each query for leakage, materializes the splits exactly as RelBench does, and
checks learnability (whether RDBLearn features + LightGBM, also given each entity's label
history, beat baselines that use no features: constants and that label history). You cannot
change the harness. A separate agent then reviews each kept task (step 3 of the loop).

## Setup (with the human)

1. Agree on the database (a Hub spec such as `rel-f1` or `org/repo/name`, or a local
   RelBench 3 folder) and the number of tasks to approve (default 2).
2. `pixi run taskgen init <dataset>` creates `workspace/<name>/`. The database's val/test
   cutoffs (in `workspace/<name>/dataset/manifest.yaml`) are shared by every task and
   cannot change later. If the dataset has none, look at the data with the human, choose
   them, and pass `--val` / `--test`.
3. Read `README.md` and `taskgen/config.py` (the thresholds you are judged by).

## Phase 1: understand the database

Explore the database however you like. The tables are plain parquet files in
`workspace/<name>/dataset/db/`, and `dataset/manifest.yaml` lists each table's primary key,
foreign keys and time column. Query them with DuckDB or pandas (`pixi run python`), and
keep any scripts you write in `workspace/<name>/scratch/`.

Write what you learn to `workspace/<name>/notes/understanding.md`. It should cover:

- **What it records.** The real-world process behind each table: who does what, when.
  For each table: entity, event, link or lookup; what one row means; what its time column
  means (creation, occurrence, schedule, last update?).
- **Entities and paths.** Which tables can be prediction entities (they have a primary
  key) and which event tables attach to them, directly or through multi-hop joins.
- **Time.** Span, granularity, seasonality and gaps; activity per entity (events per
  entity, gaps between events, share of entities seen only once, how many are active
  recently). These decide sensible horizons and cohorts.
- **Leaky columns.** Columns recorded after the fact or summarizing the future
  (lifetime totals, final statuses, timestamps later than the row's own time). Models see
  every column of past rows, so these can leak labels; list them.
- **Data quality.** Nulls, sentinel values, duplicated meanings.
- If names are anonymous (`table_3`, `feature_7`), say so and lean on statistics and
  structure.

## Phase 2: ideate

Write `notes/ideas.md`: a backlog, one line per idea: entity table · question · label ·
horizon · cohort · why someone would want it. Spread the ideas across:

- **entity tables**: every table with a primary key that has activity;
- **label sources**: direct child tables, and outcomes reached through multi-hop joins;
- **label families**: churn / inactivity; existence of a specific kind of event; counts;
  sums and means of a numeric column; rates and ratios; thresholds on aggregates; extremes;
  variety (distinct counts); novelty (a partner or category never seen before); repeat
  (same as last time); trend (next window vs previous); rank within the cohort at the
  timestamp; time until the next event; share of a total;
- **horizons**: short, medium and long relative to the typical gap between events;
- **cohorts**: recently active, all existing entities, new entities, dormant entities
  (reactivation), entities active in the window (for labels that only exist then);
- **label regimes**: balanced; rare events (1-10% positives); heavy-tailed and
  zero-inflated targets; cold start (little history per entity).

Order the backlog by usefulness × what it adds to the set, and revise it as you learn.

## Phase 3: the loop

Until the target number of tasks is approved:

1. Take the next idea. Write `workspace/<name>/candidates/<task>/manifest.yaml` (format
   below).
2. `pixi run taskgen evaluate <name> <task> > workspace/<name>/logs/<task>.log 2>&1`, then
   `grep -E "^(verdict|reasons|rows|label|lift|top_features):" workspace/<name>/logs/<task>.log`.
   The checks (listed in README.md) run in order and stop at the first that fails; each
   reason starts with the name of the check that failed, e.g. `label window: ...`. Manifest
   and leakage problems fail within seconds; only candidates that pass them reach the
   slower learnability check.
   The full report is in `candidates/<task>/report.json`. If an evaluation runs for more
   than 30 minutes, kill it and treat it as a discard.
3. **keep**: the task is now stored in `dataset/tasks/<task>/`. Have it reviewed by a new
   agent with a fresh context (for example a subagent), started with exactly this prompt and
   nothing else, with `<repo>`, `<name>` and `<task>` filled in:

   > Read `<repo>/review.md` and follow it, with `<name>` = `<name>` and `<task>` = `<task>`.

   The review must be independent of you, so:
   - add nothing to that prompt: no summary of the task, no intent, no answers to what the
     reviewer might ask, no expected verdict;
   - start a new reviewer for every review, including after a revision; never resume or
     message an earlier one;
   - read only its verdict, `workspace/<name>/reviews/<task>.json`, not its transcript, and
     never write to `reviews/` yourself;
   - do not revise the task to argue with a verdict: `revise` gets the suggested changes,
     `reject` is final.

   The verdict:
   - **approve**: the task counts toward the target.
   - **revise**: apply the suggestions to the candidate, evaluate again, and have the new
     version reviewed; this counts toward the 3 revisions per idea.
   - **reject**: delete `dataset/tasks/<task>/`, move the candidate to `rejected/`, and
     note why in `ideas.md`.
4. **discard**: read the reasons. If the problem is fixable and the idea still worth it
   (degenerate splits: a longer horizon or more eval timestamps; a leaking column:
   `remove_columns`; no signal: often a noisier restatement of a better task), revise and
   evaluate again, at most 3 times per idea. Otherwise move the candidate to
   `workspace/<name>/rejected/` and note why in `ideas.md`.
5. **error** means the harness failed, not your task. Record it in `notes/` and move on;
   do not work around it.
6. Every 5 evaluations, look at `results.tsv`, `dataset/tasks/` and `reviews/` and steer
   the next ideas toward what is under-represented.

Once the loop has started, do not stop to ask whether to continue: the human may be away.
Stop when the target is reached, or when two consecutive rounds of 5 fresh ideas yield no
approved task. Then delete any task in `dataset/tasks/` that is not approved, so that it holds
only approved tasks, and write `notes/summary.md`: what the database is, what was approved,
what was tried and why it failed, and what you would try next.

## Writing a task

A task is a RelBench 3 task manifest: one DuckDB query plus a few fields that describe its
output. `manifest.yaml` may hold only the keys in this example, for a shop database with
`users(user_id, signup_date)` and `events(user_id, ts, amount)`:

```yaml
name: users-churn-30d           # kebab-case, equal to the directory name
kind: forecast
task_type: binary_classification  # or regression
description: Whether a user who bought in the past 30 days makes no purchase in the next 30 days.
entity_table: users             # the table whose primary key identifies the entity
entity_col: user_id             # output column holding that key
target_col: churn               # output column holding the label
time_col: date                  # output column holding the timestamp
timedelta: 30 days              # window length in fixed units (days / hours), never months
num_eval_timestamps: 1          # number of val and test timestamps (see below)
remove_columns: []              # [[table, column], ...] hidden from models (see below)
sql: |-
  WITH cohort AS (
      SELECT DISTINCT t.timestamp AS ts, e.user_id
      FROM timestamps t
      JOIN events e ON e.ts > t.timestamp - INTERVAL '{timedelta}' AND e.ts <= t.timestamp
  )
  SELECT c.ts AS date, c.user_id,
         CAST(NOT EXISTS (
             SELECT 1 FROM events f
             WHERE f.user_id = c.user_id
               AND f.ts > c.ts AND f.ts <= c.ts + INTERVAL '{timedelta}'
         ) AS BIGINT) AS churn
  FROM cohort c
manifest_version: 1
```

**The shape.** A query takes a cohort of entities, computes each one's label from its rows
in the window after the timestamp, and returns one row per timestamp and entity:

- `timestamps(timestamp)` holds the timestamps, and every table is a view under its own name.
  `{timedelta}` becomes an interval string such as `30 days 00:00:00`: write
  `INTERVAL '{timedelta}'`, and `3 * INTERVAL '{timedelta}'` for multiples.
- The cohort usually comes from recent activity, as in the example: entities with rows in
  `(timestamp - k * timedelta, timestamp]`. A label that only exists when the entity acts in
  the window (a mean, the time to the next event, a finishing position) can take its cohort
  from the window itself: join the window's rows instead of left-joining them.
- The label reads the window, `x.ts > t.timestamp AND x.ts <= t.timestamp + INTERVAL
  '{timedelta}'`; rows before the timestamp may feed it too (trends, novelty).

**The rules** (`evaluate` checks them; the checks are listed in README.md):

- Output exactly `time_col` (the timestamp itself), `entity_col` (a primary key of the entity
  table) and `target_col` (BIGINT 0/1 for binary, a finite number for regression): one row
  per (timestamp, entity), no NULLs.
- Nothing after `timestamp + timedelta` may affect a label.
- RelBench computes all the timestamps of a split in one query, so partition every window
  function, rank or normalization by the timestamp.
- Label only entities that exist at the timestamp: if the entity table has a time column,
  keep it at most the timestamp.
- Quote identifiers that are SQL keywords (`"user"`, `"position"`, `"order"`, ...).

**Horizons and eval timestamps.** The val/test cutoffs belong to the database. Train
timestamps step back from the val cutoff every `timedelta`; val and test ones start at their
cutoffs. `evaluate` rejects a `timedelta` that does not fit: it must be at most test − val,
the history before the val cutoff must hold at least 3 train timestamps and at most
`MAX_TRAIN_TIMESTAMPS` (`taskgen/config.py`), and the data must extend one `timedelta` past
the test cutoff. With `num_eval_timestamps: 1`, val and test get one timestamp each: if
activity is seasonal or bursty, that window may be empty or degenerate, so use more eval
timestamps (at most (test − val) / timedelta).

**remove_columns.** RelBench runs the SQL on the task's view of the database,
without these columns, so you can only hide columns the label does not use, and never keys
or time columns; a pair naming no existing column is rejected. Use it for columns that leak
the label into features: denormalized future aggregates, final statuses.
When the learnability check reports a suspiciously accurate model, `top_features` usually
names the culprit.

## What you can and cannot do

You can:
- write anything in `workspace/<name>/` outside `dataset/` and `reviews/`: candidates,
  notes, scratch scripts, logs, rejected candidates;
- delete a kept task from `dataset/tasks/` if you decide it should not be there;
- read the data any way you like.

You cannot:
- edit anything outside `workspace/<name>/` (the harness, its tests and configuration, these
  instructions);
- write into `workspace/<name>/dataset/` by hand (only `taskgen evaluate` does);
- install packages;
- bend a task to get past a check it legitimately fails, e.g. dropping the entities that
  make a label degenerate. Fix the definition instead.

## Judgment

The harness catches mechanical problems; meaning is your responsibility.

- **Descriptions.** The description must state exactly what the SQL computes.
- **Usefulness.** Prefer predictions a practitioner would pay for (churn, demand, risk,
  engagement, quality) over arbitrary column aggregates.
- **Diversity over volume.** A new entity table or label family is worth more than a
  fifth variant of the same one. The novelty check only catches near-identical labels,
  so avoid near-restatements yourself.
- **Hard is fine.** Rare events, heavy tails and cold starts are kept as long as the model
  reliably beats the baselines, including the entity's own label history: a task whose
  label mostly repeats the entity's past labels is not worth keeping.
- **Simplicity.** Simpler SQL is better: fewer ways to be wrong, easier to review.
