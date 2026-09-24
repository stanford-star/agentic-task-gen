"""taskgen init <dataset> | taskgen evaluate <workspace> <task>"""

import argparse
import os
import sys
import warnings

# Silence the featurizer stack's progress bars, warnings and INFO logs before it is imported.
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

from taskgen.evaluate import evaluate, summary  # noqa: E402
from taskgen.workspace import Workspace  # noqa: E402


def run_init(args) -> str:
    ws = Workspace.create(args.dataset, name=args.name, val_cutoff=args.val, test_cutoff=args.test)
    return (
        f"created {ws.root}\ntables: {ws.db_dir}\n"
        f"keys, time columns, cutoffs: {ws.dataset_dir / 'manifest.yaml'}"
    )


def run_evaluate(args) -> str:
    return summary(evaluate(Workspace.open(args.workspace), args.task))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="taskgen", description=__doc__)
    commands = parser.add_subparsers(required=True)
    init = commands.add_parser("init", help="create workspace/<name> for a RelBench 3 dataset")
    init.add_argument("dataset", help="Hub spec (e.g. rel-f1) or local dataset folder")
    init.add_argument("--name", help="workspace name (default: the dataset's name)")
    init.add_argument("--val", help="val cutoff (required if the dataset has none)")
    init.add_argument("--test", help="test cutoff (required if the dataset has none)")
    init.set_defaults(run=run_init)
    evaluate_ = commands.add_parser("evaluate", help="evaluate candidates/<task>; store it if kept")
    evaluate_.add_argument("workspace")
    evaluate_.add_argument("task")
    evaluate_.set_defaults(run=run_evaluate)
    args = parser.parse_args(argv)
    try:
        print(args.run(args))
    except (OSError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
