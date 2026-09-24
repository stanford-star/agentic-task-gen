"""The checks a candidate task must pass, in order: cheap ones first, the model last."""

from taskgen.checks.base import Check
from taskgen.checks.leakage import LabelWindow, TimestampIndependence
from taskgen.checks.learnability import Learnability
from taskgen.checks.manifest import Manifest
from taskgen.checks.novelty import Novelty
from taskgen.checks.splits import Output, Splits, SplitSizes

CHECKS: tuple[Check, ...] = (
    Manifest(),
    Splits(),
    Output(),
    SplitSizes(),
    LabelWindow(),
    TimestampIndependence(),
    Novelty(),
    Learnability(),
)
