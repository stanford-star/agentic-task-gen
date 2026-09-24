"""A check: one isolated validation of a candidate task."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from taskgen.candidate import Candidate


@dataclass
class Outcome:
    issues: list[str] = field(default_factory=list)  # why the task fails the check; empty if it passes
    findings: dict = field(default_factory=dict)  # what the check measured, for the report


def violations(rules: dict[str, bool]) -> list[str]:
    """The messages of the violated rules."""
    return [message for message, violated in rules.items() if violated]


def first_line(error: Exception) -> str:
    return str(error).partition("\n")[0] or type(error).__name__


class Check(ABC):
    name: str  # starts each of its issues
    # exceptions that mean the task, not the harness, is at fault
    task_faults: tuple[type[Exception], ...] = (Exception,)

    def __call__(self, candidate: Candidate) -> Outcome:
        try:
            return self.run(candidate)
        except self.task_faults as error:
            return Outcome([first_line(error)])

    @abstractmethod
    def run(self, candidate: Candidate) -> Outcome: ...
