"""Readable proposal feedback with counted errors.

This is the history rule disclosed in arXiv:2607.24162v3 under the GSM8K
``Initial program, prompts, and history`` paragraph. It formats caller-owned
feedback; it does not alter the stored search history or its physical bills.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypedDict


def _quality(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be None or a finite score in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be None or a finite score in [0, 1]")
    return result


@dataclass(frozen=True)
class ProposalFeedback:
    """One feedback row before formatting for a later proposal.

    ``score=None`` means unevaluated. A zero score represents an evaluated
    candidate and is retained as its own row. Error strings are kept exactly,
    including whitespace, capitalization and line breaks. The input has no
    task, node, storage or administrative-ID dependencies.
    """

    score: float | None
    modification: str
    execution_errors: tuple[str, ...] = ()
    estimated_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _quality(self.score, "score"))
        object.__setattr__(self, "estimated_score", _quality(self.estimated_score, "estimated_score"))
        if not isinstance(self.modification, str):
            raise ValueError("modification must be a string")
        errors = tuple(self.execution_errors)
        if not all(isinstance(error, str) for error in errors):
            raise ValueError("execution_errors must contain exact error strings")
        object.__setattr__(self, "execution_errors", errors)


class CountedError(TypedDict):
    error: str
    count: int


class SerializedProposalFeedback(TypedDict):
    score: float | None
    modification: str
    estimated_score: float | None
    execution_failures: list[CountedError]
    occurrences: int


def serialize_proposal_history(
    history: Iterable[ProposalFeedback],
) -> list[SerializedProposalFeedback]:
    """Count exact repeated errors and consecutive unevaluated feedback.

    Distinct errors retain first-seen order within each row. Evaluated rows,
    including equal zero-score rows, always remain separate. Only consecutive
    unevaluated rows with identical formatted content share an ``occurrences``
    count. The returned dictionaries/lists are independently JSON-compatible.
    """
    rows: list[SerializedProposalFeedback] = []
    for feedback in history:
        if not isinstance(feedback, ProposalFeedback):
            raise TypeError("history must contain ProposalFeedback rows")
        counts: dict[str, int] = {}
        for error in feedback.execution_errors:
            counts[error] = counts.get(error, 0) + 1
        row: SerializedProposalFeedback = {
            "score": feedback.score,
            "modification": feedback.modification,
            "estimated_score": feedback.estimated_score,
            "execution_failures": [
                {"error": error, "count": count} for error, count in counts.items()
            ],
            "occurrences": 1,
        }
        if (
            row["score"] is None
            and rows
            and rows[-1]["score"] is None
            and rows[-1]["modification"] == row["modification"]
            and rows[-1]["estimated_score"] == row["estimated_score"]
            and rows[-1]["execution_failures"] == row["execution_failures"]
        ):
            rows[-1]["occurrences"] += 1
        else:
            rows.append(row)
    return rows
