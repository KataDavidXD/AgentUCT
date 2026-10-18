"""Actual receipts and complete reporting snapshots; no cost admission control."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Generic, Hashable, Iterable, TypeVar

State = TypeVar("State")
SEARCH_PHASES = frozenset({"search", "proposer", "repair"})
PHASES = SEARCH_PHASES | {"validation", "common_preparation", "test_measurement"}


def _cost(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("actual cost must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class ActualReceipt:
    receipt_id: Hashable
    actual_cost: float
    phase: str
    failed: bool = False


class ActualCostLedger:
    """Record already incurred costs, including charged failures and retries.

    Every physical call needs a unique receipt ID. A restored object must not
    be billed again. This ledger accepts actual receipts after execution and
    does not reserve a predicted bill or reject a call against a budget.
    """

    def __init__(self, unit: str):
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError("a nonempty declared cost unit is required")
        self.unit = unit
        self._receipts: dict[Hashable, ActualReceipt] = {}

    def record(self, receipt_id: Hashable, actual_cost: float, *,
               phase: str = "search", failed: bool = False) -> None:
        if phase not in PHASES:
            raise ValueError("unknown physical cost phase")
        receipt = ActualReceipt(receipt_id, _cost(actual_cost), phase, bool(failed))
        previous = self._receipts.get(receipt_id)
        if previous is not None and previous != receipt:
            raise ValueError("conflicting receipt ID")
        self._receipts[receipt_id] = receipt

    @property
    def receipts(self) -> tuple[ActualReceipt, ...]:
        return tuple(self._receipts.values())

    def phase_cost(self, phase: str) -> float:
        if phase not in PHASES:
            raise ValueError("unknown physical cost phase")
        return math.fsum(r.actual_cost for r in self._receipts.values() if r.phase == phase)

    @property
    def search_cost(self) -> float:
        return math.fsum(r.actual_cost for r in self._receipts.values()
                         if r.phase in SEARCH_PHASES)

    @property
    def optimization_cost(self) -> float:
        return self.search_cost + self.phase_cost("validation")


@dataclass(frozen=True)
class CompletedSnapshot(Generic[State]):
    """Independent state captured only at a complete-feedback boundary.

    The caller must include the full tree, execution cache, model observations,
    candidates and relevant RNG state when those affect its continuation.
    A partial evaluation or a failure-recovery checkpoint is not this object.
    """

    actual_search_cost: float
    completed_evaluations: int
    state: State

    def __post_init__(self):
        object.__setattr__(self, "actual_search_cost", _cost(self.actual_search_cost))
        count = self.completed_evaluations
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("completed_evaluations must be a nonnegative integer")
        object.__setattr__(self, "state", deepcopy(self.state))

    @classmethod
    def capture(cls, state: State, *, actual_search_cost: float,
                completed_evaluations: int) -> CompletedSnapshot[State]:
        return cls(actual_search_cost, completed_evaluations, state)


def last_affordable_snapshot(
    snapshots: Iterable[CompletedSnapshot[State]], budget: float,
) -> CompletedSnapshot[State] | None:
    """Copy the last complete state within an actual-search reporting budget.

    Crossing evaluations and their physical receipts remain in the caller's
    raw history. Neither their rewards nor their later cache enter this copy.
    No predicted price or anticipated reward participates in the cutoff.
    """
    budget = _cost(budget)
    retained = None
    previous_cost, previous_count = -1.0, -1
    for snapshot in snapshots:
        if not isinstance(snapshot, CompletedSnapshot):
            raise TypeError("only completed snapshots are eligible")
        if (snapshot.actual_search_cost < previous_cost or
                snapshot.completed_evaluations < previous_count):
            raise ValueError("snapshots must be in chronological cost/count order")
        previous_cost = snapshot.actual_search_cost
        previous_count = snapshot.completed_evaluations
        if snapshot.actual_search_cost <= budget:
            retained = snapshot
    return deepcopy(retained)
