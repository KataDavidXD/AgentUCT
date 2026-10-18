"""Complete-feedback cost forecasts for edits of generated programs.

This implements the prequential factors and edit-cost equations in the
``The formal edit-cost forecast`` paragraph of arXiv:2607.24162v3. It is a
receipt-data model: execution, storage, panel completion and actual billing
belong to the caller. No quality scores, model clients or held-out data are
needed to forecast a training edit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _characters(value: int, name: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class ParentQuestionCost:
    """The latest saved parent-training row for one question.

    ``input_characters`` counts the question and supplied context characters.
    ``actual_call_costs`` contains the actual bills of calls referenced by that
    row, including failed attempts and format retries. An empty tuple means no
    saved call receipts are available and selects the calibrated fallback.
    ``(0.0,)`` is an observed zero bill and stays zero. Restored rows with saved
    call references retain those calls' bills when pricing the parent cold.
    """

    query_id: str
    input_characters: int
    actual_call_costs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.query_id, "query_id")
        _characters(self.input_characters, "input_characters")
        object.__setattr__(
            self,
            "actual_call_costs",
            tuple(_nonnegative(cost, "actual call cost") for cost in self.actual_call_costs),
        )


@dataclass(frozen=True)
class EditCostObservation:
    """Receipt totals for one edit episode, with its completion boundary.

    ``request_size`` and ``parent_cold_cost`` are the values saved *before*
    generating that edit. First-call token counts and cost describe its first
    proposer call. ``proposal_cost`` includes all proposer and repair calls;
    ``execution_cost`` includes newly charged training execution, including
    failures and retries. A complete scored failure is still a complete panel.

    Only observations marked complete and training are fitted. Costs for other
    episodes must remain in the caller's bill even though they are not fitted.
    """

    observation_id: str
    request_size: int
    first_input_tokens: float
    first_output_tokens: float
    first_cost: float
    proposal_cost: float
    execution_cost: float
    parent_cold_cost: float
    complete: bool = True
    scope: Literal["training", "validation", "test"] = "training"

    def __post_init__(self) -> None:
        _identifier(self.observation_id, "observation_id")
        _characters(self.request_size, "request_size", positive=True)
        for name in (
            "first_input_tokens", "first_output_tokens", "first_cost",
            "proposal_cost", "execution_cost", "parent_cold_cost",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if not isinstance(self.complete, bool):
            raise ValueError("complete must be a boolean")
        if self.scope not in ("training", "validation", "test"):
            raise ValueError("scope must be training, validation or test")


@dataclass(frozen=True)
class QuestionCostForecast:
    query_id: str
    parent_cold_cost: float
    predicted_execution_cost: float
    uses_fallback: bool


@dataclass(frozen=True)
class EditCostForecast:
    """Pre-decision estimate, distinct from the edit's eventual actual bill."""

    request_size: int
    input_tokens_per_character: float
    first_output_tokens: float
    proposal_attempt_multiplier: float
    execution_multiplier: float
    first_proposal_cost: float
    proposal_cost: float
    parent_cold_cost: float
    execution_cost: float
    full_completion_cost: float
    training_edits: int
    question_costs: tuple[QuestionCostForecast, ...]


class PrequentialEditCostModel:
    """Forecast proposal/repair plus complete training execution.

    Both token-cost priors must be supplied by the caller's independent cost
    calibration. The v3 GSM8K study used ``prior_proposal_tokens=1768`` and
    ``prior_execution_call_tokens=531``; these are study parameters, not
    universally calibrated defaults. The first-output prior is half the total
    proposer-call prior. The input/character prior is 0.25.

    Factors are arithmetic means over complete training edits. They have no
    imposed upper cap or confidence radius. Cache reuse can lower execution's
    physical/parent-cold ratio; added calls and retries can raise it above one.
    The forecast never decides whether the caller may perform an evaluation.
    """

    def __init__(
        self,
        *,
        prior_proposal_tokens: float,
        prior_execution_call_tokens: float,
        prior_input_tokens_per_character: float = 0.25,
    ) -> None:
        self.prior_proposal_tokens = _nonnegative(prior_proposal_tokens, "proposal prior")
        self.prior_execution_call_tokens = _nonnegative(
            prior_execution_call_tokens, "execution-call prior"
        )
        self.prior_input_tokens_per_character = _nonnegative(
            prior_input_tokens_per_character, "input/character prior"
        )
        self._observations: dict[str, EditCostObservation] = {}

    @property
    def training_edits(self) -> int:
        return len(self._observations)

    def observe(self, observation: EditCostObservation) -> bool:
        """Commit a complete training episode once; return whether it was added.

        Incomplete and held-out episodes are ignored. The caller controls when
        a full training panel has completed and commits only observations in
        its retained search history. Repeating an identical observation ID is
        harmless; conflicting receipt totals for an existing ID are rejected.
        """
        if not isinstance(observation, EditCostObservation):
            raise TypeError("an EditCostObservation is required")
        if not observation.complete or observation.scope != "training":
            return False
        previous = self._observations.get(observation.observation_id)
        if previous is not None:
            if previous != observation:
                raise ValueError("conflicting complete training observation")
            return False
        self._observations[observation.observation_id] = observation
        return True

    def forecast(
        self,
        *,
        graph: str,
        prompt: str,
        operator_description: str,
        panel: Sequence[ParentQuestionCost],
    ) -> EditCostForecast:
        """Read current history and latest parent rows without changing either.

        Request size is ``len(graph) + len(prompt) + len(description) + 1800``.
        A question without saved call receipts uses
        ``0.25 * input_characters + prior_execution_call_tokens`` under the
        paper's default input/character prior. No new edit is generated here.
        """
        if not all(isinstance(value, str) for value in (graph, prompt, operator_description)):
            raise TypeError("graph, prompt and operator_description must be strings")
        questions = tuple(panel)
        if not questions or not all(isinstance(row, ParentQuestionCost) for row in questions):
            raise ValueError("a nonempty panel of ParentQuestionCost rows is required")
        if len({row.query_id for row in questions}) != len(questions):
            raise ValueError("panel query IDs must be unique")

        history = tuple(self._observations.values())
        if history:
            count = len(history)
            ratio = math.fsum(row.first_input_tokens / row.request_size for row in history) / count
            output = math.fsum(row.first_output_tokens for row in history) / count
            attempts = math.fsum(row.proposal_cost / max(row.first_cost, 1.0) for row in history) / count
            execution_multiplier = math.fsum(
                row.execution_cost / max(row.parent_cold_cost, 1.0) for row in history
            ) / count
        else:
            ratio = self.prior_input_tokens_per_character
            output = self.prior_proposal_tokens / 2.0
            attempts = execution_multiplier = 1.0

        request_size = len(graph) + len(prompt) + len(operator_description) + 1800
        first = _nonnegative(max(1.0, request_size * ratio + output), "first-call forecast")
        rows = []
        for question in questions:
            fallback = not question.actual_call_costs
            cold = (
                self.prior_input_tokens_per_character * question.input_characters
                + self.prior_execution_call_tokens
                if fallback else math.fsum(question.actual_call_costs)
            )
            cold = _nonnegative(cold, "parent cold cost")
            rows.append(QuestionCostForecast(
                question.query_id, cold,
                _nonnegative(cold * execution_multiplier, "question execution forecast"),
                fallback,
            ))
        parent_cold = _nonnegative(math.fsum(row.parent_cold_cost for row in rows), "panel parent cold cost")
        execution = _nonnegative(math.fsum(row.predicted_execution_cost for row in rows), "execution forecast")
        proposal = _nonnegative(first * attempts, "proposal/repair forecast")
        full = _nonnegative(proposal + execution, "complete-feedback forecast")
        return EditCostForecast(
            request_size, ratio, output, attempts, execution_multiplier,
            first, proposal, parent_cold, execution, full,
            len(history), tuple(rows),
        )
