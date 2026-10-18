"""Backend-neutral generated-program allocation from arXiv:2607.24162v3.

Each accepted edit receives one complete training-panel evaluation, without a
rollout. The caller supplies program generation, bounded repair, validity and
duplicate checks, execution, and a forecast of that entire feedback episode.
No task prompts, model runtime, datasets, or experiment results are included.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
import random
from typing import Any, Callable

from .scoring import CostAwareUCTScorer


def _nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class Proposal:
    """One proposal/repair episode, including every actual charged attempt.

    ``accepted=False`` means that the caller exhausted this episode's repair
    contract without a valid, distinct candidate. Its bill remains payable.
    """

    candidate: Any
    charged_cost: float
    accepted: bool = True
    failure: str = ""

    def __post_init__(self) -> None:
        _nonnegative(self.charged_cost, "proposal charged_cost")


@dataclass(frozen=True)
class Evaluation:
    """Actual panel bill and feedback, including failed execution attempts.

    ``complete=True`` attests that the whole fixed training panel was scored.
    Candidate execution errors can yield a complete, zero-score panel. If the
    panel never completes, set ``complete=False`` and ``score=None``; its cost
    is retained, but it cannot update tree statistics or a cost estimator.
    """

    score: float | None
    charged_cost: float
    complete: bool = True
    failure: str = ""

    def __post_init__(self) -> None:
        _nonnegative(self.charged_cost, "evaluation charged_cost")
        if self.complete:
            if self.score is None or not math.isfinite(self.score) or not 0 <= self.score <= 1:
                raise ValueError("a complete panel requires a finite score in [0, 1]")
        elif self.score is not None:
            raise ValueError("an incomplete panel must have score=None")


class ChargedFailure(Exception):
    """A callback could not finish, after paying the stated actual cost.

    This cost covers only the failing callback, so evaluation failure does not
    include the already returned proposal bill. Other exceptions propagate:
    the driver cannot invent a bill for an unreported backend failure.
    """

    def __init__(self, message: str, charged_cost: float):
        super().__init__(message)
        _nonnegative(charged_cost, "failure charged_cost")
        self.charged_cost = charged_cost


@dataclass(frozen=True)
class EditSearchConfig:
    c_ref: float
    strategy: str = "agentuct"
    exploration: float = 1.4
    beta: float = 2.0
    seed: int = 11

    def __post_init__(self) -> None:
        if self.strategy not in ("agentuct", "uct", "aflow_original"):
            raise ValueError("strategy must be agentuct, uct, or aflow_original")
        CostAwareUCTScorer(beta=self.beta, c_ref=self.c_ref)
        _nonnegative(self.exploration, "exploration")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


@dataclass
class EditNode:
    id: int
    parent: int | None
    artifact: Any
    own_score: float
    depth: int
    visits: int = 1
    reward_sum: float = 0.0
    children: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class EditAttempt:
    parent_id: int | None
    selection_path: tuple[int, ...]
    forecast: float | None
    proposal: Proposal | None
    evaluation: Evaluation | None

    @property
    def charged_cost(self) -> float:
        return math.fsum((self.proposal.charged_cost if self.proposal else 0,
                          self.evaluation.charged_cost if self.evaluation else 0))


@dataclass(frozen=True)
class EditSnapshot:
    """Independent tree state at one completed training-feedback boundary."""

    nodes: list[EditNode]
    total_cost: float
    attempt_count: int


@dataclass(frozen=True)
class EditSearchResult:
    nodes: list[EditNode]
    attempts: tuple[EditAttempt, ...]
    snapshots: tuple[EditSnapshot, ...]
    total_cost: float
    stop_reason: str

    def last_affordable_snapshot(self, budget: float) -> EditSnapshot | None:
        """Select after execution; crossing charges remain in ``total_cost``.

        The returned tree excludes later rewards, visits, and candidates. It is
        copied again so downstream selection cannot mutate the saved snapshot.
        No snapshot exists if even the initial complete panel is unaffordable.
        """
        _nonnegative(budget, "budget")
        affordable = [s for s in self.snapshots if s.total_cost <= budget]
        return deepcopy(affordable[-1]) if affordable else None


def progressive_width(visits: int) -> int:
    """The shared UCT/AgentUCT rule: min(8, max(1, ceil(sqrt(N))))."""
    if isinstance(visits, bool) or not isinstance(visits, int) or visits < 0:
        raise ValueError("visits must be a non-negative integer")
    return min(8, max(1, math.isqrt(visits) + (math.isqrt(visits) ** 2 < visits)))


def aflow_probabilities(scores: list[float]) -> tuple[float, ...]:
    """Original AFlow mixture: 0.3/K + 0.7 * softmax(20 * own_score)."""
    if not scores or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("scores must be nonempty and lie in [0, 1]")
    largest = max(scores)
    weights = [math.exp(20 * (s - largest)) for s in scores]
    total = math.fsum(weights)
    return tuple(0.3 / len(scores) + 0.7 * w / total for w in weights)


def _select_parent(
    nodes: list[EditNode], forecast: Callable[[Any], float],
    config: EditSearchConfig, rng: random.Random,
) -> tuple[EditNode | None, tuple[int, ...]]:
    if config.strategy == "aflow_original":
        pool = sorted((n for n in nodes if n.depth < 12 and len(n.children) < 8),
                      key=lambda n: (-n.own_score, n.id))[:4]
        if not pool:
            return None, ()
        probabilities = aflow_probabilities([n.own_score for n in pool])
        draw, cumulative = rng.random(), 0.0
        for parent, probability in zip(pool, probabilities):
            cumulative += probability
            if draw < cumulative:
                return parent, (parent.id,)
        return pool[-1], (pool[-1].id,)

    def can_expand(node):
        return node.depth < 12 and (
            len(node.children) < progressive_width(node.visits)
            or any(can_expand(nodes[i]) for i in node.children)
        )

    scorer = CostAwareUCTScorer(beta=config.beta if config.strategy == "agentuct" else 0,
                               c_ref=config.c_ref)
    parent, path = nodes[0], [0]
    while True:
        if parent.depth < 12 and len(parent.children) < progressive_width(parent.visits):
            return parent, tuple(path)
        pool = [nodes[i] for i in parent.children if can_expand(nodes[i])]
        if not pool:
            return None, tuple(path)
        parent = max(pool, key=lambda n: scorer.score(
            parent.visits, n.visits, n.reward_sum / n.visits,
            forecast(n.artifact), config.exploration))
        path.append(parent.id)


def run_edit_search(
    initial: Any, *, propose: Callable[[Any], Proposal],
    evaluate: Callable[[Any], Evaluation], cost_forecast: Callable[[Any], float],
    config: EditSearchConfig,
    max_evaluations: int = 256, max_proposals: int = 256,
    target_cost: float | None = None,
) -> EditSearchResult:
    """Run the three published allocators against one caller-owned system.

    ``propose(parent)`` includes bounded repairs, format/validity checks, and
    duplicate rejection. ``evaluate(candidate)`` returns one full-panel receipt.
    ``cost_forecast(parent)`` predicts proposal + repairs + complete feedback
    before generation; it never decides whether a call is affordable. A caller
    fitting a prequential model must commit observations only for completed
    training panels. Use the same callbacks and panel across compared methods.

    All costs use one caller-chosen unit; set ``c_ref`` in that unit. The v3
    generated-program study used beta=2 and c_ref=150954.5 raw tokens. These are
    study-specific calibration values, not universal token prices.

    ``target_cost`` stops only after a full evaluation settles. A rejected
    proposal episode may be retried within ``max_proposals``. An incomplete
    evaluation or ``ChargedFailure`` stops without a backup. All actual bills,
    including the complete crossing evaluation, remain in the result. Select
    reporting budgets retrospectively with ``last_affordable_snapshot``.
    """
    for value, name in ((max_evaluations, "max_evaluations"), (max_proposals, "max_proposals")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if target_cost is not None:
        _nonnegative(target_cost, "target_cost")
    nodes, attempts, snapshots = [], [], []
    total_cost, proposal_count = 0.0, 0
    rng = random.Random(config.seed)

    def forecast(artifact):
        value = cost_forecast(deepcopy(artifact))
        _nonnegative(value, "complete-edit forecast")
        return value

    def panel(artifact):
        try:
            result = evaluate(deepcopy(artifact))
        except ChargedFailure as failure:
            result = Evaluation(None, failure.charged_cost, False, str(failure))
        if not isinstance(result, Evaluation):
            raise TypeError("evaluate must return Evaluation")
        return result

    def record(parent_id, path, predicted, proposal, evaluation):
        nonlocal total_cost
        attempt = EditAttempt(parent_id, path, predicted, deepcopy(proposal), evaluation)
        attempts.append(attempt)
        total_cost = math.fsum((total_cost, attempt.charged_cost))

    def checkpoint():
        snapshots.append(EditSnapshot(deepcopy(nodes), total_cost, len(attempts)))

    initial_evaluation = panel(initial)
    record(None, (), None, None, initial_evaluation)
    if not initial_evaluation.complete:
        return EditSearchResult(nodes, tuple(attempts), (), total_cost, "incomplete_evaluation")
    nodes.append(EditNode(0, None, deepcopy(initial), initial_evaluation.score, 0,
                          reward_sum=initial_evaluation.score))
    checkpoint()
    stop_reason = "evaluation_limit"
    while len(nodes) < max_evaluations:
        # The preceding operation completed. There is no next-call reservation.
        if target_cost is not None and total_cost >= target_cost:
            stop_reason = "target_reached"
            break
        if proposal_count >= max_proposals:
            stop_reason = "proposal_limit"
            break
        parent, path = _select_parent(nodes, forecast, config, rng)
        if parent is None:
            stop_reason = "space_limit"
            break
        predicted = forecast(parent.artifact)
        proposal_failed = False
        while proposal_count < max_proposals:
            proposal_count += 1
            try:
                proposal = propose(deepcopy(parent.artifact))
            except ChargedFailure as failure:
                proposal = Proposal(None, failure.charged_cost, False, str(failure))
                proposal_failed = True
            if not isinstance(proposal, Proposal):
                raise TypeError("propose must return Proposal")
            if proposal.accepted:
                break
            record(parent.id, path, predicted, proposal, None)
            if proposal_failed:
                break
        if proposal_failed or not proposal.accepted:
            stop_reason = "incomplete_proposal" if proposal_failed else "proposal_limit"
            break
        evaluation = panel(proposal.candidate)
        record(parent.id, path, predicted, proposal, evaluation)
        if not evaluation.complete:
            stop_reason = "incomplete_evaluation"
            break
        score = evaluation.score
        node = EditNode(len(nodes), parent.id, deepcopy(proposal.candidate), score,
                        parent.depth + 1, reward_sum=score)
        nodes.append(node)
        parent.children.append(node.id)
        ancestor = parent
        while ancestor is not None:
            ancestor.visits += 1
            ancestor.reward_sum += score  # Raw quality, without a cost penalty.
            ancestor = nodes[ancestor.parent] if ancestor.parent is not None else None
        checkpoint()
    return EditSearchResult(nodes, tuple(attempts), tuple(snapshots), total_cost, stop_reason)
