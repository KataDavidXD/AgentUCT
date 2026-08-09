"""Public contracts for the standalone Agent-UCT search core.

The core knows about configuration states, an optional per-iteration benchmark
batch, scalar rewards/costs, and materialized reuse keys. Dataset loading,
adaptive stopping rules, workflow execution, and physical cache backends belong
to the experiment integration layer.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SearchState(Protocol):
    """A partial or complete configuration in the search tree."""

    def is_terminal(self) -> bool:
        """Return whether the configuration is complete."""
        ...

    def available_actions(self) -> list[Hashable]:
        """Return actions that extend this configuration by one level."""
        ...

    def child(self, action: Hashable) -> SearchState:
        """Return the child configuration selected by ``action``."""
        ...

    def state_key(self) -> Hashable:
        """Return a stable, hashable key for this configuration prefix."""
        ...

    def pretty(self) -> str:
        """Return a human-readable representation."""
        ...


@runtime_checkable
class ReuseChildKeyProvider(Protocol):
    """Optional, side-effect-free preview of one child prefix key.

    Environment-backed states should implement this hook when ``child`` may
    execute work.  Cost estimation can then inspect reuse without performing
    the real transition.  Legacy states remain valid and fall back to
    ``state.child(action).state_key()``.
    """

    def reuse_child_key(self, action: Hashable) -> Hashable:
        """Return the stable key of ``action``'s child without executing it."""
        ...


@dataclass(frozen=True)
class ClusterDef:
    """Benchmark-cluster weight and default one-step execution cost."""

    cluster_id: str
    weight: float = 1.0
    base_cost: float = 1.0

    def __post_init__(self) -> None:
        if not self.cluster_id:
            raise ValueError("cluster_id must be non-empty")
        if self.weight < 0:
            raise ValueError("cluster weight must be non-negative")
        if self.base_cost < 0:
            raise ValueError("cluster base_cost must be non-negative")


@dataclass(frozen=True)
class ClusterBatch:
    """Canonical item identifiers sampled from one benchmark cluster."""

    cluster_id: str
    item_ids: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        if not self.cluster_id:
            raise ValueError("cluster_id must be non-empty")
        if not self.item_ids:
            raise ValueError("a cluster batch must contain at least one item")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("item_ids must be unique within a cluster batch")


@dataclass(frozen=True)
class EvaluationBatch:
    """The paper-defined initial batch ``B_t`` for one Agent-UCT iteration."""

    clusters: tuple[ClusterBatch, ...]

    def __post_init__(self) -> None:
        cluster_ids = [batch.cluster_id for batch in self.clusters]
        if not cluster_ids:
            raise ValueError("an evaluation batch must contain at least one cluster")
        if len(set(cluster_ids)) != len(cluster_ids):
            raise ValueError("cluster_ids must be unique within an evaluation batch")

    def for_cluster(self, cluster_id: str) -> ClusterBatch | None:
        """Return the batch for ``cluster_id`` if it is active this iteration."""
        return next(
            (batch for batch in self.clusters if batch.cluster_id == cluster_id),
            None,
        )


@dataclass
class EvaluationResult:
    """Scalar search outcome and the reuse keys materialized by evaluation."""

    reward: float
    total_cost: float = 0.0
    materialized_keys: set[Hashable] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchContext:
    """Mutable state shared by one search run.

    ``current_batch`` is drawn exactly once at the start of an iteration. The
    cost model and evaluator observe the same object. ``materialized_keys`` is
    the logical view of the execution substrate's reuse graph.
    """

    materialized_keys: set[Hashable] = field(default_factory=set)
    current_batch: EvaluationBatch | None = None
    total_cost: float = 0.0
    total_evaluations: int = 0
    random: random.Random = field(default_factory=lambda: random.Random(0))
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionResult:
    """One realized environment transition and its non-reward effects.

    ``realized_cost`` and ``materialized_keys`` are accumulated across the
    expansion and rollout of one search iteration. On success, the engine
    merges them with the terminal evaluator result at one boundary. If the
    evaluator fails, already-realized transition effects are retained only for
    recovery and no reward is backed up.
    """

    state: SearchState
    realized_cost: float = 0.0
    materialized_keys: set[Hashable] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state, SearchState):
            raise TypeError("transition state must implement SearchState")
        if isinstance(self.realized_cost, bool) or not isinstance(
            self.realized_cost, (int, float)
        ):
            raise TypeError("transition realized_cost must be numeric")
        cost = float(self.realized_cost)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("transition realized_cost must be finite and non-negative")
        self.realized_cost = cost
        self.materialized_keys = set(self.materialized_keys)
        self.metadata = dict(self.metadata)


@dataclass
class SearchResult:
    """Outcome of a completed Agent-UCT search."""

    best_state: SearchState | None
    best_reward: float
    root_node: Any
    iterations: int
    total_evaluations: int
    total_cost: float
    context: SearchContext


class BatchProvider(ABC):
    """Draw the initial benchmark batch before tree selection."""

    @abstractmethod
    def draw(self, iteration: int, context: SearchContext) -> EvaluationBatch:
        """Return the single batch used by cost estimation and evaluation."""
        ...


class TransitionModel(ABC):
    """Execute one search action in a real or simulated environment.

    Implementations observe ``SearchContext`` but must report accounting
    effects in ``TransitionResult`` rather than mutating the context's cost,
    reuse keys, or evaluation counter themselves.
    """

    @abstractmethod
    def transition(
        self,
        state: SearchState,
        action: Hashable,
        context: SearchContext,
    ) -> TransitionResult:
        """Return the successor plus realized non-reward effects."""
        ...


class ChildTransitionModel(TransitionModel):
    """Backward-compatible transition using ``SearchState.child`` only."""

    def transition(
        self,
        state: SearchState,
        action: Hashable,
        context: SearchContext,
    ) -> TransitionResult:
        del context
        return TransitionResult(state=state.child(action))


class Evaluator(ABC):
    """Evaluate a completed configuration.

    When ``context.current_batch`` is present, implementations must evaluate
    that initial batch rather than draw a replacement. They may extend it with
    additional micro-batches according to an integration-owned stopping rule.
    Only keys physically or logically materialized by the evaluation should be
    returned in ``EvaluationResult.materialized_keys``.
    """

    @abstractmethod
    def evaluate(
        self,
        state: SearchState,
        context: SearchContext,
    ) -> EvaluationResult: ...


class CostModel(ABC):
    """Estimate the history-dependent marginal cost of one child prefix."""

    @abstractmethod
    def marginal_cost(
        self,
        state: SearchState,
        action: Hashable,
        context: SearchContext,
    ) -> float: ...


class ScoreFunction(ABC):
    """Compute a tree-selection score for one child."""

    @abstractmethod
    def score(
        self,
        parent_visits: int,
        child_visits: int,
        child_q: float,
        marginal_cost: float,
        exploration_constant: float,
        iteration: int = 1,
    ) -> float: ...
