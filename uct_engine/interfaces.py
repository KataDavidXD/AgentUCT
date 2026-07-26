"""Public contracts for the standalone Agent-UCT search core.

The core knows about configuration states, an optional per-iteration benchmark
batch, scalar rewards/costs, and materialized reuse keys. Dataset loading,
adaptive stopping rules, workflow execution, and physical cache backends belong
to the experiment integration layer.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Hashable, Protocol, runtime_checkable


@runtime_checkable
class SearchState(Protocol):
    """A partial or complete configuration in the search tree."""

    def is_terminal(self) -> bool:
        """Return whether the configuration is complete."""
        ...

    def available_actions(self) -> list[Hashable]:
        """Return actions that extend this configuration by one level."""
        ...

    def child(self, action: Hashable) -> "SearchState":
        """Return the child configuration selected by ``action``."""
        ...

    def state_key(self) -> Hashable:
        """Return a stable, hashable key for this configuration prefix."""
        ...

    def pretty(self) -> str:
        """Return a human-readable representation."""
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
    ) -> EvaluationResult:
        ...


class CostModel(ABC):
    """Estimate the history-dependent marginal cost of one child prefix."""

    @abstractmethod
    def marginal_cost(
        self,
        state: SearchState,
        action: Hashable,
        context: SearchContext,
    ) -> float:
        ...


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
    ) -> float:
        ...
