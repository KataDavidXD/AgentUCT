"""Protocol and ABC definitions for the UCT search engine.

Defines the extension points: SearchState, Evaluator, CostModel, ScoreFunction,
and the data structures that flow between them.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Hashable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# SearchState protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SearchState(Protocol):
    """A partially selected configuration in the search tree.

    Users implement this for their domain (e.g. RAG pipeline slots).
    The engine never assumes what the state contains internally.
    """

    def is_terminal(self) -> bool:
        """True when all slots are filled (complete configuration)."""
        ...

    def available_actions(self) -> list[Hashable]:
        """Actions that extend this partial config by one slot."""
        ...

    def child(self, action: Hashable) -> "SearchState":
        """Return a new state with *action* appended."""
        ...

    def state_key(self) -> Hashable:
        """Content-addressable key uniquely identifying this state."""
        ...

    def pretty(self) -> str:
        """Human-readable representation."""
        ...


# ---------------------------------------------------------------------------
# Evaluation data structures (benchmark-cluster oriented)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkClusterResult:
    """Result from evaluating one configuration on one benchmark cluster.

    Maps to the paper's per-cluster reward R_hat(x, z) and the path
    prefixes materialized during that evaluation.
    """
    cluster_id: str
    reward: float
    cost: float = 0.0
    materialized_keys: list[Hashable] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Aggregated result from evaluating across a cluster minibatch.

    *reward* is the weighted aggregate J_hat(x).
    *cluster_results* carries per-cluster detail so the engine can
    harvest materialized_keys for the reuse graph.
    """
    reward: float
    total_cost: float = 0.0
    cluster_results: list[BenchmarkClusterResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SearchContext -- mutable state carried through one search() call
# ---------------------------------------------------------------------------

@dataclass
class SearchContext:
    """Mutable bag of search-session state.

    *materialized_keys* maps directly to the paper's Path_t -- the set
    of (path-prefix, cluster) pairs already cached.
    """
    materialized_keys: set[Hashable] = field(default_factory=set)
    total_cost: float = 0.0
    total_evaluations: int = 0
    random: random.Random = field(default_factory=lambda: random.Random(0))
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SearchResult -- returned by engine.search()
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Outcome of a completed UCT search run."""
    best_state: Any  # SearchState
    best_reward: float
    root_node: Any  # TreeNode
    iterations: int
    total_evaluations: int
    total_cost: float
    context: SearchContext


# ---------------------------------------------------------------------------
# Abstract base classes for pluggable components
# ---------------------------------------------------------------------------

class Evaluator(ABC):
    """Evaluates a terminal (or rolled-out) configuration.

    Implementations are responsible for:
      1. Optionally sampling a benchmark cluster minibatch.
      2. Running evaluation per cluster.
      3. Returning BenchmarkClusterResult objects inside EvaluationResult.
      4. Populating materialized_keys so the engine can update Path_t.
    """

    @abstractmethod
    def evaluate(self, state: SearchState, context: SearchContext) -> EvaluationResult:
        ...


class CostModel(ABC):
    """Estimates the marginal cost of expanding *state* by *action*.

    The cost may depend on the current search history via *context*.
    """

    @abstractmethod
    def marginal_cost(
        self,
        state: SearchState,
        action: Hashable,
        context: SearchContext,
    ) -> float:
        ...


class ScoreFunction(ABC):
    """Computes the selection score for a child node during tree descent.

    Receives pre-computed numeric stats so implementations stay pure
    functions on numbers.
    """

    @abstractmethod
    def score(
        self,
        parent_visits: int,
        child_visits: int,
        child_q: float,
        marginal_cost: float,
        exploration_constant: float,
    ) -> float:
        ...
