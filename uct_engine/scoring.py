"""Agent-UCT selection score and reuse-aware marginal-cost model."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from enum import Enum

from .interfaces import (
    ClusterDef,
    CostModel,
    ScoreFunction,
    SearchContext,
    SearchState,
)

ItemCostFunction = Callable[
    [SearchState, Hashable, str, Hashable | None],
    float,
]


class CostPenaltySchedule(str, Enum):
    """Supported schedules for the reuse-cost coefficient."""

    CONSTANT = "constant"
    INVERSE_SQRT = "inverse_sqrt"


@dataclass(frozen=True)
class UCTScoreTerms:
    """Exact decomposition of one standard Agent-UCT selection score.

    ``uct_score`` is the cost-free UCT-only ablation signal.  The final
    ``selection_score`` subtracts the decayed cost penalty.  None of these
    node-local search-policy terms is a trajectory advantage.
    """

    child_q_raw: float
    exploration_bonus: float
    marginal_cost: float
    normalized_marginal_cost: float
    cost_scale: float
    lambda_t: float
    selection_cost_coefficient: float
    decayed_cost_penalty: float
    uct_score: float
    selection_score: float
    cost_schedule: str = CostPenaltySchedule.INVERSE_SQRT.value


class CostAwareUCTScorer(ScoreFunction):
    r"""Compute a cost-aware Agent-UCT tree-selection score.

    .. math::

        \widehat Q_t(s,a)
        + c_{\mathrm{UCT}}
          \sqrt{\frac{\log(N_t(s)+1)}{N_t(s,a)+1}}
        - \beta_t\frac{\widehat{\Delta C}_t(s,a)}{C_{\mathrm{scale}}}

    ``cost_scale`` is one positive global denominator.  It is never recomputed
    from the remaining candidates at a node, so costs stay comparable across
    depths, iterations, and cache states.  The default ``1`` preserves the
    historical raw-cost behavior.

    The default schedule preserves the original
    :math:`\beta_t=\lambda_0/\sqrt{t}` behavior.  ``schedule="constant"`` uses
    :math:`\beta_t=\lambda_0`, while ``lambda_0=0`` is vanilla UCT under either
    schedule.

    ``cost_aware_selection`` and ``cost_aware_expansion`` independently choose
    where the same scheduled, globally normalized coefficient is active.
    Expansion remains opt-in for backward compatibility.  Rewards stored in
    tree statistics and the final best configuration are never cost-adjusted.
    """

    def __init__(
        self,
        lambda_0: float = 0.0,
        *,
        schedule: CostPenaltySchedule | str = CostPenaltySchedule.INVERSE_SQRT,
        cost_aware_expansion: bool = False,
        cost_aware_selection: bool = True,
        cost_scale: float = 1.0,
    ) -> None:
        if not math.isfinite(lambda_0) or lambda_0 < 0:
            raise ValueError("lambda_0 must be finite and non-negative")
        if not isinstance(cost_aware_expansion, bool):
            raise TypeError("cost_aware_expansion must be boolean")
        if not isinstance(cost_aware_selection, bool):
            raise TypeError("cost_aware_selection must be boolean")
        if not math.isfinite(cost_scale) or cost_scale <= 0:
            raise ValueError("cost_scale must be finite and strictly positive")
        self.lambda_0 = float(lambda_0)
        self.cost_aware_expansion = cost_aware_expansion
        self.cost_aware_selection = cost_aware_selection
        self.cost_scale = float(cost_scale)
        try:
            self.schedule = CostPenaltySchedule(schedule)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(item.value for item in CostPenaltySchedule)
            raise ValueError(f"schedule must be one of: {allowed}") from exc

    def score(
        self,
        parent_visits: int,
        child_visits: int,
        child_q: float,
        marginal_cost: float,
        exploration_constant: float,
        iteration: int = 1,
    ) -> float:
        """Return the final Agent-UCT selection score."""
        return self.decompose(
            parent_visits=parent_visits,
            child_visits=child_visits,
            child_q=child_q,
            marginal_cost=marginal_cost,
            exploration_constant=exploration_constant,
            iteration=iteration,
        ).selection_score

    def _scheduled_cost_coefficient(self, iteration: int) -> float:
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration < 1
        ):
            raise ValueError("iteration must be an integer of at least 1")
        return (
            self.lambda_0
            if self.schedule is CostPenaltySchedule.CONSTANT
            else self.lambda_0 / math.sqrt(iteration)
        )

    def expansion_cost_coefficient(self, iteration: int) -> float:
        """Return the scheduled coefficient for the explicit expansion site."""

        coefficient = self._scheduled_cost_coefficient(iteration)
        return coefficient if self.cost_aware_expansion else 0.0

    def selection_cost_coefficient(self, iteration: int) -> float:
        """Return the scheduled coefficient for existing-child selection."""

        coefficient = self._scheduled_cost_coefficient(iteration)
        return coefficient if self.cost_aware_selection else 0.0

    def normalize_marginal_cost(self, marginal_cost: float) -> float:
        """Normalize a validated raw marginal cost by the frozen global scale."""

        if not math.isfinite(marginal_cost) or marginal_cost < 0:
            raise ValueError("marginal_cost must be finite and non-negative")
        return float(marginal_cost) / self.cost_scale

    def decompose(
        self,
        parent_visits: int,
        child_visits: int,
        child_q: float,
        marginal_cost: float,
        exploration_constant: float,
        iteration: int = 1,
    ) -> UCTScoreTerms:
        """Return every numeric term without changing scorer semantics."""
        if (
            isinstance(parent_visits, bool)
            or not isinstance(parent_visits, int)
            or parent_visits < 0
        ):
            raise ValueError("parent_visits must be a non-negative integer")
        if (
            isinstance(child_visits, bool)
            or not isinstance(child_visits, int)
            or child_visits < 0
        ):
            raise ValueError("child_visits must be a non-negative integer")
        if not math.isfinite(child_q):
            raise ValueError("child_q must be finite")
        if not math.isfinite(marginal_cost) or marginal_cost < 0:
            raise ValueError("marginal_cost must be finite and non-negative")
        if not math.isfinite(exploration_constant) or exploration_constant < 0:
            raise ValueError("exploration_constant must be finite and non-negative")
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration < 1
        ):
            raise ValueError("iteration must be an integer of at least 1")

        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits + 1) / (child_visits + 1)
        )
        lambda_t = self._scheduled_cost_coefficient(iteration)
        normalized_cost = self.normalize_marginal_cost(marginal_cost)
        selection_coefficient = self.selection_cost_coefficient(iteration)
        penalty = selection_coefficient * normalized_cost
        uct_score = child_q + exploration
        return UCTScoreTerms(
            child_q_raw=float(child_q),
            exploration_bonus=exploration,
            marginal_cost=float(marginal_cost),
            normalized_marginal_cost=normalized_cost,
            cost_scale=self.cost_scale,
            lambda_t=lambda_t,
            selection_cost_coefficient=selection_coefficient,
            decayed_cost_penalty=penalty,
            uct_score=uct_score,
            selection_score=uct_score - penalty,
            cost_schedule=self.schedule.value,
        )


class ReuseAwareCostModel(CostModel):
    r"""Estimate marginal cost from cluster- or item-level reuse keys.

    With ``context.current_batch`` set, this implements the item-level formula

    .. math::

        \widehat{\Delta C}_t(s,a)
        = \sum_z w_z \frac{1}{|B_{t,z}|}
          \sum_{q \in B_{t,z}} c_{d+1}(s,a,z,q)
          \mathbf{1}[(\mathrm{Child}(s,a),z,q) \notin E_t^Q].

    Canonical item-level keys are ``(child_prefix, cluster_id, item_id)``.
    Without a current batch, the model uses the full-pool cluster-level key
    ``(child_prefix, cluster_id)``.

    ``item_cost`` supplies the domain-specific cost
    :math:`c_{d+1}(s,a,z,q)`. The default is ``ClusterDef.base_cost``.
    """

    def __init__(
        self,
        clusters: list[ClusterDef],
        item_cost: ItemCostFunction | None = None,
    ) -> None:
        if not clusters:
            raise ValueError("clusters must be non-empty")
        cluster_ids = [cluster.cluster_id for cluster in clusters]
        if len(set(cluster_ids)) != len(cluster_ids):
            raise ValueError("cluster_ids must be unique")

        total_weight = sum(cluster.weight for cluster in clusters)
        if total_weight <= 0:
            raise ValueError("at least one cluster weight must be positive")

        self.clusters = tuple(clusters)
        self._weights = {
            cluster.cluster_id: cluster.weight / total_weight for cluster in clusters
        }
        self._base_costs = {
            cluster.cluster_id: cluster.base_cost for cluster in clusters
        }
        self._item_cost = item_cost or self._default_item_cost

    def marginal_cost(
        self,
        state: SearchState,
        action: Hashable,
        context: SearchContext,
    ) -> float:
        reuse_child_key = getattr(state, "reuse_child_key", None)
        child_prefix = (
            reuse_child_key(action)
            if callable(reuse_child_key)
            else state.child(action).state_key()
        )
        try:
            hash(child_prefix)
        except TypeError as exc:
            raise TypeError("reuse child key must be hashable") from exc
        batch = context.current_batch
        if batch is None:
            return self._cluster_level_cost(
                state,
                action,
                child_prefix,
                context,
            )

        batch_cluster_ids = {cluster.cluster_id for cluster in batch.clusters}
        expected_cluster_ids = {cluster.cluster_id for cluster in self.clusters}
        if batch_cluster_ids != expected_cluster_ids:
            missing = sorted(expected_cluster_ids - batch_cluster_ids)
            extra = sorted(batch_cluster_ids - expected_cluster_ids)
            raise ValueError(
                "current_batch clusters must match the cost-model clusters; "
                f"missing={missing}, extra={extra}"
            )

        total = 0.0
        for cluster in self.clusters:
            cluster_batch = batch.for_cluster(cluster.cluster_id)
            assert cluster_batch is not None
            uncached_cost = 0.0
            for item_id in cluster_batch.item_ids:
                reuse_key = (child_prefix, cluster.cluster_id, item_id)
                if reuse_key in context.materialized_keys:
                    continue
                uncached_cost += self._validated_item_cost(
                    state,
                    action,
                    cluster.cluster_id,
                    item_id,
                )
            total += (
                self._weights[cluster.cluster_id]
                * uncached_cost
                / len(cluster_batch.item_ids)
            )
        return total

    def _cluster_level_cost(
        self,
        state: SearchState,
        action: Hashable,
        child_prefix: Hashable,
        context: SearchContext,
    ) -> float:
        total = 0.0
        for cluster in self.clusters:
            reuse_key = (child_prefix, cluster.cluster_id)
            if reuse_key in context.materialized_keys:
                continue
            total += self._weights[cluster.cluster_id] * self._validated_item_cost(
                state,
                action,
                cluster.cluster_id,
                None,
            )
        return total

    def _default_item_cost(
        self,
        _state: SearchState,
        _action: Hashable,
        cluster_id: str,
        _item_id: Hashable | None,
    ) -> float:
        return self._base_costs[cluster_id]

    def _validated_item_cost(
        self,
        state: SearchState,
        action: Hashable,
        cluster_id: str,
        item_id: Hashable | None,
    ) -> float:
        cost = float(self._item_cost(state, action, cluster_id, item_id))
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(
                "item_cost must return a finite, non-negative value; "
                f"got {cost!r} for cluster={cluster_id!r}, item={item_id!r}"
            )
        return cost
