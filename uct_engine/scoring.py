"""AgentUCT v3 selection and a legacy direct-prefix cost helper."""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Hashable

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


class CostAwareUCTScorer(ScoreFunction):
    r"""Compute Equation (5) of arXiv:2607.24162v3.

    .. math::

        \widehat Q_t(s,a)
        + c_{\mathrm{UCT}}
          \sqrt{\frac{\log(N_t(s)+1)}{N_t(s,a)}}
        - \frac{\beta}{C_{\mathrm{ref}}}\widehat b_t(s,a)

    The cost penalty affects selection only. Rewards stored in tree statistics
    and the final best configuration are not cost-adjusted. ``marginal_cost``
    is the forecast for a complete evaluation, in the same units as ``c_ref``.
    Unvisited children have priority. ``iteration`` is retained for the scorer
    interface; the coefficient does not decay with time.

    ``lambda_0`` is a compatibility alias for the fixed coefficient beta/C_ref.
    Supplying it overrides beta/C_ref; it no longer requests the old schedule.
    """

    def __init__(
        self, lambda_0: float | None = None, *, beta: float = 2.0,
        c_ref: float = 1.0,
    ) -> None:
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and non-negative")
        if not math.isfinite(c_ref) or c_ref <= 0:
            raise ValueError("c_ref must be finite and positive")
        coefficient = beta / c_ref if lambda_0 is None else float(lambda_0)
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("cost coefficient must be finite and non-negative")
        self.beta = float(beta)
        self.c_ref = float(c_ref)
        self.lambda_0 = coefficient

    def score(
        self,
        parent_visits: int,
        child_visits: int,
        child_q: float,
        marginal_cost: float,
        exploration_constant: float,
        iteration: int = 1,
    ) -> float:
        if parent_visits < 0 or child_visits < 0:
            raise ValueError("visit counts must be non-negative")
        if not math.isfinite(marginal_cost) or marginal_cost < 0:
            raise ValueError("marginal_cost must be finite and non-negative")
        if not math.isfinite(exploration_constant) or exploration_constant < 0:
            raise ValueError("exploration_constant must be finite and non-negative")
        if iteration < 1:
            raise ValueError("iteration must be at least 1")

        if child_visits == 0:
            return math.inf
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits + 1) / child_visits
        )
        return child_q + exploration - self.lambda_0 * marginal_cost


class ReuseAwareCostModel(CostModel):
    r"""Legacy direct-prefix helper; not v3 complete-evaluation pricing.

    This estimates only the immediate child. A v3 integration must implement
    ``CostModel`` to price every missing resource needed for a complete reward.

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
            cluster.cluster_id: cluster.weight / total_weight
            for cluster in clusters
        }
        self._base_costs = {
            cluster.cluster_id: cluster.base_cost
            for cluster in clusters
        }
        self._item_cost = item_cost or self._default_item_cost

    def marginal_cost(
        self,
        state: SearchState,
        action: Hashable,
        context: SearchContext,
    ) -> float:
        child_prefix = state.child(action).state_key()
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
