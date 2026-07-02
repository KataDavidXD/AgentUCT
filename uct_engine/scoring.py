"""Concrete implementations of ScoreFunction and CostModel.

Provides the only two built-in components the engine ships with:
  - CostAwareUCTScorer  (paper Section 3.5)
  - ReuseAwareCostModel (paper Section 3.4)
"""
from __future__ import annotations

import math
from typing import Hashable

from .interfaces import ClusterDef, CostModel, ScoreFunction, SearchContext, SearchState


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class CostAwareUCTScorer(ScoreFunction):
    r"""Agent-UCT_t(s,a) = Q + c_uct * sqrt(log(N+1)/(n+1)) - (lambda_0/sqrt(t)) * cost

    .. math::

        \mathrm{Agent\text{-}UCT}_t(s,a)
        = \hat Q_t(s,a)
        + c_{\mathrm{uct}} \sqrt{\frac{\log(N_t(s)+1)}{N_t(s,a)+1}}
        - \frac{\lambda_0}{\sqrt{t}} \, \widehat{\Delta C}_t(s,a)

    Parameters
    ----------
    lambda_0 : float
        Initial cost regularization weight.  Decays as lambda_0/sqrt(t).
        When 0 this degenerates to vanilla UCT.
    lambda_t : float
        Deprecated alias for lambda_0 (backward compat).
    """

    def __init__(self, lambda_0: float = 0.0, *, lambda_t: float | None = None) -> None:
        if lambda_t is not None and lambda_0 == 0.0:
            lambda_0 = lambda_t
        self.lambda_0 = lambda_0

    def score(
        self,
        parent_visits: int,
        child_visits: int,
        child_q: float,
        marginal_cost: float,
        exploration_constant: float,
        iteration: int = 1,
    ) -> float:
        exploitation = child_q
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits + 1) / (child_visits + 1)
        )
        lambda_t = self.lambda_0 / math.sqrt(max(1, iteration))
        cost_penalty = lambda_t * marginal_cost
        return exploitation + exploration - cost_penalty


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

class ReuseAwareCostModel(CostModel):
    r"""Adaptive item-level marginal cost estimator.

    Computes per-batch hit probability for each spectrum:

    .. math::

        \widehat{\Delta C}_t(s,a)
        = \sum_{z \in \mathcal Z} \omega_z \cdot c_{d+1}(s,a,z) \cdot
          \left(1 - \frac{|\mathcal Q_t(p(s,a),z) \cap \mathcal{Batch}_z|}
                         {|\mathcal{Batch}_z|}\right)

    When ``context.extra["current_batches"]`` is populated (adaptive mode),
    the model computes exact per-batch hit rates against
    ``context.extra["cached_qids"]``.  Otherwise falls back to the legacy
    binary Path_t check for backward compatibility.

    Parameters
    ----------
    clusters : list[ClusterDef]
        Benchmark cluster definitions.  Weights are auto-normalized.
    """

    def __init__(self, clusters: list[ClusterDef]) -> None:
        self.clusters = clusters
        total_w = sum(c.weight for c in clusters)
        if total_w > 0:
            self._norm_weights = [c.weight / total_w for c in clusters]
        else:
            self._norm_weights = [1.0 / len(clusters)] * len(clusters)

    def marginal_cost(
        self,
        state: SearchState,
        action: Hashable,
        context: SearchContext,
    ) -> float:
        key_fn = getattr(state, "path_key_for_action", None)
        if key_fn is None:
            return sum(
                w * c.base_cost
                for w, c in zip(self._norm_weights, self.clusters)
            )
        prefix = key_fn(action)

        current_batches = context.extra.get("current_batches")
        if current_batches is not None:
            return self._adaptive_cost(prefix, context, current_batches)

        # Legacy fallback: binary Path_t check
        cost = 0.0
        for w, c in zip(self._norm_weights, self.clusters):
            bipartite_key = (prefix, c.cluster_id)
            if bipartite_key not in context.materialized_keys:
                cost += w * c.base_cost
        return cost

    def _adaptive_cost(
        self,
        prefix: Hashable,
        context: SearchContext,
        current_batches: dict[str, list[str]],
    ) -> float:
        """Per-batch hit probability cost: cost_z = base_cost * (1 - hit_prob)."""
        cached_qids = context.extra.get("cached_qids", {})
        cost = 0.0
        for w, c in zip(self._norm_weights, self.clusters):
            sid = c.cluster_id
            batch_z = current_batches.get(sid)
            if not batch_z:
                cost += w * c.base_cost
                continue
            cached_z = cached_qids.get((prefix, sid))
            if cached_z:
                batch_set = set(batch_z)
                hit_count = len(cached_z & batch_set)
                hit_prob = hit_count / len(batch_z)
            else:
                hit_prob = 0.0
            cost += w * c.base_cost * (1.0 - hit_prob)
        return cost
