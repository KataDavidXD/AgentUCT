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
    r"""Score_t(s,a) = Q + c_uct * sqrt(log(N_parent+1)/(N_child+1)) - lambda * marginal_cost

    .. math::

        \mathrm{Score}_t(s,a)
        = \hat Q_t(s,a)
        + c_{\mathrm{uct}} \sqrt{\frac{\log(N_t(s)+1)}{N_t(s,a)+1}}
        - \lambda_t \, \widehat{\Delta C}_t(s,a)

    Parameters
    ----------
    lambda_t : float
        Cost regularization weight (the paper's lambda_t).
        Controls the quality-cost tradeoff during search.
        When 0 this degenerates to vanilla UCT.
    """

    def __init__(self, lambda_t: float = 0.0) -> None:
        self.lambda_t = lambda_t

    def score(
        self,
        parent_visits: int,
        child_visits: int,
        child_q: float,
        marginal_cost: float,
        exploration_constant: float,
    ) -> float:
        exploitation = child_q
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits + 1) / (child_visits + 1)
        )
        cost_penalty = self.lambda_t * marginal_cost
        return exploitation + exploration - cost_penalty


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

class ReuseAwareCostModel(CostModel):
    r"""Cluster-aware, path-reuse marginal cost estimator (paper Section 3.4).

    Computes the weighted-average marginal cost across benchmark clusters:

    .. math::

        \widehat{\Delta C}_t(s,a)
        = \sum_{z \in \mathcal Z} \omega_z \cdot
          \begin{cases}
            0                    & (\text{prefix},\, z) \in \mathcal{Path}_t \\
            \text{base\_cost}_z  & \text{otherwise}
          \end{cases}

    For each cluster the model constructs a bipartite key
    ``(prefix, cluster_id)`` and checks ``context.materialized_keys``
    (the paper's Path_t).  If cached, that cluster contributes 0;
    otherwise it contributes ``base_cost``.

    The state should expose ``path_key_for_action(action)`` returning
    the config prefix after taking *action*.  If absent, the model
    falls back to the full weighted-average base cost.

    Parameters
    ----------
    clusters : list[ClusterDef]
        Benchmark cluster definitions.  Weights are auto-normalized
        so they sum to 1.
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
        cost = 0.0
        for w, c in zip(self._norm_weights, self.clusters):
            bipartite_key = (prefix, c.cluster_id)
            if bipartite_key not in context.materialized_keys:
                cost += w * c.base_cost
        return cost
