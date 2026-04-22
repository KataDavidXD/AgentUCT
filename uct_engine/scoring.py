"""Concrete implementations of ScoreFunction and CostModel.

Provides the only two built-in components the engine ships with:
  - CostAwareUCTScorer  (paper Section 3.5)
  - ReuseAwareCostModel (paper Section 3.4)
"""
from __future__ import annotations

import math
from typing import Hashable

from .interfaces import CostModel, ScoreFunction, SearchContext, SearchState


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class CostAwareUCTScorer(ScoreFunction):
    """Score_t(s,a) = Q + c_uct * sqrt(log(N_parent+1)/(N_child+1)) - lambda * marginal_cost

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
    r"""Path-reuse-aware marginal cost estimator (paper Section 3.4).

    In the paper, the true marginal cost of evaluating child(s,a) on
    cluster z is:

    .. math::

        \Delta C_t(s,a;z) = \sum_{m=1}^{M} c_m \cdot
            \mathbf{1}[(\pi_{1:m}, z) \notin \mathcal{Path}_t]

    i.e. the sum of per-workflow-node costs for nodes whose path prefix
    has NOT yet been materialized and cached.

    This model simplifies the above to a binary estimate:

    - **base_cost** if the child path key is NOT yet in Path_t
      (represents the expected full execution cost of a new,
      uncached path -- in practice this would be calibrated to
      the average per-config evaluation cost in your domain).
    - **0.0** if the child path key IS already in Path_t
      (the path prefix is cached; execution can be restored
      at zero recomputation cost via AgentGit).

    For production use, subclass this and override ``marginal_cost``
    to compute the real per-node, per-cluster cost from your AgentGit
    backend.  The ``base_cost`` scalar is a reasonable starting point
    for prototyping and unit tests.

    Expects the state to expose ``path_key_for_action(action)`` via duck
    typing.  If the method is absent the model falls back to *base_cost*.
    """

    def __init__(self, base_cost: float = 1.0) -> None:
        self.base_cost = base_cost

    def marginal_cost(
        self,
        state: SearchState,
        action: Hashable,
        context: SearchContext,
    ) -> float:
        key_fn = getattr(state, "path_key_for_action", None)
        if key_fn is None:
            return self.base_cost
        key = key_fn(action)
        if key in context.materialized_keys:
            return 0.0
        return self.base_cost
