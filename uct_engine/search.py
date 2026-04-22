"""UCT search engine with cost-aware tree policy.

Implements Algorithm 1 from the paper: selection, expansion, evaluation,
and backpropagation with reuse-aware marginal cost regularization.
"""
from __future__ import annotations

import logging
import random
from typing import Hashable, Optional

from .core import TreeNode
from .interfaces import (
    CostModel,
    Evaluator,
    ScoreFunction,
    SearchContext,
    SearchResult,
    SearchState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_rollout(state: SearchState, rng: random.Random) -> SearchState:
    """Complete *state* to a terminal config by picking random actions."""
    current = state
    while not current.is_terminal():
        actions = current.available_actions()
        current = current.child(rng.choice(actions))
    return current


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class UCTSearchEngine:
    """Cost-aware UCT search over a discrete compositional space.

    Parameters
    ----------
    evaluator : Evaluator
        Scores terminal configurations (may use benchmark clusters).
    scorer : ScoreFunction
        Tree-policy acquisition function.
    cost_model : CostModel
        Marginal cost estimator (e.g. reuse-aware).
    exploration_constant : float
        c_uct in the UCT formula.
    random_seed : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        scorer: ScoreFunction,
        cost_model: CostModel,
        exploration_constant: float = 1.4,
        random_seed: Optional[int] = 42,
    ) -> None:
        self.evaluator = evaluator
        self.scorer = scorer
        self.cost_model = cost_model
        self.exploration_constant = exploration_constant
        self.random_seed = random_seed

    # -- public API -----------------------------------------------------------

    def search(
        self,
        root_state: SearchState,
        max_iterations: int = 200,
        max_cost: Optional[float] = None,
    ) -> SearchResult:
        """Run UCT search from *root_state*.

        Stops after *max_iterations* iterations or when cumulative
        realized cost reaches *max_cost* (whichever comes first).
        """
        rng = random.Random(self.random_seed)
        context = SearchContext(random=rng)

        root = TreeNode(state=root_state)
        best_state: Optional[SearchState] = None
        best_reward: float = float("-inf")

        for iteration in range(1, max_iterations + 1):
            # --- 1. Selection ---
            node = self._select(root, context)

            # --- 2. Expansion ---
            if not node.state.is_terminal():
                unexpanded = node.unexpanded_actions()
                if unexpanded:
                    action = rng.choice(unexpanded)
                    child_state = node.state.child(action)
                    node = node.add_child(action, child_state)

            # --- 3. Evaluation ---
            eval_state = node.state
            if not eval_state.is_terminal():
                eval_state = random_rollout(eval_state, rng)

            result = self.evaluator.evaluate(eval_state, context)

            # Harvest materialized keys from cluster results -> Path_t
            for cr in result.cluster_results:
                for key in cr.materialized_keys:
                    context.materialized_keys.add(key)

            context.total_cost += result.total_cost
            context.total_evaluations += 1

            reward = result.reward

            if reward > best_reward:
                best_reward = reward
                best_state = eval_state

            # --- 4. Backpropagation ---
            self._backpropagate(node, reward)

            # --- 5. Budget check ---
            if max_cost is not None and context.total_cost >= max_cost:
                logger.info(
                    "Cost budget exhausted at iteration %d (%.2f >= %.2f)",
                    iteration, context.total_cost, max_cost,
                )
                break

        return SearchResult(
            best_state=best_state,
            best_reward=best_reward,
            root_node=root,
            iterations=iteration,
            total_evaluations=context.total_evaluations,
            total_cost=context.total_cost,
            context=context,
        )

    # -- internals ------------------------------------------------------------

    def _select(self, node: TreeNode, context: SearchContext) -> TreeNode:
        """Descend the tree using the scorer until a leaf or unexpanded node."""
        while not node.state.is_terminal():
            unexpanded = node.unexpanded_actions()
            if unexpanded:
                return node
            # All actions expanded -- pick best child by score
            node = self._best_child(node, context)
        return node

    def _best_child(self, node: TreeNode, context: SearchContext) -> TreeNode:
        best_score = float("-inf")
        best_child: Optional[TreeNode] = None
        for action, child in node.children.items():
            mc = self.cost_model.marginal_cost(node.state, action, context)
            s = self.scorer.score(
                parent_visits=node.visit_count,
                child_visits=child.visit_count,
                child_q=child.q_value,
                marginal_cost=mc,
                exploration_constant=self.exploration_constant,
            )
            if s > best_score:
                best_score = s
                best_child = child
        assert best_child is not None
        return best_child

    @staticmethod
    def _backpropagate(node: TreeNode, reward: float) -> None:
        current: Optional[TreeNode] = node
        while current is not None:
            current.update(reward)
            current = current.parent
