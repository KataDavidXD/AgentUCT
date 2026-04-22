"""Tests for ReuseAwareCostModel and cost-aware score interaction."""
from __future__ import annotations

import unittest
from typing import Any, Hashable

from uct_engine import (
    BenchmarkClusterResult,
    CostAwareUCTScorer,
    EvaluationResult,
    Evaluator,
    ReuseAwareCostModel,
    SearchContext,
    SearchState,
    UCTSearchEngine,
)


# -- Helpers ------------------------------------------------------------------

class ReusableState:
    """2-slot state that exposes path_key_for_action for reuse tracking."""
    OPTIONS = [["cheap", "expensive"], ["x", "y"]]

    def __init__(self, choices: tuple[str, ...] = ()) -> None:
        self.choices = choices

    def is_terminal(self) -> bool:
        return len(self.choices) == 2

    def available_actions(self) -> list[Hashable]:
        d = len(self.choices)
        return list(self.OPTIONS[d]) if d < 2 else []

    def child(self, action: Hashable) -> "ReusableState":
        return ReusableState(self.choices + (str(action),))

    def state_key(self) -> Hashable:
        return self.choices

    def pretty(self) -> str:
        return str(self.choices)

    def path_key_for_action(self, action: Hashable) -> Hashable:
        return self.choices + (str(action),)


# Rewards are equal so the only differentiator is cost.
EQUAL_REWARDS: dict[tuple[str, ...], float] = {
    ("cheap", "x"):     0.7,
    ("cheap", "y"):     0.7,
    ("expensive", "x"): 0.7,
    ("expensive", "y"): 0.7,
}


class CostTrackingEvaluator(Evaluator):
    """Returns equal rewards but only materializes 'cheap' branch prefixes.

    The 'expensive' branch never contributes materialized keys, so the
    ReuseAwareCostModel always returns full base_cost for it, while
    'cheap' becomes free after its first evaluation.
    """

    def evaluate(self, state: SearchState, context: SearchContext) -> EvaluationResult:
        key: tuple[str, ...] = state.state_key()  # type: ignore[assignment]
        reward = EQUAL_REWARDS.get(key, 0.0)

        new_keys: list[Hashable] = []
        cost = 0.0
        is_cheap_branch = len(key) > 0 and key[0] == "cheap"
        for depth in range(1, len(key) + 1):
            pk: Hashable = key[:depth]
            if pk not in context.materialized_keys:
                cost += 1.0
                if is_cheap_branch:
                    new_keys.append(pk)

        return EvaluationResult(
            reward=reward,
            total_cost=cost,
            cluster_results=[
                BenchmarkClusterResult(
                    cluster_id="c0",
                    reward=reward,
                    cost=cost,
                    materialized_keys=new_keys,
                ),
            ],
        )


# -- Tests --------------------------------------------------------------------

class TestReuseAwareCostModel(unittest.TestCase):

    def test_returns_base_cost_when_not_materialized(self) -> None:
        model = ReuseAwareCostModel(base_cost=2.5)
        ctx = SearchContext()
        state = ReusableState(("cheap",))
        cost = model.marginal_cost(state, "x", ctx)
        self.assertAlmostEqual(cost, 2.5)

    def test_returns_zero_when_materialized(self) -> None:
        model = ReuseAwareCostModel(base_cost=2.5)
        ctx = SearchContext()
        ctx.materialized_keys.add(("cheap", "x"))
        state = ReusableState(("cheap",))
        cost = model.marginal_cost(state, "x", ctx)
        self.assertAlmostEqual(cost, 0.0)

    def test_cluster_results_propagate_to_context(self) -> None:
        """Engine must harvest materialized_keys from EvaluationResult."""
        engine = UCTSearchEngine(
            evaluator=CostTrackingEvaluator(),
            scorer=CostAwareUCTScorer(lambda_t=0.0),
            cost_model=ReuseAwareCostModel(base_cost=1.0),
            exploration_constant=1.4,
            random_seed=0,
        )
        result = engine.search(ReusableState(), max_iterations=20)
        self.assertGreater(len(result.context.materialized_keys), 0)


class TestCostAwareBranchPreference(unittest.TestCase):

    def test_high_lambda_shifts_preference(self) -> None:
        """With high lambda_t, the engine should prefer 'cheap' over 'expensive'.

        Both branches have equal reward, but 'cheap' prefix gets materialized
        first, so subsequent visits cost 0 while 'expensive' stays costly.
        With a large lambda_t the scorer penalises 'expensive' enough to
        shift visit counts toward 'cheap'.
        """
        engine_costly = UCTSearchEngine(
            evaluator=CostTrackingEvaluator(),
            scorer=CostAwareUCTScorer(lambda_t=5.0),
            cost_model=ReuseAwareCostModel(base_cost=3.0),
            exploration_constant=1.4,
            random_seed=42,
        )
        result = engine_costly.search(ReusableState(), max_iterations=200)
        root = result.root_node

        cheap_visits = root.children["cheap"].visit_count
        expensive_visits = root.children["expensive"].visit_count

        # With strong cost penalty, cheap branch should dominate.
        self.assertGreater(cheap_visits, expensive_visits)

    def test_zero_lambda_is_balanced(self) -> None:
        """With lambda_t=0 (no cost penalty) visits should be roughly even."""
        engine_free = UCTSearchEngine(
            evaluator=CostTrackingEvaluator(),
            scorer=CostAwareUCTScorer(lambda_t=0.0),
            cost_model=ReuseAwareCostModel(base_cost=3.0),
            exploration_constant=1.4,
            random_seed=42,
        )
        result = engine_free.search(ReusableState(), max_iterations=200)
        root = result.root_node

        cheap_visits = root.children["cheap"].visit_count
        expensive_visits = root.children["expensive"].visit_count

        ratio = cheap_visits / max(expensive_visits, 1)
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 2.0)


if __name__ == "__main__":
    unittest.main()
