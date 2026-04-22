"""Basic integration tests for UCTSearchEngine."""
from __future__ import annotations

import unittest
from typing import Hashable

from uct_engine import (
    BenchmarkClusterResult,
    ClusterDef,
    CostAwareUCTScorer,
    EvaluationResult,
    Evaluator,
    ReuseAwareCostModel,
    SearchContext,
    SearchResult,
    SearchState,
    UCTSearchEngine,
)

SINGLE_CLUSTER = [ClusterDef("c0", weight=1.0, base_cost=1.0)]


# -- Minimal domain for testing -----------------------------------------------

class TinyState:
    """2-slot binary search space: 4 terminal configs total."""

    OPTIONS = [["a", "b"], ["x", "y"]]

    def __init__(self, choices: tuple[str, ...] = ()) -> None:
        self.choices = choices

    def is_terminal(self) -> bool:
        return len(self.choices) == 2

    def available_actions(self) -> list[Hashable]:
        d = len(self.choices)
        return list(self.OPTIONS[d]) if d < 2 else []

    def child(self, action: Hashable) -> "TinyState":
        return TinyState(self.choices + (str(action),))

    def state_key(self) -> Hashable:
        return self.choices

    def pretty(self) -> str:
        return str(self.choices)

    def path_key_for_action(self, action: Hashable) -> Hashable:
        return self.choices + (str(action),)


TINY_REWARDS: dict[tuple[str, ...], float] = {
    ("a", "x"): 0.3,
    ("a", "y"): 0.9,
    ("b", "x"): 0.5,
    ("b", "y"): 0.6,
}


class TinyEvaluator(Evaluator):
    def evaluate(self, state: SearchState, context: SearchContext) -> EvaluationResult:
        key: tuple[str, ...] = state.state_key()  # type: ignore[assignment]
        reward = TINY_REWARDS.get(key, 0.0)
        return EvaluationResult(
            reward=reward,
            total_cost=1.0,
            cluster_results=[
                BenchmarkClusterResult(cluster_id="c0", reward=reward, cost=1.0),
            ],
        )


# -- Tests --------------------------------------------------------------------

class TestBasicSearch(unittest.TestCase):

    def _run_search(self, iterations: int = 80) -> SearchResult:
        engine = UCTSearchEngine(
            evaluator=TinyEvaluator(),
            scorer=CostAwareUCTScorer(lambda_t=0.0),
            cost_model=ReuseAwareCostModel(clusters=SINGLE_CLUSTER),
            exploration_constant=1.4,
            random_seed=123,
        )
        return engine.search(TinyState(), max_iterations=iterations)

    def test_returns_valid_result(self) -> None:
        result = self._run_search()
        self.assertIsInstance(result, SearchResult)
        self.assertIsNotNone(result.best_state)
        self.assertGreater(result.best_reward, 0.0)

    def test_finds_best_config(self) -> None:
        result = self._run_search(iterations=200)
        self.assertAlmostEqual(result.best_reward, 0.9, places=5)
        self.assertEqual(result.best_state.state_key(), ("a", "y"))

    def test_visit_counts_positive(self) -> None:
        result = self._run_search()
        self.assertGreater(result.root_node.visit_count, 0)
        for child in result.root_node.children.values():
            self.assertGreater(child.visit_count, 0)

    def test_iterations_match(self) -> None:
        result = self._run_search(iterations=50)
        self.assertEqual(result.iterations, 50)
        self.assertEqual(result.total_evaluations, 50)

    def test_cost_budget_stops_early(self) -> None:
        engine = UCTSearchEngine(
            evaluator=TinyEvaluator(),
            scorer=CostAwareUCTScorer(lambda_t=0.0),
            cost_model=ReuseAwareCostModel(clusters=SINGLE_CLUSTER),
            exploration_constant=1.4,
            random_seed=99,
        )
        result = engine.search(TinyState(), max_iterations=1000, max_cost=5.0)
        self.assertLessEqual(result.total_cost, 6.0)
        self.assertLess(result.iterations, 1000)


if __name__ == "__main__":
    unittest.main()
