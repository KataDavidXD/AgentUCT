"""Test that a custom ScoreFunction is respected by the engine."""
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
    ScoreFunction,
    SearchContext,
    SearchState,
    UCTSearchEngine,
)

ZERO_COST_CLUSTER = [ClusterDef("c0", weight=1.0, base_cost=0.0)]


# -- Deterministic scorer that always favours the least-visited child ---------

class LeastVisitedScorer(ScoreFunction):
    """Prefers the child with fewest visits (like UCT exploration only).

    Tracks call count to verify the engine actually calls the scorer.
    """

    def __init__(self) -> None:
        self._call_count = 0

    def score(
        self,
        parent_visits: int,
        child_visits: int,
        child_q: float,
        marginal_cost: float,
        exploration_constant: float,
    ) -> float:
        self._call_count += 1
        return 1_000_000.0 / (child_visits + 1)


# -- Domain: 1-slot, 3 actions -----------------------------------------------

class OneSlotState:
    ACTIONS = ["alpha", "beta", "gamma"]

    def __init__(self, choice: str | None = None) -> None:
        self.choice = choice

    def is_terminal(self) -> bool:
        return self.choice is not None

    def available_actions(self) -> list[Hashable]:
        return list(self.ACTIONS) if self.choice is None else []

    def child(self, action: Hashable) -> "OneSlotState":
        return OneSlotState(choice=str(action))

    def state_key(self) -> Hashable:
        return self.choice

    def pretty(self) -> str:
        return str(self.choice)


REWARDS = {"alpha": 0.1, "beta": 0.5, "gamma": 0.9}


class OneSlotEvaluator(Evaluator):
    def evaluate(self, state: SearchState, context: SearchContext) -> EvaluationResult:
        r = REWARDS.get(state.state_key(), 0.0)  # type: ignore[arg-type]
        return EvaluationResult(reward=r, total_cost=0.0)


# -- Tests --------------------------------------------------------------------

class TestCustomScorer(unittest.TestCase):

    def test_custom_scorer_is_called(self) -> None:
        """Inject LeastVisitedScorer and confirm it is actually invoked."""
        scorer = LeastVisitedScorer()
        engine = UCTSearchEngine(
            evaluator=OneSlotEvaluator(),
            scorer=scorer,
            cost_model=ReuseAwareCostModel(clusters=ZERO_COST_CLUSTER),
            exploration_constant=1.4,
            random_seed=0,
        )
        result = engine.search(OneSlotState(), max_iterations=60)
        root = result.root_node

        self.assertGreater(root.children["alpha"].visit_count, 0)
        self.assertGreater(root.children["beta"].visit_count, 0)
        self.assertGreater(root.children["gamma"].visit_count, 0)
        self.assertGreater(scorer._call_count, 0)

    def test_default_scorer_finds_best(self) -> None:
        """Standard CostAwareUCTScorer should converge to gamma (reward 0.9)."""
        engine = UCTSearchEngine(
            evaluator=OneSlotEvaluator(),
            scorer=CostAwareUCTScorer(lambda_t=0.0),
            cost_model=ReuseAwareCostModel(clusters=ZERO_COST_CLUSTER),
            exploration_constant=1.4,
            random_seed=7,
        )
        result = engine.search(OneSlotState(), max_iterations=100)
        self.assertEqual(result.best_state.state_key(), "gamma")
        self.assertAlmostEqual(result.best_reward, 0.9, places=5)


if __name__ == "__main__":
    unittest.main()
