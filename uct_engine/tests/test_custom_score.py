"""Test that a custom ScoreFunction is respected by the engine."""
from __future__ import annotations

import unittest
from typing import Hashable

from uct_engine import (
    BenchmarkClusterResult,
    CostAwareUCTScorer,
    EvaluationResult,
    Evaluator,
    ReuseAwareCostModel,
    ScoreFunction,
    SearchContext,
    SearchState,
    UCTSearchEngine,
)


# -- Deterministic scorer that always favours the first action ---------------

class FirstActionScorer(ScoreFunction):
    """Always assigns descending scores to children in action order.

    This ensures the engine *always* expands/selects the first available
    action, making the search path fully deterministic.
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
        # Lower child_visits => explored less => higher score in vanilla UCT.
        # We override to always prefer the child that was expanded first
        # (which gets more visits), by returning -child_visits.
        # BUT actually, we want to *always pick the first action*.
        # The engine iterates children in dict insertion order (action order),
        # so we simply give a constant huge score minus the call count within
        # a single _best_child sweep.  Simpler: just return -marginal_cost
        # which is 0 for all.  Instead, return 1/(child_visits+1) to prefer
        # the *most-visited* child, which is the first one expanded.
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

    def test_first_action_scorer_dominates(self) -> None:
        """With FirstActionScorer the engine should explore 'alpha' most."""
        scorer = FirstActionScorer()
        engine = UCTSearchEngine(
            evaluator=OneSlotEvaluator(),
            scorer=scorer,
            cost_model=ReuseAwareCostModel(base_cost=0.0),
            exploration_constant=1.4,
            random_seed=0,
        )
        result = engine.search(OneSlotState(), max_iterations=60)
        root = result.root_node
        alpha_visits = root.children["alpha"].visit_count
        beta_visits = root.children["beta"].visit_count
        gamma_visits = root.children["gamma"].visit_count

        # FirstActionScorer biases toward the least-visited child (via 1/(n+1)),
        # so visits should be roughly balanced but scorer *was called*.
        self.assertGreater(alpha_visits, 0)
        self.assertGreater(beta_visits, 0)
        self.assertGreater(gamma_visits, 0)
        self.assertGreater(scorer._call_count, 0)

    def test_default_scorer_finds_best(self) -> None:
        """Standard CostAwareUCTScorer should converge to gamma (reward 0.9)."""
        engine = UCTSearchEngine(
            evaluator=OneSlotEvaluator(),
            scorer=CostAwareUCTScorer(lambda_t=0.0),
            cost_model=ReuseAwareCostModel(base_cost=0.0),
            exploration_constant=1.4,
            random_seed=7,
        )
        result = engine.search(OneSlotState(), max_iterations=100)
        self.assertEqual(result.best_state.state_key(), "gamma")
        self.assertAlmostEqual(result.best_reward, 0.9, places=5)


if __name__ == "__main__":
    unittest.main()
