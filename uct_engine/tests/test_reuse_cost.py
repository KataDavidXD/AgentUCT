"""Tests for ReuseAwareCostModel and cost-aware score interaction.

All tests use bipartite (prefix, cluster_id) keys matching the paper's
Path_t structure: materialized_keys contains (pi_prefix, z) tuples.
"""
from __future__ import annotations

import unittest
from typing import Hashable

from uct_engine import (
    ClusterDef,
    CostAwareUCTScorer,
    EvaluationResult,
    Evaluator,
    ReuseAwareCostModel,
    SearchContext,
    SearchState,
    UCTSearchEngine,
)

CLUSTERS = [
    ClusterDef("c0", weight=1.0, base_cost=3.0),
    ClusterDef("c1", weight=1.0, base_cost=3.0),
]


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


EQUAL_REWARDS: dict[tuple[str, ...], float] = {
    ("cheap", "x"):     0.7,
    ("cheap", "y"):     0.7,
    ("expensive", "x"): 0.7,
    ("expensive", "y"): 0.7,
}


class CostTrackingEvaluator(Evaluator):
    """Returns equal rewards; materializes bipartite (prefix, cluster) keys.

    Only the 'cheap' branch materializes keys, so the cost model sees
    'cheap' as free after its first evaluation while 'expensive' stays
    costly for all clusters.
    """

    def evaluate(self, state: SearchState, context: SearchContext) -> EvaluationResult:
        key: tuple[str, ...] = state.state_key()  # type: ignore[assignment]
        reward = EQUAL_REWARDS.get(key, 0.0)
        is_cheap_branch = len(key) > 0 and key[0] == "cheap"

        materialized_keys: set[Hashable] = set()
        total_cost = 0.0

        for c in CLUSTERS:
            cluster_cost = 0.0
            for depth in range(1, len(key) + 1):
                bk: Hashable = (key[:depth], c.cluster_id)
                if bk not in context.materialized_keys:
                    cluster_cost += 1.0
                    if is_cheap_branch:
                        materialized_keys.add(bk)
            total_cost += cluster_cost

        return EvaluationResult(
            reward=reward,
            total_cost=total_cost,
            materialized_keys=materialized_keys,
        )


# -- Tests: ReuseAwareCostModel unit tests ------------------------------------

class TestReuseAwareCostModel(unittest.TestCase):

    def test_returns_full_cost_when_not_materialized(self) -> None:
        """No keys in Path_t => weighted-average base_cost over all clusters."""
        model = ReuseAwareCostModel(clusters=[
            ClusterDef("c0", weight=1.0, base_cost=2.0),
            ClusterDef("c1", weight=1.0, base_cost=4.0),
        ])
        ctx = SearchContext()
        state = ReusableState(("cheap",))
        cost = model.marginal_cost(state, "x", ctx)
        # normalized weights: 0.5 each => 0.5*2 + 0.5*4 = 3.0
        self.assertAlmostEqual(cost, 3.0)

    def test_returns_zero_when_all_clusters_materialized(self) -> None:
        model = ReuseAwareCostModel(clusters=[
            ClusterDef("c0", weight=1.0, base_cost=2.0),
            ClusterDef("c1", weight=1.0, base_cost=4.0),
        ])
        ctx = SearchContext()
        ctx.materialized_keys.add((("cheap", "x"), "c0"))
        ctx.materialized_keys.add((("cheap", "x"), "c1"))
        state = ReusableState(("cheap",))
        cost = model.marginal_cost(state, "x", ctx)
        self.assertAlmostEqual(cost, 0.0)

    def test_bipartite_key_structure(self) -> None:
        """Materializing (prefix, 'c0') should zero out c0 but not c1."""
        model = ReuseAwareCostModel(clusters=[
            ClusterDef("c0", weight=1.0, base_cost=6.0),
            ClusterDef("c1", weight=1.0, base_cost=6.0),
        ])
        ctx = SearchContext()
        # Only c0 is materialized
        ctx.materialized_keys.add((("cheap", "x"), "c0"))
        state = ReusableState(("cheap",))
        cost = model.marginal_cost(state, "x", ctx)
        # c0: 0, c1: 6.0 => weighted avg = 0.5*0 + 0.5*6 = 3.0
        self.assertAlmostEqual(cost, 3.0)

    def test_partial_cluster_reuse(self) -> None:
        """If 2 of 3 clusters are materialized, cost is 1/3 of full cost."""
        model = ReuseAwareCostModel(clusters=[
            ClusterDef("c0", weight=1.0, base_cost=9.0),
            ClusterDef("c1", weight=1.0, base_cost=9.0),
            ClusterDef("c2", weight=1.0, base_cost=9.0),
        ])
        ctx = SearchContext()
        ctx.materialized_keys.add((("cheap", "x"), "c0"))
        ctx.materialized_keys.add((("cheap", "x"), "c1"))
        state = ReusableState(("cheap",))
        cost = model.marginal_cost(state, "x", ctx)
        # c0: 0, c1: 0, c2: 9 => (1/3)*9 = 3.0
        self.assertAlmostEqual(cost, 3.0)

    def test_evaluator_keys_propagate_to_context(self) -> None:
        """Engine must merge only evaluator-reported materialized keys."""
        engine = UCTSearchEngine(
            evaluator=CostTrackingEvaluator(),
            scorer=CostAwareUCTScorer(lambda_0=0.0),
            cost_model=ReuseAwareCostModel(clusters=CLUSTERS),
            exploration_constant=1.4,
            random_seed=0,
        )
        result = engine.search(ReusableState(), max_iterations=20)
        self.assertGreater(len(result.context.materialized_keys), 0)
        # Every key must be a (prefix_tuple, cluster_id) pair
        for key in result.context.materialized_keys:
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)
            prefix, cid = key
            self.assertIsInstance(prefix, tuple)
            self.assertIn(cid, ("c0", "c1"))


# -- Tests: cost-aware branch preference --------------------------------------

class TestCostAwareBranchPreference(unittest.TestCase):

    def test_high_lambda_shifts_preference(self) -> None:
        """With high lambda_t, the engine should prefer 'cheap' over 'expensive'.

        Both branches have equal reward. 'cheap' materializes bipartite
        keys after first eval so its marginal cost drops to 0 across all
        clusters; 'expensive' never materializes and stays costly.
        """
        engine_costly = UCTSearchEngine(
            evaluator=CostTrackingEvaluator(),
            scorer=CostAwareUCTScorer(lambda_0=5.0),
            cost_model=ReuseAwareCostModel(clusters=CLUSTERS),
            exploration_constant=1.4,
            random_seed=42,
        )
        result = engine_costly.search(ReusableState(), max_iterations=200)
        root = result.root_node

        cheap_visits = root.children["cheap"].visit_count
        expensive_visits = root.children["expensive"].visit_count
        self.assertGreater(cheap_visits, expensive_visits)

    def test_zero_lambda_is_balanced(self) -> None:
        """With lambda_t=0 (no cost penalty) visits should be roughly even."""
        engine_free = UCTSearchEngine(
            evaluator=CostTrackingEvaluator(),
            scorer=CostAwareUCTScorer(lambda_0=0.0),
            cost_model=ReuseAwareCostModel(clusters=CLUSTERS),
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


class TestScoreSemantics(unittest.TestCase):

    def test_lambda_decays_as_inverse_square_root(self) -> None:
        scorer = CostAwareUCTScorer(lambda_0=0.2)
        at_t1 = scorer.score(
            parent_visits=10,
            child_visits=5,
            child_q=0.7,
            marginal_cost=2.0,
            exploration_constant=0.0,
            iteration=1,
        )
        at_t4 = scorer.score(
            parent_visits=10,
            child_visits=5,
            child_q=0.7,
            marginal_cost=2.0,
            exploration_constant=0.0,
            iteration=4,
        )

        self.assertAlmostEqual(at_t1, 0.3)
        self.assertAlmostEqual(at_t4, 0.5)

    def test_best_configuration_is_selected_by_raw_reward(self) -> None:
        class OneSlotState:
            def __init__(self, choice: str | None = None) -> None:
                self.choice = choice

            def is_terminal(self) -> bool:
                return self.choice is not None

            def available_actions(self) -> list[Hashable]:
                return ["cheap", "expensive"] if self.choice is None else []

            def child(self, action: Hashable) -> "OneSlotState":
                return OneSlotState(str(action))

            def state_key(self) -> Hashable:
                return () if self.choice is None else (self.choice,)

            def pretty(self) -> str:
                return str(self.state_key())

        class RawRewardEvaluator(Evaluator):
            def evaluate(
                self,
                state: SearchState,
                context: SearchContext,
            ) -> EvaluationResult:
                choice = tuple(state.state_key())[0]
                reward = 0.9 if choice == "expensive" else 0.5
                keys = (
                    {(state.state_key(), "c0")}
                    if choice == "cheap"
                    else set()
                )
                return EvaluationResult(
                    reward=reward,
                    total_cost=1.0,
                    materialized_keys=keys,
                )

        engine = UCTSearchEngine(
            evaluator=RawRewardEvaluator(),
            scorer=CostAwareUCTScorer(lambda_0=100.0),
            cost_model=ReuseAwareCostModel(
                [ClusterDef("c0", weight=1.0, base_cost=10.0)],
            ),
            exploration_constant=0.0,
            random_seed=1,
        )

        result = engine.search(OneSlotState(), max_iterations=20)

        self.assertEqual(result.best_state.state_key(), ("expensive",))
        self.assertEqual(result.best_reward, 0.9)


if __name__ == "__main__":
    unittest.main()
