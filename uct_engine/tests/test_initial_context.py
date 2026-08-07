"""TDD contract for atomically handing warm-start state into Agent-UCT."""

from __future__ import annotations

import inspect
import unittest
from typing import Hashable

from uct_engine import (
    ClusterBatch,
    ClusterDef,
    CostAwareUCTScorer,
    EvaluationBatch,
    EvaluationResult,
    Evaluator,
    ReuseAwareCostModel,
    RootChildSeed,
    SearchContext,
    SearchState,
    UCTSearchEngine,
    WarmStartBundle,
)


class OneActionState:
    def __init__(self, choice: str | None = None) -> None:
        self.choice = choice

    def is_terminal(self) -> bool:
        return self.choice is not None

    def available_actions(self) -> list[Hashable]:
        return ["only"] if self.choice is None else []

    def child(self, action: Hashable) -> "OneActionState":
        return OneActionState(str(action))

    def state_key(self) -> Hashable:
        return () if self.choice is None else (self.choice,)

    def pretty(self) -> str:
        return str(self.state_key())


class TwoActionState:
    def __init__(self, choice: str | None = None) -> None:
        self.choice = choice

    def is_terminal(self) -> bool:
        return self.choice is not None

    def available_actions(self) -> list[Hashable]:
        return ["a", "b"] if self.choice is None else []

    def child(self, action: Hashable) -> "TwoActionState":
        if self.choice is not None or action not in self.available_actions():
            raise ValueError(action)
        return TwoActionState(str(action))

    def state_key(self) -> Hashable:
        return () if self.choice is None else (self.choice,)

    def pretty(self) -> str:
        return str(self.state_key())


class ThreeActionState:
    def __init__(self, choice: str | None = None) -> None:
        self.choice = choice

    def is_terminal(self) -> bool:
        return self.choice is not None

    def available_actions(self) -> list[Hashable]:
        return ["a", "b", "c"] if self.choice is None else []

    def child(self, action: Hashable) -> "ThreeActionState":
        if self.choice is not None or action not in self.available_actions():
            raise ValueError(action)
        return ThreeActionState(str(action))

    def state_key(self) -> Hashable:
        return () if self.choice is None else (self.choice,)

    def pretty(self) -> str:
        return str(self.state_key())


class OpaquePathState:
    """Two-level state whose stable key does not encode tuple ancestry."""

    def __init__(self, path: tuple[str, ...] = ()) -> None:
        self.path = path

    def is_terminal(self) -> bool:
        return len(self.path) == 2

    def available_actions(self) -> list[Hashable]:
        if not self.path:
            return ["a", "b"]
        if self.path == ("a",):
            return ["x"]
        if self.path == ("b",):
            return ["y"]
        return []

    def child(self, action: Hashable) -> "OpaquePathState":
        if action not in self.available_actions():
            raise ValueError(action)
        return OpaquePathState((*self.path, str(action)))

    def state_key(self) -> Hashable:
        return "root" + "".join(f"/{choice}" for choice in self.path)

    def pretty(self) -> str:
        return str(self.state_key())


class SpoofTerminalState:
    def is_terminal(self) -> bool:
        return True

    def available_actions(self) -> list[Hashable]:
        return []

    def child(self, action: Hashable) -> SearchState:
        raise ValueError(action)

    def state_key(self) -> Hashable:
        return ("a", "foreign")

    def pretty(self) -> str:
        return str(self.state_key())


class FixedEvaluator(Evaluator):
    def evaluate(
        self,
        state: SearchState,
        context: SearchContext,
    ) -> EvaluationResult:
        assert context.current_batch is not None
        item_id = context.current_batch.clusters[0].item_ids[0]
        return EvaluationResult(
            reward=0.75,
            total_cost=0.0,
            materialized_keys={(state.state_key(), "smoke", item_id)},
        )


class FixedBatchProvider:
    def __init__(self) -> None:
        self.batch = EvaluationBatch(
            clusters=(ClusterBatch("smoke", ("q0",)),),
        )

    def draw(
        self,
        iteration: int,
        context: SearchContext,
    ) -> EvaluationBatch:
        del iteration, context
        return self.batch


def _engine(*, with_batch: bool = True) -> UCTSearchEngine:
    return UCTSearchEngine(
        evaluator=FixedEvaluator(),
        scorer=CostAwareUCTScorer(),
        cost_model=ReuseAwareCostModel(
            clusters=[ClusterDef("smoke", weight=1.0, base_cost=1.0)],
        ),
        batch_provider=FixedBatchProvider() if with_batch else None,
        random_seed=99,
    )


class TestWarmStartBundle(unittest.TestCase):
    def test_search_exposes_one_atomic_warm_start_parameter(self) -> None:
        parameters = inspect.signature(UCTSearchEngine.search).parameters

        self.assertIn("warm_start", parameters)
        self.assertNotIn("initial_context", parameters)
        self.assertNotIn("root_child_seeds", parameters)

    def test_warm_start_reuses_ledger_and_initializes_tree_atomically(self) -> None:
        preloaded_key = (("cold",), "smoke", "q0")
        context = SearchContext(
            materialized_keys={preloaded_key},
            total_cost=2.0,
            total_evaluations=2,
        )
        warm_start = WarmStartBundle(
            context=context,
            root_child_seeds=(
                RootChildSeed("a", TwoActionState("a"), 0.9),
                RootChildSeed("b", TwoActionState("b"), 0.0),
            ),
        )

        result = _engine().search(
            TwoActionState(),
            max_iterations=0,
            warm_start=warm_start,
        )

        self.assertIs(result.context, context)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.total_evaluations, 2)
        self.assertIn(preloaded_key, result.context.materialized_keys)
        self.assertEqual(result.root_node.visit_count, 2)
        self.assertEqual(set(result.root_node.children), {"a", "b"})
        self.assertEqual(result.root_node.children["a"].visit_count, 1)
        self.assertEqual(result.root_node.children["a"].q_value, 0.9)
        self.assertEqual(result.root_node.children["b"].visit_count, 1)
        self.assertEqual(result.root_node.children["b"].q_value, 0.0)
        self.assertEqual(result.best_state.state_key(), ("a",))
        self.assertEqual(result.best_reward, 0.9)

    def test_three_warm_starts_plus_350_main_iterations_total_353(self) -> None:
        warm_start = WarmStartBundle(
            context=SearchContext(total_evaluations=3),
            root_child_seeds=(
                RootChildSeed("a", ThreeActionState("a"), 0.1),
                RootChildSeed("b", ThreeActionState("b"), 0.2),
                RootChildSeed("c", ThreeActionState("c"), 0.3),
            ),
        )
        records = []

        result = _engine().search(
            ThreeActionState(),
            max_iterations=350,
            warm_start=warm_start,
            on_iteration=records.append,
        )

        self.assertEqual(result.iterations, 350)
        self.assertEqual(result.total_evaluations, 353)
        self.assertEqual(result.root_node.visit_count, 353)
        self.assertEqual(
            [record.iteration for record in records],
            list(range(1, 351)),
        )

    def test_bundle_rejects_an_empty_seed_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            WarmStartBundle(
                context=SearchContext(total_evaluations=1),
                root_child_seeds=(),
            )

    def test_bundle_rejects_fewer_evaluations_than_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_evaluations"):
            WarmStartBundle(
                context=SearchContext(total_evaluations=1),
                root_child_seeds=(
                    RootChildSeed("a", TwoActionState("a"), 0.9),
                    RootChildSeed("b", TwoActionState("b"), 0.0),
                ),
            )

    def test_bundle_rejects_invalid_total_cost(self) -> None:
        for invalid_cost in (
            float("nan"),
            float("inf"),
            float("-inf"),
            -1.0,
            True,
            "0",
        ):
            with self.subTest(total_cost=invalid_cost):
                with self.assertRaisesRegex(ValueError, "total_cost"):
                    WarmStartBundle(
                        context=SearchContext(
                            total_cost=invalid_cost,
                            total_evaluations=1,
                        ),
                        root_child_seeds=(
                            RootChildSeed("a", TwoActionState("a"), 0.9),
                        ),
                    )

    def test_bundle_rejects_invalid_total_evaluations(self) -> None:
        for invalid_count in (float("nan"), 1.5, -1, True, "1"):
            with self.subTest(total_evaluations=invalid_count):
                with self.assertRaisesRegex(ValueError, "total_evaluations"):
                    WarmStartBundle(
                        context=SearchContext(
                            total_evaluations=invalid_count,
                        ),
                        root_child_seeds=(
                            RootChildSeed("a", TwoActionState("a"), 0.9),
                        ),
                    )

    def test_search_rejects_invalid_max_cost(self) -> None:
        for invalid_cost in (
            float("nan"),
            float("inf"),
            float("-inf"),
            -1.0,
            True,
            "0",
        ):
            with self.subTest(max_cost=invalid_cost):
                with self.assertRaisesRegex(ValueError, "max_cost"):
                    _engine(with_batch=False).search(
                        OneActionState(),
                        max_iterations=0,
                        max_cost=invalid_cost,
                    )

    def test_search_revalidates_mutated_bundle_context(self) -> None:
        context = SearchContext(total_evaluations=2)
        warm_start = WarmStartBundle(
            context=context,
            root_child_seeds=(
                RootChildSeed("a", TwoActionState("a"), 0.9),
                RootChildSeed("b", TwoActionState("b"), 0.0),
            ),
        )
        context.total_evaluations = 1

        with self.assertRaisesRegex(ValueError, "total_evaluations"):
            _engine(with_batch=False).search(
                TwoActionState(),
                max_iterations=0,
                warm_start=warm_start,
            )

    def test_search_without_warm_start_remains_generic(self) -> None:
        result = _engine().search(OneActionState(), max_iterations=1)

        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.total_evaluations, 1)
        self.assertEqual(result.root_node.visit_count, 1)

    def test_duplicate_root_child_seed_fails_fast(self) -> None:
        warm_start = WarmStartBundle(
            context=SearchContext(total_evaluations=2),
            root_child_seeds=(
                RootChildSeed("a", TwoActionState("a"), 0.9),
                RootChildSeed("a", TwoActionState("a"), 0.2),
            ),
        )

        with self.assertRaisesRegex(ValueError, "duplicate root-child seed"):
            _engine(with_batch=False).search(
                TwoActionState(),
                max_iterations=0,
                warm_start=warm_start,
            )

    def test_seed_action_must_match_evaluated_state_root_edge(self) -> None:
        warm_start = WarmStartBundle(
            context=SearchContext(total_evaluations=2),
            root_child_seeds=(
                RootChildSeed("a", TwoActionState("b"), 0.9),
                RootChildSeed("b", TwoActionState("a"), 0.2),
            ),
        )

        with self.assertRaisesRegex(ValueError, "root edge"):
            _engine(with_batch=False).search(
                TwoActionState(),
                max_iterations=0,
                warm_start=warm_start,
            )

    def test_action_path_supports_opaque_multi_level_state_keys(self) -> None:
        evaluated_state = OpaquePathState(("a", "x"))
        warm_start = WarmStartBundle(
            context=SearchContext(total_evaluations=1),
            root_child_seeds=(
                RootChildSeed(
                    action="a",
                    action_path=("a", "x"),
                    evaluated_state=evaluated_state,
                    reward=0.9,
                ),
            ),
        )

        result = _engine(with_batch=False).search(
            OpaquePathState(),
            max_iterations=0,
            warm_start=warm_start,
        )

        self.assertEqual(result.best_state.state_key(), "root/a/x")
        self.assertEqual(result.root_node.visit_count, 1)
        self.assertEqual(result.root_node.children["a"].state.state_key(), "root/a")
        self.assertEqual(result.root_node.children["a"].children, {})

    def test_action_path_rejects_an_impossible_terminal_descendant(self) -> None:
        warm_start = WarmStartBundle(
            context=SearchContext(total_evaluations=1),
            root_child_seeds=(
                RootChildSeed(
                    action="a",
                    action_path=("a", "foreign"),
                    evaluated_state=SpoofTerminalState(),
                    reward=0.9,
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "action_path"):
            _engine(with_batch=False).search(
                TwoActionState(),
                max_iterations=0,
                warm_start=warm_start,
            )

    def test_action_path_must_start_with_the_declared_root_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "start with"):
            RootChildSeed(
                action="a",
                action_path=("b",),
                evaluated_state=TwoActionState("b"),
                reward=0.9,
            )


if __name__ == "__main__":
    unittest.main()
