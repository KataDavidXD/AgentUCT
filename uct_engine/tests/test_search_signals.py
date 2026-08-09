"""TDD goldens for the public, JSON-safe UCT signal export."""

from __future__ import annotations

import json
import math
import unittest
from collections.abc import Hashable
from dataclasses import dataclass

from uct_engine import (
    ClusterBatch,
    ClusterDef,
    CostAwareUCTScorer,
    EvaluationBatch,
    EvaluationResult,
    ReuseAwareCostModel,
    RootChildSeed,
    SearchContext,
    SearchResult,
    SearchState,
    TreeNode,
    UCTSearchEngine,
    WarmStartBundle,
)


class OneSlotState:
    ACTIONS = ("cheap", "expensive")

    def __init__(self, choice: str | None = None) -> None:
        self.choice = choice

    def is_terminal(self) -> bool:
        return self.choice is not None

    def available_actions(self) -> list[Hashable]:
        return list(self.ACTIONS) if self.choice is None else []

    def child(self, action: Hashable) -> OneSlotState:
        if action not in self.available_actions():
            raise ValueError(action)
        return OneSlotState(str(action))

    def state_key(self) -> Hashable:
        return () if self.choice is None else (self.choice,)

    def pretty(self) -> str:
        return str(self.state_key())


class NeverCalledEvaluator:
    def evaluate(
        self,
        state: SearchState,
        context: SearchContext,
    ) -> EvaluationResult:
        del state, context
        raise AssertionError("zero-iteration signal fixtures must not evaluate")


def _item_cost(
    _state: SearchState,
    action: Hashable,
    _cluster_id: str,
    _item_id: Hashable | None,
) -> float:
    return 1.0 if action == "cheap" else 4.0


def _engine(*, lambda_0: float = 0.5) -> UCTSearchEngine:
    return UCTSearchEngine(
        evaluator=NeverCalledEvaluator(),
        scorer=CostAwareUCTScorer(lambda_0=lambda_0),
        cost_model=ReuseAwareCostModel(
            [ClusterDef("bench", weight=1.0, base_cost=1.0)],
            item_cost=_item_cost,
        ),
        exploration_constant=1.4,
        random_seed=7,
    )


def _warmed_result(*, context: SearchContext | None = None):
    active_context = context or SearchContext(total_evaluations=2)
    return _engine().search(
        OneSlotState(),
        max_iterations=0,
        warm_start=WarmStartBundle(
            context=active_context,
            root_child_seeds=(
                RootChildSeed("cheap", OneSlotState("cheap"), 0.25),
                RootChildSeed("expensive", OneSlotState("expensive"), 0.75),
            ),
        ),
    )


def _root_actions(payload: dict) -> dict[str, dict]:
    self_path = payload["nodes"][0]
    return {item["action"]: item for item in self_path["actions"]}


class TestSearchSignalExport(unittest.TestCase):
    def test_standard_scorer_fails_closed_on_non_finite_terms(self) -> None:
        for invalid_lambda in (float("nan"), float("inf")):
            with (
                self.subTest(lambda_0=invalid_lambda),
                self.assertRaisesRegex(ValueError, "lambda_0"),
            ):
                CostAwareUCTScorer(lambda_0=invalid_lambda)

        scorer = CostAwareUCTScorer(lambda_0=0.5)
        valid = {
            "parent_visits": 2,
            "child_visits": 1,
            "child_q": 0.5,
            "marginal_cost": 1.0,
            "exploration_constant": 1.4,
            "iteration": 4,
        }
        for field in ("child_q", "marginal_cost", "exploration_constant"):
            for invalid in (float("nan"), float("inf")):
                with self.subTest(field=field, invalid=invalid):
                    arguments = {**valid, field: invalid}
                    with self.assertRaisesRegex(ValueError, field):
                        scorer.decompose(**arguments)

        for field in ("parent_visits", "child_visits", "iteration"):
            for invalid in (True, 1.5):
                with self.subTest(field=field, invalid=invalid):
                    arguments = {**valid, field: invalid}
                    with self.assertRaisesRegex(ValueError, field):
                        scorer.decompose(**arguments)

    def test_exports_exact_uct_and_agent_uct_decomposition(self) -> None:
        engine = _engine()
        result = _warmed_result()

        payload = engine.export_search_signals(result, iteration=4)
        json.dumps(payload, allow_nan=False)

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["iteration"], 4)
        self.assertEqual(payload["semantics"]["signal_scope"], "search_snapshot")
        self.assertFalse(payload["semantics"]["is_historical_selection_trace"])
        self.assertFalse(payload["semantics"]["is_trajectory_signal"])
        self.assertFalse(payload["semantics"]["is_training_advantage"])

        node = payload["nodes"][0]
        self.assertEqual(node["path"], [])
        self.assertEqual(node["parent_visits"], 2)
        self.assertEqual(
            node["selection_mode"],
            "agent_uct_argmax_if_node_reached",
        )
        self.assertEqual(node["agent_uct_snapshot_argmax_action"], "cheap")
        self.assertEqual(node["uct_only_snapshot_argmax_action"], "expensive")
        actions = _root_actions(payload)

        exploration = 1.4 * math.sqrt(math.log(3.0) / 2.0)
        cheap = actions["cheap"]
        self.assertEqual(cheap["child_visits"], 1)
        self.assertEqual(cheap["child_q_raw"], 0.25)
        self.assertAlmostEqual(cheap["exploration_bonus"], exploration)
        self.assertEqual(cheap["marginal_cost"], 1.0)
        self.assertEqual(cheap["lambda_t"], 0.25)
        self.assertEqual(cheap["decayed_cost_penalty"], 0.25)
        self.assertAlmostEqual(cheap["uct_score"], 0.25 + exploration)
        self.assertAlmostEqual(
            cheap["agent_uct_selection_score"],
            0.25 + exploration - 0.25,
        )
        self.assertEqual(cheap["visit_policy_probability"], 0.5)
        self.assertTrue(cheap["selected_by_agent_uct_snapshot_argmax"])
        self.assertFalse(cheap["selected_by_uct_only_snapshot_argmax"])

        expensive = actions["expensive"]
        self.assertEqual(expensive["child_q_raw"], 0.75)
        self.assertEqual(expensive["marginal_cost"], 4.0)
        self.assertEqual(expensive["decayed_cost_penalty"], 1.0)
        self.assertAlmostEqual(expensive["uct_score"], 0.75 + exploration)
        self.assertAlmostEqual(
            expensive["agent_uct_selection_score"],
            0.75 + exploration - 1.0,
        )
        self.assertFalse(expensive["selected_by_agent_uct_snapshot_argmax"])
        self.assertTrue(expensive["selected_by_uct_only_snapshot_argmax"])

    def test_bipartite_reuse_edges_only_change_cost_side_of_selection(self) -> None:
        batch = EvaluationBatch(
            (ClusterBatch("bench", ("q0", "q1")),),
        )
        context = SearchContext(
            current_batch=batch,
            total_evaluations=2,
        )
        engine = _engine(lambda_0=1.0)
        result = engine.search(
            OneSlotState(),
            max_iterations=0,
            warm_start=WarmStartBundle(
                context=context,
                root_child_seeds=(
                    RootChildSeed("cheap", OneSlotState("cheap"), 0.5),
                    RootChildSeed("expensive", OneSlotState("expensive"), 0.5),
                ),
            ),
        )

        cold = engine.export_search_signals(result, iteration=4)
        self.assertEqual(
            cold["context_current_batch_cluster_sizes"],
            [{"cluster_id": "bench", "item_count": 2}],
        )
        before_stats = {
            action: (signal["child_visits"], signal["child_q_raw"])
            for action, signal in _root_actions(cold).items()
        }

        evaluator_report = EvaluationResult(
            reward=0.5,
            materialized_keys={
                (("cheap",), "bench", "q0"),
                (("cheap",), "bench", "q1"),
            },
        )
        context.materialized_keys.update(evaluator_report.materialized_keys)
        warm = engine.export_search_signals(result, iteration=4)

        cold_actions = _root_actions(cold)
        warm_actions = _root_actions(warm)
        self.assertEqual(cold_actions["cheap"]["marginal_cost"], 1.0)
        self.assertEqual(warm_actions["cheap"]["marginal_cost"], 0.0)
        self.assertEqual(cold_actions["cheap"]["decayed_cost_penalty"], 0.5)
        self.assertEqual(warm_actions["cheap"]["decayed_cost_penalty"], 0.0)
        self.assertGreater(
            warm_actions["cheap"]["agent_uct_selection_score"],
            cold_actions["cheap"]["agent_uct_selection_score"],
        )
        self.assertEqual(
            {
                action: (signal["child_visits"], signal["child_q_raw"])
                for action, signal in warm_actions.items()
            },
            before_stats,
        )
        self.assertEqual(result.root_node.visit_count, 2)
        self.assertEqual(result.root_node.q_value, 0.5)

    def test_unexpanded_actions_are_visible_without_claiming_score_was_used(self) -> None:
        result = _engine().search(OneSlotState(), max_iterations=0)

        with self.assertRaisesRegex(TypeError, "iteration"):
            _engine().export_search_signals(result)  # type: ignore[call-arg]

        payload = _engine().export_search_signals(result, iteration=1)

        node = payload["nodes"][0]
        self.assertEqual(node["selection_mode"], "random_expansion")
        self.assertIsNone(node["agent_uct_snapshot_argmax_action"])
        self.assertIsNone(node["uct_only_snapshot_argmax_action"])
        self.assertFalse(node["visit_policy_defined"])
        self.assertEqual(
            [action["action"] for action in node["actions"]],
            ["cheap", "expensive"],
        )
        self.assertTrue(all(not action["expanded"] for action in node["actions"]))
        self.assertTrue(
            all(
                not action["eligible_for_engine_argmax_if_node_reached"]
                for action in node["actions"]
            )
        )

    def test_walks_and_exports_an_arbitrary_deeper_actionable_node(self) -> None:
        class TwoLevelState:
            def __init__(self, path: tuple[str, ...] = ()) -> None:
                self.path = path

            def is_terminal(self) -> bool:
                return len(self.path) == 2

            def available_actions(self) -> list[Hashable]:
                if not self.path:
                    return ["left", "right"]
                return ["x", "y"] if len(self.path) == 1 else []

            def child(self, action: Hashable) -> TwoLevelState:
                if action not in self.available_actions():
                    raise ValueError(action)
                return TwoLevelState((*self.path, str(action)))

            def state_key(self) -> Hashable:
                return self.path

            def pretty(self) -> str:
                return str(self.path)

        context = SearchContext(total_evaluations=2)
        root = TreeNode(TwoLevelState())
        left = root.add_child("left", TwoLevelState(("left",)))
        x = left.add_child("x", TwoLevelState(("left", "x")))
        y = left.add_child("y", TwoLevelState(("left", "y")))
        for node, reward in ((x, 0.2), (y, 0.8)):
            node.update(reward)
            left.update(reward)
            root.update(reward)
        result = SearchResult(
            best_state=TwoLevelState(("left", "y")),
            best_reward=0.8,
            root_node=root,
            iterations=0,
            total_evaluations=2,
            total_cost=0.0,
            context=context,
        )

        payload = _engine().export_search_signals(result, iteration=2)
        deeper = next(node for node in payload["nodes"] if node["path"] == ["left"])

        self.assertEqual(deeper["parent_visits"], 2)
        self.assertEqual(deeper["parent_q_raw"], 0.5)
        self.assertEqual(
            deeper["selection_mode"],
            "agent_uct_argmax_if_node_reached",
        )
        self.assertEqual(deeper["agent_uct_snapshot_argmax_action"], "y")
        self.assertEqual(
            [(item["action"], item["child_visits"], item["child_q_raw"])
             for item in deeper["actions"]],
            [("x", 1, 0.2), ("y", 1, 0.8)],
        )

    def test_custom_value_encoder_keeps_structured_actions_json_safe(self) -> None:
        @dataclass(frozen=True)
        class Action:
            name: str

        class StructuredState:
            def __init__(self, action: Action | None = None) -> None:
                self.action = action

            def is_terminal(self) -> bool:
                return self.action is not None

            def available_actions(self) -> list[Hashable]:
                return [Action("a")] if self.action is None else []

            def child(self, action: Hashable) -> StructuredState:
                assert isinstance(action, Action)
                return StructuredState(action)

            def state_key(self) -> Hashable:
                return () if self.action is None else (self.action,)

            def pretty(self) -> str:
                return str(self.state_key())

        action = Action("a")
        engine = UCTSearchEngine(
            evaluator=NeverCalledEvaluator(),
            scorer=CostAwareUCTScorer(),
            cost_model=ReuseAwareCostModel([ClusterDef("bench")]),
        )
        result = engine.search(
            StructuredState(),
            max_iterations=0,
            warm_start=WarmStartBundle(
                context=SearchContext(total_evaluations=1),
                root_child_seeds=(
                    RootChildSeed(action, StructuredState(action), 1.0),
                ),
            ),
        )

        payload = engine.export_search_signals(
            result,
            iteration=1,
            encode_value=lambda value: (
                {"action": value.name} if isinstance(value, Action) else value
            ),
        )

        self.assertEqual(payload["nodes"][0]["actions"][0]["action"], {"action": "a"})
        json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
