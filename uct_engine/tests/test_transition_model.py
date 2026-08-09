"""TDD contract for environment-backed Agent-UCT transitions."""

from __future__ import annotations

import unittest
from collections.abc import Hashable

from uct_engine import (
    ClusterDef,
    CostAwareUCTScorer,
    CostPenaltySchedule,
    EvaluationResult,
    Evaluator,
    ReuseAwareCostModel,
    SearchContext,
    SearchState,
    TransitionModel,
    TransitionResult,
    UCTSearchEngine,
)


class EnvironmentState:
    """Two real transitions; ``child`` must never execute in this test."""

    ACTIONS = ("expand", "rollout")

    def __init__(self, path: tuple[str, ...] = ()) -> None:
        self.path = path

    def is_terminal(self) -> bool:
        return len(self.path) == len(self.ACTIONS)

    def available_actions(self) -> list[Hashable]:
        return [] if self.is_terminal() else [self.ACTIONS[len(self.path)]]

    def child(self, action: Hashable) -> EnvironmentState:
        raise AssertionError(f"SearchState.child executed for {action!r}")

    def reuse_child_key(self, action: Hashable) -> Hashable:
        return self.path + (str(action),)

    def state_key(self) -> Hashable:
        return self.path

    def pretty(self) -> str:
        return repr(self.path)


class RecordingEnvironment(TransitionModel):
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def transition(
        self,
        state: SearchState,
        action: Hashable,
        context: SearchContext,
    ) -> TransitionResult:
        del context
        path = tuple(state.state_key())
        action_id = str(action)
        self.calls.append((path, action_id))
        return TransitionResult(
            state=EnvironmentState(path + (action_id,)),
            realized_cost=2.0 if action_id == "expand" else 3.0,
            materialized_keys={
                (path + (action_id,), "hotpotqa", "q0"),
            },
        )


class RawRewardEvaluator(Evaluator):
    def __init__(self) -> None:
        self.keys_seen_at_entry: set[Hashable] | None = None

    def evaluate(
        self,
        state: SearchState,
        context: SearchContext,
    ) -> EvaluationResult:
        self.keys_seen_at_entry = set(context.materialized_keys)
        self.assert_terminal_path = tuple(state.state_key())
        return EvaluationResult(
            reward=0.75,
            total_cost=5.0,
            materialized_keys={
                (("expand", "rollout"), "hotpotqa", "q0"),
                ("terminal-evaluation", "hotpotqa", "q0"),
            },
        )


class PreviewState:
    def __init__(self) -> None:
        self.child_calls = 0
        self.preview_calls = 0

    def is_terminal(self) -> bool:
        return False

    def available_actions(self) -> list[Hashable]:
        return ["a"]

    def child(self, action: Hashable) -> PreviewState:
        del action
        self.child_calls += 1
        return self

    def reuse_child_key(self, action: Hashable) -> Hashable:
        self.preview_calls += 1
        return ("preview", str(action))

    def state_key(self) -> Hashable:
        return ()

    def pretty(self) -> str:
        return "preview"


class LegacyPreviewState(PreviewState):
    reuse_child_key = None  # type: ignore[assignment]


class TestEnvironmentTransitions(unittest.TestCase):
    def test_expansion_and_rollout_use_transition_model_and_merge_effects_once(
        self,
    ) -> None:
        environment = RecordingEnvironment()
        evaluator = RawRewardEvaluator()
        records = []
        engine = UCTSearchEngine(
            evaluator=evaluator,
            scorer=CostAwareUCTScorer(lambda_0=0.0),
            cost_model=ReuseAwareCostModel([ClusterDef("hotpotqa")]),
            transition_model=environment,
            random_seed=7,
        )

        result = engine.search(
            EnvironmentState(),
            max_iterations=1,
            on_iteration=records.append,
        )

        self.assertEqual(
            environment.calls,
            [((), "expand"), (("expand",), "rollout")],
        )
        self.assertEqual(evaluator.assert_terminal_path, ("expand", "rollout"))
        # Transition keys stay pending until the terminal evaluation succeeds.
        self.assertEqual(evaluator.keys_seen_at_entry, set())
        self.assertEqual(result.total_cost, 10.0)
        self.assertEqual(len(result.context.materialized_keys), 3)
        self.assertEqual(result.total_evaluations, 1)

        root = result.root_node
        expanded = root.children["expand"]
        self.assertEqual((root.visit_count, expanded.visit_count), (1, 1))
        self.assertEqual((root.q_value, expanded.q_value), (0.75, 0.75))
        self.assertNotIn("rollout", expanded.children)

        record = records[0]
        self.assertEqual(record.reward, 0.75)
        self.assertEqual(record.evaluation_cost, 10.0)
        self.assertEqual(record.transition_cost, 5.0)
        self.assertEqual(record.terminal_evaluation_cost, 5.0)
        audit = record.algorithm_audit
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(
            [step.phase for step in audit.transition_steps],
            ["expansion", "rollout"],
        )
        self.assertEqual(audit.cache_cost.transition_calls, 2)
        self.assertEqual(audit.cache_cost.transition_cost, 5.0)
        self.assertEqual(audit.cache_cost.terminal_evaluation_cost, 5.0)
        self.assertEqual(audit.cache_cost.evaluation_cost, 10.0)
        self.assertEqual(audit.cache_cost.materialized_keys_reported, 3)
        self.assertEqual(audit.cache_cost.materialized_keys_added, 3)

    def test_reuse_cost_preview_uses_side_effect_free_child_key(self) -> None:
        state = PreviewState()
        model = ReuseAwareCostModel([ClusterDef("hotpotqa")])

        cost = model.marginal_cost(state, "a", SearchContext())

        self.assertEqual(cost, 1.0)
        self.assertEqual(state.preview_calls, 1)
        self.assertEqual(state.child_calls, 0)

    def test_selection_preview_does_not_execute_an_environment_transition(
        self,
    ) -> None:
        environment = RecordingEnvironment()
        records = []
        engine = UCTSearchEngine(
            evaluator=RawRewardEvaluator(),
            scorer=CostAwareUCTScorer(lambda_0=0.5),
            cost_model=ReuseAwareCostModel([ClusterDef("hotpotqa")]),
            transition_model=environment,
            random_seed=7,
        )

        result = engine.search(
            EnvironmentState(),
            max_iterations=2,
            on_iteration=records.append,
        )

        # Iteration one realizes expansion+rollout. Iteration two previews the
        # existing root edge, then realizes only the previously rollout-only
        # child. A fourth call would prove selection executed the environment.
        self.assertEqual(
            environment.calls,
            [
                ((), "expand"),
                (("expand",), "rollout"),
                (("expand",), "rollout"),
            ],
        )
        second_audit = records[1].algorithm_audit
        self.assertIsNotNone(second_audit)
        assert second_audit is not None
        self.assertEqual(len(second_audit.selection_steps), 1)
        self.assertEqual(second_audit.transition_steps[0].phase, "expansion")
        self.assertEqual(result.root_node.q_value, 0.75)

    def test_legacy_state_without_preview_hook_still_falls_back_to_child(self) -> None:
        state = LegacyPreviewState()
        model = ReuseAwareCostModel([ClusterDef("hotpotqa")])

        cost = model.marginal_cost(state, "a", SearchContext())

        self.assertEqual(cost, 1.0)
        self.assertEqual(state.child_calls, 1)


class TestCostPenaltySchedules(unittest.TestCase):
    def test_constant_beta_is_not_decayed(self) -> None:
        scorer = CostAwareUCTScorer(
            lambda_0=0.8,
            schedule=CostPenaltySchedule.CONSTANT,
        )

        terms = scorer.decompose(
            parent_visits=3,
            child_visits=1,
            child_q=0.4,
            marginal_cost=2.0,
            exploration_constant=0.0,
            iteration=16,
        )

        self.assertEqual(terms.cost_schedule, "constant")
        self.assertEqual(terms.lambda_t, 0.8)
        self.assertEqual(terms.decayed_cost_penalty, 1.6)
        self.assertAlmostEqual(terms.selection_score, -1.2)

    def test_default_remains_inverse_sqrt_and_zero_is_vanilla_uct(self) -> None:
        default_terms = CostAwareUCTScorer(lambda_0=0.8).decompose(
            parent_visits=3,
            child_visits=1,
            child_q=0.4,
            marginal_cost=2.0,
            exploration_constant=0.0,
            iteration=16,
        )
        vanilla_terms = CostAwareUCTScorer(
            lambda_0=0.0,
            schedule="constant",
        ).decompose(
            parent_visits=3,
            child_visits=1,
            child_q=0.4,
            marginal_cost=2.0,
            exploration_constant=0.0,
            iteration=16,
        )

        self.assertEqual(default_terms.cost_schedule, "inverse_sqrt")
        self.assertEqual(default_terms.lambda_t, 0.2)
        self.assertEqual(default_terms.decayed_cost_penalty, 0.4)
        self.assertEqual(vanilla_terms.decayed_cost_penalty, 0.0)
        self.assertEqual(vanilla_terms.selection_score, vanilla_terms.uct_score)


if __name__ == "__main__":
    unittest.main()
