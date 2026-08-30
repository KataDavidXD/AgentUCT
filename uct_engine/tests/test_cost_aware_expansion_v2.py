"""TDD contract for explicit, globally scaled AgentUCT cost sites."""

from __future__ import annotations

import math
import random
import unittest
from collections.abc import Hashable

from uct_engine import (
    CostAwareUCTScorer,
    CostModel,
    CostPenaltySchedule,
    EvaluationResult,
    Evaluator,
    SearchContext,
    SearchState,
    UCTSearchEngine,
)


class _OneStepState:
    ACTIONS = ("expensive", "cheap")

    def __init__(self, action: str | None = None) -> None:
        self.action = action

    def is_terminal(self) -> bool:
        return self.action is not None

    def available_actions(self) -> list[Hashable]:
        return [] if self.is_terminal() else list(self.ACTIONS)

    def child(self, action: Hashable) -> _OneStepState:
        if action not in self.ACTIONS:
            raise ValueError(f"unknown action: {action!r}")
        return _OneStepState(str(action))

    def state_key(self) -> Hashable:
        return () if self.action is None else (self.action,)

    def pretty(self) -> str:
        return repr(self.state_key())


class _ConstantEvaluator(Evaluator):
    def evaluate(
        self,
        state: SearchState,
        context: SearchContext,
    ) -> EvaluationResult:
        del context
        return EvaluationResult(
            reward=1.0,
            total_cost=1.0,
            materialized_keys={state.state_key()},
        )


class _RewardByActionEvaluator(Evaluator):
    def evaluate(
        self,
        state: SearchState,
        context: SearchContext,
    ) -> EvaluationResult:
        del context
        action = next(iter(tuple(state.state_key())))
        return EvaluationResult(
            reward=1.0 if action == "expensive" else 0.9,
            total_cost=1.0,
            materialized_keys={state.state_key()},
        )


class _ActionCost(CostModel):
    def marginal_cost(
        self,
        state: SearchState,
        action: Hashable,
        context: SearchContext,
    ) -> float:
        del state, context
        return 10.0 if action == "expensive" else 0.0


class TestExplicitCostSites(unittest.TestCase):
    def test_constant_beta_uses_one_fixed_global_cost_scale(self) -> None:
        scorer = CostAwareUCTScorer(
            lambda_0=0.8,
            schedule=CostPenaltySchedule.CONSTANT,
            cost_aware_selection=True,
            cost_aware_expansion=True,
            cost_scale=4.0,
        )

        terms = scorer.decompose(
            parent_visits=3,
            child_visits=1,
            child_q=0.4,
            marginal_cost=2.0,
            exploration_constant=0.0,
            iteration=16,
        )

        self.assertEqual(terms.lambda_t, 0.8)
        self.assertEqual(terms.cost_scale, 4.0)
        self.assertEqual(terms.normalized_marginal_cost, 0.5)
        self.assertEqual(terms.selection_cost_coefficient, 0.8)
        self.assertEqual(terms.decayed_cost_penalty, 0.4)
        self.assertAlmostEqual(terms.selection_score, 0.0)
        self.assertEqual(scorer.expansion_cost_coefficient(16), 0.8)

    def test_selection_can_be_disabled_without_changing_raw_uct_terms(self) -> None:
        scorer = CostAwareUCTScorer(
            lambda_0=2.0,
            schedule="constant",
            cost_aware_selection=False,
            cost_aware_expansion=True,
            cost_scale=5.0,
        )

        terms = scorer.decompose(
            parent_visits=3,
            child_visits=1,
            child_q=0.4,
            marginal_cost=10.0,
            exploration_constant=0.0,
            iteration=4,
        )

        self.assertEqual(terms.lambda_t, 2.0)
        self.assertEqual(terms.selection_cost_coefficient, 0.0)
        self.assertEqual(terms.decayed_cost_penalty, 0.0)
        self.assertEqual(terms.selection_score, terms.uct_score)
        self.assertEqual(scorer.expansion_cost_coefficient(4), 2.0)

    def test_expansion_cost_tilts_the_uniform_proposal_and_is_audited(self) -> None:
        records = []
        engine = UCTSearchEngine(
            evaluator=_ConstantEvaluator(),
            scorer=CostAwareUCTScorer(
                lambda_0=50.0,
                schedule="constant",
                cost_aware_expansion=True,
                cost_scale=10.0,
            ),
            cost_model=_ActionCost(),
            exploration_constant=0.0,
            random_seed=7,
        )

        result = engine.search(
            _OneStepState(),
            max_iterations=1,
            on_iteration=records.append,
        )

        self.assertEqual(result.best_state.state_key(), ("cheap",))
        self.assertEqual(len(records), 1)
        audit = records[0].algorithm_audit
        self.assertIsNotNone(audit)
        assert audit is not None
        choice = audit.expansion_choice
        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertEqual(choice.chosen_action, "cheap")
        self.assertEqual(choice.sampling_mode, "cost_tilted_uniform")
        self.assertEqual(choice.cost_coefficient, 50.0)
        candidates = {
            candidate.action: candidate for candidate in choice.cost_candidates
        }
        self.assertEqual(set(candidates), {"expensive", "cheap"})
        self.assertEqual(candidates["expensive"].marginal_cost, 10.0)
        self.assertEqual(candidates["expensive"].normalized_marginal_cost, 1.0)
        self.assertEqual(candidates["expensive"].cost_scale, 10.0)
        self.assertAlmostEqual(candidates["expensive"].log_bias, -50.0)
        self.assertAlmostEqual(
            math.exp(candidates["cheap"].final_logprob)
            + math.exp(candidates["expensive"].final_logprob),
            1.0,
        )
        self.assertTrue(candidates["cheap"].selected)

    def test_expansion_opt_out_preserves_the_historical_seeded_choice(self) -> None:
        expected = random.Random(42).choice(list(_OneStepState.ACTIONS))
        records = []
        engine = UCTSearchEngine(
            evaluator=_ConstantEvaluator(),
            scorer=CostAwareUCTScorer(
                lambda_0=50.0,
                schedule="constant",
                cost_aware_expansion=False,
                cost_scale=10.0,
            ),
            cost_model=_ActionCost(),
            exploration_constant=0.0,
            random_seed=42,
        )

        result = engine.search(
            _OneStepState(),
            max_iterations=1,
            on_iteration=records.append,
        )

        self.assertEqual(result.best_state.state_key(), (expected,))
        choice = records[0].algorithm_audit.expansion_choice
        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertEqual(choice.sampling_mode, "uniform")
        self.assertEqual(choice.cost_coefficient, 0.0)
        self.assertEqual(choice.cost_candidates, ())

    def test_selection_audit_exposes_cost_free_argmax_and_cost_treatment(self) -> None:
        records = []
        engine = UCTSearchEngine(
            evaluator=_RewardByActionEvaluator(),
            scorer=CostAwareUCTScorer(
                lambda_0=1.0,
                schedule="constant",
                cost_aware_selection=True,
                cost_aware_expansion=False,
                cost_scale=10.0,
            ),
            cost_model=_ActionCost(),
            exploration_constant=0.0,
            random_seed=3,
        )

        engine.search(
            _OneStepState(),
            max_iterations=3,
            on_iteration=records.append,
        )

        step = records[-1].algorithm_audit.selection_steps[0]
        candidates = {candidate.action: candidate for candidate in step.candidates}
        self.assertEqual(step.selected_action, "cheap")
        self.assertGreater(
            candidates["expensive"].uct_score,
            candidates["cheap"].uct_score,
        )
        self.assertLess(
            candidates["expensive"].score,
            candidates["cheap"].score,
        )
        self.assertEqual(candidates["expensive"].normalized_marginal_cost, 1.0)
        self.assertEqual(candidates["expensive"].cost_scale, 10.0)
        self.assertEqual(candidates["expensive"].cost_coefficient, 1.0)
        self.assertTrue(step.cost_penalty_changed_argmax)

    def test_signal_snapshot_labels_cost_tilted_expansion_and_global_scale(self) -> None:
        engine = UCTSearchEngine(
            evaluator=_ConstantEvaluator(),
            scorer=CostAwareUCTScorer(
                lambda_0=1.0,
                schedule="constant",
                cost_aware_selection=True,
                cost_aware_expansion=True,
                cost_scale=10.0,
            ),
            cost_model=_ActionCost(),
            exploration_constant=0.0,
            random_seed=1,
        )
        result = engine.search(_OneStepState(), max_iterations=0)

        payload = engine.export_search_signals(result, iteration=16)

        node = payload["nodes"][0]
        self.assertEqual(node["selection_mode"], "cost_tilted_expansion")
        actions = {item["action"]: item for item in node["actions"]}
        self.assertEqual(actions["expensive"]["normalized_marginal_cost"], 1.0)
        self.assertEqual(actions["expensive"]["cost_scale"], 10.0)
        self.assertEqual(actions["expensive"]["selection_cost_coefficient"], 1.0)
        self.assertEqual(actions["expensive"]["expansion_cost_coefficient"], 1.0)
        self.assertEqual(actions["expensive"]["expansion_log_bias"], -1.0)
        self.assertTrue(payload["semantics"]["cost_aware_selection"])
        self.assertTrue(payload["semantics"]["cost_aware_expansion"])
        self.assertEqual(payload["semantics"]["cost_scale"], 10.0)

    def test_cost_site_flags_and_scale_fail_closed(self) -> None:
        for invalid in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(cost_scale=invalid), self.assertRaisesRegex(
                ValueError, "cost_scale"
            ):
                CostAwareUCTScorer(cost_scale=invalid)

        with self.assertRaisesRegex(TypeError, "cost_aware_selection"):
            CostAwareUCTScorer(cost_aware_selection=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "cost_aware_expansion"):
            CostAwareUCTScorer(cost_aware_expansion=1)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
