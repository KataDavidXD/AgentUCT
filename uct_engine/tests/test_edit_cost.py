"""Receipt-only examples for the disclosed prequential edit-cost equations."""

from dataclasses import replace
import unittest

from uct_engine.edit_cost import (
    EditCostObservation,
    ParentQuestionCost,
    PrequentialEditCostModel,
)


class PrequentialEditCostTests(unittest.TestCase):
    def model(self):
        return PrequentialEditCostModel(
            prior_proposal_tokens=1768,
            prior_execution_call_tokens=531,
        )

    def observation(self, **changes):
        values = dict(
            observation_id="edit-1", request_size=100,
            first_input_tokens=50, first_output_tokens=20,
            first_cost=70, proposal_cost=140,
            execution_cost=120, parent_cold_cost=60,
        )
        values.update(changes)
        return EditCostObservation(**values)

    def forecast(self, model, panel=None):
        return model.forecast(
            graph="abc", prompt="de", operator_description="f",
            panel=panel or [ParentQuestionCost("q", 100, (20,))],
        )

    def test_empty_history_uses_explicit_study_priors_and_full_panel(self):
        result = self.forecast(self.model(), [
            ParentQuestionCost("missing", 100),
            ParentQuestionCost("observed-zero", 100, (0,)),
            ParentQuestionCost("paid", 100, (10, 7, 3)),
        ])
        self.assertEqual(result.request_size, 1806)
        self.assertEqual(result.first_proposal_cost, 1806 * .25 + 884)
        self.assertEqual(result.parent_cold_cost, 556 + 0 + 20)
        self.assertEqual(result.full_completion_cost, result.first_proposal_cost + 576)
        self.assertEqual(result.training_edits, 0)
        self.assertEqual([row.uses_fallback for row in result.question_costs], [True, False, False])
        self.assertEqual(result.question_costs[1].predicted_execution_cost, 0)

    def test_factors_average_episode_ratios_and_preserve_complete_forecast(self):
        model = self.model()
        model.observe(self.observation())
        model.observe(self.observation(
            observation_id="edit-2", request_size=200,
            first_input_tokens=50, first_output_tokens=40,
            first_cost=90, proposal_cost=90,
            execution_cost=90, parent_cold_cost=30,
        ))
        result = self.forecast(model)
        self.assertEqual(result.input_tokens_per_character, .375)
        self.assertEqual(result.first_output_tokens, 30)
        self.assertEqual(result.proposal_attempt_multiplier, 1.5)
        self.assertEqual(result.execution_multiplier, 2.5)
        self.assertEqual(result.first_proposal_cost, 707.25)
        self.assertEqual(result.proposal_cost, 1060.875)
        self.assertEqual(result.execution_cost, 50)
        self.assertEqual(result.full_completion_cost, 1110.875)

    def test_incomplete_and_heldout_receipts_do_not_fit_factors(self):
        model = self.model()
        before = self.forecast(model)
        for scope, complete in (("training", False), ("validation", True), ("test", True)):
            with self.subTest(scope=scope, complete=complete):
                self.assertFalse(model.observe(self.observation(scope=scope, complete=complete)))
                self.assertEqual(self.forecast(model), before)
        self.assertTrue(model.observe(self.observation()))
        self.assertEqual(model.training_edits, 1)

    def test_observation_ids_make_repeated_commits_idempotent(self):
        model = self.model()
        observation = self.observation()
        self.assertTrue(model.observe(observation))
        self.assertFalse(model.observe(observation))
        self.assertEqual(model.training_edits, 1)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            model.observe(replace(observation, execution_cost=121))

    def test_zero_denominators_are_guarded_and_ratios_are_uncapped(self):
        model = self.model()
        model.observe(self.observation(
            first_input_tokens=0, first_output_tokens=0,
            first_cost=0, proposal_cost=3,
            parent_cold_cost=0, execution_cost=5,
        ))
        result = self.forecast(model)
        self.assertEqual(result.first_proposal_cost, 1)
        self.assertEqual(result.proposal_attempt_multiplier, 3)
        self.assertEqual(result.execution_multiplier, 5)
        self.assertEqual(result.full_completion_cost, 3 + 5 * 20)

    def test_execution_fallback_keeps_the_prior_character_ratio(self):
        model = self.model()
        model.observe(self.observation(first_input_tokens=1000))
        result = self.forecast(model, [ParentQuestionCost("q", 100)])
        self.assertEqual(result.input_tokens_per_character, 10)
        self.assertEqual(result.parent_cold_cost, 100 * .25 + 531)
        self.assertEqual(result.question_costs[0].predicted_execution_cost, 556 * 2)

    def test_forecast_is_read_only_and_retains_failed_retry_bills(self):
        model = self.model()
        model.observe(self.observation())
        costs = [0, 12, 8]  # Restored/failed/retried calls are caller-provided actual bills.
        question = ParentQuestionCost("q", 100, costs)
        costs.append(1000)
        first = self.forecast(model, [question])
        self.assertEqual(first.parent_cold_cost, 20)
        self.assertEqual(first, self.forecast(model, [question]))
        self.assertEqual(model.training_edits, 1)

    def test_invalid_receipts_and_panel_coordinates_are_rejected(self):
        for value in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ParentQuestionCost("q", 10, (value,))
                with self.assertRaises(ValueError):
                    self.observation(execution_cost=value)
        with self.assertRaises(ValueError):
            ParentQuestionCost("q", -1)
        with self.assertRaises(ValueError):
            self.observation(request_size=0)
        with self.assertRaises(ValueError):
            self.model().forecast(graph="", prompt="", operator_description="", panel=[])
        with self.assertRaisesRegex(ValueError, "unique"):
            self.forecast(self.model(), [ParentQuestionCost("q", 1), ParentQuestionCost("q", 2)])

    def test_priors_are_required_and_caller_owned(self):
        with self.assertRaises(TypeError):
            PrequentialEditCostModel()
        model = PrequentialEditCostModel(
            prior_proposal_tokens=20, prior_execution_call_tokens=7,
            prior_input_tokens_per_character=.5,
        )
        result = self.forecast(model, [ParentQuestionCost("q", 10)])
        self.assertEqual(result.first_proposal_cost, 1806 * .5 + 10)
        self.assertEqual(result.parent_cold_cost, 12)


if __name__ == "__main__":
    unittest.main()
