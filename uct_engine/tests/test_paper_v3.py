"""Numerical and execution checks for the publicly described v3 semantics."""
import math
import unittest

from uct_engine import CostAwareUCTScorer, EvaluationResult, SearchContext, UCTSearchEngine
from uct_engine.examples.rag_mock_example import (
    CompleteColdCostModel, MockEvaluator, RAGSearchState, resources,
)


class TestPaperV3(unittest.TestCase):
    def test_equation_five_uses_actual_child_count_and_fixed_coefficient(self):
        scorer = CostAwareUCTScorer(beta=2.0, c_ref=10.0)
        expected = 0.7 + 1.4 * math.sqrt(math.log(11) / 5) - 0.2 * 2
        for iteration in (1, 4, 100):
            self.assertAlmostEqual(scorer.score(10, 5, 0.7, 2, 1.4, iteration), expected)
        self.assertEqual(scorer.score(10, 0, 0, 1000, 1.4), math.inf)

    def test_invalid_price_inputs_are_rejected(self):
        for kwargs in ({"beta": math.nan}, {"beta": -1}, {"c_ref": 0},
                       {"c_ref": math.inf}, {"lambda_0": math.nan}):
            with self.assertRaises(ValueError):
                CostAwareUCTScorer(**kwargs)

    def test_complete_price_includes_suffix_after_a_cached_child(self):
        context = SearchContext()
        prefix = RAGSearchState(("sparse",))
        context.materialized_keys.update(resources(prefix))
        # The child is cached; four queries still need mean rewrite price 0.5
        # plus mean answer price 2.0 to complete the workflow.
        self.assertEqual(CompleteColdCostModel().marginal_cost(
            RAGSearchState(), "sparse", context), 10.0)

    def test_complete_replay_has_zero_cold_cost_and_same_raw_quality(self):
        state = RAGSearchState(("dense", "rewrite", "grounded"))
        context = SearchContext()
        evaluator = MockEvaluator()
        first = evaluator.evaluate(state, context)
        context.materialized_keys.update(first.materialized_keys)
        replay = evaluator.evaluate(state, context)
        self.assertGreater(first.total_cost, 0)
        self.assertEqual(replay.total_cost, 0)
        self.assertEqual(first.reward, replay.reward)

    def test_crossing_is_completed_and_retained_record_excludes_future_reward(self):
        class BilledEvaluator:
            calls = 0

            def evaluate(self, state, context):
                self.calls += 1
                return EvaluationResult(reward=self.calls / 10, total_cost=3)

        evaluator = BilledEvaluator()
        records = []
        result = UCTSearchEngine(
            evaluator=evaluator, scorer=CostAwareUCTScorer(beta=0),
            cost_model=CompleteColdCostModel(),
        ).search(RAGSearchState(), max_iterations=10, max_cost=5,
                 on_iteration=records.append)
        self.assertEqual(evaluator.calls, 2)
        self.assertEqual(result.total_cost, 6)
        self.assertEqual(result.best_reward, 0.2)
        affordable = [r for r in records
                      if r.algorithm_audit.cache_cost.total_cost_after_evaluation <= 5]
        self.assertEqual(len(affordable), 1)
        self.assertEqual(affordable[-1].best_reward, 0.1)


if __name__ == "__main__":
    unittest.main()
