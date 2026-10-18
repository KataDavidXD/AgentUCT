"""Check completed-feedback pricing against small hand-computable grammars."""
import unittest

from uct_engine import SearchContext
from uct_engine.accounting import ActualCostLedger
from uct_engine.completion_cost import (
    HistoricalCompletionCostModel, OnlineResourcePrices, ResourceReceipt,
)


class State:
    EDGES = {(): ("a",), ("a",): ("b",), ("a", "b"): ("c",)}

    def __init__(self, path=()):
        self.path = path

    def state_key(self):
        return self.path

    def available_actions(self):
        return self.EDGES.get(self.path, ())

    def is_terminal(self):
        return not self.available_actions()

    def child(self, action):
        return type(self)(self.path + (action,))


class BranchingState(State):
    EDGES = {(): ("start",), ("start",): ("left", "right"),
             ("start", "right"): ("x", "y")}


def model(**kwargs):
    return HistoricalCompletionCostModel(
        ("q0", "q1"), {"a": 2, "b": 3, "c": 5},
        {"a": 20, "b": 20, "c": 20}, **kwargs)


class TestCompletionPrices(unittest.TestCase):
    def test_price_pooling_retains_one_prior_at_each_tier(self):
        prices = OnlineResourcePrices({"a": 10}, {"a": 20})
        receipt = ResourceReceipt("r0", "q0", ("a",), 4)
        prices.observe(receipt)
        prices.observe(receipt)
        self.assertEqual(prices.price("q0", ("a",)), 7)  # exact query-prefix
        self.assertEqual(prices.price("q1", ("a",)), 7)  # same prefix, another query
        self.assertEqual(prices.price("q1", ("other", "a")), 7)  # action fallback
        prices.observe(ResourceReceipt("r1", "q1", ("a",), 8))
        self.assertEqual(prices.price("q0", ("a",)), 7)
        self.assertAlmostEqual(prices.price("q2", ("a",)), 22 / 3)

    def test_caps_change_predictions_and_leave_actual_bill_intact(self):
        prices = OnlineResourcePrices({"a": 10}, {"a": 20})
        prices.observe(ResourceReceipt("r", "q0", ("a",), 100))
        ledger = ActualCostLedger("tokens")
        ledger.record("r", 100, failed=True)
        self.assertEqual(prices.price("q0", ("a",)), 15)
        self.assertEqual(ledger.search_cost, 100)

    def test_complete_and_next_use_same_samples_but_different_suffixes(self):
        context = SearchContext(materialized_keys={("q0", ("a",)), ("q1", ("a",))})
        estimator = model()
        full = estimator.forecast(State(), "a", context)
        next_cost = estimator.forecast(State(), "a", context, target="next_unpaid")
        self.assertEqual(full.sampled_paths, next_cost.sampled_paths)
        self.assertEqual(len(full.sampled_paths), 16)
        self.assertEqual(full.cost, 16)  # 2 queries * (3 + 5)
        self.assertEqual(next_cost.cost, 6)

    def test_next_skips_structural_zero_and_full_cache_replay_is_free(self):
        estimator = model(structural_zero=lambda _q, p: p[-1] == "b")
        context = SearchContext(materialized_keys={(q, ("a",)) for q in ("q0", "q1")})
        self.assertEqual(estimator.forecast(State(), "a", context).cost, 10)
        self.assertEqual(estimator.forecast(
            State(), "a", context, target="next_unpaid").cost, 10)
        context.materialized_keys.update({(q, ("a", "b", "c")) for q in ("q0", "q1")})
        self.assertEqual(estimator.forecast(State(), "a", context).cost, 0)

    def test_forecasting_does_not_advance_search_randomness(self):
        context = SearchContext()
        before = context.random.getstate()
        full = model().forecast(State(), "a", context)
        self.assertEqual(context.random.getstate(), before)
        self.assertEqual(full, model().forecast(State(), "a", context))

    def test_observed_zero_work_is_not_a_cache_publication_and_static_ignores_it(self):
        context = SearchContext(extra={"zero_work_resources": {
            (q, ("a", "b")) for q in ("q0", "q1")}})
        self.assertEqual(model().forecast(State(), "a", context).cost, 14)
        self.assertEqual(model(reference="static_full").forecast(State(), "a", context).cost, 20)
        self.assertEqual(context.materialized_keys, set())

    def test_static_full_freezes_prices_law_tape_and_empty_pricing_cache(self):
        estimator = model(reference="static_full")
        context = SearchContext()
        before = estimator.forecast(State(), "a", context)
        estimator.observe_completed("e", ("a", "b", "c"),
                                    (ResourceReceipt("r", "q0", ("a",), 0),))
        context.materialized_keys.add(("q0", ("a", "b", "c")))
        context.total_evaluations = 100
        self.assertEqual(before, estimator.forecast(State(), "a", context))
        self.assertEqual(before.cost, 20)
        self.assertIn(("q0", ("a", "b", "c")), context.materialized_keys)

    def test_only_complete_training_feedback_fits_prices(self):
        estimator = model()
        receipt = ResourceReceipt("r", "q0", ("a",), 8)
        estimator.observe_completed("partial", ("a", "b", "c"), (receipt,), complete=False)
        estimator.observe_completed("held-out", ("a", "b", "c"), (receipt,), scope="test")
        self.assertEqual(estimator.prices.price("q0", ("a",)), 2)
        estimator.observe_completed("complete", ("a", "b", "c"), (receipt,))
        estimator.observe_completed("complete", ("a", "b", "c"), (receipt,))
        self.assertEqual(estimator.prices.price("q0", ("a",)), 5)

    def test_conflicting_receipts_leave_a_completed_commit_unmodified(self):
        estimator = model()
        with self.assertRaisesRegex(ValueError, "conflicting"):
            estimator.observe_completed("e", ("a", "b", "c"), (
                ResourceReceipt("r", "q0", ("a",), 1),
                ResourceReceipt("r", "q0", ("a",), 2)))
        self.assertEqual(estimator.prices.price("q0", ("a",)), 2)

    def test_a_held_out_query_cannot_enter_training_price_pools(self):
        estimator = model()
        with self.assertRaisesRegex(ValueError, "fixed panel"):
            estimator.observe_completed("e", ("a", "b", "c"), (
                ResourceReceipt("held-out", "test-query", ("a",), 100),))
        self.assertEqual(estimator.prices.price("q0", ("a",)), 2)

    def test_uniform_next_actions_differ_from_uniform_terminal_weights(self):
        actions = ("start", "left", "right", "x", "y")
        kwargs = dict(query_ids=("q",), priors=dict.fromkeys(actions, 1),
                      caps=dict.fromkeys(actions, 10), samples=1000)
        uniform = HistoricalCompletionCostModel(reference="uniform", **kwargs)
        paths = uniform.forecast(BranchingState(), "start", SearchContext()).sampled_paths
        left_fraction = sum(p[1] == "left" for p in paths) / len(paths)
        self.assertGreater(left_fraction, 0.44)
        self.assertLess(left_fraction, 0.56)
        historical = HistoricalCompletionCostModel(**kwargs)
        for i in range(50):
            historical.observe_completed(i, ("start", "right", "x"))
        paths = historical.forecast(BranchingState(), "start", SearchContext()).sampled_paths
        self.assertGreater(sum(p[1:] == ("right", "x") for p in paths) / len(paths), 0.9)


if __name__ == "__main__":
    unittest.main()
