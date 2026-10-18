"""Actual bills and independent complete snapshots at retrospective cutoffs."""
import unittest

from uct_engine.accounting import (
    ActualCostLedger, CompletedSnapshot, last_affordable_snapshot,
)


class TestActualAccounting(unittest.TestCase):
    def test_all_search_work_is_charged_with_phases_kept_separate(self):
        ledger = ActualCostLedger("weighted_tokens")
        ledger.record("setup", 100, phase="common_preparation")
        ledger.record("proposal", 2, phase="proposer")
        ledger.record("failed", 3, phase="repair", failed=True)
        ledger.record("panel", 5)
        ledger.record("validation", 7, phase="validation")
        ledger.record("measurement", 11, phase="test_measurement")
        self.assertEqual(ledger.search_cost, 10)
        self.assertEqual(ledger.optimization_cost, 17)
        self.assertEqual(ledger.phase_cost("test_measurement"), 11)
        ledger.record("panel", 5)
        self.assertEqual(ledger.search_cost, 10)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            ledger.record("panel", 6)

    def test_crossing_reward_and_future_cache_cannot_enter_retained_copy(self):
        state = {"best": 0.2, "cache": ["old"], "model": {"observations": 1}}
        first = CompletedSnapshot.capture(state, actual_search_cost=3, completed_evaluations=1)
        state["best"] = 1.0
        state["cache"].append("future")
        state["model"]["observations"] = 2
        crossing = CompletedSnapshot.capture(state, actual_search_cost=9, completed_evaluations=2)
        history = [first, crossing]
        retained = last_affordable_snapshot(history, 5)
        self.assertEqual(retained.state, {"best": 0.2, "cache": ["old"],
                                          "model": {"observations": 1}})
        retained.state["cache"].append("caller edit")
        self.assertEqual(first.state["cache"], ["old"])
        self.assertEqual(crossing.state["best"], 1.0)

    def test_zero_cost_replays_retain_last_complete_equal_cost_state(self):
        snapshots = [CompletedSnapshot.capture(i, actual_search_cost=3, completed_evaluations=i)
                     for i in range(1, 4)]
        self.assertEqual(last_affordable_snapshot(snapshots, 3).state, 3)
        self.assertIsNone(last_affordable_snapshot(snapshots, 2))

    def test_incomplete_object_and_out_of_order_history_are_ineligible(self):
        with self.assertRaises(TypeError):
            last_affordable_snapshot([{"cost": 1}], 5)
        first = CompletedSnapshot.capture({}, actual_search_cost=3, completed_evaluations=1)
        later = CompletedSnapshot.capture({}, actual_search_cost=9, completed_evaluations=2)
        with self.assertRaises(ValueError):
            last_affordable_snapshot([later, first], 10)
        with self.assertRaises(ValueError):
            ActualCostLedger("tokens").record("invalid", -1)


if __name__ == "__main__":
    unittest.main()
