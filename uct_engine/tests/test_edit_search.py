"""Allocation, feedback boundaries, and actual-cost reporting for program edits."""
import math
import random
import unittest

from uct_engine.edit_search import (
    ChargedFailure, EditNode, EditSearchConfig, Evaluation, Proposal,
    _select_parent, aflow_probabilities, progressive_width, run_edit_search,
)


class TestEditAllocation(unittest.TestCase):
    def test_progressive_width_is_the_published_bounded_square_root(self):
        for visits, width in ((0, 1), (1, 1), (2, 2), (4, 2), (5, 3),
                              (49, 7), (50, 8), (64, 8), (10000, 8)):
            self.assertEqual(progressive_width(visits), width)
        for invalid in (-1, 1.2, True):
            with self.assertRaises(ValueError):
                progressive_width(invalid)

    def test_original_aflow_uses_own_score_mixture(self):
        self.assertEqual(aflow_probabilities([0.4] * 4), (0.25,) * 4)
        probabilities = aflow_probabilities([1, 0])
        self.assertAlmostEqual(probabilities[0], 0.15 + 0.7 / (1 + math.exp(-20)))
        self.assertAlmostEqual(sum(probabilities), 1)
        nodes = [EditNode(i, None, i, i / 10, 0, reward_sum=100 - i)
                 for i in range(8)]
        nodes[7].depth = 12
        nodes[6].children = list(range(8))
        # Only top-four eligible *own* scores enter the AFlow draw. The huge
        # backed-up means on low-score nodes must not affect that pool.
        for seed in range(25):
            parent, _ = _select_parent(nodes, lambda _: 0,
                                      EditSearchConfig(c_ref=1, strategy="aflow_original"), random.Random(seed))
            self.assertIn(parent.id, (2, 3, 4, 5))

    def test_cost_pricing_changes_a_choice_below_the_root(self):
        nodes = [
            EditNode(0, None, 0, 0.1, 0, 6, 2.5, [1, 2, 3]),
            EditNode(1, 0, 1, 1.0, 1, 3, 2.4, [4, 5]),
            EditNode(2, 0, 2, 0.0, 1, reward_sum=0),
            EditNode(3, 0, 3, 0.0, 1, reward_sum=0),
            EditNode(4, 1, 4, 0.9, 2, reward_sum=0.9),
            EditNode(5, 1, 5, 0.5, 2, reward_sum=0.5),
        ]
        forecast = lambda artifact: 10 if artifact == 4 else 0
        for strategy, expected in (("agentuct", 5), ("uct", 4)):
            parent, path = _select_parent(
                nodes, forecast, EditSearchConfig(strategy=strategy, exploration=0,
                                                  beta=2, c_ref=1), random.Random(11))
            self.assertEqual(parent.id, expected)
            self.assertEqual(path, (0, 1, expected))

    def test_tree_stops_expanding_at_depth_twelve(self):
        # Each node has filled its eight slots. All depth-12 leaves are
        # terminal even when their visits would permit further widening.
        nodes = [EditNode(0, None, "root", 0.5, 11, 64, 32, list(range(1, 9)))]
        nodes.extend(EditNode(i, 0, i, 0.5, 12, reward_sum=0.5) for i in range(1, 9))
        for strategy in ("agentuct", "uct", "aflow_original"):
            parent, _ = _select_parent(nodes, lambda _: 0,
                                      EditSearchConfig(c_ref=1, strategy=strategy), random.Random(11))
            self.assertIsNone(parent)


class TestEditFeedbackAndBills(unittest.TestCase):
    def test_crossing_completes_and_earlier_snapshot_excludes_future_state(self):
        evaluated = []
        observed_parents = []

        def propose(parent):
            observed_parents.append(parent["version"])
            return Proposal({"version": parent["version"] + 1}, 2)

        def evaluate(candidate):
            evaluated.append(candidate["version"])
            return Evaluation(0.2 + candidate["version"] / 10, 3)

        result = run_edit_search(
            {"version": 0}, propose=propose, evaluate=evaluate,
            # A deliberately enormous forecast cannot block either callback.
            cost_forecast=lambda _: 1_000_000, config=EditSearchConfig(c_ref=100),
            target_cost=7, max_evaluations=10)
        self.assertEqual(evaluated, [0, 1])
        self.assertEqual(observed_parents, [0])
        self.assertEqual(result.total_cost, 8)
        self.assertEqual(result.stop_reason, "target_reached")
        self.assertEqual(result.nodes[0].visits, 2)
        self.assertAlmostEqual(result.nodes[0].reward_sum, 0.5)
        retained = result.last_affordable_snapshot(7)
        self.assertEqual(retained.total_cost, 3)
        self.assertEqual(len(retained.nodes), 1)
        self.assertEqual(retained.nodes[0].visits, 1)
        self.assertEqual(retained.nodes[0].reward_sum, 0.2)
        self.assertEqual(retained.nodes[0].children, [])
        self.assertIsNone(result.last_affordable_snapshot(2))
        result.nodes[0].artifact["version"] = 99
        retained.nodes[0].artifact["version"] = 88
        self.assertEqual(result.snapshots[0].nodes[0].artifact["version"], 0)
        self.assertEqual(result.last_affordable_snapshot(7).nodes[0].artifact["version"], 0)

    def test_initial_panel_runs_even_if_it_alone_crosses_the_target(self):
        result = run_edit_search(
            "initial", propose=lambda _: self.fail("no edit is needed"),
            evaluate=lambda _: Evaluation(0.5, 10), cost_forecast=lambda _: 100,
            config=EditSearchConfig(c_ref=1), target_cost=0)
        self.assertEqual(result.total_cost, 10)
        self.assertIsNone(result.last_affordable_snapshot(0))

    def test_rejected_and_repaired_proposals_keep_every_charge(self):
        proposals = iter([Proposal(None, 2, False, "invalid program"),
                          Proposal("valid", 4)])
        evaluations = []

        def evaluate(artifact):
            evaluations.append(artifact)
            return Evaluation(0.5, 3)

        result = run_edit_search(
            "initial", propose=lambda _: next(proposals), evaluate=evaluate,
            cost_forecast=lambda _: 1, config=EditSearchConfig(c_ref=1), max_evaluations=2)
        self.assertEqual(evaluations, ["initial", "valid"])
        self.assertEqual(result.total_cost, 12)
        self.assertEqual([a.charged_cost for a in result.attempts], [3, 2, 7])
        self.assertEqual(result.nodes[0].visits, 2)
        self.assertEqual(len(result.snapshots), 2)
        self.assertFalse(result.attempts[1].proposal.accepted)

    def test_incomplete_feedback_keeps_bills_without_backup(self):
        panels = iter([Evaluation(0.8, 3), Evaluation(None, 9, False, "unfinished panel")])
        result = run_edit_search(
            "initial", propose=lambda _: Proposal("candidate", 2),
            evaluate=lambda _: next(panels), cost_forecast=lambda _: 1,
            config=EditSearchConfig(c_ref=1))
        self.assertEqual(result.total_cost, 14)
        self.assertEqual(result.stop_reason, "incomplete_evaluation")
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(result.nodes[0].visits, 1)
        self.assertEqual(len(result.snapshots), 1)
        self.assertEqual(result.last_affordable_snapshot(14).total_cost, 3)

    def test_charged_failure_reports_only_the_failing_callback_bill(self):
        def evaluate(candidate):
            if candidate == "candidate":
                raise ChargedFailure("transport stopped panel", 7)
            return Evaluation(0.5, 3)

        result = run_edit_search(
            "initial", propose=lambda _: Proposal("candidate", 2),
            evaluate=evaluate, cost_forecast=lambda _: 1, config=EditSearchConfig(c_ref=1))
        self.assertEqual(result.total_cost, 12)
        self.assertEqual(result.attempts[-1].evaluation.charged_cost, 7)
        self.assertEqual(result.nodes[0].visits, 1)

    def test_completed_execution_errors_still_back_up_raw_zero_reward(self):
        result = run_edit_search(
            "initial", propose=lambda _: Proposal("failed-program", 2),
            evaluate=lambda p: Evaluation(0.8 if p == "initial" else 0.0, 5,
                                          failure="" if p == "initial" else "execution errors"),
            cost_forecast=lambda _: 100, config=EditSearchConfig(c_ref=1), max_evaluations=2)
        self.assertEqual(result.total_cost, 12)
        self.assertEqual(result.nodes[0].visits, 2)
        self.assertEqual(result.nodes[0].reward_sum, 0.8)
        self.assertEqual(result.nodes[1].own_score, 0)
        self.assertEqual(len(result.snapshots), 2)

    def test_rejection_limit_has_no_fabricated_candidate_or_snapshot(self):
        result = run_edit_search(
            "initial", propose=lambda _: Proposal(None, 2, False, "invalid"),
            evaluate=lambda _: Evaluation(0.5, 3), cost_forecast=lambda _: 1,
            config=EditSearchConfig(c_ref=1), max_proposals=2)
        self.assertEqual(result.total_cost, 7)
        self.assertEqual(result.stop_reason, "proposal_limit")
        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(len(result.snapshots), 1)

    def test_incomplete_panel_cannot_smuggle_a_partial_score(self):
        with self.assertRaises(ValueError):
            Evaluation(0.8, 5, complete=False)
        for cost in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                Proposal("candidate", cost)


if __name__ == "__main__":
    unittest.main()
