"""The disclosed proposal-history grouping rules, without a task runner."""

from dataclasses import replace
import json
import unittest

from uct_engine.proposal_history import ProposalFeedback, serialize_proposal_history


class ProposalHistoryTests(unittest.TestCase):
    def feedback(self, **changes):
        values = dict(score=None, modification="try a revision", execution_errors=("B", "A", "B"))
        values.update(changes)
        return ProposalFeedback(**values)

    def test_exact_errors_are_counted_in_first_seen_order(self):
        row = self.feedback(execution_errors=("B", "A", "B", "a", "A\n", "", "A", " A"))
        result = serialize_proposal_history([row])[0]
        self.assertEqual(result["execution_failures"], [
            {"error": "B", "count": 2},
            {"error": "A", "count": 2},
            {"error": "a", "count": 1},
            {"error": "A\n", "count": 1},
            {"error": "", "count": 1},
            {"error": " A", "count": 1},
        ])

    def test_evaluated_siblings_including_zero_scores_remain_separate(self):
        zero = self.feedback(score=0)
        scored = self.feedback(score=.75)
        result = serialize_proposal_history([zero, zero, scored, scored])
        self.assertEqual([row["score"] for row in result], [0, 0, .75, .75])
        self.assertEqual([row["occurrences"] for row in result], [1, 1, 1, 1])

    def test_only_consecutive_identical_unevaluated_rows_are_combined(self):
        first = self.feedback()
        other = self.feedback(modification="different revision")
        result = serialize_proposal_history([first, first, other, first, first, first])
        self.assertEqual([row["occurrences"] for row in result], [2, 1, 3])
        self.assertEqual([row["modification"] for row in result], [
            "try a revision", "different revision", "try a revision",
        ])

    def test_evaluated_row_breaks_an_unevaluated_group(self):
        row = self.feedback()
        result = serialize_proposal_history([row, replace(row, score=0), row])
        self.assertEqual([item["score"] for item in result], [None, 0, None])
        self.assertEqual([item["occurrences"] for item in result], [1, 1, 1])

    def test_grouping_compares_the_formatted_error_counts(self):
        first = self.feedback(execution_errors=("B", "B", "A"))
        same = self.feedback(execution_errors=("B", "A", "B"))
        different_first_seen_order = self.feedback(execution_errors=("A", "B", "B"))
        result = serialize_proposal_history([first, same, different_first_seen_order])
        self.assertEqual([row["occurrences"] for row in result], [2, 1])

    def test_modification_estimate_and_error_counts_prevent_grouping(self):
        first = self.feedback()
        for different in (
            replace(first, modification="try a revision "),
            replace(first, estimated_score=0),
            replace(first, execution_errors=("B", "A")),
            replace(first, execution_errors=("B", "a", "B")),
        ):
            with self.subTest(different=different):
                self.assertEqual(len(serialize_proposal_history([first, different])), 2)

    def test_output_is_json_compatible_and_does_not_mutate_input(self):
        errors = ["B", "B"]
        feedback = self.feedback(execution_errors=errors)
        errors.append("later mutation")
        result = serialize_proposal_history([feedback, feedback])
        self.assertEqual(json.loads(json.dumps(result)), result)
        result[0]["execution_failures"][0]["count"] = 999
        result[0]["occurrences"] = 999
        again = serialize_proposal_history([feedback, feedback])
        self.assertEqual(again[0]["execution_failures"], [{"error": "B", "count": 2}])
        self.assertEqual(again[0]["occurrences"], 2)
        self.assertEqual(feedback.execution_errors, ("B", "B"))

    def test_empty_history_and_iterators(self):
        self.assertEqual(serialize_proposal_history([]), [])
        self.assertEqual(len(serialize_proposal_history(iter([self.feedback()]))), 1)

    def test_invalid_scores_and_non_string_errors_are_rejected(self):
        for score in (True, -1, float("nan"), float("inf"), 1.1):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    self.feedback(score=score)
        with self.assertRaises(ValueError):
            self.feedback(execution_errors=(None,))
        with self.assertRaises(TypeError):
            serialize_proposal_history([{}])


if __name__ == "__main__":
    unittest.main()
