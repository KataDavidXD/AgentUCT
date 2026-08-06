"""Regression coverage for rollout-tail tree and reuse-ledger semantics."""

from __future__ import annotations

import unittest
from typing import Hashable

from uct_engine import (
    ClusterBatch,
    ClusterDef,
    CostAwareUCTScorer,
    EvaluationBatch,
    EvaluationResult,
    Evaluator,
    IterationRecord,
    ReuseAwareCostModel,
    SearchContext,
    SearchState,
    UCTSearchEngine,
)


class LinearState:
    """Three-level tree with one valid action at every level."""

    ACTIONS = ("a", "b", "c")

    def __init__(self, key: tuple[str, ...] = ()) -> None:
        self.key = key

    def is_terminal(self) -> bool:
        return len(self.key) == len(self.ACTIONS)

    def available_actions(self) -> list[Hashable]:
        return [] if self.is_terminal() else [self.ACTIONS[len(self.key)]]

    def child(self, action: Hashable) -> "LinearState":
        if action not in self.available_actions():
            raise ValueError(action)
        return LinearState(self.key + (str(action),))

    def state_key(self) -> Hashable:
        return self.key

    def pretty(self) -> str:
        return str(self.key)


class ChangingBatchProvider:
    """Use q0 once, then reuse q1 for the next two iterations."""

    def __init__(self) -> None:
        self.item_ids = ("q0", "q1", "q1")

    def draw(
        self,
        iteration: int,
        context: SearchContext,
    ) -> EvaluationBatch:
        del context
        return EvaluationBatch(
            clusters=(ClusterBatch("z", (self.item_ids[iteration - 1],)),),
        )


class RecordingCostModel(ReuseAwareCostModel):
    def __init__(self) -> None:
        super().__init__([ClusterDef("z", weight=1.0, base_cost=1.0)])
        self.calls: list[tuple[str, tuple[str, ...], str, float]] = []

    def marginal_cost(
        self,
        state: SearchState,
        action: Hashable,
        context: SearchContext,
    ) -> float:
        batch = context.current_batch
        assert batch is not None
        item_id = str(batch.clusters[0].item_ids[0])
        cost = super().marginal_cost(state, action, context)
        self.calls.append(
            (item_id, tuple(state.state_key()), str(action), cost)
        )
        return cost


class PrefixMaterializingEvaluator(Evaluator):
    """Return every prefix key produced by the terminal rollout."""

    def __init__(self, cost_model: RecordingCostModel) -> None:
        self.cost_model = cost_model
        self.evaluation_count = 0
        self.cost_call_counts_at_entry: list[int] = []

    def evaluate(
        self,
        state: SearchState,
        context: SearchContext,
    ) -> EvaluationResult:
        batch = context.current_batch
        assert batch is not None
        self.cost_call_counts_at_entry.append(len(self.cost_model.calls))
        self.evaluation_count += 1
        key = tuple(state.state_key())
        item_id = batch.clusters[0].item_ids[0]
        return EvaluationResult(
            reward=float(self.evaluation_count),
            materialized_keys={
                (key[:depth], "z", item_id)
                for depth in range(1, len(key) + 1)
            },
        )


class TestRolloutCacheSemantics(unittest.TestCase):
    def test_rollout_tail_stays_outside_tree_but_warms_future_cost(self) -> None:
        cost_model = RecordingCostModel()
        evaluator = PrefixMaterializingEvaluator(cost_model)
        records: list[IterationRecord] = []
        cost_call_counts_after_iteration: list[int] = []

        def capture_iteration(record: IterationRecord) -> None:
            records.append(record)
            cost_call_counts_after_iteration.append(len(cost_model.calls))

        engine = UCTSearchEngine(
            evaluator=evaluator,
            scorer=CostAwareUCTScorer(lambda_0=0.1),
            cost_model=cost_model,
            batch_provider=ChangingBatchProvider(),
            random_seed=42,
        )

        result = engine.search(
            LinearState(),
            max_iterations=3,
            on_iteration=capture_iteration,
        )

        root = result.root_node
        node_a = root.children["a"]
        node_b = node_a.children["b"]
        node_c = node_b.children["c"]

        # Iteration 1 expands only a; b/c are rollout-only. Iteration 2 first
        # creates b, and iteration 3 first creates c, so neither tail node
        # inherits reward or visits from an earlier rollout.
        self.assertEqual(
            (
                root.visit_count,
                node_a.visit_count,
                node_b.visit_count,
                node_c.visit_count,
            ),
            (3, 3, 2, 1),
        )
        self.assertEqual(
            (
                root.value_sum,
                node_a.value_sum,
                node_b.value_sum,
                node_c.value_sum,
            ),
            (6.0, 6.0, 5.0, 3.0),
        )

        terminal_key = ("a", "b", "c")
        self.assertEqual(
            [tuple(record.evaluated_state.state_key()) for record in records],
            [terminal_key] * 3,
        )
        self.assertEqual([record.reward for record in records], [1.0, 2.0, 3.0])
        self.assertEqual(
            [record.materialized_keys_added for record in records],
            [3, 3, 0],
        )
        self.assertEqual(result.best_state.state_key(), terminal_key)
        self.assertEqual(result.best_reward, 3.0)

        expected_keys = {
            (terminal_key[:depth], "z", item_id)
            for item_id in ("q0", "q1")
            for depth in range(1, len(terminal_key) + 1)
        }
        self.assertEqual(result.context.materialized_keys, expected_keys)

        # q1 is cold during iteration 2 selection. Evaluation then adds all
        # q1 tail keys, and iteration 3 sees those same-QID prefixes as free.
        self.assertEqual(
            cost_model.calls,
            [
                ("q1", (), "a", 1.0),
                ("q1", (), "a", 0.0),
                ("q1", ("a",), "b", 0.0),
            ],
        )

        # Cost is computed only before evaluation. Materializing q1 during an
        # evaluation does not trigger a same-iteration score recomputation.
        self.assertEqual(evaluator.cost_call_counts_at_entry, [0, 1, 3])
        self.assertEqual(cost_call_counts_after_iteration, [0, 1, 3])


if __name__ == "__main__":
    unittest.main()
