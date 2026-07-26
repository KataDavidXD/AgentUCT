"""Paper-boundary regression tests for the standalone Agent-UCT core."""
from __future__ import annotations

import unittest
from typing import Hashable

from uct_engine import (
    BatchProvider,
    ClusterBatch,
    ClusterDef,
    CostAwareUCTScorer,
    CostModel,
    EvaluationBatch,
    EvaluationResult,
    Evaluator,
    ReuseAwareCostModel,
    SearchContext,
    SearchState,
    UCTSearchEngine,
)


class OneActionState:
    """A one-slot search space used to expose the iteration contract."""

    def __init__(self, choice: str | None = None) -> None:
        self.choice = choice

    def is_terminal(self) -> bool:
        return self.choice is not None

    def available_actions(self) -> list[Hashable]:
        return ["only"] if self.choice is None else []

    def child(self, action: Hashable) -> "OneActionState":
        return OneActionState(str(action))

    def state_key(self) -> Hashable:
        return () if self.choice is None else (self.choice,)

    def pretty(self) -> str:
        return str(self.state_key())


class RecordingBatchProvider(BatchProvider):
    def __init__(self) -> None:
        self.batches: list[EvaluationBatch] = []

    def draw(self, iteration: int, context: SearchContext) -> EvaluationBatch:
        batch = EvaluationBatch(
            clusters=(ClusterBatch("z", (f"q{iteration}",)),),
        )
        self.batches.append(batch)
        return batch


class RecordingCostModel(CostModel):
    def __init__(self) -> None:
        self.seen_batches: list[EvaluationBatch | None] = []

    def marginal_cost(
        self,
        state: SearchState,
        action: Hashable,
        context: SearchContext,
    ) -> float:
        self.seen_batches.append(context.current_batch)
        return 0.0


class RecordingEvaluator(Evaluator):
    def __init__(self) -> None:
        self.seen_batches: list[EvaluationBatch | None] = []

    def evaluate(
        self,
        state: SearchState,
        context: SearchContext,
    ) -> EvaluationResult:
        self.seen_batches.append(context.current_batch)
        assert context.current_batch is not None
        item_id = context.current_batch.clusters[0].item_ids[0]
        return EvaluationResult(
            reward=1.0,
            total_cost=1.0,
            materialized_keys={(state.state_key(), "z", item_id)},
        )


class RaisingEvaluator(Evaluator):
    def evaluate(
        self,
        state: SearchState,
        context: SearchContext,
    ) -> EvaluationResult:
        raise RuntimeError("evaluation failed")


class TestSamplingContract(unittest.TestCase):
    def test_cost_and_evaluator_share_the_exact_iteration_batch(self) -> None:
        provider = RecordingBatchProvider()
        cost_model = RecordingCostModel()
        evaluator = RecordingEvaluator()
        engine = UCTSearchEngine(
            evaluator=evaluator,
            scorer=CostAwareUCTScorer(lambda_0=0.1),
            cost_model=cost_model,
            batch_provider=provider,
            random_seed=7,
        )

        result = engine.search(OneActionState(), max_iterations=2)

        self.assertIs(evaluator.seen_batches[0], provider.batches[0])
        self.assertIs(evaluator.seen_batches[1], provider.batches[1])
        self.assertIs(cost_model.seen_batches[0], provider.batches[1])
        self.assertEqual(
            result.context.materialized_keys,
            {
                (("only",), "z", "q1"),
                (("only",), "z", "q2"),
            },
        )

    def test_evaluator_failures_are_not_rewritten_as_nan_rewards(self) -> None:
        engine = UCTSearchEngine(
            evaluator=RaisingEvaluator(),
            scorer=CostAwareUCTScorer(),
            cost_model=ReuseAwareCostModel(
                clusters=[ClusterDef("z", weight=1.0, base_cost=1.0)],
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            engine.search(OneActionState(), max_iterations=1)

    def test_zero_iterations_is_a_valid_empty_search(self) -> None:
        engine = UCTSearchEngine(
            evaluator=RecordingEvaluator(),
            scorer=CostAwareUCTScorer(),
            cost_model=ReuseAwareCostModel(
                clusters=[ClusterDef("z", weight=1.0, base_cost=1.0)],
            ),
        )

        result = engine.search(OneActionState(), max_iterations=0)

        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.total_evaluations, 0)
        self.assertIsNone(result.best_state)


class TestItemLevelMarginalCost(unittest.TestCase):
    def test_formula_uses_cluster_qid_keys_and_per_item_cost(self) -> None:
        costs = {
            ("z0", "q1"): 10.0,
            ("z0", "q2"): 4.0,
            ("z1", "q3"): 9.0,
        }
        model = ReuseAwareCostModel(
            clusters=[
                ClusterDef("z0", weight=2.0),
                ClusterDef("z1", weight=1.0),
            ],
            item_cost=lambda _state, _action, cluster_id, item_id: costs[
                (cluster_id, str(item_id))
            ],
        )
        context = SearchContext(
            materialized_keys={(("only",), "z0", "q1")},
            current_batch=EvaluationBatch(
                clusters=(
                    ClusterBatch("z0", ("q1", "q2")),
                    ClusterBatch("z1", ("q3",)),
                ),
            ),
        )

        cost = model.marginal_cost(OneActionState(), "only", context)

        # z0: (0 + 4) / 2, weighted 2/3; z1: 9, weighted 1/3.
        self.assertAlmostEqual(cost, 13.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
