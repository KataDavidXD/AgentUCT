"""Full-tree checkpoint/resume contract for deterministic Agent-UCT runs."""

from __future__ import annotations

import inspect
import json
import math
import random
import unittest
from dataclasses import dataclass
from typing import Any, Hashable

from uct_engine import (
    BatchProvider,
    CheckpointCodec,
    ClusterBatch,
    ClusterDef,
    CostAwareUCTScorer,
    EvaluationBatch,
    EvaluationResult,
    Evaluator,
    ReuseAwareCostModel,
    RootChildSeed,
    SearchCheckpoint,
    SearchContext,
    SearchState,
    TreeNode,
    UCTSearchEngine,
    WarmStartBundle,
)


@dataclass(frozen=True)
class SlotAction:
    slot: int
    name: str


@dataclass(frozen=True)
class ResumeState:
    choices: tuple[SlotAction, ...] = ()

    OPTIONS = (
        (SlotAction(0, "passage"), SlotAction(0, "semantic")),
        (SlotAction(1, "identity"), SlotAction(1, "rewrite")),
        (SlotAction(2, "bm25"), SlotAction(2, "hybrid")),
    )

    def is_terminal(self) -> bool:
        return len(self.choices) == len(self.OPTIONS)

    def available_actions(self) -> list[Hashable]:
        return [] if self.is_terminal() else list(self.OPTIONS[len(self.choices)])

    def child(self, action: Hashable) -> "ResumeState":
        # A historical string-action replay cannot pass this boundary.
        if not isinstance(action, SlotAction) or action not in self.available_actions():
            raise TypeError(f"expected a structured SlotAction, got {action!r}")
        return ResumeState((*self.choices, action))

    def state_key(self) -> Hashable:
        return tuple((choice.slot, choice.name) for choice in self.choices)

    def pretty(self) -> str:
        return "/".join(choice.name for choice in self.choices) or "<root>"


class ResumeCodec(CheckpointCodec):
    def encode_state(self, state: SearchState) -> Any:
        assert isinstance(state, ResumeState)
        return {"choices": [self.encode_value(action) for action in state.choices]}

    def decode_state(self, payload: Any) -> SearchState:
        return ResumeState(
            tuple(self.decode_value(item) for item in payload["choices"])
        )

    def encode_value(self, value: Any) -> Any:
        if isinstance(value, SlotAction):
            return {"type": "slot-action", "slot": value.slot, "name": value.name}
        if isinstance(value, tuple):
            return {"type": "tuple", "items": [self.encode_value(v) for v in value]}
        if isinstance(value, list):
            return {"type": "list", "items": [self.encode_value(v) for v in value]}
        if isinstance(value, dict):
            return {
                "type": "dict",
                "items": [
                    [self.encode_value(key), self.encode_value(item)]
                    for key, item in value.items()
                ],
            }
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise TypeError(f"unsupported checkpoint value: {value!r}")

    def decode_value(self, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        value_type = payload.get("type")
        if value_type == "slot-action":
            return SlotAction(int(payload["slot"]), str(payload["name"]))
        if value_type == "tuple":
            return tuple(self.decode_value(item) for item in payload["items"])
        if value_type == "list":
            return [self.decode_value(item) for item in payload["items"]]
        if value_type == "dict":
            return {
                self.decode_value(key): self.decode_value(item)
                for key, item in payload["items"]
            }
        raise ValueError(f"unknown checkpoint value type: {value_type!r}")


class PreviousBatchAwareProvider(BatchProvider):
    def __init__(self) -> None:
        self.previous_items: list[str | None] = []

    def draw(self, iteration: int, context: SearchContext) -> EvaluationBatch:
        previous = context.current_batch
        self.previous_items.append(
            None
            if previous is None
            else str(previous.clusters[0].item_ids[0])
        )
        return EvaluationBatch(
            clusters=(ClusterBatch("hotpot", (f"q{iteration % 4}",)),)
        )


class ResumeEvaluator(Evaluator):
    QUALITY = {
        "passage": 0.08,
        "semantic": 0.12,
        "identity": 0.04,
        "rewrite": 0.10,
        "bm25": 0.06,
        "hybrid": 0.14,
    }

    def __init__(self) -> None:
        self.trajectory: list[tuple[Hashable, float]] = []

    def evaluate(
        self,
        state: SearchState,
        context: SearchContext,
    ) -> EvaluationResult:
        assert isinstance(state, ResumeState)
        assert context.current_batch is not None
        seen = int(context.extra.get("evaluations_seen", 0)) + 1
        context.extra["evaluations_seen"] = seen
        reward = (
            sum(self.QUALITY[action.name] for action in state.choices)
            + context.random.random() * 0.001
        )
        item_id = context.current_batch.clusters[0].item_ids[0]
        keys = {
            (state.state_key()[:depth], "hotpot", item_id)
            for depth in range(1, len(state.choices) + 1)
        }
        new_keys = keys.difference(context.materialized_keys)
        self.trajectory.append((state.state_key(), reward))
        return EvaluationResult(
            reward=reward,
            total_cost=float(len(new_keys)),
            materialized_keys=new_keys,
            metadata={"evaluations_seen": seen},
        )


class MutatingFailureEvaluator(Evaluator):
    def evaluate(
        self,
        state: SearchState,
        context: SearchContext,
    ) -> EvaluationResult:
        context.total_cost += 2.5
        context.total_evaluations += 1
        context.extra.setdefault("aborted_evaluations", []).append(
            {"realized_cost": 2.5}
        )
        raise RuntimeError("provider failed after durable side effects")


def _engine(evaluator: ResumeEvaluator, provider: PreviousBatchAwareProvider):
    return UCTSearchEngine(
        evaluator=evaluator,
        scorer=CostAwareUCTScorer(lambda_0=0.1),
        cost_model=ReuseAwareCostModel(
            [ClusterDef("hotpot", weight=1.0, base_cost=1.0)]
        ),
        exploration_constant=1.1,
        random_seed=928,
        batch_provider=provider,
    )


def _warm_start() -> WarmStartBundle:
    passage, semantic = ResumeState.OPTIONS[0]
    identity, rewrite = ResumeState.OPTIONS[1]
    bm25, hybrid = ResumeState.OPTIONS[2]
    return WarmStartBundle(
        context=SearchContext(
            materialized_keys={("cold-prefix", "hotpot", "cold-q")},
            current_batch=EvaluationBatch(
                clusters=(ClusterBatch("hotpot", ("cold-q",)),)
            ),
            total_cost=6.0,
            total_evaluations=2,
            random=random.Random(928),
            extra={"evaluations_seen": 2, "warm": True},
        ),
        root_child_seeds=(
            RootChildSeed(
                action=passage,
                action_path=(passage, identity, bm25),
                evaluated_state=ResumeState((passage, identity, bm25)),
                reward=0.18,
            ),
            RootChildSeed(
                action=semantic,
                action_path=(semantic, rewrite, hybrid),
                evaluated_state=ResumeState((semantic, rewrite, hybrid)),
                reward=0.36,
            ),
        ),
    )


def _tree_signature(node: TreeNode):
    return (
        node.state.state_key(),
        node.action_from_parent,
        node.visit_count,
        node.value_sum,
        node.best_value,
        None
        if node._available_actions_cache is None
        else tuple(node._available_actions_cache),
        tuple(
            (action, _tree_signature(child))
            for action, child in node.children.items()
        ),
    )


def _record_signature(record):
    return (
        record.iteration,
        record.evaluated_state.state_key(),
        record.reward,
        record.evaluation_cost,
        record.selection_path,
        record.expanded,
        record.best_state.state_key(),
        record.best_reward,
        record.algorithm_audit,
    )


class TestSearchCheckpointResume(unittest.TestCase):
    def test_search_and_resume_expose_keyword_only_checkpoint_callbacks(self) -> None:
        search_parameters = inspect.signature(UCTSearchEngine.search).parameters
        resume_parameters = inspect.signature(UCTSearchEngine.resume).parameters

        self.assertIn("on_checkpoint", search_parameters)
        self.assertIn("on_checkpoint", resume_parameters)
        self.assertEqual(
            search_parameters["on_checkpoint"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        self.assertEqual(
            resume_parameters["on_checkpoint"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )

    def test_checkpoint_callback_sees_each_fully_updated_search_iteration(self) -> None:
        progress = []
        payloads = []
        codec = ResumeCodec()

        result = _engine(ResumeEvaluator(), PreviousBatchAwareProvider()).search(
            ResumeState(),
            max_iterations=3,
            on_checkpoint=lambda current: (
                progress.append(
                    (
                        current.iterations,
                        current.root_node.visit_count,
                        current.total_evaluations,
                        current.context.total_evaluations,
                        current.total_cost,
                        (
                            None
                            if current.best_state is None
                            else current.best_state.state_key()
                        ),
                        (
                            None
                            if current.best_state is None
                            else current.best_reward
                        ),
                    )
                ),
                payloads.append(SearchCheckpoint.capture(current, codec).to_payload()),
            ),
        )

        self.assertEqual([item[0] for item in progress], [0, 1, 2, 3])
        self.assertEqual([item[1] for item in progress], [0, 1, 2, 3])
        self.assertEqual([item[2] for item in progress], [0, 1, 2, 3])
        self.assertEqual([item[3] for item in progress], [0, 1, 2, 3])
        self.assertTrue(all(item[4] >= 0 for item in progress))
        self.assertIsNone(progress[0][5])
        self.assertIsNone(progress[0][6])
        self.assertTrue(all(item[5] for item in progress[1:]))
        self.assertTrue(all(math.isfinite(item[6]) for item in progress[1:]))
        self.assertEqual(
            [payload["completed_iterations"] for payload in payloads],
            [0, 1, 2, 3],
        )
        self.assertEqual(result.iterations, 3)

    def test_resume_checkpoint_callback_keeps_global_iteration_numbers(self) -> None:
        codec = ResumeCodec()
        first = _engine(ResumeEvaluator(), PreviousBatchAwareProvider()).search(
            ResumeState(),
            max_iterations=2,
        )
        checkpoint = SearchCheckpoint.from_payload(
            json.loads(json.dumps(SearchCheckpoint.capture(first, codec).to_payload()))
        )
        iteration_records = []
        progress = []

        result = _engine(ResumeEvaluator(), PreviousBatchAwareProvider()).resume(
            checkpoint,
            codec,
            max_iterations=2,
            on_iteration=iteration_records.append,
            on_checkpoint=lambda current: progress.append(
                (
                    current.iterations,
                    current.root_node.visit_count,
                    current.total_evaluations,
                )
            ),
        )

        self.assertEqual([record.iteration for record in iteration_records], [3, 4])
        self.assertEqual(progress, [(2, 2, 2), (3, 3, 3), (4, 4, 4)])
        self.assertEqual(result.iterations, 4)

    def test_zero_iterations_emit_recoverable_initial_boundaries(self) -> None:
        calls = []
        engine = _engine(ResumeEvaluator(), PreviousBatchAwareProvider())
        empty = engine.search(
            ResumeState(),
            max_iterations=0,
            on_checkpoint=calls.append,
        )
        checkpoint = SearchCheckpoint.capture(empty, ResumeCodec())
        engine.resume(
            checkpoint,
            ResumeCodec(),
            max_iterations=0,
            on_checkpoint=calls.append,
        )

        self.assertEqual(
            [(result.iterations, result.total_evaluations) for result in calls],
            [(0, 0), (0, 0)],
        )

    def test_warm_start_boundary_contains_seeded_tree_before_main_iteration(self) -> None:
        calls = []

        result = _engine(
            ResumeEvaluator(),
            PreviousBatchAwareProvider(),
        ).search(
            ResumeState(),
            max_iterations=0,
            warm_start=_warm_start(),
            on_checkpoint=calls.append,
        )

        self.assertEqual(calls, [result])
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.total_evaluations, 2)
        self.assertEqual(result.total_cost, 6.0)
        self.assertEqual(result.root_node.visit_count, 2)
        self.assertEqual(len(result.root_node.children), 2)
        self.assertEqual(result.best_reward, 0.36)

    def test_initial_checkpoint_callback_exception_prevents_evaluation(self) -> None:
        observed = []

        def fail(current):
            observed.append(
                (
                    current.iterations,
                    current.root_node.visit_count,
                    current.total_evaluations,
                )
            )
            raise RuntimeError("checkpoint storage failed")

        with self.assertRaisesRegex(RuntimeError, "checkpoint storage failed"):
            _engine(ResumeEvaluator(), PreviousBatchAwareProvider()).search(
                ResumeState(),
                max_iterations=3,
                on_checkpoint=fail,
            )

        self.assertEqual(observed, [(0, 0, 0)])

    def test_resume_checkpoint_callback_exception_also_propagates(self) -> None:
        codec = ResumeCodec()
        first = _engine(ResumeEvaluator(), PreviousBatchAwareProvider()).search(
            ResumeState(),
            max_iterations=1,
        )
        checkpoint = SearchCheckpoint.capture(first, codec)
        observed = []

        def fail(current):
            observed.append(current.iterations)
            raise RuntimeError("resume checkpoint storage failed")

        with self.assertRaisesRegex(
            RuntimeError,
            "resume checkpoint storage failed",
        ):
            _engine(ResumeEvaluator(), PreviousBatchAwareProvider()).resume(
                checkpoint,
                codec,
                max_iterations=2,
                on_checkpoint=fail,
            )

        self.assertEqual(observed, [1])

    def test_evaluator_failure_emits_checkpoint_with_mutated_accounting(self) -> None:
        checkpoints = []
        engine = UCTSearchEngine(
            evaluator=MutatingFailureEvaluator(),
            scorer=CostAwareUCTScorer(lambda_0=0.1),
            cost_model=ReuseAwareCostModel(
                [ClusterDef("hotpot", weight=1.0, base_cost=1.0)]
            ),
            random_seed=928,
            batch_provider=PreviousBatchAwareProvider(),
        )

        with self.assertRaisesRegex(RuntimeError, "durable side effects"):
            engine.search(
                ResumeState(),
                max_iterations=1,
                on_checkpoint=checkpoints.append,
            )

        self.assertEqual(len(checkpoints), 2)
        initial, failed = checkpoints
        self.assertEqual((initial.iterations, initial.total_cost), (0, 0.0))
        self.assertEqual(failed.iterations, 0)
        self.assertEqual(failed.total_cost, 2.5)
        self.assertEqual(failed.total_evaluations, 1)
        self.assertEqual(failed.root_node.visit_count, 0)
        self.assertEqual(len(failed.root_node.children), 1)
        payload = SearchCheckpoint.capture(failed, ResumeCodec()).to_payload()
        restored = SearchCheckpoint.from_payload(
            json.loads(json.dumps(payload))
        )._restore(ResumeCodec())
        self.assertEqual(restored[1].total_cost, 2.5)
        self.assertEqual(restored[1].total_evaluations, 1)

    def test_zero_iteration_checkpoint_resumes_at_iteration_one(self) -> None:
        empty = _engine(ResumeEvaluator(), PreviousBatchAwareProvider()).search(
            ResumeState(),
            max_iterations=0,
        )
        payload = SearchCheckpoint.capture(empty, ResumeCodec()).to_payload()
        records = []

        resumed = _engine(ResumeEvaluator(), PreviousBatchAwareProvider()).resume(
            SearchCheckpoint.from_payload(json.loads(json.dumps(payload))),
            ResumeCodec(),
            max_iterations=2,
            on_iteration=records.append,
        )

        self.assertEqual([record.iteration for record in records], [1, 2])
        self.assertEqual(resumed.iterations, 2)
        self.assertEqual(resumed.total_evaluations, 2)

    def test_json_round_trip_resume_matches_one_uninterrupted_trajectory(self) -> None:
        uninterrupted_evaluator = ResumeEvaluator()
        uninterrupted_provider = PreviousBatchAwareProvider()
        uninterrupted_records = []
        uninterrupted = _engine(
            uninterrupted_evaluator,
            uninterrupted_provider,
        ).search(
            ResumeState(),
            max_iterations=14,
            on_iteration=uninterrupted_records.append,
        )

        first_evaluator = ResumeEvaluator()
        first_provider = PreviousBatchAwareProvider()
        first_records = []
        first = _engine(first_evaluator, first_provider).search(
            ResumeState(),
            max_iterations=6,
            on_iteration=first_records.append,
        )
        codec = ResumeCodec()
        checkpoint = SearchCheckpoint.capture(first, codec)
        wire_payload = json.loads(json.dumps(checkpoint.to_payload()))
        restored_checkpoint = SearchCheckpoint.from_payload(wire_payload)

        resumed_evaluator = ResumeEvaluator()
        resumed_provider = PreviousBatchAwareProvider()
        resumed_records = []
        resumed = _engine(resumed_evaluator, resumed_provider).resume(
            restored_checkpoint,
            codec,
            max_iterations=8,
            on_iteration=resumed_records.append,
        )

        self.assertEqual(
            [_record_signature(record) for record in uninterrupted_records],
            [
                _record_signature(record)
                for record in (*first_records, *resumed_records)
            ],
        )
        self.assertEqual(
            uninterrupted_evaluator.trajectory,
            first_evaluator.trajectory + resumed_evaluator.trajectory,
        )
        self.assertEqual(
            resumed_provider.previous_items[0],
            first.context.current_batch.clusters[0].item_ids[0],
        )
        self.assertEqual(resumed.iterations, 14)
        self.assertEqual(resumed.total_evaluations, 14)
        self.assertEqual(resumed.total_cost, uninterrupted.total_cost)
        self.assertEqual(
            resumed.best_state.state_key(),
            uninterrupted.best_state.state_key(),
        )
        self.assertEqual(resumed.best_reward, uninterrupted.best_reward)
        self.assertEqual(
            resumed.context.materialized_keys,
            uninterrupted.context.materialized_keys,
        )
        self.assertEqual(resumed.context.extra, uninterrupted.context.extra)
        self.assertEqual(
            resumed.context.random.getstate(),
            uninterrupted.context.random.getstate(),
        )
        self.assertEqual(
            _tree_signature(resumed.root_node),
            _tree_signature(uninterrupted.root_node),
        )
        self.assertEqual(
            [record.iteration for record in resumed_records],
            list(range(7, 15)),
        )

    def test_warm_start_counts_stay_distinct_from_resumed_iterations(self) -> None:
        uninterrupted_records = []
        uninterrupted = _engine(
            ResumeEvaluator(),
            PreviousBatchAwareProvider(),
        ).search(
            ResumeState(),
            max_iterations=10,
            warm_start=_warm_start(),
            on_iteration=uninterrupted_records.append,
        )

        first_records = []
        first = _engine(ResumeEvaluator(), PreviousBatchAwareProvider()).search(
            ResumeState(),
            max_iterations=3,
            warm_start=_warm_start(),
            on_iteration=first_records.append,
        )
        resumed_records = []
        resumed = _engine(ResumeEvaluator(), PreviousBatchAwareProvider()).resume(
            SearchCheckpoint.from_payload(
                json.loads(
                    json.dumps(
                        SearchCheckpoint.capture(first, ResumeCodec()).to_payload()
                    )
                )
            ),
            ResumeCodec(),
            max_iterations=7,
            on_iteration=resumed_records.append,
        )

        self.assertEqual(first.iterations, 3)
        self.assertEqual(first.total_evaluations, 5)
        self.assertEqual(resumed.iterations, 10)
        self.assertEqual(resumed.total_evaluations, 12)
        self.assertEqual(
            [_record_signature(record) for record in uninterrupted_records],
            [
                _record_signature(record)
                for record in (*first_records, *resumed_records)
            ],
        )
        self.assertEqual(
            _tree_signature(resumed.root_node),
            _tree_signature(uninterrupted.root_node),
        )
        self.assertEqual(
            resumed.context.materialized_keys,
            uninterrupted.context.materialized_keys,
        )
        self.assertEqual(
            resumed.context.random.getstate(),
            uninterrupted.context.random.getstate(),
        )

    def test_checkpoint_restores_every_tree_stat_and_structured_action(self) -> None:
        result = _engine(ResumeEvaluator(), PreviousBatchAwareProvider()).search(
            ResumeState(),
            max_iterations=7,
        )
        codec = ResumeCodec()
        checkpoint = SearchCheckpoint.capture(result, codec)
        payload = checkpoint.to_payload()

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["completed_iterations"], 7)
        self.assertEqual(
            set(payload["root"]),
            {
                "state",
                "visit_count",
                "value_sum",
                "best_value",
                "available_actions",
                "children",
            },
        )
        self.assertTrue(payload["root"]["children"])
        self.assertEqual(
            set(payload["root"]["children"][0]),
            {"action", "node"},
        )
        self.assertEqual(
            set(payload["context"]),
            {
                "materialized_keys",
                "current_batch",
                "total_cost",
                "total_evaluations",
                "random_state",
                "extra",
            },
        )

        restored = _engine(ResumeEvaluator(), PreviousBatchAwareProvider()).resume(
            SearchCheckpoint.from_payload(json.loads(json.dumps(payload))),
            codec,
            max_iterations=0,
        )

        self.assertEqual(
            _tree_signature(restored.root_node),
            _tree_signature(result.root_node),
        )
        self.assertTrue(
            all(
                isinstance(action, SlotAction)
                for action in restored.root_node.children
            )
        )
        self.assertEqual(
            restored.best_state.state_key(),
            result.best_state.state_key(),
        )
        self.assertEqual(restored.best_reward, result.best_reward)
        self.assertTrue(math.isfinite(restored.root_node.best_value))

    def test_payload_rejects_unknown_schema_version(self) -> None:
        result = _engine(ResumeEvaluator(), PreviousBatchAwareProvider()).search(
            ResumeState(),
            max_iterations=1,
        )
        payload = SearchCheckpoint.capture(result, ResumeCodec()).to_payload()
        payload["schema_version"] = 999

        with self.assertRaisesRegex(ValueError, "schema_version"):
            SearchCheckpoint.from_payload(payload)

    def test_payload_rejects_missing_best_value_for_a_visited_node(self) -> None:
        result = _engine(ResumeEvaluator(), PreviousBatchAwareProvider()).search(
            ResumeState(),
            max_iterations=1,
        )
        payload = SearchCheckpoint.capture(result, ResumeCodec()).to_payload()
        payload["root"]["best_value"] = None

        with self.assertRaisesRegex(ValueError, "best_value"):
            SearchCheckpoint.from_payload(payload)


if __name__ == "__main__":
    unittest.main()
