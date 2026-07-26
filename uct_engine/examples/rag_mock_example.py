"""Self-contained item-level Agent-UCT example with no RAG dependencies."""
from __future__ import annotations

import hashlib
from typing import Hashable

from uct_engine import (
    BatchProvider,
    ClusterBatch,
    ClusterDef,
    CostAwareUCTScorer,
    EvaluationBatch,
    EvaluationResult,
    Evaluator,
    ReuseAwareCostModel,
    SearchContext,
    SearchState,
    UCTSearchEngine,
)


class RAGSearchState:
    """Three toy RAG slots used only to demonstrate compositional search."""

    OPTIONS = (
        ("sparse", "dense"),
        ("identity", "rewrite"),
        ("short", "grounded"),
    )

    def __init__(self, choices: tuple[str, ...] = ()) -> None:
        self.choices = choices

    def is_terminal(self) -> bool:
        return len(self.choices) == len(self.OPTIONS)

    def available_actions(self) -> list[Hashable]:
        return (
            list(self.OPTIONS[len(self.choices)])
            if not self.is_terminal()
            else []
        )

    def child(self, action: Hashable) -> "RAGSearchState":
        return RAGSearchState(self.choices + (str(action),))

    def state_key(self) -> Hashable:
        return self.choices

    def pretty(self) -> str:
        return " / ".join(self.choices) if self.choices else "<root>"


class RotatingBatchProvider(BatchProvider):
    """Draw deterministic toy batches before selection."""

    def draw(self, iteration: int, context: SearchContext) -> EvaluationBatch:
        offset = (iteration - 1) % 4
        return EvaluationBatch(
            clusters=(
                ClusterBatch(
                    "fact",
                    tuple(f"fact-{(offset + i) % 6}" for i in range(2)),
                ),
                ClusterBatch(
                    "reason",
                    tuple(f"reason-{(offset + i) % 6}" for i in range(2)),
                ),
            ),
        )


class MockEvaluator(Evaluator):
    """Return raw quality reward and exact item-level materialization keys."""

    _quality = {
        "sparse": 0.10,
        "dense": 0.20,
        "identity": 0.05,
        "rewrite": 0.15,
        "short": 0.05,
        "grounded": 0.25,
    }

    def evaluate(
        self,
        state: SearchState,
        context: SearchContext,
    ) -> EvaluationResult:
        assert context.current_batch is not None
        config = tuple(state.state_key())
        rewards: list[float] = []
        new_keys: set[Hashable] = set()
        realized_cost = 0.0

        for cluster_batch in context.current_batch.clusters:
            for item_id in cluster_batch.item_ids:
                digest = hashlib.sha256(
                    f"{config}|{cluster_batch.cluster_id}|{item_id}".encode()
                ).digest()
                noise = (digest[0] / 255.0 - 0.5) * 0.04
                rewards.append(
                    0.2 + sum(self._quality[choice] for choice in config) + noise
                )
                for depth in range(1, len(config) + 1):
                    key = (
                        config[:depth],
                        cluster_batch.cluster_id,
                        item_id,
                    )
                    if key not in context.materialized_keys:
                        new_keys.add(key)
                        realized_cost += 1.0

        return EvaluationResult(
            reward=sum(rewards) / len(rewards),
            total_cost=realized_cost,
            materialized_keys=new_keys,
        )


def main() -> None:
    clusters = [
        ClusterDef("fact", weight=1.0, base_cost=1.0),
        ClusterDef("reason", weight=1.0, base_cost=1.0),
    ]
    engine = UCTSearchEngine(
        evaluator=MockEvaluator(),
        scorer=CostAwareUCTScorer(lambda_0=0.1),
        cost_model=ReuseAwareCostModel(clusters),
        batch_provider=RotatingBatchProvider(),
        exploration_constant=1.4,
        random_seed=42,
    )
    result = engine.search(RAGSearchState(), max_iterations=100)

    print("Agent-UCT search complete")
    print(f"Best config: {result.best_state.pretty()}")
    print(f"Best raw reward: {result.best_reward:.4f}")
    print(f"Evaluations: {result.total_evaluations}")
    print(f"Realized cost: {result.total_cost:.1f}")
    print(f"Materialized item-prefix keys: {len(result.context.materialized_keys)}")


if __name__ == "__main__":
    main()
