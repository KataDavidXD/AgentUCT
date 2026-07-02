"""Standalone mock RAG pipeline search demonstrating Agent-UCT.

This demo intentionally does not use WTB. It shows the minimum contract needed
by the AG-UCT package:

* SearchState defines a discrete prefix tree.
* Evaluator scores terminal configurations.
* Evaluator returns newly materialized ``(prefix, cluster_id)`` keys.
* ReuseAwareCostModel reads ``context.materialized_keys`` as the logical
  bipartite reuse graph.

Slots
-----
  retriever : bm25 | dense
  reranker  : none | bge
  topk      : 5 | 10

The evaluator simulates 3 benchmark clusters and returns per-cluster results
with ``materialized_keys`` so ``ReuseAwareCostModel`` can track path-prefix
reuse across iterations. A real evaluator could replace these logical keys with
RAG/question cache keys, or with WTB-backed file/state checkpoint keys.

Run::

    python -m uct_engine.examples.rag_mock_example
"""
from __future__ import annotations

import sys
from typing import Hashable

from uct_engine import (
    BenchmarkClusterResult,
    ClusterDef,
    CostAwareUCTScorer,
    EvaluationResult,
    Evaluator,
    ReuseAwareCostModel,
    SearchContext,
    SearchState,
    UCTSearchEngine,
)

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

SLOT_NAMES: list[str] = ["retriever", "reranker", "topk"]
SLOT_OPTIONS: list[list[str]] = [
    ["bm25", "dense"],
    ["none", "bge"],
    ["5", "10"],
]

# Deterministic reward table keyed by full config tuple.
# Higher is better; max is 0.92 for ("dense", "bge", "10").
REWARD_TABLE: dict[tuple[str, ...], float] = {
    ("bm25", "none", "5"):   0.45,
    ("bm25", "none", "10"):  0.50,
    ("bm25", "bge", "5"):    0.65,
    ("bm25", "bge", "10"):   0.70,
    ("dense", "none", "5"):  0.60,
    ("dense", "none", "10"): 0.68,
    ("dense", "bge", "5"):   0.85,
    ("dense", "bge", "10"):  0.92,
}


# ---------------------------------------------------------------------------
# SearchState implementation
# ---------------------------------------------------------------------------

class RAGSearchState:
    """Partial RAG configuration represented as a growing tuple."""

    def __init__(self, choices: tuple[str, ...] = ()) -> None:
        self.choices = choices

    def is_terminal(self) -> bool:
        return len(self.choices) == len(SLOT_NAMES)

    def available_actions(self) -> list[Hashable]:
        depth = len(self.choices)
        if depth >= len(SLOT_OPTIONS):
            return []
        return list(SLOT_OPTIONS[depth])

    def child(self, action: Hashable) -> "RAGSearchState":
        return RAGSearchState(self.choices + (str(action),))

    def state_key(self) -> Hashable:
        return self.choices

    def pretty(self) -> str:
        parts = [f"{SLOT_NAMES[i]}={v}" for i, v in enumerate(self.choices)]
        return "(" + ", ".join(parts) + ")"

    def path_key_for_action(self, action: Hashable) -> Hashable:
        """Key for reuse tracking -- the prefix path after taking *action*."""
        return self.choices + (str(action),)


# ---------------------------------------------------------------------------
# Mock benchmark evaluator
# ---------------------------------------------------------------------------

CLUSTER_IDS = ["cluster_A", "cluster_B", "cluster_C"]

# Per-cluster noise offsets to simulate variance across clusters.
CLUSTER_NOISE: dict[str, float] = {
    "cluster_A": -0.02,
    "cluster_B":  0.00,
    "cluster_C":  0.03,
}

# Per-cluster execution cost (simulates heterogeneous cluster sizes).
CLUSTER_COST: dict[str, float] = {
    "cluster_A": 1.0,
    "cluster_B": 1.5,
    "cluster_C": 0.8,
}

EXPECTED_BEST = ("dense", "bge", "10")


class MockBenchmarkEvaluator(Evaluator):
    """Simulates benchmark-cluster-based evaluation.

    For each cluster the evaluator:
      1. Looks up the base reward from REWARD_TABLE + cluster noise.
      2. Computes realized cost (0 if all path prefixes are already
         materialized in context, else CLUSTER_COST).
      3. Reports which path prefixes were newly materialized.
    """

    def evaluate(self, state: SearchState, context: SearchContext) -> EvaluationResult:
        assert state.is_terminal(), "evaluator expects a terminal state"
        choices: tuple[str, ...] = state.state_key()  # type: ignore[assignment]

        cluster_results: list[BenchmarkClusterResult] = []
        total_reward = 0.0
        total_cost = 0.0
        weight = 1.0 / len(CLUSTER_IDS)

        for cid in CLUSTER_IDS:
            base_reward = REWARD_TABLE.get(choices, 0.0)
            reward = base_reward + CLUSTER_NOISE[cid]

            # Logical bipartite reuse graph edge: (configuration prefix, cluster).
            # The engine harvests these edges into context.materialized_keys.
            new_keys: list[Hashable] = []
            cluster_cost = 0.0
            for depth in range(1, len(choices) + 1):
                prefix_key: Hashable = (choices[:depth], cid)
                if prefix_key not in context.materialized_keys:
                    new_keys.append(prefix_key)
                    cluster_cost += CLUSTER_COST[cid] / len(choices)

            cluster_results.append(BenchmarkClusterResult(
                cluster_id=cid,
                reward=reward,
                cost=cluster_cost,
                materialized_keys=new_keys,
            ))
            total_reward += weight * reward
            total_cost += cluster_cost

        return EvaluationResult(
            reward=total_reward,
            total_cost=total_cost,
            cluster_results=cluster_results,
        )


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def _materialized_keys_by_depth(keys: set[Hashable]) -> dict[int, int]:
    """Count logical reuse edges by configuration-prefix depth."""
    counts: dict[int, int] = {}
    for key in keys:
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        prefix, _cluster_id = key
        if isinstance(prefix, tuple):
            counts[len(prefix)] = counts.get(len(prefix), 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    evaluator = MockBenchmarkEvaluator()
    scorer = CostAwareUCTScorer(lambda_0=0.1)
    clusters = [
        ClusterDef("cluster_A", weight=1.0, base_cost=1.0),
        ClusterDef("cluster_B", weight=1.0, base_cost=1.5),
        ClusterDef("cluster_C", weight=1.0, base_cost=0.8),
    ]
    cost_model = ReuseAwareCostModel(clusters=clusters)

    engine = UCTSearchEngine(
        evaluator=evaluator,
        scorer=scorer,
        cost_model=cost_model,
        exploration_constant=1.4,
        random_seed=42,
    )

    root = RAGSearchState()
    result = engine.search(root, max_iterations=200)

    print("=" * 60, flush=True)
    print("  Standalone Agent-UCT Search Complete", flush=True)
    print("=" * 60, flush=True)
    print("  Cache backend: in-memory logical bipartite graph", flush=True)
    print(f"  Best config : {result.best_state.pretty()}", flush=True)
    print(f"  Expected    : {EXPECTED_BEST}", flush=True)
    print(f"  Best reward : {result.best_reward:.4f}", flush=True)
    print(f"  Iterations  : {result.iterations}", flush=True)
    print(f"  Evaluations : {result.total_evaluations}", flush=True)
    print(f"  Total cost  : {result.total_cost:.2f}", flush=True)
    print(f"  Materialized keys: {len(result.context.materialized_keys)}", flush=True)
    print(
        f"  Keys by prefix depth: "
        f"{_materialized_keys_by_depth(result.context.materialized_keys)}",
        flush=True,
    )
    print("=" * 60, flush=True)

    # Show root children stats
    print("\nRoot children stats:", flush=True)
    for action, child in result.root_node.children.items():
        print(
            f"  action={action!s:6s}  visits={child.visit_count:4d}  "
            f"Q={child.q_value:.4f}  best={child.best_value:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
    sys.exit(0)
