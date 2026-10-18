"""Synthetic program-edit allocation; no model, dataset, or measured result.

Run: python -m uct_engine.examples.edit_mock_example
"""
from uct_engine.edit_search import (
    EditSearchConfig, Evaluation, Proposal, run_edit_search,
)


def run_demo(strategy="agentuct"):
    next_revision = 0
    panel = range(4)

    def propose(_parent):
        nonlocal next_revision
        next_revision += 1
        # One valid, distinct synthetic program; this unit includes any repair.
        return Proposal(candidate=next_revision, charged_cost=1)

    def evaluate(revision):
        # Every callback completes the same panel, producing one raw reward.
        score = sum((revision + q) % 5 != 0 for q in panel) / len(panel)
        bill = sum(1 + (revision + q) % 3 for q in panel)
        return Evaluation(score=score, charged_cost=bill)

    return run_edit_search(
        0, propose=propose, evaluate=evaluate,
        cost_forecast=lambda parent: 1 + sum(1 + (parent + q) % 3 for q in panel),
        config=EditSearchConfig(strategy=strategy, beta=2, c_ref=10),
        max_evaluations=16, target_cost=45,
    )


if __name__ == "__main__":
    for strategy in ("aflow_original", "uct", "agentuct"):
        result = run_demo(strategy)
        retained = result.last_affordable_snapshot(45)
        best = max(retained.nodes, key=lambda node: node.own_score)
        print(f"{strategy}: full bill={result.total_cost:g}; "
              f"retained bill={retained.total_cost:g}; "
              f"retained panels={len(retained.nodes)}; "
              f"best synthetic score={best.own_score:.2f}")
