"""Deterministic UCT-only versus Agent-UCT signal export example."""
from __future__ import annotations

import json
from collections.abc import Hashable

from uct_engine import (
    ClusterDef,
    CostAwareUCTScorer,
    EvaluationResult,
    Evaluator,
    ReuseAwareCostModel,
    RootChildSeed,
    SearchContext,
    SearchState,
    UCTSearchEngine,
    WarmStartBundle,
)


class ChoiceState:
    def __init__(self, choice: str | None = None) -> None:
        self.choice = choice

    def is_terminal(self) -> bool:
        return self.choice is not None

    def available_actions(self) -> list[Hashable]:
        return ["cheap", "expensive"] if self.choice is None else []

    def child(self, action: Hashable) -> ChoiceState:
        if action not in self.available_actions():
            raise ValueError(action)
        return ChoiceState(str(action))

    def state_key(self) -> Hashable:
        return () if self.choice is None else (self.choice,)

    def pretty(self) -> str:
        return str(self.state_key())


class NoEvaluation(Evaluator):
    def evaluate(
        self,
        state: SearchState,
        context: SearchContext,
    ) -> EvaluationResult:
        del state, context
        raise AssertionError("this zero-iteration example must not evaluate")


def build_payload() -> dict:
    """Build one exact t=4 snapshot without provider calls or mutation."""

    def action_cost(
        _state: SearchState,
        action: Hashable,
        _cluster_id: str,
        _item_id: Hashable | None,
    ) -> float:
        return 1.0 if action == "cheap" else 4.0

    engine = UCTSearchEngine(
        evaluator=NoEvaluation(),
        scorer=CostAwareUCTScorer(lambda_0=0.5),
        cost_model=ReuseAwareCostModel(
            [ClusterDef("demo")],
            item_cost=action_cost,
        ),
        exploration_constant=1.4,
        random_seed=7,
    )
    result = engine.search(
        ChoiceState(),
        max_iterations=0,
        warm_start=WarmStartBundle(
            context=SearchContext(total_evaluations=2),
            root_child_seeds=(
                RootChildSeed("cheap", ChoiceState("cheap"), 0.25),
                RootChildSeed("expensive", ChoiceState("expensive"), 0.75),
            ),
        ),
    )
    return engine.export_search_signals(result, iteration=4)


def main() -> None:
    print(json.dumps(build_payload(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
