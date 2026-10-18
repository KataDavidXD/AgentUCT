"""Fixed-panel synthetic example of complete-feedback pricing; no model calls."""
from __future__ import annotations

from typing import Hashable

from uct_engine import (
    CostAwareUCTScorer,
    CostModel,
    EvaluationResult,
    Evaluator,
    SearchContext,
    UCTSearchEngine,
)

PANEL = ("q0", "q1", "q2", "q3")
PRICES = {
    "sparse": 1.0, "dense": 2.0,
    "identity": 0.0, "rewrite": 1.0,
    "short": 1.0, "grounded": 3.0,
}
QUALITY = {
    "sparse": 0.10, "dense": 0.20,
    "identity": 0.05, "rewrite": 0.15,
    "short": 0.05, "grounded": 0.25,
}


class RAGSearchState:
    """Three independent binary slots; eight fixed terminal workflows."""

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
        return list(self.OPTIONS[len(self.choices)]) if not self.is_terminal() else []

    def child(self, action: Hashable) -> RAGSearchState:
        if action not in self.available_actions():
            raise ValueError(f"illegal action: {action!r}")
        return RAGSearchState(self.choices + (str(action),))

    def state_key(self) -> Hashable:
        return self.choices

    def pretty(self) -> str:
        return " / ".join(self.choices) if self.choices else "<root>"


def resources(state: RAGSearchState) -> dict[Hashable, float]:
    """Query-specific full-prefix resource keys and their fixed physical prices."""
    return {
        (query, state.choices[:depth]): PRICES[state.choices[depth - 1]]
        for query in PANEL
        for depth in range(1, len(state.choices) + 1)
    }


class CompleteColdCostModel(CostModel):
    """Exact legal-completion expectation for this tiny binary grammar.

    Uniform next-action rollout induces a uniform distribution on these leaves.
    Enumeration here replaces sampling only because the example has eight leaves.
    """

    def marginal_cost(self, state, action, context: SearchContext) -> float:
        def expected_bill(node: RAGSearchState) -> float:
            if node.is_terminal():
                return sum(
                    price for key, price in resources(node).items()
                    if key not in context.materialized_keys
                )
            actions = node.available_actions()
            return sum(expected_bill(node.child(a)) for a in actions) / len(actions)

        return expected_bill(state.child(action))


class MockEvaluator(Evaluator):
    """Complete the fixed panel and publish every successful prefix."""

    def evaluate(self, state, context: SearchContext) -> EvaluationResult:
        required = resources(state)
        new_keys = set(required) - context.materialized_keys
        return EvaluationResult(
            reward=0.2 + sum(QUALITY[choice] for choice in state.choices),
            total_cost=sum(required[key] for key in new_keys),
            materialized_keys=new_keys,
        )


def main() -> None:
    engine = UCTSearchEngine(
        evaluator=MockEvaluator(),
        scorer=CostAwareUCTScorer(beta=2.0, c_ref=40.0),
        cost_model=CompleteColdCostModel(),
        random_seed=42,
    )
    result = engine.search(RAGSearchState(), max_iterations=100)
    print("Synthetic AgentUCT example (not a benchmark reproduction)")
    print(f"Best workflow: {result.best_state.pretty()}")
    print(f"Best raw reward: {result.best_reward:.4f}")
    print(f"Complete evaluations: {result.total_evaluations}")
    print(f"Actual cold cost: {result.total_cost:.1f}")
    print(f"Published query-prefix keys: {len(result.context.materialized_keys)}")


if __name__ == "__main__":
    main()
