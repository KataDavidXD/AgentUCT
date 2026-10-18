"""Synthetic fixed-panel search with online complete-feedback pricing."""
from uct_engine import CostAwareUCTScorer, EvaluationResult, UCTSearchEngine
from uct_engine.completion_cost import HistoricalCompletionCostModel, ResourceReceipt
from uct_engine.examples.rag_mock_example import (
    PANEL, PRICES, QUALITY, RAGSearchState, resources,
)


class TrainingEvaluator:
    def __init__(self, model):
        self.model, self.completed = model, 0

    def evaluate(self, state, context):
        demanded = resources(state)
        new_keys = set(demanded) - context.materialized_keys
        actual_cost = sum(demanded[key] for key in new_keys)
        self.completed += 1
        receipts = tuple(ResourceReceipt(
            (self.completed, query, prefix), query, prefix, demanded[(query, prefix)],
        ) for query, prefix in new_keys)
        # The entire declared panel has now joined. Fit only its physical
        # observations; replays contribute a path count but no new price sample.
        self.model.observe_completed(self.completed, state.state_key(), receipts)
        return EvaluationResult(
            reward=0.2 + sum(QUALITY[action] for action in state.choices),
            total_cost=actual_cost,
            materialized_keys=new_keys,
        )


def run_demo(reference="historical", target="complete"):
    model = HistoricalCompletionCostModel(
        PANEL, PRICES, {action: 20 for action in PRICES},
        reference=reference, target=target, samples=16, seed=11,
    )
    engine = UCTSearchEngine(
        evaluator=TrainingEvaluator(model), cost_model=model,
        scorer=CostAwareUCTScorer(beta=2, c_ref=40), random_seed=11,
    )
    return engine.search(RAGSearchState(), max_iterations=100)


if __name__ == "__main__":
    print("Synthetic fixed-panel example; no benchmark or model calls")
    for reference, target in (("historical", "complete"), ("historical", "next_unpaid"),
                              ("uniform", "complete"), ("static_full", "complete")):
        result = run_demo(reference, target)
        print(f"{reference}/{target}: raw quality={result.best_reward:.2f}; "
              f"actual cold bill={result.total_cost:g}; "
              f"complete panels={result.total_evaluations}")
