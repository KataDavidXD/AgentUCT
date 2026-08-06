# agent-uct

`agent-uct` is a research-oriented Python algorithm package implementing
reuse-aware UCT for discrete, compositional workflow search. RAG is the leading
case; the broader method applies to workflows whose evaluated prefixes can be
reused. The package owns the search policy, while datasets, execution,
checkpoints, and reward computation stay in the caller's integration layer.

## Why agent-uct

Ordinary UCT treats every candidate evaluation as equally expensive. Agent-UCT
keeps raw task reward as its optimization target while adding a decaying
marginal-cost penalty to tree selection. Integrations can therefore expose real
reuse without coupling the search core to one RAG stack or execution backend.

## How the method works

At iteration `t`, the engine scores a child with:

```text
Q(s, a)
+ c_uct * sqrt(log(N(s) + 1) / (N(s, a) + 1))
- (lambda_0 / sqrt(t)) * marginal_cost(s, a)
```

The cost term changes selection only. Evaluation, backpropagation, and final
best-state selection use the evaluator's finite raw reward.

In sampling mode, a `BatchProvider` draws one `EvaluationBatch` before
selection. The cost model and evaluator receive that exact object through
`SearchContext.current_batch`. Item-level reuse keys use the canonical shape:

```text
(configuration_prefix, cluster_id, item_id)
```

Only keys returned by the evaluator are materialized; the search core never
invents physical cache hits.

For the RAG experiments, OminiRAG/RAGSpace is the integration layer. WTB is
optional node-level cache/replay infrastructure rather than the search
algorithm itself; at item level, that reuse can appear as a question cache.

## Quick start

Install the package and run the included deterministic example:

```powershell
python -m pip install -e .
python -m uct_engine.examples.rag_mock_example
```

Construct an engine with your state, evaluator, cost model, and optional batch
provider:

```python
from uct_engine import (
    CostAwareUCTScorer,
    ReuseAwareCostModel,
    UCTSearchEngine,
)

engine = UCTSearchEngine(
    evaluator=my_evaluator,
    scorer=CostAwareUCTScorer(lambda_0=0.05),
    cost_model=ReuseAwareCostModel(my_clusters),
    batch_provider=my_batch_provider,
    random_seed=42,
)
result = engine.search(root_state, max_iterations=20)
```

## Integration contract

An integration supplies:

- `SearchState`, describing valid actions and stable prefix keys;
- `Evaluator`, returning raw reward, realized cost, and materialized keys;
- optionally `BatchProvider`, drawing exactly one initial batch per iteration;
- `CostModel`, estimating marginal cost from the current batch and reuse keys.

Cold-start evaluation remains integration-owned. A warmed ledger and the
root-child visits/Q values produced by those evaluations enter the core as one
atomic `WarmStartBundle`:

```python
from uct_engine import RootChildSeed, WarmStartBundle

warm_start = WarmStartBundle(
    context=cold_context,
    root_child_seeds=(
        RootChildSeed(
            action="kg_extraction",
            action_path=cold_action_path_a,
            evaluated_state=cold_state_a,
            reward=cold_reward_a,
        ),
        RootChildSeed(
            action="standard_passage",
            action_path=cold_action_path_b,
            evaluated_state=cold_state_b,
            reward=cold_reward_b,
        ),
    ),
)
result = engine.search(
    root_state,
    max_iterations=2,
    warm_start=warm_start,
)
```

The engine reuses the exact context object, including its random state,
`materialized_keys`, cost, and evaluation count, while seeding root/child
visits and Q values from the same bundle. Each seed's complete `action_path` is
replayed from the root and must reproduce its terminal `evaluated_state`.
`total_cost` must be finite and non-negative; `total_evaluations` must be a
non-negative integer at least as large as the non-empty seed count. Ordinary
`engine.search(root_state, ...)` remains a fresh, backend-agnostic search when
no warm start is supplied.

The paper's sampling protocol supplies three full-pool root-child rollouts as
iteration 0, then calls the engine with `max_iterations=350`; the shared ledger
therefore reports 353 terminal evaluations while main-loop callbacks remain
numbered `1..350`. Full-pool mode has no separate warm-start phase.

## Scope and non-goals

The package includes tree nodes, scoring, search iteration ordering, batch
contracts, and reuse-key accounting. It intentionally does not include RAG
components, benchmark loaders, model clients, stopping rules, checkpoint
storage, resume orchestration, result analysis, or experiment launchers.

Physical reuse must be implemented and verified by the integration backend.
Logical key membership alone is not evidence that executor work was skipped.

## Testing

Run the standard-library test suite:

```powershell
python -m unittest discover -s uct_engine\tests -p "test_*.py" -v
```

The suite covers search behavior, custom scoring, raw-reward semantics, the
inverse-square-root cost schedule, item-level keys, exact batch identity,
context handoff, budget limits, evaluator failures, and repository hygiene.
