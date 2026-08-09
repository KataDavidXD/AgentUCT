# agent-uct

`agent-uct` is a research-oriented Python algorithm package implementing
reuse-aware UCT for discrete, compositional workflow search. RAG is the leading
case; the broader method applies to workflows whose evaluated prefixes can be
reused. The package owns the search policy, while datasets, execution,
checkpoint storage, and reward computation stay in the caller's integration
layer.

## Why agent-uct

Ordinary UCT treats every candidate evaluation as equally expensive. Agent-UCT
keeps raw task reward as its optimization target while adding a decaying
marginal-cost penalty to tree selection. Integrations can therefore expose real
reuse without coupling the search core to one RAG stack or execution backend.

## How the method works

At iteration `t`, the default inverse-square-root schedule scores a child with:

```text
Q(s, a)
+ c_uct * sqrt(log(N(s) + 1) / (N(s, a) + 1))
- (lambda_0 / sqrt(t)) * marginal_cost(s, a)
```

The cost term changes selection only. Evaluation, backpropagation, and final
best-state selection use the evaluator's finite raw reward.

`CostAwareUCTScorer(lambda_0=beta_c, schedule="constant")` instead uses the
same constant `beta_c` at every iteration. The default remains
`schedule="inverse_sqrt"` for backward compatibility. With `lambda_0=0`, both
schedules reduce exactly to vanilla UCT.

## UCT signal export and ablation

The standard scorer exposes an additive, observational signal boundary:

```python
payload = engine.export_search_signals(result, iteration=next_iteration)
```

The returned dictionary is strict-JSON-safe and contains every actionable
node and action: parent/child `N`, raw `Q`, exploration bonus, marginal cost,
`lambda_t`, decayed cost penalty, the cost-free `uct_score`, and the final
`agent_uct_selection_score`. A caller-supplied `encode_value` handles custom
structured actions. Run the deterministic payload example with:

```powershell
python -m uct_engine.examples.export_signals_example
```

`uct_score = Q + exploration` is the independent UCT-only ablation signal.
For a real UCT-only experiment, rerun the engine with `lambda_0=0`; the
cost-free score exported from an Agent-UCT run is a counterfactual audit and
does not replace the trajectory that UCT-only selection would produce.

At temperature 1, `visit_policy_probability` is `N(s,a)` normalized over the
node's children. It can be used by an integration as an MCTS policy target or
behavior diagnostic. It is not a scalar advantage. Neither `Q`, exploration,
cost, penalty, UCT score, Agent-UCT score, nor visit policy is a trajectory
return or a PPO/GRPO/AT-PPO advantage. Any such training estimator remains an
explicit integration-layer computation.

The exporter does not draw a batch or mutate/evaluate/backpropagate the tree.
It is a snapshot projection over the final tree, the explicitly supplied
`iteration`, and the supplied context's `current_batch` and logical reuse
keys. It is not a historical selection trace; use
`IterationAlgorithmAudit.selection_steps` for the path and scores actually
observed during an iteration. In particular, a completed result may still
carry the previous iteration's batch, so callers projecting another batch
must pass a matching `context` explicitly.

Logical reuse edges reported by evaluators show the selection-side
bipartite-graph effect, but do not prove that a physical cache or SQLite
backend skipped work. At a node with unexpanded actions, the engine uses
random expansion; the payload marks those scores
`eligible_for_engine_argmax_if_node_reached=false`. Fully expanded nodes expose
both `uct_only_snapshot_argmax_action` and
`agent_uct_snapshot_argmax_action`.

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

Environment-backed search can additionally supply a `TransitionModel`.
Expansion and rollout both call `transition(state, action, context)` and receive
a `TransitionResult` containing the successor state, realized cost, and
materialized reuse keys. The default `ChildTransitionModel` preserves the old
`SearchState.child(action)` behavior with zero realized effects.

## Integration contract

An integration supplies:

- `SearchState`, describing valid actions and stable prefix keys;
- optionally `SearchState.reuse_child_key(action)`, a side-effect-free key
  preview used by reuse-cost selection when `child` may execute real work;
- optionally `TransitionModel`, executing expansion and rollout actions and
  reporting their realized cost/reuse effects;
- `Evaluator`, returning raw reward, realized cost, and materialized keys;
- optionally `BatchProvider`, drawing exactly one initial batch per iteration;
- `CostModel`, estimating marginal cost from the current batch and reuse keys.

On a successful iteration, transition effects and the terminal evaluator's
effects are unioned/summed into one context merge. Their cost is audited but is
never subtracted from the reward backed up into `N`, `value_sum`, or `Q`.

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

## Complete checkpoint and resume

`SearchCheckpoint` captures the complete tree, search context, RNG state,
best result, and completed main-loop iteration. The integration provides a
lossless `CheckpointCodec` that maps domain states, structured actions, cache
keys, batch item IDs, and context extras to JSON values:

```python
from uct_engine import SearchCheckpoint

checkpoint = SearchCheckpoint.capture(result, my_checkpoint_codec)
payload = checkpoint.to_payload()

loaded = SearchCheckpoint.from_payload(payload_from_json)
result = engine.resume(
    loaded,
    my_checkpoint_codec,
    max_iterations=20,  # 20 additional iterations
)
```

For crash-safe long runs, persist inside the keyword-only `on_checkpoint`
callback. It receives the current complete `SearchResult` at the initialized
or restored boundary, after every fully committed iteration, and after an
evaluator failure that may have changed cost/cache accounting:

```python
def persist(current_result):
    checkpoint = SearchCheckpoint.capture(current_result, my_checkpoint_codec)
    atomic_write_json(checkpoint.to_payload())

result = engine.search(
    root_state,
    max_iterations=350,
    on_checkpoint=persist,
)
```

Zero-iteration searches and resumes therefore still emit one recoverable
checkpoint. Callback exceptions are not swallowed, so a failed durable write
fails the run visibly. A failed evaluation is not assigned reward or
backpropagated; its checkpoint retains the exact tree/RNG/context state and any
accounting changes reported by the evaluator.

The engine restores encoded node states and structured edge actions directly;
it never rebuilds the tree by stringifying actions. Child order, cached action
order, `N`, `value_sum`, per-node `best_value`, the previous batch, ledger,
totals, `context.extra`, and exact `random.Random` state are retained. Callback
iteration numbers continue at `completed_iterations + 1`, while `max_cost`
remains an absolute whole-run ceiling.

Durable storage, encryption, retention, provider/model compatibility checks,
and atomic file/database writes remain integration responsibilities.

The RAGSpace reference integration supplies three full-pool root-child rollouts
as iteration 0, then calls the engine with `max_iterations=350`; its shared
ledger therefore reports 353 terminal evaluations while main-loop callbacks
remain numbered `1..350`. This warm-start mapping is an integration choice that
is compatible with the paper's three full-pool initial evaluations; it is not a
generic requirement imposed by this engine.

## Scope and non-goals

The package includes tree nodes, scoring, search iteration ordering, batch
contracts, reuse-key accounting, and backend-neutral JSON checkpoint/resume.
It intentionally does not include RAG components, benchmark loaders, model
clients, stopping rules, checkpoint persistence, result analysis, or
experiment launchers.

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
Checkpoint tests compare uninterrupted and JSON-round-tripped continuations at
every iteration, including warmed-ledger runs.
