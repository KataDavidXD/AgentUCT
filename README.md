# agent-uct

> Anonymous submission artifact. Identity-bearing links and attribution are
> intentionally omitted during double-blind review.

`agent-uct` is the standalone algorithm core for reuse-aware UCT over discrete,
compositional workflow configurations. The full RAG experiments were run in the
separate OminiRAG/RAGSpace artifact; this repository intentionally does not
duplicate that experiment stack.

## Method

At iteration `t`, Agent-UCT selects an action with

```text
score_t(s, a)
  = Q_t(s, a)
  + c_uct * sqrt(log(N_t(s) + 1) / (N_t(s, a) + 1))
  - (lambda_0 / sqrt(t)) * marginal_cost_t(s, a)
```

The cost term changes the tree policy only. Tree statistics, backpropagation,
and the final best configuration use the raw task reward, not
`reward - cost`.

In sampling mode, the marginal-cost estimate is

```text
sum_z w_z / |B_t,z|
  * sum_{q in B_t,z}
      c(s, a, z, q)
      * 1[(child_prefix, z, q) is not materialized]
```

The package uses the canonical item key
`(child_prefix, cluster_id, item_id)`. Full-pool mode uses the cluster-level key
`(child_prefix, cluster_id)`.

## Exact scope

The core owns:

- the prefix search tree and visit/Q statistics;
- one optional initial batch `B_t` drawn before selection;
- selection, expansion, random completion, and raw-reward backpropagation;
- the `lambda_0 / sqrt(t)` schedule;
- logical reuse-key updates reported by the evaluator;
- iteration and atomic-evaluation cost limits.

The experiment integration owns:

- RAG components, prompts, datasets, spectra, and validity constraints;
- canonical source QIDs and batch contents;
- sequential sampling and the definition of SE/CL stopping;
- reward aggregation and physical cost calibration;
- WTB/AgentGit execution, checkpoints, replay, SQLite ledgers, and VDB caches;
- resume, concurrency, logging, baselines, ablations, and result analysis.

This separation is deliberate. WTB can make a logical reuse key correspond to a
restorable workflow prefix, but WTB is an execution backend rather than part of
the Agent-UCT tree policy.

## Iteration contract

1. An optional `BatchProvider` draws one `EvaluationBatch`.
2. The engine stores that exact object in `SearchContext.current_batch`.
3. The `CostModel` reads it during tree selection.
4. The `Evaluator` evaluates the same initial batch. It must not replace it
   with a separately sampled batch.
5. The evaluator may add integration-defined micro-batches, but they cannot
   retroactively change the current iteration's selection cost.
6. Only keys returned in `EvaluationResult.materialized_keys` enter the reuse
   graph. The core never fabricates prefix/QID hits.

Evaluator failures and invalid `NaN` rewards fail fast. Silent simulated
fallbacks belong neither to the algorithm nor to a trustworthy real experiment.

## Package layout

```text
uct_engine/
  interfaces.py   public state, batch, evaluator, and cost contracts
  core.py         internal tree node and statistics
  scoring.py      Agent-UCT score and reuse-aware marginal cost
  search.py       paper-ordered search loop
  examples/
    rag_mock_example.py
  tests/
```

The standalone package does not contain the former real-RAG driver, resume
driver, or persistence module. Their authoritative historical versions remain
with the OminiRAG experiment artifact.

## Minimal usage

Implement:

- `SearchState` for the compositional configuration;
- `Evaluator` for raw reward, realized cost, and exact materialized keys;
- optionally `BatchProvider` for sampling mode;
- optionally an `item_cost` callback for the paper's
  `c(s, a, z, q)` term.

Then construct:

```python
from uct_engine import (
    ClusterDef,
    CostAwareUCTScorer,
    ReuseAwareCostModel,
    UCTSearchEngine,
)

clusters = [
    ClusterDef("cluster_a", weight=1.0, base_cost=1.0),
    ClusterDef("cluster_b", weight=1.0, base_cost=1.0),
]

engine = UCTSearchEngine(
    evaluator=my_evaluator,
    scorer=CostAwareUCTScorer(lambda_0=0.05),
    cost_model=ReuseAwareCostModel(
        clusters,
        item_cost=my_per_item_cost,
    ),
    batch_provider=my_batch_provider,
    exploration_constant=1.4,
    random_seed=42,
)

result = engine.search(root_state, max_iterations=350)
```

`ReuseAwareCostModel` normalizes non-negative cluster weights. Its default item
cost is `ClusterDef.base_cost`; real integrations should pass a calibrated
per-item/per-action cost callback.

## Run

```powershell
python -m pip install -e .
python -m unittest discover -s uct_engine\tests -v
python -m uct_engine.examples.rag_mock_example
```

The current suite has 20 tests covering basic search, custom scoring,
cluster-level reuse, exact item-level cost, shared-batch identity, raw-reward
selection, budget stopping, fail-fast evaluation, and empty-search behavior.

