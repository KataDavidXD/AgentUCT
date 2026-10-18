# AgentUCT

A small Python search core for **Agent-UCT: Upper Confidence Bounds Applied to
Trees for Cost-Aware Agentic Workflow Optimization**.

**Paper:** [arXiv:2607.24162v3](https://arxiv.org/abs/2607.24162v3),
10 October 2026. This version, including its supplement, defines the public
method and claim scope. [Lean proofs](https://github.com/KataDavidXD/AgentUCT-Lean-Proofs)
provide the companion mathematical development.

This repository contains the allocation core and a deterministic example. The
benchmark executor, calibrated estimators, model clients, datasets and experiment
outputs are outside this release. Running the example checks the API; it does
not reproduce the paper's HotpotQA or GSM8K results.

## Why AgentUCT

Workflow evaluations share reusable execution prefixes. Their remaining costs
change with the cache, so an inexpensive next step need not lead to an
inexpensive complete quality observation. AgentUCT prices complete feedback
during tree selection and backs up the evaluator's raw quality.

## How the method works

Unvisited legal children have priority. For an initialized child, Equation (5)
of v3 is

$$I_t(p,a)=Q_t(pa)+c\sqrt{\frac{\log(N_t(p)+1)}{N_t(pa)}}
-\frac{\beta}{C_{\rm ref}}\widehat b_t(p,a).$$

- `Q` is the mean of raw rewards backed up through expanded ancestors.
- `c` controls exploration; the theorem requires `c > 0`.
- `beta / c_ref` is fixed. The forecast changes with observed history.
- `b_hat` prices the remaining work needed for a complete evaluation after the
  child. The finite RAG setting prices all missing cold resources. The general
  model also permits recurring work.

The allocation tree and execution cache are distinct. A rollout may publish
prefixes that have not become tree nodes. Those prefixes receive no additional
tree visits merely because their execution objects were saved.

## Quick start

Python 3.10 or later; no runtime dependencies or model credentials are needed.

```sh
git clone https://github.com/KataDavidXD/AgentUCT.git
cd AgentUCT
python -m pip install -e .
python -m uct_engine.examples.rag_mock_example
```

The example uses a fixed synthetic panel, fixed terminal rewards, successful
prefix publication and an exact uniform-completion cold-cost forecast. It is
deliberately small enough to read in one file.

## Integration contract

```python
from uct_engine import CostAwareUCTScorer, UCTSearchEngine

engine = UCTSearchEngine(
    evaluator=my_full_panel_evaluator,
    scorer=CostAwareUCTScorer(beta=2.0, c_ref=my_calibrated_scale),
    cost_model=my_complete_evaluation_cost_model,
    random_seed=42,
)
result = engine.search(root_state, max_iterations=100)
```

An integration supplies these interfaces:

| Interface | Responsibility |
| --- | --- |
| `SearchState` | Legal actions, terminal states and stable prefix keys |
| `CostModel.marginal_cost` | A finite nonnegative **complete-evaluation forecast**, read from the pre-decision history |
| `Evaluator.evaluate` | Complete the declared panel, return raw reward, actual charges and successfully materialized keys |
| Optional `BatchProvider` | Provide the same panel to prediction and execution; the fixed-reward theorem requires a fixed panel and reward table |

The historical API name `marginal_cost` refers to the entire remaining
evaluation bill in a v3 integration. `ReuseAwareCostModel` is retained as a
legacy direct-child helper; it does not implement the v3 completion forecast.
The example shows a complete forecast through `CostModel`.

The paper's RAG predictor uses online resource-price means and 16 sampled legal
completions. Those benchmark components are not included here. WTB supplies the
paper's execution/checkpoint service; this package records only keys reported
by its evaluator. A logical key alone cannot establish physical reuse.

### Actual costs and reporting budgets

`max_cost` is an actual-cost stopping target. Each evaluation completes before
its receipts and reward are committed; no estimated-cost admission check is
performed. A run may cross the target, and the raw `SearchResult` includes that
complete crossing evaluation and bill.

For each reporting budget, retain an independent copy of the **last complete
snapshot whose cumulative actual search cost is at most that budget**. Preserve
the crossing evaluation and charges in the raw record. Its reward, best state,
tree and cache must not enter the earlier retained snapshot. `max_cost` does
not itself construct a budget-filtered result or select a held-out endpoint.

### Warm starts and checkpoints

`WarmStartBundle` couples an evaluator-owned initialized ledger with the
`RootChildSeed` observations that produced its visits and quality means.
`SearchCheckpoint.capture(result, codec)` serializes the tree, context and RNG;
`engine.resume(checkpoint, codec, ...)` continues that state. The integration
supplies a lossless `CheckpointCodec` and durable storage.

`on_checkpoint` emits recovery state at initialization, after committed
iterations and after evaluator failures. Failure recovery state is not a
completed reporting snapshot. `on_iteration` marks successful complete returns.

## Scope

The paper's logarithmic quality-regret result assumes a finite fixed-reward
tree, bounded adaptive cold-cost forecasts, legal expansion, raw backup and
successful persistent publication. Its horizon counts complete evaluations,
including free replays. It does not establish uniform finite-budget test
superiority, optimal endpoint net value, or a theorem for arbitrary growing
programs. Python-to-Lean program refinement is not machine-verified.

The companion [proof repository](https://github.com/KataDavidXD/AgentUCT-Lean-Proofs)
maps the mathematical statements to their assumptions. This public core is
maintained separately from the research workspace; this release imports no
workspace snapshots, benchmark records or later research results.

## Compatibility

Version 0.2.0 aligns the scorer with v3: the cost coefficient is constant and
the exploration denominator is `N_child`. Older versions used
`lambda_0 / sqrt(t)` and `N_child + 1`. The `lambda_0` keyword remains an alias
for the fixed coefficient `beta / c_ref`; it no longer requests time decay.
These changes alter search trajectories. Old checkpoints retain their data
format, but continuation under a changed scorer has changed policy semantics.

## Testing

```sh
python -m unittest discover -s uct_engine/tests -p "test_*.py"
```

The tests cover scoring, raw rewards, prefix publication, panel identity,
warm starts, checkpoint continuation and completed cost crossings. They are
software checks, separate from the paper's empirical results and Lean proofs.

## Citation

Please cite [the paper](https://doi.org/10.48550/arXiv.2607.24162), using
[v3](https://arxiv.org/abs/2607.24162v3) when referring to this release's scope.
The Python code is distributed under the [MIT license](LICENSE).
