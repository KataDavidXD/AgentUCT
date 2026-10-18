# Components described in the paper

The reference is [arXiv:2607.24162v3](https://arxiv.org/abs/2607.24162v3),
including its supplement. These components implement the published allocation
and prediction rules. They do not include model clients, datasets, executor
sandboxes, calibrated benchmark files or saved experimental trajectories.

| Paper description | Implementation and boundary |
| --- | --- |
| Sections 2-3: raw quality, complete feedback and the selection index | The existing `UCTSearchEngine` and `CostAwareUCTScorer` |
| Section 3: online resource means and sampled completion costs | `completion_cost`: one prior plus capped physical samples, query-prefix/prefix/action pooling, smoothed historical continuation |
| Section 3: first positive unpaid resource per query | The same completion samples with `target="next_unpaid"` |
| Supplement: StaticFull ablation | `reference="static_full"`: initial prices, zero-history law and fixed tape against an empty pricing cache |
| Supplement: formal generated-program edit-cost forecast | `edit_cost`: prequential first-call, proposal/repair and complete training-execution factors |
| Supplement: generated-program allocation | `edit_search`: width `min(8, max(1, ceil(sqrt(N))))`, depth limit 12, no rollout, raw backup |
| Supplement: AFlow control and proposal history | Eligible top-four own-score mixture; counted exact errors and consecutive unevaluated feedback |
| Algorithm 1 and the actual-cost reporting protocol | `accounting`: actual phase receipts and the last complete affordable snapshot |

## Fixed-panel cold-cost prediction

```python
from uct_engine.completion_cost import HistoricalCompletionCostModel, ResourceReceipt

model = HistoricalCompletionCostModel(
    query_ids=("q0", "q1"),
    priors={"retrieve": 4.0, "answer": 10.0},
    caps={"retrieve": 20.0, "answer": 100.0},
    samples=16,
    seed=11,
)
```

These are illustrative prices. A real integration supplies independent cost
calibration in one declared unit. States expose pure `child(action)` previews
and tuple-valued prefix keys. Successful cache entries have the shape
`(query_id, full_prefix)`. Call `model.marginal_cost(state, action, context)`
through the core's `CostModel` interface to predict a complete cold evaluation.

After the **whole training panel** returns a score, commit its observations:

```python
model.observe_completed(
    evaluation_id="completed-1",
    path=("retrieve", "answer"),
    receipts=(ResourceReceipt("physical-call-1", "q0", ("retrieve",), 6.0),),
)
```

Supply only fresh physical receipts. A restored object contributes no new
price sample. Query IDs outside the fixed training panel are rejected.
Incomplete evaluations and validation/test observations do not fit the model;
their actual bills remain in the caller's accounting. Price caps clip forecast
samples, never physical charges. The caller commits execution publications to
`context.materialized_keys` only when they are actually available.

Known zero-work slots can be supplied as exact `(query_id, prefix)` keys in
`context.extra["zero_work_resources"]`. A zero-work observation does not itself
materialize a checkpoint. Pure structural zero rules can instead be declared
with `structural_zero`. The model freezes its cache, zero-work view, prices and
transition counts for each forecast.

| Paper configuration | Model settings |
| --- | --- |
| AgentUCT cold prediction | `reference="historical", target="complete"` |
| AgentUCT-NextC | `reference="historical", target="next_unpaid"` |
| Uniform-complete control | `reference="uniform", target="complete"` |
| AgentUCT-StaticFull | `reference="static_full", target="complete"` |

Historical transitions use `(n(p,a) + 1/|A(p)|) / (sum(n(p,*)) + 1)`.
Uniform-complete chooses successive legal actions uniformly; terminals need
not have equal probabilities. Repeated sampled paths retain their multiplicity.
Complete and NextC forecasts use identical sampled paths in the same state.
Sampling uses a separate RNG and does not advance search randomness.

StaticFull ignores later prices, transition counts, observed zero-work slots
and execution-cache membership. Its structural zeros remain free. Its pricing
cache is empty while the caller's real execution cache continues to operate.

This module implements the **cold-cost** part used by the finite RAG study,
whose recurring-work forecast is empty. It supplies no general overhead model
or calibrated error radius. Its random tape is a portable implementation of
the disclosed rule; it is not a claim of replaying a saved benchmark run.

Run the complete synthetic integration:

```sh
python -m uct_engine.examples.online_cost_example
```

## Generated-program editing

`PrequentialEditCostModel` in `uct_engine.edit_cost` reads parent program text,
operator-description length and the latest parent-training question receipts.
The caller supplies proposal and executor priors. The paper's specific GSM8K
values were 1768 and 531 raw tokens; its first-output prior was 884. Those
values are not universal calibration defaults.

Each `EditCostObservation` contains the request-size proxy saved before
generation, first proposer input/output tokens and bill, total proposal/repair
bill, new physical training-execution bill and saved parent-cold bill. Only
complete training episodes update the four prequential factors. The model uses
means of episode ratios and imposes no upper cap or confidence radius.

`ParentQuestionCost(actual_call_costs=())` means no saved parent receipts and
uses the character-count fallback. `(0.0,)` means an observed zero physical
bill and remains zero. Include failed attempts and format retries in supplied
actual costs. Neither forecast model sees task scores or held-out outcomes.

The editing driver accepts caller-owned callbacks:

```python
from uct_engine.edit_search import EditSearchConfig, Evaluation, Proposal, run_edit_search

result = run_edit_search(
    initial_program,
    propose=propose_and_repair,       # returns Proposal(candidate, actual_cost)
    evaluate=score_entire_panel,      # returns Evaluation(raw_score, actual_cost)
    cost_forecast=forecast_full_edit,
    config=EditSearchConfig(c_ref=calibrated_scale, strategy="agentuct"),
    target_cost=reporting_budget,
)
retained_tree = result.last_affordable_snapshot(reporting_budget)
```

Use the same proposal, repair, validation/duplicate checks, training panel and
executor for compared methods. The driver supplies only allocation. Strategies
are `agentuct`, `uct` and `aflow_original`. AFlow uses eligible parents ranked
by their own completed score, then `0.3/K + 0.7 * softmax(20 * score)` over its
top four. UCT and AgentUCT recursively allocate through expandable children.
Every newly evaluated program receives one complete panel, without rollout.

`Proposal.charged_cost` includes the entire proposal/repair episode. A rejected
proposal, `ChargedFailure`, partial panel or complete crossing retains its
actual costs. Only a complete panel inserts a node and backs up raw quality.
Caller-owned prequential observations must be committed at that same complete
training boundary. A forecast must be pure and read the pre-decision history.

`proposal_history` supplies `ProposalFeedback` and
`serialize_proposal_history()`. It preserves exact errors and their first-seen
order, retains all evaluated rows including zero scores, and groups only
consecutive identical unevaluated rows. This is a readable prompt view; it
does not replace the original history or erase rejected-attempt bills.

```sh
python -m uct_engine.examples.edit_mock_example
```

## Actual bills and reporting snapshots

`ActualCostLedger` accepts already incurred receipts. Search includes execution,
proposal and repair, including charged failures. Validation contributes to the
optimization bill; common preparation and test measurement remain separate.
The ledger has no reservation or predicted-cost admission mechanism.

`CompletedSnapshot.capture(payload, actual_search_cost=..., completed_evaluations=...)`
copies a caller-provided state after a complete evaluation.
`last_affordable_snapshot(snapshots, budget)` returns an independent copy of the
last such state with actual cumulative search cost at most the budget. Equal
costs from free replays retain the latest complete state. Preserve crossing
evaluations and their full bills in the raw history.

There are three different state boundaries:

| Object | Captured state |
| --- | --- |
| `SearchCheckpoint` | The fixed-tree allocation core's tree, context and RNG |
| Generated-program `EditSnapshot` | Program-tree nodes, actual cost and attempt count for endpoint reporting |
| Generic `CompletedSnapshot` | Whatever complete payload the caller explicitly supplies |

An `EditSnapshot` is a reporting tree, not a resumable backend checkpoint.
Neither driver automatically captures an external execution cache, price model,
proposer RNG, sandbox or storage state. For full recovery, persist these with
the tree at a common boundary using a lossless caller codec. For an earlier
budget endpoint, do not consult later cache entries, model observations,
rewards, candidates or held-out choices.

## Scientific scope

The finite-tree theorem retains its fixed rewards, bounded cold forecasts,
successful persistent publication and complete-feedback assumptions. The
generated-program code implements the empirical allocation protocol; it does
not transfer that theorem to arbitrary growing programs. These examples and
software tests establish neither benchmark performance nor Python-to-Lean
program refinement.
