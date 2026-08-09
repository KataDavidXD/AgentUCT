# UCT signal export TDD checklist

This slice adds an observational, JSON-safe signal boundary. It does not add a
training algorithm and does not reinterpret a tree-selection score as an
advantage.

## Large checklist

- [x] Preserve the raw-reward contract: evaluation reward alone updates `N`,
  `value_sum`, `Q`, and final best-state selection.
- [x] Export a cost-free UCT-only score independently from the Agent-UCT score.
- [x] Export every standard score term for root and deeper actionable nodes.
- [x] Export a visit-count policy at temperature 1 when at least one child has
  visits.
- [x] Require an explicit projection iteration and label the supplied
  context/current-batch provenance; do not call the snapshot a historical
  selection trace.
- [x] Keep the payload pure-data and JSON-safe, with a caller encoder for
  structured actions.
- [x] Distinguish random expansion from the fully-expanded UCT argmax path.
- [x] Demonstrate item-level logical reuse edges changing only the cost side.
- [x] Keep custom `ScoreFunction` search behavior backward compatible without
  falsely decomposing arbitrary custom math as standard Agent-UCT.
- [x] Synchronize both README copies.
- [x] Run the complete standard-library test suite.

## Small RED to GREEN checklist

- [x] RED: four new tests failed with `AttributeError` because
  `UCTSearchEngine.export_search_signals` did not exist.
- [x] GREEN: exact t=4 golden checks `N`, raw `Q`, exploration, marginal cost,
  `lambda_t`, decayed penalty, UCT score, Agent-UCT score, argmax, and visit
  probability.
- [x] GREEN: adding evaluator-shaped item-level keys lowers the matching
  branch's marginal cost and penalty while its `N` and raw `Q` remain fixed.
- [x] GREEN: unexpanded actions are visible but marked as random-expansion
  candidates whose displayed score was not used by the engine.
- [x] GREEN: structured actions require and pass a JSON value encoder.
- [x] GREEN: deterministic example payload round-trips through strict JSON.
- [x] RED then GREEN: reject non-finite `lambda_0`, raw Q, marginal cost, and
  exploration inputs, plus bool/non-integer visits and iteration, before they
  can create ambiguous or NaN/Infinity signals.

## Mathematical golden

For `N(s)=2`, `N(s,a)=1`, `c_uct=1.4`, and `t=4`:

```text
exploration = 1.4 * sqrt(log(3) / 2) = 1.037612665157258
lambda_t   = 0.5 / sqrt(4)           = 0.25
```

The seeded actions therefore decompose as:

| action | N | raw Q | cost | penalty | UCT-only | Agent-UCT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cheap | 1 | 0.25 | 1.0 | 0.25 | 1.287612665157258 | 1.037612665157258 |
| expensive | 1 | 0.75 | 4.0 | 1.0 | 1.787612665157258 | 0.787612665157258 |

The actual Agent-UCT argmax is `cheap`; the cost-free UCT-only argmax is
`expensive`. A true UCT-only experiment must rerun search with `lambda_0=0`;
reading `uct_score` from an Agent-UCT trajectory is an audit/counterfactual,
not a replacement trajectory.

The snapshot argmax is conditional on reaching that fully expanded node under
the explicit iteration and supplied context. Historical actions and scores
come from `IterationAlgorithmAudit.selection_steps`; a result's attached
`current_batch` may still be the previous iteration's batch.

## Bipartite behavior golden

The item-level logical reuse key is `(child_prefix, cluster_id, item_id)`. For
two active items, adding both evaluator-reported keys for the `cheap` prefix
changes its marginal cost from `1.0` to `0.0` and its t=4 penalty from `0.5`
to `0.0`. Visits and raw Q remain `(1, 0.5)` for both actions. This proves the
logical reuse-ledger effect used by selection; it is not evidence of a
physical SQLite/cache hit.

## Environment-transition TDD

- [x] RED: importing `TransitionModel`, `TransitionResult`, and
  `CostPenaltySchedule` failed before the runtime hook existed.
- [x] GREEN: expansion and rollout both execute the configured transition
  model; the default still delegates to `SearchState.child` with zero effects.
- [x] GREEN: selection cost preview uses optional
  `reuse_child_key(action)` without executing either `child` or the environment
  transition; legacy states retain the old `child(action).state_key()` fallback.
- [x] GREEN: transition and terminal-evaluator costs/keys have one successful
  iteration merge boundary and separate audit fields.
- [x] GREEN: a `2 + 3` transition-cost path plus evaluator cost `5` records
  total realized cost `10`, while every backed-up Q remains the raw reward
  `0.75`.
- [x] GREEN: `schedule="constant"` keeps `beta_c` unchanged, the default
  `inverse_sqrt` schedule preserves `lambda_0/sqrt(t)`, and `lambda_0=0` is
  vanilla UCT under both schedules.
