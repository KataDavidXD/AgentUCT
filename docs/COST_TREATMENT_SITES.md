# Cost-treatment sites and compatibility contract

## Stable boundary

AgentUCT keeps evaluator reward, backed-up Q, visit counts, and best-state
ranking cost-free. Cost changes only the search policy at explicitly enabled
decision sites.

The generic defaults preserve the historical engine:

- `schedule="inverse_sqrt"`;
- `cost_aware_selection=True`;
- `cost_aware_expansion=False`;
- `cost_scale=1.0`;
- uniform, cost-blind rollout.

An integration that wants the AT-PPO/RAGSpace V2 treatment must configure it
explicitly:

```python
CostAwareUCTScorer(
    lambda_0=beta_c,
    schedule="constant",
    cost_aware_selection=True,
    cost_aware_expansion=True,
    cost_scale=frozen_global_scale,
)
```

The expansion distribution is

```text
p(a | s) proportional to uniform(a | unexpanded(s))
                    * exp(-beta_t * DeltaC(s, a) / cost_scale)
```

Equal costs, zero beta, or disabled expansion retain the historical seeded
uniform-choice path. The engine records the full expansion distribution and
the existing-child UCT-versus-cost-aware argmax relation, so integrations can
measure treatment exposure rather than infer it from configuration alone.

## Deliberate non-change

Rollout remains cost-blind. Evidence from a separate hyper-heuristic adapter
shows that rollout treatment is domain- and proposal-dependent; it is not a
safe generic default. A future rollout policy must be introduced as its own
labelled API and ablation while preserving raw reward/Q semantics.
