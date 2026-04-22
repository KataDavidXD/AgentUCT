# AG-UCT (AgentUCT)

Path-reuse regularized UCT for discrete configuration search in agentic workflows and RAG pipelines.

A lightweight, extensible UCT (Upper Confidence bounds applied to Trees) search
engine for discrete compositional search spaces. Implements the  **AG-UCT (AgentUCT)**, a
**Path-Reuse Regularized UCT** algorithm for RAG pipeline configuration search
with history-dependent, reuse-aware cost regularization based on WTB and Agent Git https://github.com/KataDavidXD/WTB-AgenticWorkflowTestBench.

**Zero external dependencies** -- stdlib only (`dataclasses`, `typing`, `abc`,
`math`, `random`, `logging`).

---

## Background: the algorithm

This engine implements the method described in the paper. The core idea:

> UCT searches a discrete RAG component space; AgentGit provides path-level
> state reuse; UCT's tree policy subtracts a history-dependent marginal cost
> term. This is not static cost-aware UCT -- it is
> **history-dependent, path-reuse-aware UCT**.

### Problem formulation

A full RAG configuration is a tuple of component choices across L slots:

```
x = (o_1, o_2, ..., o_L),    o_l in O_l
```

For example: (embedding_model, chunking_strategy, retriever, reranker, top_k,
generation_mode). The search space is the Cartesian product O_1 x O_2 x ... x O_L.

### Search states as selected configurations

A search state `s` is a **selected config** -- a partially specified configuration:

```
s = (o_1, ..., o_d),    d in {0, 1, ..., L}
```

The root state is the empty config `s_0 = ()`. An action `a in O_{d+1}` appends
one component choice:

```
child(s, a) = (o_1, ..., o_d, a)
```

A state is terminal when `d = L` (complete configuration).

### Benchmark clusters

The evaluation set is organized into **benchmark clusters** Z = {z_1, ..., z_K}.
Each cluster contains a batch of benchmark instances evaluated together. The
overall objective is:

```
J(x) = sum_z  w_z * R(x, z)
```

Clustering serves two purposes: (1) reducing evaluation variance by aggregating
instances, and (2) providing a natural unit for reuse accounting under AgentGit.

### Path-level reuse (AgentGit)

Given configuration `x` and cluster `z`, execution produces a workflow path:

```
pi(x, z) = (v_1, v_2, ..., v_M)
```

Each prefix `pi_{1:m}(x, z)` is content-addressable. If two evaluations produce
the same prefix hash under the same cluster, the cached state is restored
instead of recomputed.

The engine maintains a **reuse graph** Path_t -- the set of (path-prefix, cluster)
pairs already materialized. In code this is `context.materialized_keys`.

### Marginal cost (history-dependent)

The cost of evaluating a new configuration is **not static** -- it depends on
what has been cached. The marginal newly incurred cost is:

```
delta_C_t(s, a; z) = sum_m  c_m(child(s,a), z) * 1[(pi_{1:m}, z) not in Path_t]
```

A workflow prefix contributes zero cost if already materialized for the same
cluster. Aggregated over a cluster minibatch:

```
delta_C_hat_t(s, a) = sum_{z in Z_t}  w_z * delta_C_t(s, a; z)
```

### Reuse-regularized UCT score

The selection score at state `s` for action `a`:

```
Score_t(s, a) = Q_hat(s, a)
              + c_uct * sqrt(log(N(s) + 1) / (N(s, a) + 1))
              - lambda_t * delta_C_hat_t(s, a)
```

Where:

- `Q_hat(s, a)` -- empirical value estimate (exploitation)
- `c_uct * sqrt(...)` -- UCT exploration bonus
- `lambda_t * delta_C_hat(...)` -- marginal cost penalty (the regularizer)

The regularizer acts only on the **search policy**, not the final objective.
The returned configuration is always the best by observed reward:

```
x_best = argmax_{x in X_visited}  J_hat(x)
```

This separates the optimization target from the search heuristic.

### Algorithm pseudocode

```
Algorithm 1: Path-Reuse Regularized UCT

Input:  component sets {O_1, ..., O_L}, benchmark clusters Z,
        total budget B_tot, c_uct, cost schedule {lambda_t}
Output: best configuration x_best

1:  s_0 = ()                           -- root selected config
2:  Path_0 = empty                     -- reuse graph
3:  N(s), N(s,a), Q(s,a) = 0          -- search tree stats
4:  spent_cost = 0

5:  while spent_cost < B_tot do
6:      s = s_0
7:
8:      -- SELECTION: descend tree
9:      while s is non-terminal do
10:         if s has unexpanded actions then
11:             choose unexpanded a, s = child(s, a), break
12:         else
13:             for each a in A(s):
14:                 estimate delta_C_hat_t(s, a) from Path_t
15:                 Score = Q(s,a) + c_uct*sqrt(log(N(s)+1)/(N(s,a)+1))
16:                         - lambda_t * delta_C_hat_t(s,a)
17:             a* = argmax Score, s = child(s, a*)
18:
19:     -- EVALUATION
20:     complete s to terminal x if necessary (random rollout)
21:     sample cluster minibatch Z_t
22:     for each z in Z_t:
23:         execute workflow path pi(x, z)
24:         for each node m:
25:             if (pi_{1:m}, z) in Path_t: restore cached state
26:             else: execute, materialize, insert into Path_t
27:                   spent_cost += c_m
28:         reward_sum += w_z * R_hat(x, z)
29:
30:     -- BACKPROPAGATION
31:     backpropagate reward_sum along search path
32:
33: return x_best = argmax observed terminal reward
```

---

## Architecture

```
uct_engine/
    __init__.py           public re-exports
    interfaces.py         SearchState protocol, ABCs, data structures
    core.py               TreeNode (engine-internal)
    scoring.py            CostAwareUCTScorer, ReuseAwareCostModel
    search.py             UCTSearchEngine (main loop)
    examples/
        rag_mock_example.py
    tests/
        test_basic_search.py
        test_custom_score.py
        test_reuse_cost.py
```

### Component diagram

```
                        +-------------------+
                        | UCTSearchEngine   |
                        |   (search.py)     |
                        +--------+----------+
                                 |
              +------------------+-------------------+
              |                  |                    |
     +--------v-------+  +------v--------+  +--------v---------+
     | ScoreFunction   |  | Evaluator     |  | CostModel        |
     | (ABC)           |  | (ABC)         |  | (ABC)            |
     +--------+-------+  +------+--------+  +--------+---------+
              |                  |                    |
   +----------v----------+      |         +----------v-----------+
   | CostAwareUCTScorer  |      |         | ReuseAwareCostModel  |
   | Q + explore         |      |         | 0 if in Path_t       |
   | - lambda_t * cost   |      |         | else base_cost       |
   +---------------------+      |         +----------------------+
                                |
                   +------------v-----------+
                   | EvaluationResult       |
                   |   reward (aggregate)   |
                   |   total_cost           |
                   |   cluster_results: []  |
                   |     BenchmarkCluster   |
                   |     Result per z       |
                   +------------------------+
```

### Mapping from paper to code


| Paper notation        | Code location                          | Description                      |
| --------------------- | -------------------------------------- | -------------------------------- |
| `s = (o_1, ..., o_d)` | `SearchState` protocol                 | Partial config (selected config) |
| `child(s, a)`         | `SearchState.child(action)`            | State extension                  |
| `Path_t`              | `SearchContext.materialized_keys`      | Reuse graph                      |
| `delta_C_t(s, a)`     | `CostModel.marginal_cost()`            | History-dependent marginal cost  |
| `lambda_t`            | `CostAwareUCTScorer.lambda_t`          | Cost regularization weight       |
| `c_uct`               | `UCTSearchEngine.exploration_constant` | Exploration coefficient          |
| `N(s)`                | `TreeNode.visit_count`                 | Parent visit count               |
| `N(s,a)`              | `TreeNode.children[a].visit_count`     | Child visit count                |
| `Q_hat(s,a)`          | `TreeNode.children[a].q_value`         | Empirical mean reward            |
| `R_hat(x, z)`         | `BenchmarkClusterResult.reward`        | Per-cluster reward               |
| `J_hat(x)`            | `EvaluationResult.reward`              | Weighted aggregate reward        |
| `B_tot`               | `UCTSearchEngine.search(max_cost=...)` | Total compute budget             |
| `Z = {z_1, ..., z_K}` | `BenchmarkClusterResult.cluster_id`    | Benchmark clusters               |
| `x_best`              | `SearchResult.best_state`              | Best observed config by reward   |


### Data flow per iteration

```
1. SELECT    -- descend tree using scorer until leaf / unexpanded node
2. EXPAND    -- pick random unexpanded action, create child TreeNode
3. EVALUATE  -- rollout to terminal if needed, call evaluator.evaluate()
               engine harvests materialized_keys from cluster results
               into context.materialized_keys  (=  Path_t update)
4. BACKPROP  -- walk parent chain, update visit counts and value sums
5. BUDGET    -- stop if max_iterations or max_cost reached
```

### Extension points


| Interface                | What to swap                          | When                         |
| ------------------------ | ------------------------------------- | ---------------------------- |
| `SearchState` (Protocol) | Different search space representation | New domain (not RAG)         |
| `ScoreFunction` (ABC)    | Different acquisition function        | Alternative tree policy      |
| `Evaluator` (ABC)        | Different evaluation backend          | Real benchmarks vs. mock     |
| `CostModel` (ABC)        | Different cost structure              | Per-node cost, no cost, etc. |


`SearchState` is a Protocol (duck-typed, no inheritance required). The other
three are ABCs with a single abstract method each.

---

## Key concepts

### SearchState

A partially selected configuration. The engine never assumes what it contains.

```python
class MyState:
    def is_terminal(self) -> bool: ...
    def available_actions(self) -> list[Hashable]: ...
    def child(self, action: Hashable) -> "MyState": ...
    def state_key(self) -> Hashable: ...
    def pretty(self) -> str: ...

    # Optional: enables ReuseAwareCostModel
    def path_key_for_action(self, action: Hashable) -> Hashable: ...
```

### Benchmark cluster evaluation

The evaluation interface mirrors the paper's cluster-based model.
`EvaluationResult` contains a list of `BenchmarkClusterResult`, each carrying:

- `cluster_id` -- which benchmark cluster z
- `reward` -- R_hat(x, z), the cluster-level reward
- `cost` -- realized execution cost for this cluster
- `materialized_keys` -- newly cached (path-prefix, cluster) pairs

The engine automatically harvests `materialized_keys` from all cluster results
and inserts them into `context.materialized_keys` (= Path_t update, matching
Algorithm 1 line 26).

### CostAwareUCTScorer and lambda_t

The scorer implements the paper's Equation from Section 3.5:

```
Score(s,a) = Q(s,a)
           + c_uct * sqrt(log(N(s)+1) / (N(s,a)+1))
           - lambda_t * marginal_cost(s,a)
```

`lambda_t` controls the quality-cost tradeoff. Higher values make the search
prefer branches extensible from cached paths. When `lambda_t = 0` this
degenerates to vanilla UCT.

### ReuseAwareCostModel and base_cost

The paper's marginal cost (Section 3.4) sums per-workflow-node costs for
uncached prefix-cluster pairs:

```
delta_C_t(s,a;z) = sum_m c_m * 1[(pi_{1:m}, z) not in Path_t]
```

`ReuseAwareCostModel` simplifies this to a binary estimate:

- **base_cost** when the child path key is NOT in Path_t -- represents the
expected full execution cost of evaluating a new, uncached configuration.
In practice, calibrate this to the average per-config evaluation cost in
your domain (e.g. seconds of compute, API cost, etc.).
- **0.0** when the child path key IS in Path_t -- the prefix is cached and
can be restored at zero cost via AgentGit.

For production, subclass `ReuseAwareCostModel` and override `marginal_cost()`
to compute real per-node, per-cluster costs from your backend. The scalar
`base_cost` is a reasonable default for prototyping.

### SearchContext

A mutable bag carried through one `search()` call:

- `materialized_keys: set` -- the reuse graph Path_t
- `total_cost: float` -- cumulative realized cost (= spent_cost in Algorithm 1)
- `total_evaluations: int`
- `random: Random` -- seeded RNG
- `extra: dict` -- user extension slot

---

## Quick start

```python
from uct_engine import (
    UCTSearchEngine,
    CostAwareUCTScorer,
    ReuseAwareCostModel,
)
from my_domain import MySearchState, MyEvaluator

engine = UCTSearchEngine(
    evaluator=MyEvaluator(),
    scorer=CostAwareUCTScorer(lambda_t=0.1),
    cost_model=ReuseAwareCostModel(base_cost=1.0),
    exploration_constant=1.4,
    random_seed=42,
)

result = engine.search(
    root_state=MySearchState(),
    max_iterations=200,
    max_cost=50.0,    # optional: B_tot budget
)

print(f"Best: {result.best_state.pretty()}")
print(f"Reward: {result.best_reward:.4f}")
print(f"Cost: {result.total_cost:.2f}")
```

---

## Example: mock RAG pipeline search

The included example searches over a 3-slot RAG pipeline:


| Slot      | Options     | Paper notation |
| --------- | ----------- | -------------- |
| retriever | bm25, dense | O_1            |
| reranker  | none, bge   | O_2            |
| topk      | 5, 10       | O_3            |


The `MockBenchmarkEvaluator` simulates 3 benchmark clusters (Z = {cluster_A,
cluster_B, cluster_C}) with per-cluster noise and heterogeneous execution costs.
It returns `BenchmarkClusterResult` objects that populate Path_t, demonstrating
that shared-prefix evaluations become free after the first materialization.

Run it:

```bash
cd AG-UCT
python -m uct_engine.examples.rag_mock_example
```

Output:

```
============================================================
  UCT Search Complete
============================================================
  Best config : (retriever=dense, reranker=bge, topk=10)
  Best reward : 0.9233
  Iterations  : 200
  Evaluations : 200
  Total cost  : 15.40
  Materialized keys: 42
============================================================

Root children stats:
  action=bm25    visits=  44  Q=0.6113  best=0.7033
  action=dense   visits= 156  Q=0.8388  best=0.9233
```

The engine finds the optimal config `(dense, bge, 10)` and allocates most visits
to the better branch (dense: 156 vs bm25: 44), guided by both reward estimates
and cost-aware path reuse.

---

## Running tests

```bash
cd AG-UCT
python -m unittest discover -s uct_engine/tests -v
```

12 tests covering:

- Basic search correctness and best-result tracking
- Cost budget termination
- Custom scorer injection
- Reuse-aware cost model (0 for materialized, base_cost otherwise)
- lambda_t shifting branch preference toward cheaper branches
- Materialized key propagation from BenchmarkClusterResult to context

---

## Adapting for the real RAG paper setting

1. **Evaluator**: Replace `MockBenchmarkEvaluator` with a real evaluator that
  runs configurations through WTB on actual benchmark clusters. Each cluster
   evaluation should return a `BenchmarkClusterResult` with real
   `materialized_keys` from AgentGit's content-addressable state hashing.
2. **SearchState**: Replace `RAGSearchState` slots with your real component
  option lists (embedding model, chunking strategy, retriever, reranker,
   top-k, generation mode, etc.).
3. **CostModel**: Subclass `ReuseAwareCostModel` to compute real per-node,
  per-cluster marginal costs from AgentGit, matching the paper's
   delta_C_t(s,a;z) = sum_m c_m * 1[(pi_{1:m}, z) not in Path_t].
4. **SearchContext**: `context.materialized_keys` maps directly to Path_t.
  The evaluator populates `materialized_keys` in each cluster result;
   the engine harvests them automatically.
5. **lambda_t schedule**: `CostAwareUCTScorer.lambda_t` maps to the paper's
  lambda_t. For a dynamic schedule {lambda_t}, subclass `CostAwareUCTScorer`
   and read the current step from `context.total_evaluations`.
6. **Budget mode**: Pass `max_cost=B_tot` to `engine.search()` to use the
  paper's cost-budget termination (Algorithm 1 line 5) instead of or in
   addition to iteration-based termination.

