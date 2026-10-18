# uct_engine

The Python allocation core for [AgentUCT](../README.md), aligned with
[arXiv:2607.24162v3](https://arxiv.org/abs/2607.24162v3).

Read the root README for the method, quick start, integration contract,
compatibility changes and actual-cost reporting semantics.

| File | Purpose |
| --- | --- |
| [scoring.py](scoring.py) | Fixed-coefficient v3 index; legacy direct-child cost helper |
| [search.py](search.py) | Selection, one-node expansion, completion and raw backup |
| [interfaces.py](interfaces.py) | Caller-owned states, complete forecasts, evaluation and cache contracts |
| [checkpoint.py](checkpoint.py) | Backend-neutral state serialization and resume |
| [completion_cost.py](completion_cost.py) | Online cold prices, historical completions, NextC and StaticFull |
| [edit_cost.py](edit_cost.py) | Prequential proposal/repair and full-execution forecasts |
| [edit_search.py](edit_search.py) | Generated-program UCT/AgentUCT and AFlow allocation |
| [proposal_history.py](proposal_history.py) | Exact counted-error feedback for later proposals |
| [accounting.py](accounting.py) | Actual receipts and independent complete reporting snapshots |
| [examples/rag_mock_example.py](examples/rag_mock_example.py) | Small fixed-panel synthetic example |

See the [component guide](../docs/COMPONENTS.md) for the paper mapping and input
contracts. Benchmark execution and calibration data remain caller-owned.
