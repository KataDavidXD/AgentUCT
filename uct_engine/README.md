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
| [examples/rag_mock_example.py](examples/rag_mock_example.py) | Small fixed-panel synthetic example |

Benchmark execution and calibrated predictors are outside this package.
