"""AG-UCT SearchState + Evaluator for the OminiRAG 5-dimension configuration space.

Search space (5 dimensions -- aligned with RAG survey taxonomy)
----------------------------------------------------------------
  chunking       : standard_passage | longrag_4k | kg_extraction
  query          : identity | lightrag_keywords
  retrieval      : bm25 | dense_e5 | bm25_dense_hybrid | lightrag_graph
  post_retrieval : identity | cross_encoder | lightrag_vector_enrich | selfrag_critique
  generation     : longrag_reader | lightrag_answer | selfrag_generator | simple_llm
                   | simple_llm_ud_words_40/80/140 (explicit slot override only)

Hard constraint: lightrag_graph requires kg_extraction chunking.

Framework presets:
  LongRAG  = longrag_4k      + identity          + bm25/dense     + identity          + longrag_reader
  LightRAG = kg_extraction   + lightrag_keywords + lightrag_graph  + lightrag_vector_enrich + lightrag_answer
  Self-RAG = standard_passage + identity          + dense_e5        + selfrag_critique  + selfrag_generator

The evaluator uses simulated rewards by default.  When ``--real`` is passed,
``build_pipeline_from_config()`` maps slot values to actual adapter instances
and evaluates against real benchmark adapters.

When ``--wtb-reuse`` is passed, the search uses the ``RAGCacheAwareEvaluator``
from ``ominirag_wtb`` which physically caches intermediate LangGraph states
via WTB and forks from shared prefixes to avoid redundant computation.

Run::

    python -m uct_engine.examples.rag_pipeline_search                     # simulated
    python -m uct_engine.examples.rag_pipeline_search --real              # real pipeline (sample_data)
    python -m uct_engine.examples.rag_pipeline_search --data-dir /data1/ragworkspace/dataset  # real datasets
    python -m uct_engine.examples.rag_pipeline_search --wtb-reuse         # with WTB reuse
"""

from __future__ import annotations

import random
import sys
from typing import Any, Hashable

from uct_engine import (
    BenchmarkClusterResult,
    ClusterDef,
    CostAwareUCTScorer,
    EvaluationResult,
    Evaluator,
    ReuseAwareCostModel,
    SearchContext,
    SearchState,
    UCTSearchEngine,
)


# ---------------------------------------------------------------------------
# Configuration space: 5 dimensions aligned with RAG survey taxonomy
# ---------------------------------------------------------------------------

SLOT_NAMES: list[str] = [
    "chunking", "query", "retrieval", "post_retrieval", "generation",
]

SLOT_OPTIONS: list[list[str]] = [
    # Chunking (offline corpus preparation)
    ["standard_passage", "longrag_4k", "kg_extraction"],
    # Query (expansion / decomposition)
    ["identity", "lightrag_keywords"],
    # Retrieval (real corpus search)
    ["bm25", "dense_e5", "bm25_dense_hybrid", "lightrag_graph"],
    # Post-Retrieval (reranking / vector enrichment / critique)
    ["identity", "cross_encoder", "lightrag_vector_enrich", "selfrag_critique"],
    # Generation
    ["longrag_reader", "lightrag_answer", "selfrag_generator", "simple_llm"],
]

# Mutable runtime search-space view.  Default runs keep the historical
# 384-config space.  Supplementary controlled experiments may replace this
# with a smaller Cartesian product via ``--slot-options-json``.
ACTIVE_SLOT_OPTIONS: list[list[str]] = [list(opts) for opts in SLOT_OPTIONS]


def set_active_slot_options(slot_options: list[list[str]]) -> None:
    """Install a runtime search-space override for this process."""
    if len(slot_options) != len(SLOT_NAMES):
        raise ValueError(
            f"Expected {len(SLOT_NAMES)} slot option lists, got "
            f"{len(slot_options)}"
        )
    cleaned: list[list[str]] = []
    for idx, opts in enumerate(slot_options):
        vals = [str(v) for v in opts]
        if not vals:
            raise ValueError(f"Slot {idx} ({SLOT_NAMES[idx]}) has no options")
        cleaned.append(vals)
    ACTIVE_SLOT_OPTIONS[:] = cleaned


def active_slot_options() -> list[list[str]]:
    return [list(opts) for opts in ACTIVE_SLOT_OPTIONS]

# Backward-compatible old slot names for migration
SLOT_NAMES_OLD: list[str] = ["frame", "query", "retrieval", "reranking", "generation"]

# Hard constraint: lightrag_graph requires kg_extraction chunking
CHUNKING_RETRIEVAL_CONSTRAINTS: dict[str, str] = {
    "lightrag_graph": "kg_extraction",
}

# Component costs: approximate compute units per query
COMPONENT_COSTS: dict[str, float] = {
    # Chunking (offline, amortised to near-zero at query time)
    "standard_passage":   0.0,
    "longrag_4k":         0.0,
    "kg_extraction":      0.0,
    # Query
    "identity":           0.0,
    "lightrag_keywords":  1.0,
    # Retrieval
    "bm25":               0.1,   # CPU-only lexical search
    "dense_e5":           0.3,   # one embedding inference
    "bm25_dense_hybrid":  0.4,   # BM25 + embedding + RRF merge
    "lightrag_graph":     1.0,   # embedding + KG traversal
    # Post-Retrieval
    "cross_encoder":      0.5,   # N forward passes on cross-encoder
    "lightrag_vector_enrich": 0.5,  # embedding search, no LLM
    "selfrag_critique":   0.5,   # scoring is lighter than generation
    # Generation
    "longrag_reader":     1.0,
    "lightrag_answer":    1.0,
    "selfrag_generator":  1.5,   # generate + score
    "simple_llm":         1.0,
    "simple_llm_ud_words_40": 1.0,
    "simple_llm_ud_words_80": 1.0,
    "simple_llm_ud_words_140": 1.0,
}

# Reward lookup: simulated quality scores for key configurations.
# Keys are 5-tuples (chunking, query, retrieval, post_retrieval, generation).
REWARD_TABLE: dict[tuple[str, ...], float] = {
    # -- Framework preset baselines --
    # LongRAG preset: longrag_4k + identity + bm25 + identity + longrag_reader
    ("longrag_4k", "identity", "bm25", "identity", "longrag_reader"):                     0.72,
    # LightRAG mix preset: kg_extraction + lightrag_keywords + lightrag_graph + lightrag_vector_enrich + lightrag_answer
    ("kg_extraction", "lightrag_keywords", "lightrag_graph", "lightrag_vector_enrich", "lightrag_answer"): 0.88,
    # Self-RAG preset: standard_passage + identity + dense_e5 + selfrag_critique + selfrag_generator
    ("standard_passage", "identity", "dense_e5", "selfrag_critique", "selfrag_generator"):  0.85,

    # -- Cross-framework combinations --
    # BM25 + cross-encoder reranking (classic IR pipeline)
    ("standard_passage", "identity", "bm25", "cross_encoder", "longrag_reader"):           0.79,
    ("standard_passage", "identity", "bm25", "cross_encoder", "simple_llm"):               0.78,
    # Dense + cross-encoder
    ("standard_passage", "identity", "dense_e5", "cross_encoder", "longrag_reader"):       0.82,
    ("standard_passage", "identity", "dense_e5", "cross_encoder", "simple_llm"):           0.81,
    # Hybrid BM25+Dense + cross-encoder (expected strongest standard pipeline)
    ("standard_passage", "identity", "bm25_dense_hybrid", "cross_encoder", "longrag_reader"):  0.86,
    ("standard_passage", "identity", "bm25_dense_hybrid", "cross_encoder", "simple_llm"):      0.85,
    # Hybrid + selfrag critique
    ("standard_passage", "identity", "bm25_dense_hybrid", "selfrag_critique", "selfrag_generator"): 0.87,
    # LightRAG retrieval + cross-encoder
    ("kg_extraction", "lightrag_keywords", "lightrag_graph", "cross_encoder", "lightrag_answer"):  0.90,
    # LightRAG retrieval + selfrag critique + selfrag gen
    ("kg_extraction", "lightrag_keywords", "lightrag_graph", "selfrag_critique", "selfrag_generator"): 0.89,
    # LongRAG 4k chunks + dense retrieval
    ("longrag_4k", "identity", "dense_e5", "identity", "longrag_reader"):                  0.74,
    ("longrag_4k", "identity", "dense_e5", "cross_encoder", "longrag_reader"):             0.80,
    # LightRAG graph-only (paper hybrid, no vector enrich)
    ("kg_extraction", "lightrag_keywords", "lightrag_graph", "identity", "lightrag_answer"): 0.82,
    # BM25+dense hybrid + vector enrich
    ("standard_passage", "identity", "bm25_dense_hybrid", "lightrag_vector_enrich", "lightrag_answer"): 0.83,
}

# Default reward for configs not in the table.  USED ONLY BY SIMULATED MODE
# (REWARD_TABLE + noise).  Real-mode evaluators MUST NOT return this on
# failure -- they should raise _RealEvalFailure so the caller can mark the
# spectrum reward as 0.0 instead of an unearned mid-table 0.55.  See the
# 2026-05-13 v2 silent-failure audit (EXPERIMENT_LOG.md §9-B).
DEFAULT_REWARD = 0.55

# Incompatible: lightrag retrieval without kg_extraction gets penalised
INCOMPATIBLE_PENALTY = 0.20


class _RealEvalFailure(RuntimeError):
    """Raised when real evaluation cannot proceed for structural reasons.

    Replaces every former ``except Exception: return DEFAULT_REWARD`` site
    in ``evaluate_config_real`` / ``_evaluate_frozen`` / ``_try_build_pipeline``.
    ``RAGPipelineEvaluator.evaluate`` catches this and converts to
    ``reward=0.0`` with a printed [FAIL] line, so a broken config never
    silently scores 0.55 and contaminates AG-UCT Q estimates.
    """


def _check_constraints(choices: tuple[str, ...]) -> bool:
    """Return True if the configuration satisfies hard constraints."""
    if len(choices) >= 3:
        chunking, _, retrieval = choices[0], choices[1], choices[2]
        required = CHUNKING_RETRIEVAL_CONSTRAINTS.get(retrieval)
        if required and chunking != required:
            return False
    return True


def _compute_reward(choices: tuple[str, ...]) -> float:
    if not _check_constraints(choices):
        return DEFAULT_REWARD - INCOMPATIBLE_PENALTY

    if choices in REWARD_TABLE:
        return REWARD_TABLE[choices]

    base = DEFAULT_REWARD
    for component in choices:
        base += COMPONENT_COSTS.get(component, 0.0) * 0.02
    return min(base, 0.70)


def _compute_cost(choices: tuple[str, ...]) -> float:
    return sum(COMPONENT_COSTS.get(c, 0.5) for c in choices)


# ---------------------------------------------------------------------------
# Pipeline builder: maps slot values to real adapter instances
# ---------------------------------------------------------------------------

from rag_contracts.component_registry import (  # noqa: E402
    build_longrag_generation as _build_longrag_generation,
    build_pipeline_from_config,
    build_selfrag_components as _build_selfrag_components,
    build_simple_llm as _build_simple_llm,
)


_CORPUS_CACHE: dict[str, "CorpusIndex"] = {}


def _load_corpus_for_benchmark(benchmark: str, chunks: dict | None = None):
    """Load a CorpusIndex for the given benchmark.

    Priority: inline chunks > real corpus index > KG sample fallback.
    Real corpus indexes are loaded once and cached for the process lifetime.
    """
    from rag_contracts import CorpusIndex
    from pathlib import Path
    import os

    if chunks:
        return CorpusIndex.from_chunks_dict(chunks)

    if benchmark in _CORPUS_CACHE:
        return _CORPUS_CACHE[benchmark]

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_root = Path(os.environ.get("OMINIRAG_DATA_ROOT", "/data1/ragworkspace/train"))

    real_paths = {
        "hotpotqa": [
            project_root / "data" / "corpus_indexes" / "fullwiki_corpus_index.json",
            data_root / "fullwiki" / "fullwiki_corpus_index.json",
        ],
        "ultradomain": [
            project_root / "data" / "corpus_indexes" / "ultradomain_corpus_index.json",
            project_root / "data" / "corpus_indexes" / "ultradomain_mix_corpus_index.json",
            data_root / "ultradomain" / "ultradomain_corpus_index.json",
        ],
    }
    sample_paths = {
        "hotpotqa": project_root / "benchmark" / "sample_data" / "hotpotqa_kg_sample" / "chunks.json",
        "ultradomain": project_root / "benchmark" / "sample_data" / "ultradomain_kg_sample" / "chunks.json",
    }

    for p in real_paths.get(benchmark, []):
        if p.exists():
            corpus = CorpusIndex.from_json_file(p)
            _CORPUS_CACHE[benchmark] = corpus
            print(f"  [corpus] Loaded real index for {benchmark}: {len(corpus)} chunks from {p}", flush=True)
            return corpus

    sample = sample_paths.get(benchmark)
    if sample and sample.exists():
        corpus = CorpusIndex.from_json_file(sample)
        _CORPUS_CACHE[benchmark] = corpus
        print(f"  [corpus] Loaded sample index for {benchmark}: {len(corpus)} chunks", flush=True)
        return corpus

    return None


def evaluate_config_real(
    choices: tuple[str, ...],
    benchmark: str = "hotpotqa",
    frozen_sample: list | None = None,
) -> float:
    """Build a real pipeline from config, run on sample data, return score.

    This is the bridge between AG-UCT and real rag_contracts evaluation.
    Returns a normalized reward in [0, 1].

    The full pipeline (retrieval + post-retrieval + generation) is exercised
    when a corpus is available.  For ALCE, the graph_factory pattern with
    ALCEDocRetrieval is used.  Falls back to generation-only evaluation
    when corpus loading or pipeline construction fails.

    Parameters
    ----------
    frozen_sample:
        Optional list of ``BenchmarkItem`` pre-drawn via ``SamplingEngine``.
        When provided, items are converted to eval dicts on-the-fly instead
        of loading from ``sample_data/``.
    """
    if frozen_sample is not None:
        return _evaluate_frozen(frozen_sample, benchmark, choices)

    try:
        corpus = _load_corpus_for_benchmark(benchmark)
        components = build_pipeline_from_config(choices, benchmark, corpus=corpus)
    except _RealEvalFailure:
        raise
    except Exception as e:
        raise _RealEvalFailure(
            f"evaluate_config_real({choices}, {benchmark}) build failed: "
            f"{type(e).__name__}: {e}"
        ) from e

    generation = components.get("generation")
    if generation is None:
        raise _RealEvalFailure(
            f"evaluate_config_real({choices}, {benchmark}): generation is None "
            f"after build (component_registry did not return a generator)"
        )

    try:
        if benchmark == "hotpotqa":
            from benchmark.hotpotqa_adapter import (
                HotpotQABenchmarkAdapter,
                load_hotpotqa_sample,
            )
            data = load_hotpotqa_sample("benchmark/sample_data/hotpotqa_kg_sample")
            adapter = HotpotQABenchmarkAdapter()
            graph = _build_eval_graph(choices, components, benchmark="hotpotqa")
            if graph is not None:
                result = adapter.evaluate_pipeline(data, graph)
            else:
                result = adapter.evaluate_generation(data, generation)
            return result.avg_f1 / 100.0
        elif benchmark == "ultradomain":
            from benchmark.ultradomain_adapter import (
                UltraDomainBenchmarkAdapter,
                load_ultradomain_sample,
            )
            data = load_ultradomain_sample("benchmark/sample_data/ultradomain_kg_sample")
            adapter = UltraDomainBenchmarkAdapter(judge_disabled=True)
            graph = _build_eval_graph(choices, components, benchmark="ultradomain")
            if graph is not None:
                result = adapter.evaluate_pipeline(data, graph)
            else:
                result = adapter.evaluate_generation(data, generation)
            return result.avg_f1 / 100.0
        elif benchmark == "alce":
            from benchmark.alce_adapter import ALCEBenchmarkAdapter
            import json
            from pathlib import Path as _P
            docs_path = _P("benchmark/sample_data/alce_kg_sample/alce_docs.json")
            if not docs_path.exists():
                raise _RealEvalFailure(
                    f"evaluate_config_real(alce): missing sample data at {docs_path}"
                )
            with open(docs_path, encoding="utf-8") as f:
                all_docs = json.load(f)
            queries_path = _P("benchmark/sample_data/alce_kg_sample/queries.jsonl")
            data = []
            with open(queries_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        q = json.loads(line)
                        data.append({
                            "question": q["query"],
                            "answer": q.get("ground_truth", ""),
                            "docs": all_docs.get(q["query_id"], []),
                        })
            adapter = ALCEBenchmarkAdapter()
            factory = _build_alce_graph_factory(choices, components)
            if factory is not None:
                result = adapter.evaluate_pipeline(data, factory)
            else:
                result = adapter.evaluate_generation(data, generation)
            return result.avg_f1 / 100.0
        else:
            raise _RealEvalFailure(
                f"evaluate_config_real: unknown benchmark {benchmark!r}"
            )
    except _RealEvalFailure:
        raise
    except Exception as e:
        raise _RealEvalFailure(
            f"evaluate_config_real({choices}, {benchmark}) eval failed: "
            f"{type(e).__name__}: {e}"
        ) from e


def _build_eval_graph(choices: tuple[str, ...], components: dict, corpus=None, benchmark: str = "hotpotqa"):
    """Try to build a compiled LangGraph from components. Returns None on failure."""
    try:
        from ominirag_wtb.config_types import RAGConfig
        from ominirag_wtb.graph_factories import config_to_graph_factory
        config = RAGConfig.from_tuple(choices if len(choices) == 5 else ("standard_passage",) + choices)
        factory = config_to_graph_factory(config, corpus=corpus, benchmark=benchmark)
        return factory()
    except Exception:
        return None


def _build_alce_graph_factory(choices: tuple[str, ...], components: dict):
    """Build a graph_factory callable for ALCE (injects ALCEDocRetrieval per item)."""
    try:
        from ominirag_wtb.config_types import RAGConfig
        from ominirag_wtb.graph_factories import _infer_frame, _get_frame_builder
        from rag_contracts import ALCEDocRetrieval

        config = RAGConfig.from_tuple(choices if len(choices) == 5 else ("standard_passage",) + choices)
        frame = _infer_frame(config)
        builder = _get_frame_builder(frame)

        def factory(retrieval=None):
            ret = retrieval if retrieval is not None else components["retrieval"]
            return builder(
                retrieval=ret,
                generation=components["generation"],
                reranking=components.get("post_retrieval"),
                query=components.get("query"),
            )
        return factory
    except Exception:
        return None


def _evaluate_frozen(frozen_sample: list, benchmark: str, choices: tuple[str, ...]) -> float:
    """Evaluate a frozen BenchmarkItem sample with the full pipeline when possible.

    Supports two item formats:

    * **BenchmarkItem objects** (from ``SamplingEngine`` / HuggingFace) --
      converted to eval dicts using ``payload`` / ``target`` / ``metadata``.
    * **Plain eval dicts** (from ``build_frozen_samples_real``) --
      already carry ``question``, ``answer``, ``context_results``, etc.
      These are passed through directly.
    """
    try:
        if benchmark == "hotpotqa":
            from benchmark.hotpotqa_adapter import HotpotQABenchmarkAdapter
            eval_items = _frozen_to_eval_items_hotpotqa(frozen_sample)
            generation, graph = _try_build_pipeline(eval_items, choices, benchmark)

            adapter = HotpotQABenchmarkAdapter()
            if graph is not None:
                result = adapter.evaluate_pipeline(eval_items, graph)
            else:
                result = adapter.evaluate_generation(eval_items, generation)
            return result.avg_f1 / 100.0

        elif benchmark == "ultradomain":
            from benchmark.ultradomain_adapter import UltraDomainBenchmarkAdapter
            eval_items = _frozen_to_eval_items_ultradomain(frozen_sample)
            generation, graph = _try_build_pipeline(eval_items, choices, benchmark)

            adapter = UltraDomainBenchmarkAdapter(judge_disabled=True)
            if graph is not None:
                result = adapter.evaluate_pipeline(eval_items, graph)
            else:
                result = adapter.evaluate_generation(eval_items, generation)
            return result.avg_f1 / 100.0

        elif benchmark == "alce":
            from benchmark.alce_adapter import ALCEBenchmarkAdapter
            eval_items = []
            for bitem in frozen_sample:
                eval_items.append({
                    "question": bitem.payload["question"],
                    "answer": bitem.target["answer"],
                    "docs": bitem.payload.get("docs", [])[:5],
                    "qa_pairs": bitem.target.get("qa_pairs", []),
                    "query_id": bitem.item_id,
                })

            components = build_pipeline_from_config(choices, benchmark)
            generation = components.get("generation")
            if generation is None:
                raise _RealEvalFailure(
                    f"_evaluate_frozen(alce, {choices}): generation is None"
                )

            adapter = ALCEBenchmarkAdapter()
            factory = _build_alce_graph_factory(choices, components)
            if factory is not None:
                result = adapter.evaluate_pipeline(eval_items, factory)
            else:
                result = adapter.evaluate_generation(eval_items, generation)
            return result.avg_f1 / 100.0
        else:
            raise _RealEvalFailure(
                f"_evaluate_frozen: unknown benchmark {benchmark!r}"
            )
    except _RealEvalFailure:
        raise
    except Exception as e:
        raise _RealEvalFailure(
            f"_evaluate_frozen({choices}, {benchmark}): "
            f"{type(e).__name__}: {e}"
        ) from e


def _try_build_pipeline(eval_items, choices, benchmark):
    """Build pipeline components.

    Preferred path: full ``build_pipeline_from_config`` with corpus.  If
    that raises (e.g. missing optional dependency), we fall back to a
    *minimal* generation-only path that still allows scoring on
    pre-retrieved context, but only when the generator itself is well
    known.  Both paths now raise :class:`_RealEvalFailure` on hard
    failure -- callers no longer get ``(None, None)`` and silently
    decide what to do.
    """
    try:
        all_chunks = _collect_chunks(eval_items)
        corpus = _load_corpus_for_benchmark(benchmark, all_chunks or None)
        components = build_pipeline_from_config(choices, benchmark, corpus=corpus)
        generation = components.get("generation")
        if generation is None:
            raise _RealEvalFailure(
                f"_try_build_pipeline({choices}): generation is None after build"
            )
        graph = _build_eval_graph(
            choices, components, corpus=corpus, benchmark=benchmark
        )
        return generation, graph
    except _RealEvalFailure:
        raise
    except Exception as build_err:
        # Best-effort: try a minimal generator-only pipeline, useful when
        # the failure is in an optional retrieval/reranking dependency
        # but the eval items already carry ``context_results``.
        print(
            f"  [warn] _try_build_pipeline full-build failed for {choices} "
            f"on {benchmark}: {type(build_err).__name__}: {build_err}. "
            f"Falling back to generation-only.",
            flush=True,
        )
        try:
            gen_name = choices[-1] if choices else "simple_llm"
            if gen_name == "longrag_reader":
                generation = _build_longrag_generation()
            elif gen_name == "simple_llm":
                from rag_contracts.common_components import SimpleLLMGeneration
                generation = SimpleLLMGeneration(llm=_build_simple_llm())
            elif gen_name.startswith("simple_llm_ud_words_"):
                from rag_contracts.common_components import SimpleLLMGeneration
                word_count = int(gen_name.rsplit("_", 1)[1])
                generation = SimpleLLMGeneration(
                    llm=_build_simple_llm(),
                    benchmark=benchmark,
                    ud_word_count=word_count,
                )
            elif gen_name == "selfrag_generator":
                _, generation = _build_selfrag_components()
            else:
                from rag_contracts.identity import IdentityGeneration
                generation = IdentityGeneration()
            return generation, None
        except Exception as fallback_err:
            raise _RealEvalFailure(
                f"_try_build_pipeline({choices}, {benchmark}): "
                f"build failed ({type(build_err).__name__}: {build_err}) "
                f"and minimal-gen fallback also failed "
                f"({type(fallback_err).__name__}: {fallback_err})"
            ) from fallback_err


def _is_plain_dict(item) -> bool:
    """True when *item* is a plain eval dict (from ``load_*_real``)."""
    return isinstance(item, dict) and "question" in item


def _frozen_to_eval_items_hotpotqa(frozen_sample: list) -> list[dict]:
    """Convert frozen sample to HotpotQA eval dicts.

    Plain dicts (from ``load_hotpotqa_real``) are passed through unchanged.
    BenchmarkItem objects are converted using their ``payload`` / ``target``.
    """
    eval_items: list[dict] = []
    for bitem in frozen_sample:
        if _is_plain_dict(bitem):
            eval_items.append(bitem)
            continue
        p = bitem.payload
        chunks: dict = {}
        for idx, (title, sents) in enumerate(
            zip(p.get("context_titles", []), p.get("context_sentences", []))
        ):
            chunk_id = f"c{bitem.item_id}_{idx}"
            chunk_data = {"content": f"[{title}] " + " ".join(sents), "doc_ids": [title]}
            chunks[chunk_id] = chunk_data
        eval_items.append({
            "question": p["question"], "answer": bitem.target["answer"],
            "query_id": bitem.item_id, "chunks": chunks,
        })
    return eval_items


def _frozen_to_eval_items_ultradomain(frozen_sample: list) -> list[dict]:
    """Convert frozen sample to UltraDomain eval dicts."""
    eval_items: list[dict] = []
    for bitem in frozen_sample:
        if _is_plain_dict(bitem):
            eval_items.append(bitem)
            continue
        ctx = (bitem.payload.get("context") or "")[:3000]
        domain = bitem.metadata.get("domain", "unknown")
        chunk_id = f"c{bitem.item_id}"
        if ctx:
            chunk_data = {"content": ctx, "doc_ids": [domain]}
            chunks = {chunk_id: chunk_data}
        else:
            chunks = {}
        eval_items.append({
            "question": bitem.payload.get("query") or "",
            "answer": bitem.target.get("answer") or "",
            "domain": domain, "query_id": bitem.item_id, "chunks": chunks,
        })
    return eval_items


def _collect_chunks(eval_items: list[dict]) -> dict:
    """Aggregate all ``chunks`` dicts from eval items (for corpus building)."""
    all_chunks: dict = {}
    for item in eval_items:
        for k, v in item.get("chunks", {}).items():
            all_chunks[k] = v
    return all_chunks


def build_frozen_samples(budget: int = 30, seed: int = 42) -> dict[str, list]:
    """Pre-draw a frozen sample per cluster using SamplingEngine.

    Returns ``{cluster_id: list[BenchmarkItem]}``.
    Only builds samples for clusters whose HF data is available locally.
    """
    import os
    from pathlib import Path
    from bsamp.sampling.engine import SamplingEngine

    _hf = Path(os.environ.get("HF_HUB_DIR", str(Path.home() / ".cache" / "huggingface" / "hub")))
    samples: dict[str, list] = {}

    hq_root = _hf / "datasets--hotpotqa--hotpot_qa" / "snapshots" / "1908d6afbbead072334abe2965f91bd2709910ab"
    if hq_root.exists():
        from bsamp.sampling.adapters.hotpotqa import HotpotQAAdapter
        adapter = HotpotQAAdapter(str(hq_root))
        result = SamplingEngine(adapter=adapter, method="proportional", budget=budget, seed=seed).run()
        samples["hotpotqa"] = result.items

    ud_root = _hf / "datasets--TommyChien--UltraDomain" / "snapshots" / "aa8a51d523f8fc3c5a0ab90dd16b7f6b9dbb5d0d"
    if ud_root.exists():
        from bsamp.sampling.adapters.ultradomain import UltraDomainAdapter
        adapter = UltraDomainAdapter(str(ud_root), target_domains=["physics", "cs", "legal"])
        result = SamplingEngine(adapter=adapter, method="proportional", budget=budget, seed=seed).run()
        samples["ultradomain"] = result.items

    alce_root = _hf / "datasets--princeton-nlp--ALCE-data" / "snapshots" / "334fa2e7dd32040c3fef931a123c4be1a81e91a0"
    if alce_root.exists():
        from bsamp.sampling.adapters.alce import ALCEAdapter
        adapter = ALCEAdapter(str(alce_root), subsets=["asqa"])
        result = SamplingEngine(adapter=adapter, method="proportional", budget=budget, seed=seed).run()
        samples["alce"] = result.items

    return samples


def build_frozen_samples_real(
    data_dir: str,
    max_items: int = 30,
    ud_domain: str = "mix",
) -> dict[str, list]:
    """Load real datasets from ``data_dir`` and wrap them as frozen eval dicts.

    Unlike ``build_frozen_samples()`` (which uses HuggingFace parquet via
    ``SamplingEngine``), this reads the raw JSON/JSONL files produced by the
    new ``load_hotpotqa_real`` / ``load_ultradomain_real`` loaders and
    returns plain eval-item dicts that ``_evaluate_frozen_real()`` consumes
    directly.

    Supported layouts::

        data_dir/
          all_data/hotpotqa/hotpotqa.json    (original)
          UltraDomain/mix.jsonl

        OR (fullwiki layout):

        data_dir/
          fullwiki/fullwiki_sample_500_uniform.parquet
          ultradomain/mix.jsonl
    """
    from pathlib import Path
    samples: dict[str, list] = {}

    hotpotqa_dir = Path(data_dir) / "all_data" / "hotpotqa"
    fullwiki_parquet = Path(data_dir) / "fullwiki" / "fullwiki_sample_500_uniform.parquet"

    if hotpotqa_dir.exists():
        from benchmark.hotpotqa_adapter import load_hotpotqa_real
        items = load_hotpotqa_real(hotpotqa_dir, max_items=max_items)
        samples["hotpotqa"] = items
    elif fullwiki_parquet.exists():
        items = _load_hotpotqa_parquet(fullwiki_parquet, max_items=max_items)
        samples["hotpotqa"] = items

    ud_dir = Path(data_dir) / "UltraDomain"
    ud_dir_alt = Path(data_dir) / "ultradomain"
    if ud_dir.exists():
        from benchmark.ultradomain_adapter import load_ultradomain_real
        items = load_ultradomain_real(ud_dir, domain=ud_domain, max_items=max_items)
        samples["ultradomain"] = items
    elif ud_dir_alt.exists():
        from benchmark.ultradomain_adapter import load_ultradomain_real
        items = load_ultradomain_real(ud_dir_alt, domain=ud_domain, max_items=max_items)
        samples["ultradomain"] = items

    return samples


def _load_hotpotqa_parquet(path, max_items: int | None = None) -> list[dict]:
    """Load HotpotQA items from the fullwiki parquet file."""
    import pandas as pd
    from benchmark.base_adapter import hotpotqa_context_to_retrieval_results

    df = pd.read_parquet(path)
    if max_items is not None:
        df = df.head(max_items)

    items: list[dict] = []
    for _, row in df.iterrows():
        context_raw = row.get("context", [])
        if isinstance(context_raw, list) and context_raw:
            context_results = hotpotqa_context_to_retrieval_results(context_raw)
        else:
            context_results = []
        items.append({
            "question": row["question"],
            "answer": row["answer"],
            "query_id": row.get("id", ""),
            "type": row.get("type", ""),
            "level": row.get("level", ""),
            "context_results": context_results,
        })
    return items


# ---------------------------------------------------------------------------
# SearchState
# ---------------------------------------------------------------------------

class RAGPipelineSearchState:
    """Partial RAG pipeline config being assembled slot by slot."""

    def __init__(self, choices: tuple[str, ...] = ()) -> None:
        self.choices = choices

    def is_terminal(self) -> bool:
        return len(self.choices) == len(SLOT_NAMES)

    def available_actions(self) -> list[Hashable]:
        depth = len(self.choices)
        if depth >= len(ACTIVE_SLOT_OPTIONS):
            return []
        options = list(ACTIVE_SLOT_OPTIONS[depth])

        # Prune retrieval options that violate chunking constraint
        if depth == 2 and len(self.choices) >= 1:
            chunking = self.choices[0]
            options = [
                o for o in options
                if CHUNKING_RETRIEVAL_CONSTRAINTS.get(o, chunking) == chunking
            ]

        return options

    def child(self, action: Hashable) -> "RAGPipelineSearchState":
        return RAGPipelineSearchState(self.choices + (str(action),))

    def state_key(self) -> Hashable:
        return self.choices

    def pretty(self) -> str:
        parts = [f"{SLOT_NAMES[i]}={v}" for i, v in enumerate(self.choices)]
        return "(" + ", ".join(parts) + ")"

    def path_key_for_action(self, action: Hashable) -> Hashable:
        return self.choices + (str(action),)


# ---------------------------------------------------------------------------
# Benchmark spectrums (7 fine-grained evaluation clusters)
# ---------------------------------------------------------------------------

SPECTRUM_IDS = [
    "hq_easy", "hq_medium", "hq_hard",
    "ud_agriculture", "ud_cs", "ud_legal", "ud_mix",
]

# Backward-compatible alias
CLUSTER_IDS = SPECTRUM_IDS

SPECTRUM_TO_BENCHMARK: dict[str, str] = {
    "hq_easy": "hotpotqa", "hq_medium": "hotpotqa", "hq_hard": "hotpotqa",
    "ud_agriculture": "ultradomain", "ud_cs": "ultradomain",
    "ud_legal": "ultradomain", "ud_mix": "ultradomain",
}

CLUSTER_NOISE: dict[str, float] = {
    "hq_easy":  0.04, "hq_medium":  0.02, "hq_hard": -0.02,
    "ud_agriculture": 0.00, "ud_cs": -0.01, "ud_legal": -0.02, "ud_mix": 0.01,
}

CLUSTER_COST: dict[str, float] = {
    "hq_easy": 1.0, "hq_medium": 1.0, "hq_hard": 1.0,
    "ud_agriculture": 1.2, "ud_cs": 1.2, "ud_legal": 1.2, "ud_mix": 1.2,
}

DATASET_AFFINITY: dict[tuple[str, str], float] = {
    ("lightrag_graph", "ud_agriculture"):  0.04,
    ("lightrag_graph", "ud_cs"):           0.04,
    ("lightrag_graph", "ud_legal"):        0.04,
    ("lightrag_graph", "ud_mix"):          0.04,
    ("selfrag_critique", "hq_easy"):       0.01,
    ("selfrag_critique", "hq_medium"):     0.03,
    ("selfrag_critique", "hq_hard"):       0.05,
    ("cross_encoder", "hq_easy"):          0.01,
    ("cross_encoder", "hq_medium"):        0.02,
    ("cross_encoder", "hq_hard"):          0.03,
    ("bm25", "hq_easy"):                   0.03,
    ("bm25", "hq_medium"):                 0.02,
    ("bm25", "hq_hard"):                   0.01,
    ("bm25_dense_hybrid", "ud_agriculture"): 0.03,
    ("bm25_dense_hybrid", "ud_cs"):          0.03,
    ("bm25_dense_hybrid", "ud_legal"):       0.03,
    ("bm25_dense_hybrid", "ud_mix"):         0.03,
}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGPipelineEvaluator(Evaluator):
    """Evaluates a complete RAG pipeline configuration across 7 spectrum clusters.

    Supports two modes:
    - Full evaluation: evaluates ALL items per spectrum (legacy behavior)
    - Sequential batch sampling: draws initial_batch items, then accumulates
      in steps of +5 until SE < se_threshold or max_items_per_spectrum reached.

    The evaluator reports ``drawn_qids_per_spectrum`` in result.metadata
    so the engine can update cached_qids at all prefix depths.
    """

    def __init__(self, use_real: bool = False,
                 frozen_samples: dict[str, list] | None = None,
                 *,
                 batch_size: int = 0,
                 se_threshold: float = 0.05,
                 max_items_per_spectrum: int = 30,
                 batch_step: int = 5) -> None:
        self.use_real = use_real
        self.frozen_samples = frozen_samples or {}
        self.batch_size = batch_size
        self.se_threshold = se_threshold
        self.max_items_per_spectrum = max_items_per_spectrum
        self.batch_step = batch_step

        # Layer B: per-item score cache  (config_key, qid) -> reward
        self.score_cache: dict[tuple[str, str], float] = {}
        self._score_cache_stats = {"hits": 0, "misses": 0}

    def score_cache_stats(self) -> dict[str, int]:
        """Return Layer-B score cache hit/miss counters."""
        return dict(self._score_cache_stats)

    def evaluate(self, state: SearchState, context: SearchContext) -> EvaluationResult:
        assert state.is_terminal(), "evaluator expects a terminal state"
        choices: tuple[str, ...] = state.state_key()  # type: ignore[assignment]

        # Determine effective batch size (cold_start override or configured)
        effective_batch = context.extra.get("_cold_start_budget", self.batch_size)

        cluster_results: list[BenchmarkClusterResult] = []
        total_cost = 0.0
        active_spectrums = [s for s in SPECTRUM_IDS if self.frozen_samples.get(s)]
        if not active_spectrums:
            active_spectrums = SPECTRUM_IDS

        group_rewards: dict[str, list[float]] = {}
        all_drawn_qids: dict[str, list[str]] = {}

        iter_count = context.total_evaluations
        base_seed = 42 + iter_count * 7919

        for sid in active_spectrums:
            benchmark = SPECTRUM_TO_BENCHMARK.get(sid, "hotpotqa")

            if self.use_real:
                frozen = self.frozen_samples.get(sid)
                if frozen and effective_batch > 0:
                    reward, drawn_qids = self._evaluate_spectrum_sequential(
                        choices, sid, frozen, benchmark, base_seed)
                elif frozen:
                    reward, drawn_qids = self._evaluate_spectrum_fullpool(
                        choices, sid, frozen, benchmark)
                else:
                    reward = 0.0
                    drawn_qids = []
            else:
                reward = _compute_reward(choices) + CLUSTER_NOISE.get(sid, 0.0)
                for component in choices:
                    reward += DATASET_AFFINITY.get((component, sid), 0.0)
                drawn_qids = []

            all_drawn_qids[sid] = drawn_qids
            group_rewards.setdefault(benchmark, []).append(reward)

            new_keys: list[Hashable] = []
            cluster_cost = 0.0
            for depth in range(1, len(choices) + 1):
                prefix_key: Hashable = (choices[:depth], sid)
                if prefix_key not in context.materialized_keys:
                    new_keys.append(prefix_key)
                    cluster_cost += CLUSTER_COST.get(sid, 1.0) / len(choices)

            cluster_results.append(BenchmarkClusterResult(
                cluster_id=sid,
                reward=reward,
                cost=cluster_cost,
                materialized_keys=new_keys,
                metadata={
                    "config": {SLOT_NAMES[i]: v for i, v in enumerate(choices)},
                    "spectrum": sid,
                    "benchmark": benchmark,
                    "real_eval": self.use_real,
                    "n_items_evaluated": len(drawn_qids),
                },
            ))
            total_cost += cluster_cost

        group_means = {
            g: sum(rs) / len(rs) for g, rs in group_rewards.items()
        }
        total_reward = sum(group_means.values()) / len(group_means) if group_means else 0.0

        return EvaluationResult(
            reward=total_reward,
            total_cost=total_cost,
            cluster_results=cluster_results,
            metadata={
                "config_pretty": state.pretty(),
                "real_eval": self.use_real,
                "group_means": group_means,
                "drawn_qids_per_spectrum": all_drawn_qids,
            },
        )

    @staticmethod
    def _item_qid(it: dict) -> str:
        return str(it.get("query_id") or it.get("_id") or it.get("id", ""))

    def _eval_item_with_cache(
        self,
        choices: tuple[str, ...],
        benchmark: str,
        it: dict,
    ) -> float:
        """Evaluate a single item, using Layer-B score cache when available."""
        config_key = "/".join(choices)
        qid = self._item_qid(it)
        sc_key = (config_key, qid)
        if sc_key in self.score_cache:
            self._score_cache_stats["hits"] += 1
            return self.score_cache[sc_key]
        try:
            r = evaluate_config_real(choices, benchmark=benchmark, frozen_sample=[it])
        except _RealEvalFailure:
            r = 0.0
        self.score_cache[sc_key] = r
        self._score_cache_stats["misses"] += 1
        return r

    def _evaluate_spectrum_sequential(
        self,
        choices: tuple[str, ...],
        sid: str,
        all_items: list,
        benchmark: str,
        base_seed: int,
    ) -> tuple[float, list[str]]:
        """Sequential accumulation until SE < threshold or max reached.

        Uses Layer-B score cache: items already evaluated for this exact
        (config, qid) return cached reward without pipeline execution.

        Returns (mean_reward, list_of_drawn_qids).
        """
        import math as _math

        rng = random.Random(base_seed + hash(sid))
        pool = list(all_items)
        rng.shuffle(pool)

        initial_n = min(self.batch_size, len(pool))
        max_n = min(self.max_items_per_spectrum, len(pool))

        drawn_items: list[dict] = []
        item_rewards: list[float] = []
        idx = 0

        # Phase 1: initial batch
        batch = pool[idx:idx + initial_n]
        idx += initial_n
        drawn_items.extend(batch)

        for it in batch:
            item_rewards.append(
                self._eval_item_with_cache(choices, benchmark, it)
            )

        # Phase 2: sequential accumulation until CL convergence
        while idx < max_n and len(item_rewards) >= 2:
            n = len(item_rewards)
            mean = sum(item_rewards) / n
            variance = sum((x - mean) ** 2 for x in item_rewards) / (n - 1)
            se = _math.sqrt(variance / n) if variance > 0 else 0.0
            if se < self.se_threshold:
                print(
                    f"  [sampling] {sid}: n={n}/{max_n} "
                    f"SE={se:.4f} converged=True",
                    flush=True,
                )
                break
            step_n = min(self.batch_step, max_n - idx)
            if step_n <= 0:
                break
            next_batch = pool[idx:idx + step_n]
            idx += step_n
            drawn_items.extend(next_batch)
            for it in next_batch:
                item_rewards.append(
                    self._eval_item_with_cache(choices, benchmark, it)
                )
        else:
            if len(item_rewards) >= 2:
                n = len(item_rewards)
                mean = sum(item_rewards) / n
                variance = sum((x - mean) ** 2 for x in item_rewards) / (n - 1)
                se = _math.sqrt(variance / n) if variance > 0 else 0.0
                print(
                    f"  [sampling] {sid}: n={n}/{max_n} "
                    f"SE={se:.4f} converged={se < self.se_threshold}",
                    flush=True,
                )

        mean_reward = sum(item_rewards) / len(item_rewards) if item_rewards else 0.0
        drawn_qids = [
            self._item_qid(it)
            for it in drawn_items if isinstance(it, dict)
        ]
        return mean_reward, drawn_qids

    def _evaluate_spectrum_fullpool(
        self,
        choices: tuple[str, ...],
        sid: str,
        all_items: list,
        benchmark: str,
    ) -> tuple[float, list[str]]:
        """Full-pool evaluation with Layer-B score cache.

        Each item is individually checked against the cache. Only uncached
        items go through ``evaluate_config_real``. Returns (mean_reward,
        list_of_drawn_qids).
        """
        item_rewards: list[float] = []
        for it in all_items:
            if isinstance(it, dict):
                item_rewards.append(
                    self._eval_item_with_cache(choices, benchmark, it)
                )
            else:
                try:
                    reward = evaluate_config_real(
                        choices, benchmark=benchmark, frozen_sample=[it])
                except _RealEvalFailure as fail:
                    print(
                        f"  [FAIL (reward=0)] {':'.join(choices)} @ {sid}: "
                        f"{fail}",
                        flush=True,
                    )
                    reward = 0.0
                item_rewards.append(reward)

        mean_reward = sum(item_rewards) / len(item_rewards) if item_rewards else 0.0
        drawn_qids = [
            str(it.get("query_id") or it.get("_id") or it.get("id", ""))
            for it in all_items if isinstance(it, dict)
        ]
        return mean_reward, drawn_qids


# ---------------------------------------------------------------------------
# WTB Cache-Aware Evaluator factory
# ---------------------------------------------------------------------------

def _build_wtb_evaluator(
    use_real: bool = False,
    frozen_samples: dict[str, list] | None = None,
    *,
    batch_size: int = 0,
    se_threshold: float = 0.05,
    max_items_per_spectrum: int = 30,
    batch_step: int = 5,
    score_cache_log_path: str | None = None,
    ledger_path: str | None = None,
    wtb_data_dir: str | None = None,
) -> "Evaluator":
    """Build a ``RAGCacheAwareEvaluator`` backed by a WTB bench + ledger.

    ``ledger_path`` and ``wtb_data_dir`` default to a fresh tempdir
    (matches legacy behaviour).  Phase-1 launchers and the resume
    driver pass run-dir paths so the Layer-2 ledger and WTB checkpoint
    store survive across processes (see resume plan section B).
    """
    import os as _os
    import tempfile
    from ominirag_wtb import RAGCacheAwareEvaluator, ReuseLedger

    if ledger_path is None:
        tmp = tempfile.mkdtemp(prefix="wtb_reuse_")
        ledger_path = str(__import__("pathlib").Path(tmp) / "reuse_ledger.db")
        if wtb_data_dir is None:
            wtb_data_dir = tmp
    else:
        _os.makedirs(_os.path.dirname(ledger_path) or ".", exist_ok=True)
        if wtb_data_dir is None:
            wtb_data_dir = _os.path.dirname(ledger_path) or "."
        _os.makedirs(wtb_data_dir, exist_ok=True)

    ledger = ReuseLedger(db_path=ledger_path)

    bench = None
    if use_real:
        from wtb.sdk import WTBTestBench
        bench = WTBTestBench.create(mode="development", data_dir=wtb_data_dir)

    bq_samples: dict[str, list] = {}
    if frozen_samples:
        from ominirag_wtb.config_types import BenchmarkQuestion
        for cid, items in frozen_samples.items():
            bq_list = []
            for it in items:
                if isinstance(it, dict):
                    qid = it.get("query_id", it.get("_id", ""))
                    bq = BenchmarkQuestion(
                        question_id=f"{cid}::{qid}",
                        cluster_id=cid,
                        stratum=cid,
                        question=it.get("question", it.get("input", "")),
                        payload=it,
                        target={"answer": it.get("answer", "")},
                        metadata={},
                    )
                else:
                    bq = BenchmarkQuestion.from_benchmark_item(it, cid)
                bq_list.append(bq)
            bq_samples[cid] = bq_list

    return RAGCacheAwareEvaluator(
        ledger=ledger,
        bench=bench,
        cluster_ids=CLUSTER_IDS,
        frozen_samples=bq_samples,
        cluster_costs=CLUSTER_COST,
        use_real=use_real,
        reward_table=REWARD_TABLE,
        default_reward=DEFAULT_REWARD,
        spectrum_to_benchmark=SPECTRUM_TO_BENCHMARK,
        batch_size=batch_size,
        se_threshold=se_threshold,
        max_items_per_spectrum=max_items_per_spectrum,
        batch_step=batch_step,
        score_cache_log_path=score_cache_log_path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _assign_hq_spectrum(entry: dict) -> str:
    level = str(entry.get("level", "nan")).lower()
    if level == "easy":
        return "hq_easy"
    if level == "medium":
        return "hq_medium"
    return "hq_hard"


def _assign_ud_spectrum(entry: dict) -> str:
    domain = entry.get("_domain", entry.get("domain", "unknown"))
    return f"ud_{domain}"


def load_search_set(path: str) -> dict[str, list]:
    """Load a pre-built search set JSON and assign items to spectrum IDs.

    Items carry a ``_spectrum`` field from ``build_stratified_search_set.py``.
    If absent, the spectrum is inferred from ``level`` (HotpotQA) or
    ``_domain`` (UltraDomain).

    Returns ``{spectrum_id: list[eval_dict]}``.
    """
    import json as _json
    from pathlib import Path as _P

    path_obj = _P(path)
    with open(path_obj, "r", encoding="utf-8") as f:
        items = _json.load(f)

    if not items:
        return {}

    result: dict[str, list] = {}

    if "question" in items[0]:
        for entry in items:
            spectrum = entry.get("_spectrum") or _assign_hq_spectrum(entry)
            eval_item = {
                "question": entry["question"],
                "answer": entry["answer"],
                "query_id": entry.get("id", ""),
                "type": entry.get("type", ""),
                "level": entry.get("level", ""),
            }
            result.setdefault(spectrum, []).append(eval_item)
    elif "input" in items[0]:
        for entry in items:
            spectrum = entry.get("_spectrum") or _assign_ud_spectrum(entry)
            eval_item = {
                "question": entry.get("input", ""),
                "answer": (entry.get("answers", [""])[0]
                           if isinstance(entry.get("answers"), list)
                           else entry.get("answers", "")),
                "query_id": entry.get("_id", ""),
                "domain": entry.get("_domain", "unknown"),
            }
            result.setdefault(spectrum, []).append(eval_item)

    return result


def _serialize_tree(root_node, slot_names: list[str]) -> dict:
    """Recursively serialize a TreeNode into a JSON-safe dict."""
    configs_visited: list[dict] = []

    def _collect(node, depth=0, prefix=()):
        if depth == len(slot_names):
            if node.visit_count > 0:
                configs_visited.append({
                    "config": {slot_names[i]: v for i, v in enumerate(prefix)},
                    "config_tuple": list(prefix),
                    "q_value": node.q_value,
                    "visit_count": node.visit_count,
                    "value_sum": node.value_sum,
                    "best_value": node.best_value,
                })
            return
        for act, ch in node.children.items():
            _collect(ch, depth + 1, prefix + (str(act),))

    _collect(root_node)
    configs_visited.sort(key=lambda x: x["q_value"], reverse=True)

    slot_breakdown = {}
    for action, child in root_node.children.items():
        slot_breakdown[str(action)] = {
            "visit_count": child.visit_count,
            "q_value": child.q_value,
            "best_value": child.best_value,
        }

    return {
        "slot_breakdown": slot_breakdown,
        "configs_visited": configs_visited,
    }


def main() -> None:
    from pathlib import Path as _PP
    _env = _PP(__file__).resolve().parents[3] / ".env"
    if _env.exists():
        from dotenv import load_dotenv
        load_dotenv(_env)

    import argparse
    import json as _json
    import random

    parser = argparse.ArgumentParser(description="AG-UCT RAG pipeline search")
    parser.add_argument("--real", action="store_true", help="Use real pipeline evaluation (sample_data)")
    parser.add_argument("--use-hf", action="store_true",
                        help="Use HuggingFace datasets via SamplingEngine (implies --real)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to real dataset root (e.g. /data1/ragworkspace/dataset). "
                             "Loads HotpotQA and UltraDomain from raw JSON/JSONL. Implies --real.")
    parser.add_argument("--search-set", type=str, nargs="+", default=None,
                        help="Path(s) to pre-built search set JSON files "
                             "(from build_stratified_search_set.py). "
                             "Pass one per dataset. Implies --real.")
    parser.add_argument("--ud-domain", type=str, default="mix",
                        help="UltraDomain domain file to load when --data-dir is set (default: mix)")
    parser.add_argument("--wtb-reuse", action="store_true", default=None,
                        help="Use WTB cache-aware evaluator with bipartite reuse ledger "
                             "(auto-enabled when --real is set; use --no-wtb-reuse to disable)")
    parser.add_argument("--no-wtb-reuse", action="store_true",
                        help="Disable WTB reuse even when --real is set")
    parser.add_argument("--budget", type=int, default=None,
                        help="Cap items per spectrum for real evaluation "
                             "(default: None = use all items in each spectrum)")
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iter-log", type=str, default=None,
                        help="Path for per-iteration JSONL log "
                             "(auto-generated if omitted: results/logs/uct_iter_log_<ts>.jsonl)")
    parser.add_argument("--save-tree", type=str, default=None,
                        help="Save search tree (node visits, Q-values, configs) to JSON file")
    parser.add_argument("--output-stats", type=str, default=None,
                        help="Save SearchResult summary to JSON file")
    parser.add_argument("--tree-snapshot-every", type=int, default=0,
                        help="If >0, every N iters reconstruct a (visits, Q-value) "
                             "tree from the live iter_log.jsonl and write it to "
                             "<iter_log_dir>/tree_snapshot.json (atomic temp+rename). "
                             "Survives mid-run crashes; final --save-tree still wins.")
    parser.add_argument("--snapshot-out", type=str, default=None,
                        help="Override path for the periodic tree_snapshot.json "
                             "(default: sibling of --iter-log)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of parallel items to evaluate per spectrum "
                             "(default: 1 = serial). Sets OMINIRAG_EVAL_CONCURRENCY.")
    parser.add_argument("--cold-start", type=int, default=0,
                        help="Enable cold-start: if >0, seeds all root children "
                             "with full-pool evaluation (value is ignored, just >0 to enable)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Initial batch size per spectrum per iteration for "
                             "sequential CL sampling (default: 10, 0 for full eval)")
    parser.add_argument("--se-threshold", type=float, default=0.05,
                        help="SE convergence threshold for sequential sampling "
                             "(default: 0.05)")
    parser.add_argument("--max-items-per-spectrum", type=int, default=30,
                        help="Hard cap on items per spectrum per iteration "
                             "(default: 30)")
    parser.add_argument("--lambda-0", type=float, default=0.05,
                        help="Initial lambda_0 for cost penalty decay "
                             "(lambda_t = lambda_0/sqrt(t), default: 0.05)")
    parser.add_argument("--score-cache-log", type=str, default=None,
                        help="Sidecar JSONL path receiving every fresh "
                             "(config_key,qid)->f1 entry from the Layer-B "
                             "score cache. Required for resume warm-start; "
                             "default sibling of --iter-log "
                             "(score_cache.jsonl).")
    parser.add_argument("--ledger-path", type=str, default=None,
                        help="Persist the Layer-2 ReuseLedger SQLite at "
                             "this path so WTB physical reuse survives "
                             "resume. Default: tempfile (legacy ephemeral).")
    parser.add_argument("--wtb-data-dir", type=str, default=None,
                        help="WTB checkpoint store directory. Default: "
                             "alongside --ledger-path (or tempfile).")
    parser.add_argument("--slot-options-json", type=str, default=None,
                        help="JSON array overriding the 5-slot search space. "
                             "Use this for controlled ablations, e.g. "
                             "'[[\"standard_passage\"],[\"identity\"],...]'. "
                             "Default: historical 384-config space.")
    args = parser.parse_args()

    if args.slot_options_json:
        try:
            override = _json.loads(args.slot_options_json)
            set_active_slot_options(override)
        except Exception as exc:
            raise SystemExit(
                f"  [slot-options] FATAL: invalid --slot-options-json: {exc}"
            ) from exc
        print("  [slot-options] ACTIVE override:", flush=True)
        for name, opts in zip(SLOT_NAMES, ACTIVE_SLOT_OPTIONS):
            print(f"    {name}: {opts}", flush=True)

    use_real = args.real or args.use_hf or args.data_dir is not None or args.search_set is not None

    if args.concurrency > 1:
        import os as _os
        _os.environ["OMINIRAG_EVAL_CONCURRENCY"] = str(args.concurrency)
        print(f"  [concurrency] {args.concurrency}x parallel items per spectrum", flush=True)

    frozen_samples: dict[str, list] | None = None

    if args.search_set is not None:
        frozen_samples = {}
        for ss_path in args.search_set:
            print(f"Loading search set from {ss_path} ...", flush=True)
            partial = load_search_set(ss_path)
            for sid, sitems in partial.items():
                frozen_samples.setdefault(sid, []).extend(sitems)
        if args.budget is not None:
            _rng = random.Random(args.seed)
            for sid in frozen_samples:
                if len(frozen_samples[sid]) > args.budget:
                    _rng.shuffle(frozen_samples[sid])
                    frozen_samples[sid] = frozen_samples[sid][:args.budget]
            print(f"  Budget cap: {args.budget} items per spectrum", flush=True)
        for sid in sorted(frozen_samples):
            print(f"  {sid}: {len(frozen_samples[sid])} items", flush=True)
        total_items = sum(len(v) for v in frozen_samples.values())
        print(f"  Total: {total_items} items across {len(frozen_samples)} spectrums", flush=True)
        if not frozen_samples:
            print("  WARNING: Empty search set. Falling back to sample_data.", flush=True)
    elif args.data_dir is not None:
        print(f"Loading real datasets from {args.data_dir} ...", flush=True)
        frozen_samples = build_frozen_samples_real(
            data_dir=args.data_dir,
            max_items=args.budget or 30,
            ud_domain=args.ud_domain,
        )
        for cid, items in frozen_samples.items():
            print(f"  {cid}: {len(items)} items", flush=True)
        if not frozen_samples:
            print("  WARNING: No real datasets found. Falling back to sample_data.", flush=True)
    elif args.use_hf:
        print("Building frozen HF samples ...", flush=True)
        frozen_samples = build_frozen_samples(budget=args.budget or 30, seed=args.seed)
        for cid, items in frozen_samples.items():
            print(f"  {cid}: {len(items)} items", flush=True)
        if not frozen_samples:
            print("  WARNING: No HF datasets found locally. Falling back to sample_data.", flush=True)

    # Resolve iter_log path eagerly so resume artifacts can default to
    # its sibling directory (score_cache.jsonl, wtb_reuse/reuse_ledger.db).
    if args.iter_log is None:
        import datetime as _dt
        _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.iter_log = f"results/logs/uct_iter_log_{_ts}.jsonl"
    _iter_log_dir = _PP(args.iter_log).parent
    _iter_log_dir.mkdir(parents=True, exist_ok=True)

    if args.score_cache_log is None:
        args.score_cache_log = str(_iter_log_dir / "score_cache.jsonl")
    if args.ledger_path is None and use_real:
        args.ledger_path = str(_iter_log_dir / "wtb_reuse" / "reuse_ledger.db")
    if args.wtb_data_dir is None and args.ledger_path is not None:
        args.wtb_data_dir = str(_PP(args.ledger_path).parent)

    enable_wtb = args.wtb_reuse or (use_real and not args.no_wtb_reuse)
    if enable_wtb:
        print("  [wtb-reuse] ENABLED (physical caching via WTB)", flush=True)
        print(f"    ledger     : {args.ledger_path}", flush=True)
        print(f"    wtb_data   : {args.wtb_data_dir}", flush=True)
        print(f"    score_log  : {args.score_cache_log}", flush=True)
        evaluator = _build_wtb_evaluator(
            use_real=use_real, frozen_samples=frozen_samples,
            batch_size=args.batch_size,
            se_threshold=args.se_threshold,
            max_items_per_spectrum=args.max_items_per_spectrum,
            score_cache_log_path=args.score_cache_log,
            ledger_path=args.ledger_path,
            wtb_data_dir=args.wtb_data_dir,
        )
    else:
        print("  [wtb-reuse] DISABLED (logical cost only)", flush=True)
        evaluator = RAGPipelineEvaluator(
            use_real=use_real,
            frozen_samples=frozen_samples,
            batch_size=args.batch_size,
            se_threshold=args.se_threshold,
            max_items_per_spectrum=args.max_items_per_spectrum,
        )
    scorer = CostAwareUCTScorer(lambda_0=args.lambda_0)
    clusters = [
        ClusterDef(cid, weight=1.0, base_cost=CLUSTER_COST[cid])
        for cid in CLUSTER_IDS
    ]
    cost_model = ReuseAwareCostModel(clusters=clusters)

    engine = UCTSearchEngine(
        evaluator=evaluator,
        scorer=scorer,
        cost_model=cost_model,
        exploration_constant=1.4,
        random_seed=args.seed,
    )

    # --- Per-iteration JSONL logger ---
    from uct_engine import IterationRecord
    import datetime

    # Install the process-global token tracker BEFORE any pipeline LLM call.
    # Idempotent; the call publishes the install banner to stdout.
    try:
        from ominirag_wtb.token_tracker import install as _tt_install
        from ominirag_wtb.token_tracker import (  # noqa: E402
            snapshot_delta as _tt_snapshot_delta,
            get_totals as _tt_get_totals,
        )
        _tt_install()
        _HAS_TOKEN_TRACKER = True
    except Exception as _tt_exc:
        print(f"  [token-tracker] WARN: not installed ({_tt_exc})", flush=True)
        _tt_snapshot_delta = lambda: None  # type: ignore[assignment]
        _tt_get_totals = lambda: None  # type: ignore[assignment]
        _HAS_TOKEN_TRACKER = False

    iter_log_path = args.iter_log  # resolved earlier (before evaluator build)
    iter_log_pp = _PP(iter_log_path)
    iter_log_pp.parent.mkdir(parents=True, exist_ok=True)
    _iter_log_fh = open(iter_log_pp, "w", encoding="utf-8")

    snapshot_every = max(0, int(args.tree_snapshot_every or 0))
    if args.snapshot_out:
        snapshot_path = _PP(args.snapshot_out)
    else:
        snapshot_path = iter_log_pp.with_name("tree_snapshot.json")

    def _atomic_snapshot(iter_num: int) -> None:
        """Reconstruct a (visits, Q-value) tree from the live iter_log
        and write atomically via temp file + rename.  Cheap (linear scan
        of iter_log.jsonl) and resilient: a crash here cannot leave
        a half-written tree_snapshot.json, because rename is atomic on
        POSIX."""
        try:
            import importlib, sys as _sys
            _repo = _PP(__file__).resolve().parents[3]
            _scripts = _repo / "scripts"
            if str(_scripts) not in _sys.path:
                _sys.path.insert(0, str(_scripts))
            mod = importlib.import_module("reconstruct_tree_from_iter_log")
            tmp = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
            mod.reconstruct(iter_log_pp, tmp, verbose=False)
            tmp.replace(snapshot_path)
            print(f"  [tree-snapshot] {snapshot_path.name} updated "
                  f"at iter {iter_num}", flush=True)
        except Exception as exc:
            print(f"  [tree-snapshot] WARN: {exc}", flush=True)

    _score_cache_prev: dict[str, int] = {"hits": 0, "misses": 0}
    _llm_cache_prev: dict[str, int] = {"hits": 0, "misses": 0}

    def _result_cache_delta() -> dict:
        """Compute per-iter delta for Layer-A (LLM) and Layer-B (score) caches."""
        out: dict = {}
        try:
            from rag_contracts.llm_helpers import get_llm_cache_stats
            llm_snap = get_llm_cache_stats()
            llm_delta = {k: llm_snap.get(k, 0) - _llm_cache_prev.get(k, 0)
                         for k in llm_snap}
            _llm_cache_prev.update(llm_snap)
            out["llm_cache_delta"] = llm_delta
            out["llm_cache_cumulative"] = llm_snap
        except Exception as exc:
            out["llm_cache_error"] = str(exc)
        try:
            if hasattr(evaluator, "score_cache_stats"):
                sc_snap = evaluator.score_cache_stats()
                sc_delta = {k: sc_snap.get(k, 0) - _score_cache_prev.get(k, 0)
                            for k in sc_snap}
                _score_cache_prev.update(sc_snap)
                out["score_cache_delta"] = sc_delta
                out["score_cache_cumulative"] = sc_snap
        except Exception as exc:
            out["score_cache_error"] = str(exc)
        return out

    _bipartite_prev: dict[str, dict] = {"snapshot": {}}

    def _bipartite_delta() -> dict:
        """Read the evaluator's bipartite cache trigger counters and
        compute the per-iter delta.  These counters are populated by
        ``RAGCacheAwareEvaluator.evaluate()`` as it walks every
        ``(prefix, qid)`` and ``(prefix, cluster_id)`` against the live
        ``SearchContext.materialized_keys`` set.  We also surface the
        SQLite-backed ``ReuseLedger.stats_snapshot()`` if available, so
        any future write-through path is observed too."""
        out: dict = {}
        try:
            if hasattr(evaluator, "cache_stats"):
                snap = evaluator.cache_stats()
                prev = _bipartite_prev["snapshot"] or {}
                delta = {k: snap.get(k, 0) - prev.get(k, 0) for k in snap}
                _bipartite_prev["snapshot"] = snap
                out["delta"] = delta
                out["cumulative"] = snap
            led = getattr(evaluator, "ledger", None)
            if led is not None and hasattr(led, "stats_snapshot"):
                out["ledger_cumulative"] = led.stats_snapshot()
        except Exception as exc:
            out["error"] = str(exc)
        return out

    def _on_iteration(rec: IterationRecord) -> None:
        d = rec.to_dict()
        # Attach per-iter bipartite cache trigger stats (hit/miss/insert).
        _bp = _bipartite_delta()
        if _bp:
            d["bipartite_cache"] = _bp
        # Attach Layer-A / Layer-B result cache stats
        _rc = _result_cache_delta()
        if _rc:
            d["result_cache"] = _rc
        # Attach per-iter token + call accountancy (delta since prev iter)
        # so iter_log.jsonl is the single self-contained run record.
        if _HAS_TOKEN_TRACKER:
            try:
                _delta = _tt_snapshot_delta()
                if _delta is not None:
                    d["llm_tokens_delta"] = {
                        "n_chat_calls": _delta.n_chat_calls,
                        "n_embedding_calls": _delta.n_embedding_calls,
                        "prompt_tokens": _delta.prompt_tokens,
                        "completion_tokens": _delta.completion_tokens,
                        "reasoning_tokens": _delta.reasoning_tokens,
                        "total_tokens": _delta.total_tokens,
                        "embedding_tokens": _delta.embedding_tokens,
                        "chat_call_seconds": round(_delta.chat_call_seconds, 3),
                        "embedding_call_seconds": round(_delta.embedding_call_seconds, 3),
                    }
                _tot = _tt_get_totals()
                if _tot is not None:
                    d["llm_tokens_cumulative"] = {
                        "n_chat_calls": _tot.n_chat_calls,
                        "n_embedding_calls": _tot.n_embedding_calls,
                        "prompt_tokens": _tot.prompt_tokens,
                        "completion_tokens": _tot.completion_tokens,
                        "reasoning_tokens": _tot.reasoning_tokens,
                        "total_tokens": _tot.total_tokens,
                        "embedding_tokens": _tot.embedding_tokens,
                    }
            except Exception as _tt_err:
                d["llm_tokens_error"] = str(_tt_err)

        line = _json.dumps(d, ensure_ascii=False, default=str)
        _iter_log_fh.write(line + "\n")
        _iter_log_fh.flush()

        # Flush the Layer-B score cache sidecar log for this iter so
        # a kill -9 between iterations can never lose entries that the
        # iter_log line claims were materialised.
        try:
            if hasattr(evaluator, "flush_score_cache_log"):
                evaluator.flush_score_cache_log()
        except Exception as _exc:
            print(f"  [score-cache-log] WARN: {_exc}", flush=True)

        gm = d.get("group_means", {})
        gm_str = ", ".join(f"{k}={v:.4f}" for k, v in gm.items()) if gm else ""
        tok_str = ""
        if "llm_tokens_delta" in d:
            td = d["llm_tokens_delta"]
            tc = d.get("llm_tokens_cumulative", {})
            tok_str = (
                f"  tok=+{td['total_tokens']}/{tc.get('total_tokens', 0)} "
                f"(chat={td['n_chat_calls']}, embed={td['n_embedding_calls']})"
            )
        bp_str = ""
        bp = d.get("bipartite_cache", {})
        if bp and "delta" in bp:
            bd = bp["delta"]
            n_lk = bd.get("n_qid_lookups_total", 0)
            n_hit = bd.get("n_qid_lookups_hit", 0)
            n_miss = bd.get("n_qid_lookups_miss", 0)
            full_hit = bd.get("n_configs_full_hit_per_qid", 0)
            hit_pct = (n_hit / n_lk * 100.0) if n_lk > 0 else 0.0
            bp_str = (f"  bp_lookups={n_lk}(hit={n_hit}/miss={n_miss},"
                      f"{hit_pct:.0f}%; full_hit_qids={full_hit})")
        rc_str = ""
        rc = d.get("result_cache", {})
        sc_d = rc.get("score_cache_delta", {})
        lc_d = rc.get("llm_cache_delta", {})
        sc_h, sc_m = sc_d.get("hits", 0), sc_d.get("misses", 0)
        lc_h, lc_m = lc_d.get("hits", 0), lc_d.get("misses", 0)
        if sc_h + sc_m > 0 or lc_h + lc_m > 0:
            sc_tot = sc_h + sc_m
            lc_tot = lc_h + lc_m
            rc_str = (
                f"  score_cache={sc_h}/{sc_tot}"
                f"({sc_h/sc_tot*100:.0f}%hit)" if sc_tot else ""
            ) + (
                f"  llm_cache={lc_h}/{lc_tot}"
                f"({lc_h/lc_tot*100:.0f}%hit)" if lc_tot else ""
            )
        print(
            f"  [iter {rec.iteration:4d}]  reward={rec.total_reward:.4f}  "
            f"cost={rec.total_cost:.2f}  keys=+{rec.new_materialized_keys}/"
            f"{rec.cumulative_materialized_keys}{bp_str}{rc_str}{tok_str}  "
            f"best={rec.best_reward_so_far:.4f}  "
            f"wall={rec.wall_time_s:.1f}s  "
            f"conc={args.concurrency}  "
            f"({gm_str})",
            flush=True,
        )
        if snapshot_every > 0 and rec.iteration % snapshot_every == 0:
            _atomic_snapshot(rec.iteration)

    print(f"  Iteration log: {iter_log_pp}", flush=True)

    # --- Pre-search validation ---
    if use_real and frozen_samples:
        import os as _os
        print("\n  === Pre-Search Validation ===", flush=True)
        _validation_configs = [
            ("bm25+simple", ("standard_passage", "identity", "bm25", "identity", "simple_llm")),
            ("full_lightrag", ("kg_extraction", "lightrag_keywords", "lightrag_graph", "lightrag_vector_enrich", "lightrag_answer")),
            ("full_selfrag", ("standard_passage", "identity", "dense_e5", "selfrag_critique", "selfrag_generator")),
        ]
        _vllm_url = _os.environ.get("SELFRAG_VLLM_URL", "")
        if _vllm_url:
            print(f"  SELFRAG_VLLM_URL = {_vllm_url}", flush=True)
        else:
            print("  WARNING: SELFRAG_VLLM_URL not set, selfrag uses fake OpenAI shim!", flush=True)

        _any_sample = None
        for _sid, _sitems in frozen_samples.items():
            if _sitems:
                _any_sample = (_sid, _sitems[0])
                break

        _val_fail = False
        if _any_sample:
            _val_sid, _val_item = _any_sample
            _val_bm = SPECTRUM_TO_BENCHMARK.get(_val_sid, "hotpotqa")
            for _vname, _vchoices in _validation_configs:
                if not _check_constraints(_vchoices):
                    print(f"  [SKIP] {_vname}: constraint violation (expected)", flush=True)
                    continue
                try:
                    _vreward = evaluate_config_real(
                        _vchoices, benchmark=_val_bm, frozen_sample=[_val_item])
                    _status = "PASS" if _vreward > 0.0 else "FAIL (reward=0)"
                    if _vreward == 0.0:
                        _val_fail = True
                    print(f"  [{_status}] {_vname}: reward={_vreward:.4f}", flush=True)
                except Exception as _ve:
                    _val_fail = True
                    print(f"  [FAIL] {_vname}: {_ve}", flush=True)
        if _val_fail:
            print("  WARNING: Some validation configs failed. Check component setup.", flush=True)
        print("  === End Pre-Search Validation ===\n", flush=True)

    root = RAGPipelineSearchState()

    # Print adaptive sampling config
    if args.batch_size > 0:
        print(f"  [adaptive-sampling] batch_size={args.batch_size}, "
              f"se_threshold={args.se_threshold}, "
              f"max_items={args.max_items_per_spectrum}", flush=True)
    if args.cold_start > 0:
        print(f"  [cold-start] budget={args.cold_start}/spectrum "
              f"(seeds {len(root.available_actions())} root children)", flush=True)
    print(f"  [lambda-decay] lambda_0={args.lambda_0} "
          f"(lambda_t = {args.lambda_0}/sqrt(t))", flush=True)

    result = engine.search(
        root,
        max_iterations=args.max_iterations,
        on_iteration=_on_iteration,
        cold_start_budget=args.cold_start,
        batch_size=args.batch_size,
        frozen_samples=frozen_samples,
    )
    _iter_log_fh.close()

    print("=" * 70, flush=True)
    print("  RAG Pipeline UCT Search Complete", flush=True)
    print("=" * 70, flush=True)
    # ``result.best_state`` is ``None`` when every evaluation in this run
    # returned a failure (e.g. all UltraDomain runs crashing on the
    # LLM-judge safety check). Rendering it via ``.pretty()`` masks the
    # real failure with an ``AttributeError`` traceback, so report the
    # degenerate-run state explicitly instead.
    if result.best_state is None:
        print("  Best config : <NONE> (no successful evaluations -- see "
              "FAILURE entries above)", flush=True)
        print(f"  Best reward : N/A (all evaluations failed)", flush=True)
    else:
        print(f"  Best config : {result.best_state.pretty()}", flush=True)
        print(f"  Best reward : {result.best_reward:.4f}", flush=True)
    print(f"  Iterations  : {result.iterations}", flush=True)
    print(f"  Evaluations : {result.total_evaluations}", flush=True)
    print(f"  Total cost  : {result.total_cost:.2f}", flush=True)
    print(f"  Materialized keys: {len(result.context.materialized_keys)}", flush=True)
    print("=" * 70, flush=True)

    total_configs = 1
    for opts in ACTIVE_SLOT_OPTIONS:
        total_configs *= len(opts)
    print(f"\n  Search space : {total_configs} total configurations", flush=True)
    print(f"  Explored     : {result.total_evaluations} evaluations", flush=True)
    print(f"  Efficiency   : {result.total_evaluations / total_configs:.1%} of space", flush=True)

    print("\n  Slot breakdown:", flush=True)
    for action, child in result.root_node.children.items():
        print(
            f"    {SLOT_NAMES[0]}={action!s:20s}  visits={child.visit_count:4d}  "
            f"Q={child.q_value:.4f}  best={child.best_value:.4f}",
            flush=True,
        )

    print("\n  Top-5 configs by Q-value:", flush=True)
    configs_visited = []

    def _collect(node, depth=0, prefix=()):
        if depth == len(SLOT_NAMES):
            if node.visit_count > 0:
                configs_visited.append((prefix, node.q_value, node.visit_count))
            return
        for act, ch in node.children.items():
            _collect(ch, depth + 1, prefix + (str(act),))

    _collect(result.root_node)
    configs_visited.sort(key=lambda x: x[1], reverse=True)
    for cfg, q, visits in configs_visited[:5]:
        label = ", ".join(f"{SLOT_NAMES[i]}={v}" for i, v in enumerate(cfg))
        print(f"    Q={q:.4f}  visits={visits:3d}  ({label})", flush=True)

    if args.save_tree:
        # Tree save must survive a degenerate run where every iteration's
        # evaluator raised PipelineFailure (-> NaN reward -> best_state
        # stays None).  The earlier code dereferenced
        # ``result.best_state.state_key()`` unconditionally and crashed
        # with AttributeError, losing the whole serialized tree and
        # masking the actual upstream failure in the FailureTracker.
        # Guard each best_* field independently so the tree.json is
        # always written; downstream readers see explicit ``null`` /
        # NaN markers when the search degenerated.
        tree_path = _PP(args.save_tree)
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        tree_data = _serialize_tree(result.root_node, SLOT_NAMES)
        if result.best_state is not None:
            tree_data["best_config"] = list(result.best_state.state_key())
            tree_data["best_reward"] = result.best_reward
        else:
            tree_data["best_config"] = None
            tree_data["best_reward"] = None
            tree_data["best_state_is_none"] = True
        tree_data["iterations"] = result.iterations
        tree_data["total_evaluations"] = result.total_evaluations
        tree_data["total_cost"] = result.total_cost
        tree_data["materialized_keys_count"] = len(result.context.materialized_keys)
        tree_data["search_space_size"] = total_configs
        with open(tree_path, "w", encoding="utf-8") as f:
            _json.dump(tree_data, f, ensure_ascii=False, indent=2,
                       default=lambda o: None)
        print(f"\n  Search tree saved to {tree_path}", flush=True)

    if args.output_stats:
        # Same NoneType guard as the save_tree branch above: stats.json
        # must always be produced so the post-mortem from a degenerate
        # run is not lost.
        stats_path = _PP(args.output_stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        if result.best_state is not None:
            best_config = list(result.best_state.state_key())
            best_config_pretty = result.best_state.pretty()
            best_reward = result.best_reward
        else:
            best_config = None
            best_config_pretty = None
            best_reward = None
        stats = {
            "best_config": best_config,
            "best_config_pretty": best_config_pretty,
            "best_reward": best_reward,
            "best_state_is_none": result.best_state is None,
            "iterations": result.iterations,
            "total_evaluations": result.total_evaluations,
            "total_cost": result.total_cost,
            "materialized_keys_count": len(result.context.materialized_keys),
            "search_space_size": total_configs,
            "efficiency": result.total_evaluations / total_configs,
        }
        with open(stats_path, "w", encoding="utf-8") as f:
            _json.dump(stats, f, ensure_ascii=False, indent=2,
                       default=lambda o: None)
        print(f"  Search stats saved to {stats_path}", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
