"""AG-UCT resume driver.

A standalone main entry-point that picks up an interrupted
``rag_pipeline_search.py --real --search-set ...`` run from any
``iter_log.jsonl`` and continues it to ``--max-iterations`` using the
:class:`ResumableUCTSearchEngine`.

Key isolation guarantees
------------------------

* ``rag_pipeline_search.py`` is **not modified**.  This driver only
  imports the helpers it needs (``RAGPipelineSearchState``,
  ``_build_wtb_evaluator``, etc.) -- those helpers are pure with respect
  to their inputs and have no side effects on import.
* ``uct_engine/search.py`` is **not modified**.  We use the
  :class:`ResumableUCTSearchEngine` subclass shipped in
  ``uct_engine/resume.py``.
* The on-disk Layer-1 VDB caches (LightRAG native ``vdb_chunks.json``
  + Qwen3 auto-built ``vdb_auto_chunks_{chunking}.json``) are reused
  in-place; nothing here mutates them.
* The Layer-2 SQLite ledger is recreated fresh (in a new tempdir) --
  the existing concurrent search run leaves the in-memory
  ``materialized_keys`` set as the canonical source of truth, and we
  rebuild that set via :func:`scripts.resume_uct_state.replay_materialized_keys`.

Example
-------

::

    python AG-UCT/uct_engine/examples/rag_pipeline_search_resume.py \
        --resume-from-run results/uct_350_concurrent_20260515_063251 \
        --resume-from-iter 100 \
        --max-iterations 350 \
        --search-set /data1/.../fullwiki_search_350.json \
                     /data1/.../ultradomain_search_350.json \
        --concurrency 5
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "AG-UCT"))
sys.path.insert(0, str(_REPO / "Benchmark_Sampling"))
sys.path.insert(0, str(_REPO / "A-Simplified-Core-Workflow-for-Enhancing-RAG"))
sys.path.insert(0, str(_REPO / "scripts"))

from dotenv import load_dotenv as _ld
_env = _REPO / ".env"
if _env.exists():
    _ld(_env)

from uct_engine import (
    ClusterDef,
    CostAwareUCTScorer,
    ReuseAwareCostModel,
    SearchContext,
)
from uct_engine.resume import ResumableUCTSearchEngine, rebuild_tree_node

# Pure helpers (no module-level side effects):
from uct_engine.examples.rag_pipeline_search import (
    CLUSTER_COST,
    CLUSTER_IDS,
    RAGPipelineSearchState,
    REWARD_TABLE,
    SLOT_NAMES,
    SLOT_OPTIONS,
    SPECTRUM_TO_BENCHMARK,
    _build_wtb_evaluator,
    _serialize_tree,
    active_slot_options,
    load_search_set,
    set_active_slot_options,
)

import resume_uct_state  # scripts/resume_uct_state.py


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AG-UCT resume driver")
    p.add_argument("--resume-from-run", type=str, required=True,
                   help="Existing run dir (must contain iter_log.jsonl + hparams.json)")
    p.add_argument("--resume-from-iter", type=int, default=None,
                   help="Last iter to replay (default: last completed iter in log)")
    p.add_argument("--out-run-dir", type=str, default=None,
                   help="Output dir (default: <src>_resume_<TS>)")
    p.add_argument("--search-set", type=str, nargs="+", required=True,
                   help="Search-set JSONs -- MUST match the original run")
    p.add_argument("--max-iterations", type=int, default=None,
                   help="Absolute iter target (default: read from hparams.json)")
    p.add_argument("--budget", type=int, default=None,
                   help="Per-spectrum cap (default: full set, matching original)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Parallel items per spectrum (default 1 = serial)")
    p.add_argument("--seed", type=int, default=None,
                   help="Override base seed (default: read from hparams.json)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Sampling batch size (default: read from hparams; 0 = full-pool)")
    p.add_argument("--se-threshold", type=float, default=None,
                   help="SE convergence threshold (default: read from hparams)")
    p.add_argument("--max-items-per-spectrum", type=int, default=None,
                   help="Max items per spectrum in sampling (default: read from hparams)")
    p.add_argument("--batch-step", type=int, default=None,
                   help="Batch step for sequential accumulation (default: read from hparams)")
    p.add_argument("--tree-snapshot-every", type=int, default=5,
                   help="Atomic snapshot frequency (default 5)")
    p.add_argument("--no-validate-hparams", action="store_true",
                   help="Skip the strict hparams cross-check (use with care)")
    p.add_argument("--score-cache-log", type=str, default=None,
                   help="Sidecar Layer-B score-cache log to warm-start "
                        "from (default: <src_run_dir>/score_cache.jsonl)")
    p.add_argument("--ledger-path", type=str, default=None,
                   help="Persistent Layer-2 ReuseLedger SQLite to reopen "
                        "(default: <src_run_dir>/wtb_reuse/reuse_ledger.db). "
                        "When the file is absent, a fresh ledger is created "
                        "alongside the resume out_dir and a WARNING printed.")
    p.add_argument("--slot-options-json", type=str, default=None,
                   help="JSON array overriding the 5-slot search space. "
                        "Default: read agent_uct.slot_options from hparams.json "
                        "when present, otherwise use the historical space.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# hparams cross-check
# ---------------------------------------------------------------------------

def _validate_hparams(hparams: dict, args: argparse.Namespace) -> None:
    """Loudly fail if the resume CLI drifts from the original run.

    Mismatched search set / seed / max-iterations would produce a
    formally valid trajectory but a meaningless comparison, so we
    refuse rather than silently proceed.
    """
    agent = hparams.get("agent_uct", {}) or {}
    src_seed = agent.get("random_seed")
    src_max = agent.get("max_iterations")

    if args.seed is None:
        args.seed = int(src_seed) if src_seed is not None else 42
    if args.max_iterations is None:
        args.max_iterations = int(src_max) if src_max is not None else 350

    # Resolve sampling params from hparams if not explicitly provided
    sampling = agent.get("sampling", {}) or {}
    if args.batch_size is None:
        args.batch_size = int(sampling.get("batch_size", 0))
    if args.se_threshold is None:
        args.se_threshold = float(sampling.get("se_threshold", 0.05))
    if args.max_items_per_spectrum is None:
        args.max_items_per_spectrum = int(sampling.get("max_items_per_spectrum", 30))
    if args.batch_step is None:
        args.batch_step = int(sampling.get("batch_step", 5))

    data_split = hparams.get("data_split", {}) or {}
    hq = (data_split.get("hotpotqa_search") or {}).get("path")
    ud = (data_split.get("ultradomain_search") or {}).get("path")
    expected = [p for p in (hq, ud) if p]
    given = [os.path.abspath(p) for p in args.search_set]
    if expected and not all(any(e.endswith(Path(g).name) for g in given) for e in expected):
        raise SystemExit(
            f"  [resume] FATAL: --search-set drifts from original run.\n"
            f"     original: {expected}\n"
            f"     given   : {given}\n"
            f"     pass --no-validate-hparams to override."
        )


# ---------------------------------------------------------------------------
# Iter-log copy-with-truncate
# ---------------------------------------------------------------------------

def _copy_iter_log_prefix(src: Path, dst: Path, up_to_iter: int) -> int:
    """Copy iter_log lines for iters 1..up_to_iter (inclusive) verbatim.

    Kept for the rare case where ``--resume-from-iter`` is explicitly
    passed and we need a plain truncating copy.  The default path uses
    :func:`resume_uct_state.truncate_iter_log_to_last_valid` which
    additionally validates iteration ordering and reports trailing
    junk loudly (see plan section E).

    Returns the number of lines copied.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line_s = line.rstrip("\n")
            if not line_s:
                continue
            try:
                rec = json.loads(line_s)
            except Exception:
                continue
            it = int(rec.get("iteration", 0))
            if it <= up_to_iter:
                fout.write(line_s + "\n")
                n_copied += 1
            else:
                break
    return n_copied


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    src_dir = Path(args.resume_from_run).resolve()
    src_log = src_dir / "iter_log.jsonl"
    src_hparams = src_dir / "hparams.json"
    if not src_log.exists():
        raise SystemExit(f"  [resume] FATAL: {src_log} not found")

    hparams: dict = {}
    if src_hparams.exists():
        with open(src_hparams, "r", encoding="utf-8") as f:
            hparams = json.load(f)
    slot_options_override = args.slot_options_json
    if slot_options_override is None:
        agent_block = hparams.get("agent_uct", {}) or {}
        stored_slot_options = agent_block.get("slot_options")
        if stored_slot_options:
            slot_options_override = json.dumps(stored_slot_options)
    if slot_options_override:
        try:
            set_active_slot_options(json.loads(slot_options_override))
        except Exception as exc:
            raise SystemExit(
                f"  [resume] FATAL: invalid slot-options override: {exc}"
            ) from exc
    if not args.no_validate_hparams:
        _validate_hparams(hparams, args)
    else:
        if args.seed is None:
            args.seed = 42
        if args.max_iterations is None:
            args.max_iterations = 350
        if args.batch_size is None:
            args.batch_size = 0
        if args.se_threshold is None:
            args.se_threshold = 0.05
        if args.max_items_per_spectrum is None:
            args.max_items_per_spectrum = 30
        if args.batch_step is None:
            args.batch_step = 5

    # Output dir (created BEFORE truncation so the merged log lives in it)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_run_dir:
        out_dir = Path(args.out_run_dir).resolve()
    else:
        out_dir = src_dir.with_name(f"{src_dir.name}_resume_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_iter_log = out_dir / "iter_log.jsonl"
    out_snapshot = out_dir / "tree_snapshot.json"
    out_tree = out_dir / "tree.json"
    out_stats = out_dir / "stats.json"
    out_resume_info = out_dir / "resume_info.json"

    # --- Loud truncate copy: produces the merged iter_log AND the
    #     "last valid iter" used for resume bookkeeping.  Replaces the
    #     previous _copy_iter_log_prefix path which silently shortened
    #     mid-write phase-1 crashes.
    last_valid_iter, n_truncated_trailing = (
        resume_uct_state.truncate_iter_log_to_last_valid(src_log, out_iter_log)
    )
    if args.resume_from_iter is not None:
        # Caller wants to clip earlier; respect it but never go past
        # last_valid_iter (would replay imaginary data).
        up_to_iter = min(int(args.resume_from_iter), last_valid_iter)
        if up_to_iter < last_valid_iter:
            # Re-truncate the merged log to the requested cutoff.
            tmp_log = out_iter_log.with_suffix(".jsonl.cutoff")
            _copy_iter_log_prefix(out_iter_log, tmp_log, up_to_iter)
            tmp_log.replace(out_iter_log)
    else:
        up_to_iter = last_valid_iter

    if up_to_iter < 1:
        raise SystemExit("  [resume] FATAL: no replayable iters in iter_log.jsonl")
    if up_to_iter >= args.max_iterations:
        raise SystemExit(
            f"  [resume] FATAL: --resume-from-iter {up_to_iter} >= "
            f"--max-iterations {args.max_iterations}; nothing to do."
        )

    print("=" * 70, flush=True)
    print(f"  AG-UCT RESUME", flush=True)
    print("=" * 70, flush=True)
    print(f"  src       : {src_dir}", flush=True)
    print(f"  out       : {out_dir}", flush=True)
    print(f"  up_to_iter: {up_to_iter}", flush=True)
    print(f"  max_iters : {args.max_iterations}", flush=True)
    print(f"  seed      : {args.seed}", flush=True)
    print(f"  conc      : {args.concurrency}", flush=True)
    print(f"  search-set: {args.search_set}", flush=True)
    print(f"  slot opts : {active_slot_options()}", flush=True)
    print(f"  [resume] iter_log truncate: last_valid={last_valid_iter}, "
          f"trailing_skipped={n_truncated_trailing}", flush=True)

    # All downstream replay_* helpers MUST read from this clean,
    # truncated copy -- not src_log -- because src_log may still
    # contain the half-written line that crashed phase-1's last write.
    # (Re-using src_log here is the bug fixed alongside plan section E.)
    replay_log = out_iter_log

    # Concurrency env (matches rag_pipeline_search.py)
    if args.concurrency > 1:
        os.environ["OMINIRAG_EVAL_CONCURRENCY"] = str(args.concurrency)

    # Load frozen samples deterministically
    frozen_samples: dict[str, list] = {}
    for ss_path in args.search_set:
        partial = load_search_set(ss_path)
        for sid, sitems in partial.items():
            frozen_samples.setdefault(sid, []).extend(sitems)
    if args.budget is not None:
        rng_b = random.Random(args.seed)
        for sid in frozen_samples:
            if len(frozen_samples[sid]) > args.budget:
                rng_b.shuffle(frozen_samples[sid])
                frozen_samples[sid] = frozen_samples[sid][: args.budget]
    print("  [resume] frozen_samples:", flush=True)
    for sid in sorted(frozen_samples):
        print(f"    {sid}: {len(frozen_samples[sid])} items", flush=True)
    total_items = sum(len(v) for v in frozen_samples.values())
    print(f"  [resume] total: {total_items} items across "
          f"{len(frozen_samples)} spectrums", flush=True)

    # Install token tracker + seed baseline
    try:
        from ominirag_wtb import token_tracker as _tt
        _tt.install()
        tok_snapshot = resume_uct_state.replay_token_totals(replay_log, up_to_iter)
        if tok_snapshot:
            _tt.seed_totals(tok_snapshot)
        snapshot_delta = _tt.snapshot_delta
        get_totals = _tt.get_totals
        _has_tt = True
    except Exception as exc:
        print(f"  [token-tracker] WARN: disabled ({exc})", flush=True)
        _has_tt = False
        snapshot_delta = lambda: None  # type: ignore
        get_totals = lambda: None  # type: ignore

    # Build evaluator (same wiring as the original run).  The resume
    # process gets *its own* score-cache sidecar log (so post-resume
    # entries are auditable separately) but *reuses* the source run's
    # Layer-2 ReuseLedger SQLite so WTB physical fork reuse survives.
    if args.score_cache_log is None:
        # Default: write new entries into the resume out_dir; we still
        # warm-start from the source dir's score_cache.jsonl below.
        args.score_cache_log = str(out_dir / "score_cache.jsonl")

    src_score_log = src_dir / "score_cache.jsonl"

    if args.ledger_path is None:
        candidate = src_dir / "wtb_reuse" / "reuse_ledger.db"
        if candidate.exists():
            args.ledger_path = str(candidate)
        else:
            print(
                f"  WARNING [resume] no persistent ledger at {candidate}; "
                f"creating a fresh ledger in the resume out_dir.  "
                f"Phase-1 WTB checkpoints will not be reused.",
                flush=True,
            )
            args.ledger_path = str(out_dir / "wtb_reuse" / "reuse_ledger.db")
    wtb_data_dir = str(Path(args.ledger_path).parent)

    print("  [resume] building WTB cache-aware evaluator (real, wtb-reuse on)",
          flush=True)
    print(f"    ledger     : {args.ledger_path}", flush=True)
    print(f"    wtb_data   : {wtb_data_dir}", flush=True)
    print(f"    score_log  : {args.score_cache_log}", flush=True)
    if args.batch_size > 0:
        print(f"  [resume] sampling mode: batch_size={args.batch_size}, "
              f"se_threshold={args.se_threshold}, "
              f"max_items_per_spectrum={args.max_items_per_spectrum}, "
              f"batch_step={args.batch_step}", flush=True)
    evaluator = _build_wtb_evaluator(
        use_real=True,
        frozen_samples=frozen_samples,
        batch_size=args.batch_size,
        se_threshold=args.se_threshold,
        max_items_per_spectrum=args.max_items_per_spectrum,
        batch_step=args.batch_step,
        score_cache_log_path=args.score_cache_log,
        ledger_path=args.ledger_path,
        wtb_data_dir=wtb_data_dir,
    )

    # Warm-start Layer-B score_cache from the source run's sidecar log.
    n_score_cache_replayed = 0
    if src_score_log.exists():
        replayed_sc = resume_uct_state.replay_score_cache(
            src_score_log, up_to_iter=up_to_iter,
        )
        evaluator.score_cache.update(replayed_sc)
        # Reset hit/miss counters so post-resume rates reflect only
        # post-resume work; the cumulative continues from the iter_log
        # via _result_cache_delta wiring inside the search driver.
        evaluator._score_cache_stats = {"hits": 0, "misses": 0}
        n_score_cache_replayed = len(replayed_sc)
        print(f"  [resume] warm-loaded {n_score_cache_replayed} "
              f"(config_key,qid) entries into score_cache from "
              f"{src_score_log}", flush=True)
    else:
        print(f"  WARNING [resume] no score_cache.jsonl at {src_score_log}; "
              f"repeat (config,qid) evaluations will re-call the LLM",
              flush=True)
    scorer = CostAwareUCTScorer(lambda_t=0.05)
    clusters = [
        ClusterDef(cid, weight=1.0, base_cost=CLUSTER_COST[cid])
        for cid in CLUSTER_IDS
    ]
    cost_model = ReuseAwareCostModel(clusters=clusters)

    engine = ResumableUCTSearchEngine(
        evaluator=evaluator,
        scorer=scorer,
        cost_model=cost_model,
        exploration_constant=1.4,
        random_seed=args.seed,
    )

    # Replay tree + materialized keys
    print(f"  [resume] replaying tree from iter_log up to iter {up_to_iter} ...",
          flush=True)
    tree_state = resume_uct_state.replay_tree(replay_log, up_to_iter)
    root_state = RAGPipelineSearchState()
    pre_root = rebuild_tree_node(root_state, tree_state["root"])
    print(f"  [resume]   tree visit_count = {pre_root.visit_count}; "
          f"best_reward = {tree_state['best_reward']}", flush=True)

    print(f"  [resume] replaying materialized_keys ...", flush=True)
    if args.batch_size > 0:
        materialized_keys_method = "sampling_aware"
        mk = resume_uct_state.replay_materialized_keys_sampling(
            replay_log, frozen_samples, up_to_iter, cluster_ids=CLUSTER_IDS)
    else:
        materialized_keys_method = "full_pool"
        mk = resume_uct_state.replay_materialized_keys(
            replay_log, frozen_samples, up_to_iter, cluster_ids=CLUSTER_IDS)
    print(f"  [resume]   |materialized_keys| = {len(mk)} "
          f"(method={materialized_keys_method})", flush=True)

    pre_context = SearchContext(
        random=random.Random(args.seed * 7919 + up_to_iter),
        materialized_keys=mk,
    )

    # Replay cached_qids for sampling mode (no-op for full-pool)
    if args.batch_size > 0:
        print(f"  [resume] replaying cached_qids (sampling) ...", flush=True)
        cached_qids_raw = resume_uct_state.replay_cached_qids(replay_log, up_to_iter)
        pre_context.extra["cached_qids"] = cached_qids_raw
        total_cached_entries = sum(len(v) for v in cached_qids_raw.values())
        print(f"  [resume]   |cached_qids keys| = {len(cached_qids_raw)}, "
              f"total entries = {total_cached_entries}", flush=True)

    # Replay total_evaluations
    pre_context.total_evaluations = resume_uct_state.replay_total_evaluations(
        replay_log, up_to_iter)
    print(f"  [resume]   total_evaluations = {pre_context.total_evaluations}",
          flush=True)

    # Rebuild best_state from the replayed best cfg
    best_state_baseline = None
    best_cfg = tree_state.get("best_cfg")
    if best_cfg is not None:
        s = RAGPipelineSearchState()
        try:
            for action in best_cfg:
                s = s.child(action)
            best_state_baseline = s
        except Exception:
            best_state_baseline = None
    best_reward_baseline = tree_state.get("best_reward", float("-inf"))
    if best_reward_baseline == float("-inf"):
        best_reward_baseline = -math.inf
    cum_wall_baseline = float(tree_state.get("cumulative_wall_s") or 0.0)

    # Write a small resume_info.json for later analysis
    with open(out_resume_info, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": 2,
            "created_at": ts,
            "src_run_dir": str(src_dir),
            "out_run_dir": str(out_dir),
            "resume_from_iter": up_to_iter,
            "last_valid_iter_in_src": last_valid_iter,
            "truncated_trailing_lines": n_truncated_trailing,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
            "rng_seed_for_resume": args.seed * 7919 + up_to_iter,
            "search_set": args.search_set,
            "concurrency": args.concurrency,
            "replayed_tree_visit_count": pre_root.visit_count,
            "replayed_materialized_keys_count": len(mk),
            "materialized_keys_method": materialized_keys_method,
            "score_cache_replayed_count": n_score_cache_replayed,
            "ledger_path": args.ledger_path,
            "score_cache_log": args.score_cache_log,
            "replayed_best_reward": (
                None if best_reward_baseline == -math.inf
                else float(best_reward_baseline)
            ),
            "replayed_best_cfg": list(best_cfg) if best_cfg else None,
            "cum_wall_baseline_s": cum_wall_baseline,
            "token_snapshot_seeded": _has_tt and bool(
                resume_uct_state.replay_token_totals(replay_log, up_to_iter)
            ),
        }, f, ensure_ascii=False, indent=2)

    # Streaming iter-log writer + atomic tree-snapshot
    iter_log_fh = open(out_iter_log, "a", encoding="utf-8")
    snapshot_every = max(0, int(args.tree_snapshot_every or 0))

    def _atomic_snapshot(iter_num: int) -> None:
        try:
            from reconstruct_tree_from_iter_log import reconstruct  # noqa: WPS433
            tmp = out_snapshot.with_suffix(out_snapshot.suffix + ".tmp")
            reconstruct(out_iter_log, tmp, verbose=False)
            tmp.replace(out_snapshot)
            print(f"  [tree-snapshot] {out_snapshot.name} updated at iter {iter_num}",
                  flush=True)
        except Exception as exc:
            print(f"  [tree-snapshot] WARN: {exc}", flush=True)

    _last_total = TokenSnapshotShim()
    if _has_tt:
        from ominirag_wtb.token_tracker import TokenCounters as _TC
        _last_total = _TC(
            **{k: int(tok_snapshot.get(k, 0)) for k in (
                "n_chat_calls", "n_embedding_calls",
                "prompt_tokens", "completion_tokens", "reasoning_tokens",
                "total_tokens", "embedding_tokens",
            )}
        )

    def _on_iteration(rec) -> None:
        d = rec.to_dict()
        # Bipartite cache delta (best-effort)
        try:
            cache = evaluator.cache_stats() if hasattr(evaluator, "cache_stats") else {}
            d["bipartite_cache"] = {"cumulative": cache}
        except Exception:
            pass
        # Token deltas
        if _has_tt:
            try:
                delta = snapshot_delta()
                if delta is not None:
                    d["llm_tokens_delta"] = {
                        "n_chat_calls": delta.n_chat_calls,
                        "n_embedding_calls": delta.n_embedding_calls,
                        "prompt_tokens": delta.prompt_tokens,
                        "completion_tokens": delta.completion_tokens,
                        "reasoning_tokens": delta.reasoning_tokens,
                        "total_tokens": delta.total_tokens,
                        "embedding_tokens": delta.embedding_tokens,
                        "chat_call_seconds": round(delta.chat_call_seconds, 3),
                        "embedding_call_seconds": round(delta.embedding_call_seconds, 3),
                    }
                tot = get_totals()
                if tot is not None:
                    d["llm_tokens_cumulative"] = {
                        "n_chat_calls": tot.n_chat_calls,
                        "n_embedding_calls": tot.n_embedding_calls,
                        "prompt_tokens": tot.prompt_tokens,
                        "completion_tokens": tot.completion_tokens,
                        "reasoning_tokens": tot.reasoning_tokens,
                        "total_tokens": tot.total_tokens,
                        "embedding_tokens": tot.embedding_tokens,
                    }
            except Exception as exc:
                d["llm_tokens_error"] = str(exc)
        d["resume_marker"] = {"resumed_from_iter": up_to_iter}
        line = json.dumps(d, ensure_ascii=False, default=str)
        iter_log_fh.write(line + "\n")
        iter_log_fh.flush()
        # Flush score-cache sidecar so a kill -9 between iters can
        # never lose entries that the merged iter_log already claims.
        try:
            if hasattr(evaluator, "flush_score_cache_log"):
                evaluator.flush_score_cache_log()
        except Exception as _exc:
            print(f"  [score-cache-log] WARN: {_exc}", flush=True)
        gm = d.get("group_means", {}) or {}
        gm_str = ", ".join(f"{k}={v:.4f}" for k, v in gm.items()) if gm else ""
        print(
            f"  [iter {rec.iteration:4d}]  reward={rec.total_reward:.4f}  "
            f"cost={rec.total_cost:.2f}  keys=+{rec.new_materialized_keys}/"
            f"{rec.cumulative_materialized_keys}  "
            f"best={rec.best_reward_so_far:.4f}  "
            f"wall={rec.wall_time_s:.1f}s  ({gm_str})",
            flush=True,
        )
        if snapshot_every > 0 and rec.iteration % snapshot_every == 0:
            _atomic_snapshot(rec.iteration)

    # Run!
    print(f"\n  === Resuming UCT search at iter {up_to_iter + 1} -> "
          f"{args.max_iterations} ===\n", flush=True)
    result = engine.search_resume(
        root_state,
        pre_root=pre_root,
        pre_context=pre_context,
        start_iter=up_to_iter + 1,
        max_iterations=args.max_iterations,
        cumulative_wall_baseline=cum_wall_baseline,
        best_reward_baseline=best_reward_baseline,
        best_state_baseline=best_state_baseline,
        on_iteration=_on_iteration,
        batch_size=args.batch_size,
        frozen_samples=frozen_samples if args.batch_size > 0 else None,
    )
    iter_log_fh.close()

    # Final artifacts
    tree_data = _serialize_tree(result.root_node, SLOT_NAMES)
    if result.best_state is not None:
        tree_data["best_config"] = list(result.best_state.state_key())
        tree_data["best_reward"] = float(result.best_reward)
    else:
        tree_data["best_config"] = None
        tree_data["best_reward"] = None
        tree_data["best_state_is_none"] = True
    tree_data["iterations"] = result.iterations
    tree_data["total_evaluations"] = result.total_evaluations
    tree_data["total_cost"] = float(result.total_cost)
    tree_data["materialized_keys_count"] = len(result.context.materialized_keys)
    tree_data["resumed_from_iter"] = up_to_iter
    with open(out_tree, "w", encoding="utf-8") as f:
        json.dump(tree_data, f, ensure_ascii=False, indent=2, default=str)

    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump({
            "best_reward": float(result.best_reward)
            if result.best_state is not None else None,
            "best_config": list(result.best_state.state_key())
            if result.best_state is not None else None,
            "iterations": result.iterations,
            "total_evaluations": result.total_evaluations,
            "total_cost": float(result.total_cost),
            "materialized_keys_count": len(result.context.materialized_keys),
            "resumed_from_iter": up_to_iter,
        }, f, ensure_ascii=False, indent=2, default=str)

    print("=" * 70, flush=True)
    print(f"  RESUME COMPLETE: {result.iterations} iters total "
          f"(resumed at {up_to_iter + 1})", flush=True)
    print(f"  Best reward : "
          f"{(float(result.best_reward) if result.best_state else None)}",
          flush=True)
    print(f"  Best config : "
          f"{result.best_state.pretty() if result.best_state else '<NONE>'}",
          flush=True)
    print(f"  Materialized keys: {len(result.context.materialized_keys)}", flush=True)
    print(f"  Wrote: {out_iter_log}", flush=True)
    print(f"  Wrote: {out_tree}", flush=True)
    print(f"  Wrote: {out_stats}", flush=True)
    print(f"  Wrote: {out_resume_info}", flush=True)
    print("=" * 70, flush=True)


class TokenSnapshotShim:
    """Trivial fallback when the openai token tracker is unavailable."""
    n_chat_calls = 0
    n_embedding_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    total_tokens = 0
    embedding_tokens = 0
    chat_call_seconds = 0.0
    embedding_call_seconds = 0.0


if __name__ == "__main__":
    main()
