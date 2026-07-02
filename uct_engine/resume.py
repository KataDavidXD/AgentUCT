"""Resume capability for UCTSearchEngine.

This module is **purely additive** -- it does not modify
``uct_engine/search.py`` or any other shipped engine code.  Callers
that do not import ``resume`` see exactly the original behaviour.

Usage
-----

::

    from uct_engine import UCTSearchEngine
    from uct_engine.resume import ResumableUCTSearchEngine, rebuild_tree_node

    engine = ResumableUCTSearchEngine(evaluator=..., scorer=..., cost_model=...)
    pre_root = rebuild_tree_node(root_state, replayed_tree_dict)
    pre_context = SearchContext(
        random=random.Random(seed * 7919 + start_iter),
        materialized_keys=materialized_keys_set,
    )
    result = engine.search_resume(
        root_state,
        pre_root=pre_root,
        pre_context=pre_context,
        start_iter=start_iter,
        max_iterations=350,
        cumulative_wall_baseline=cum_wall,
        best_reward_baseline=best_reward,
        best_state_baseline=best_state,
        on_iteration=on_iter_cb,
    )

The body of :meth:`search_resume` is a deliberate copy-and-modify of
:meth:`UCTSearchEngine.search` -- duplicated to keep the original
method untouched.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any, Hashable, Optional

from .core import TreeNode
from .interfaces import EvaluationResult, SearchContext, SearchResult, SearchState
from .search import (
    UCTSearchEngine,
    _draw_batches,
    _record_evaluator_failure,
    random_rollout,
    update_cached_qids,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tree rebuild helper -- bridges scripts/resume_uct_state.replay_tree() to
# the engine's in-memory TreeNode shape.
# ---------------------------------------------------------------------------

def rebuild_tree_node(
    root_state: SearchState,
    replayed: dict,
    parent: Optional[TreeNode] = None,
    action_from_parent: Optional[Hashable] = None,
) -> TreeNode:
    """Recursively rebuild a ``TreeNode`` tree from a replayed dict.

    ``replayed`` is the same nested ``{visit_count, value_sum,
    best_value, children}`` shape produced by
    :func:`scripts.resume_uct_state.replay_tree` (specifically its
    ``"root"`` field).  Children are walked by calling
    ``state.child(action)`` on the user-provided ``root_state`` so the
    user's domain (e.g. ``RAGPipelineSearchState``) reconstructs the
    correct sub-state at each depth.
    """
    node = TreeNode(
        state=root_state,
        parent=parent,
        action_from_parent=action_from_parent,
    )
    node.visit_count = int(replayed.get("visit_count", 0))
    node.value_sum = float(replayed.get("value_sum", 0.0))
    bv = replayed.get("best_value", float("-inf"))
    try:
        node.best_value = float(bv)
    except (TypeError, ValueError):
        node.best_value = float("-inf")

    for action_str, child_replayed in (replayed.get("children") or {}).items():
        # The actions are hashables on the engine side; replayed dicts
        # store them as their stringified form (matches the iter_log
        # convention).  Most domains (including RAGPipelineSearchState)
        # use plain strings already, so this is a no-op cast.
        try:
            child_state = root_state.child(action_str)
        except Exception as exc:
            logger.warning(
                "rebuild_tree_node: state.child(%r) failed (%s: %s); "
                "dropping subtree.",
                action_str, type(exc).__name__, exc,
            )
            continue
        child_node = rebuild_tree_node(
            child_state, child_replayed,
            parent=node, action_from_parent=action_str,
        )
        node.children[action_str] = child_node
    return node


# ---------------------------------------------------------------------------
# ResumableUCTSearchEngine
# ---------------------------------------------------------------------------

class ResumableUCTSearchEngine(UCTSearchEngine):
    """``UCTSearchEngine`` with an extra :meth:`search_resume` entry point.

    Inherits all selection/expansion/evaluation/backprop logic from the
    base class and adds a single new method.  The base class is not
    monkey-patched and its ``search()`` method is not overridden.
    """

    def search_resume(
        self,
        root_state: SearchState,
        *,
        pre_root: TreeNode,
        pre_context: SearchContext,
        start_iter: int,
        max_iterations: int,
        cumulative_wall_baseline: float = 0.0,
        best_reward_baseline: float = float("-inf"),
        best_state_baseline: Optional[SearchState] = None,
        max_cost: Optional[float] = None,
        on_iteration=None,
        batch_size: int = 0,
        frozen_samples: Optional[dict] = None,
    ) -> SearchResult:
        """Continue an interrupted search using a rehydrated tree and context.

        Parameters
        ----------
        root_state:
            Original root SearchState (caller must build it the same way
            as the original run).  Used to satisfy ``SearchResult.best_state``
            type when no new best is seen.
        pre_root:
            Tree built by :func:`rebuild_tree_node` from a replayed log.
        pre_context:
            Pre-populated ``SearchContext`` -- typically holds the
            replayed ``materialized_keys`` set and a freshly seeded RNG.
        start_iter:
            First iteration number to emit (= last replayed iter + 1).
        max_iterations:
            Absolute upper bound (e.g. 350).  ``max_iterations -
            start_iter + 1`` iterations will be executed.
        cumulative_wall_baseline:
            Continuation point for ``cumulative_wall_time_s`` in the
            iter_log -- typically the value from the last replayed iter.
        best_reward_baseline / best_state_baseline:
            Best reward / state recovered from the replay; new iters
            update these via the standard ``reward > best`` check.
        max_cost / on_iteration:
            Same as in :meth:`UCTSearchEngine.search`.
        """
        if start_iter < 1:
            raise ValueError(f"start_iter must be >= 1, got {start_iter}")
        if max_iterations < start_iter:
            raise ValueError(
                f"max_iterations ({max_iterations}) < start_iter "
                f"({start_iter}); nothing to do."
            )

        rng = pre_context.random
        root = pre_root
        context = pre_context

        if batch_size > 0:
            context.extra["_sampling_mode"] = True

        best_state: Optional[SearchState] = best_state_baseline
        best_reward: float = float(best_reward_baseline)
        cumulative_wall = float(cumulative_wall_baseline)
        iteration = start_iter - 1  # so the trailing return uses the last value

        for iteration in range(start_iter, max_iterations + 1):
            t0 = time.time()
            keys_before = len(context.materialized_keys)

            # --- 0. Draw batch for this iteration (sampling mode) ---
            if batch_size > 0 and frozen_samples:
                iter_rng = random.Random(self.random_seed + iteration * 7919)
                current_batches = _draw_batches(frozen_samples, batch_size, iter_rng)
                context.extra["current_batches"] = current_batches
            else:
                context.extra.pop("current_batches", None)

            context.extra["iteration"] = iteration

            # --- 1. Selection ---
            # Forward `iteration` so cost-penalty lambda decay matches
            # `UCTSearchEngine.search()` (search.py:369).  Without this,
            # post-resume selection is stuck at lambda(t=1) and over-explores.
            node = self._select(root, context, iteration=iteration)
            selection_path = self._trace_path(root, node)

            # --- 2. Expansion ---
            is_expansion = False
            if not node.state.is_terminal():
                unexpanded = node.unexpanded_actions()
                if unexpanded:
                    action = rng.choice(unexpanded)
                    child_state = node.state.child(action)
                    node = node.add_child(action, child_state)
                    is_expansion = True

            # --- 3. Evaluation ---
            eval_state = node.state
            if not eval_state.is_terminal():
                eval_state = random_rollout(eval_state, rng)

            evaluation_failed = False
            try:
                result = self.evaluator.evaluate(eval_state, context)
            except Exception as exc:
                _record_evaluator_failure(eval_state, exc, iteration)
                logger.error(
                    "iter=%d EVALUATOR FAILED: %s: %s",
                    iteration, type(exc).__name__, exc,
                    exc_info=True,
                )
                result = EvaluationResult(
                    reward=float("nan"),
                    total_cost=0.0,
                    cluster_results=[],
                    metadata={
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "uct_iteration": iteration,
                    },
                )
                evaluation_failed = True

            for cr in result.cluster_results:
                for key in cr.materialized_keys:
                    context.materialized_keys.add(key)

            # --- Update cached_qids for sampling cost model ---
            # Mirror search.py:417-421: update whenever drawn_qids are
            # present (the cost model only consults the cache when
            # `_sampling_mode` is on, so the extra writes are harmless
            # in full-pool mode but let us share the code path).
            drawn_qids = result.metadata.get("drawn_qids_per_spectrum")
            if drawn_qids:
                try:
                    choices = eval_state.state_key()
                    if isinstance(choices, tuple):
                        update_cached_qids(context, choices, drawn_qids)
                except Exception:
                    pass

            context.total_cost += result.total_cost
            context.total_evaluations += 1

            reward = result.reward
            reward_is_finite = isinstance(reward, (int, float)) and not (
                isinstance(reward, float) and math.isnan(reward)
            )

            if reward_is_finite and reward > best_reward:
                best_reward = reward
                best_state = eval_state

            # --- 4. Backpropagation ---
            if reward_is_finite:
                self._backpropagate(node, reward)
            else:
                logger.warning(
                    "iter=%d skipping backprop: reward=%r failed=%s",
                    iteration, reward, evaluation_failed,
                )

            wall_s = time.time() - t0
            cumulative_wall += wall_s

            # --- 5. Emit iteration record ---
            if on_iteration is not None:
                rec = self._build_record(
                    iteration=iteration,
                    eval_state=eval_state,
                    result=result,
                    root=root,
                    selection_path=selection_path,
                    is_expansion=is_expansion,
                    best_reward=best_reward,
                    best_state=best_state,
                    keys_before=keys_before,
                    context=context,
                    wall_s=wall_s,
                    cumulative_wall=cumulative_wall,
                )
                try:
                    on_iteration(rec)
                except Exception as exc:
                    logger.error(
                        "iter=%d on_iteration callback failed: %s: %s",
                        iteration, type(exc).__name__, exc,
                        exc_info=True,
                    )
                    _record_evaluator_failure(eval_state, exc, iteration)

            # --- 6. Budget check ---
            if max_cost is not None and context.total_cost >= max_cost:
                logger.info(
                    "Cost budget exhausted at iteration %d (%.2f >= %.2f)",
                    iteration, context.total_cost, max_cost,
                )
                break

        return SearchResult(
            best_state=best_state,
            best_reward=best_reward,
            root_node=root,
            iterations=iteration,
            total_evaluations=context.total_evaluations,
            total_cost=context.total_cost,
            context=context,
        )


__all__ = ["ResumableUCTSearchEngine", "rebuild_tree_node"]
