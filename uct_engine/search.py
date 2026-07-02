"""UCT search engine with cost-aware tree policy.

Implements Algorithm 1 from the paper: selection, expansion, evaluation,
and backpropagation with reuse-aware marginal cost regularization.

Enhanced with:
  - Cold-start: seed root children with initial rollouts before main loop.
  - Per-batch adaptive cost: draw Batch_z before selection, compute hit-prob.
  - Lambda decay: lambda_t = lambda_0 / sqrt(t).
"""
from __future__ import annotations

import logging
import math
import random
import time
from typing import Any, Callable, Dict, Hashable, Optional

from .core import TreeNode
from .interfaces import (
    CostModel,
    EvaluationResult,
    Evaluator,
    ScoreFunction,
    SearchContext,
    SearchResult,
    SearchState,
)

logger = logging.getLogger(__name__)


def _record_evaluator_failure(
    eval_state: SearchState,
    exc: BaseException,
    iteration: int,
) -> None:
    """Forward an evaluator exception to the shared FailureTracker."""
    try:
        from rag_contracts.failure_tracker import get_tracker
    except Exception:
        logger.error(
            "iter=%d evaluator exception (no FailureTracker available): %r",
            iteration, exc,
            exc_info=True,
        )
        return
    try:
        config_str = eval_state.pretty()
    except Exception:
        config_str = str(eval_state)
    get_tracker().record(
        stage="uct_engine.evaluate",
        error=exc,
        config=config_str,
        extra={"iteration": iteration},
    )


# ---------------------------------------------------------------------------
# Per-iteration record (passed to on_iteration callback)
# ---------------------------------------------------------------------------

class IterationRecord:
    """All data captured for one UCT iteration."""

    __slots__ = (
        "iteration", "config_key", "config_dict",
        "total_reward", "group_means", "spectrum_rewards",
        "total_cost", "new_materialized_keys", "cumulative_materialized_keys",
        "selection_path", "is_expansion",
        "best_reward_so_far", "best_config_so_far",
        "wall_time_s", "cumulative_wall_time_s",
        "per_spectrum_cost",
        "q_values_at_root", "visit_counts_at_root",
        "drawn_qids_per_spectrum",
    )

    def __init__(self) -> None:
        for attr in self.__slots__:
            setattr(self, attr, None)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__ if getattr(self, k) is not None}


# Type alias for the callback
OnIterationCallback = Callable[[IterationRecord], None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_rollout(state: SearchState, rng: random.Random) -> SearchState:
    """Complete *state* to a terminal config by picking random actions."""
    current = state
    while not current.is_terminal():
        actions = current.available_actions()
        current = current.child(rng.choice(actions))
    return current


def _draw_batches(
    frozen_samples: dict[str, list],
    batch_size: int,
    rng: random.Random,
) -> dict[str, list[str]]:
    """Draw batch_size random qids per spectrum for cost computation.

    Returns {spectrum_id: [qid, ...]} where qids use the prefixed format
    ``{sid}::{raw_id}`` to match cached_qids produced by evaluators.
    """
    batches: dict[str, list[str]] = {}
    for sid, items in frozen_samples.items():
        if not items:
            continue
        k = min(batch_size, len(items))
        drawn = rng.sample(items, k)
        qids = []
        for it in drawn:
            if isinstance(it, dict):
                raw = it.get("query_id") or it.get("_id") or it.get("id", "")
                qids.append(f"{sid}::{raw}")
            else:
                qids.append(f"{sid}::{it}")
        batches[sid] = qids
    return batches


def update_cached_qids(
    context: SearchContext,
    choices: tuple[str, ...],
    drawn_qids_per_spectrum: dict[str, list[str]],
) -> None:
    """Register evaluated qids at ALL prefix depths into cached_qids.

    After evaluating config (A,B,C,D,E) on qids for each spectrum,
    adds them to cached_qids at depths 1..5:
      ((A,), sid), ((A,B), sid), ..., ((A,B,C,D,E), sid)
    """
    cached_qids = context.extra.setdefault("cached_qids", {})
    for depth in range(1, len(choices) + 1):
        prefix = choices[:depth]
        for sid, qids in drawn_qids_per_spectrum.items():
            key = (prefix, sid)
            if key not in cached_qids:
                cached_qids[key] = set()
            cached_qids[key].update(qids)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class UCTSearchEngine:
    """Cost-aware UCT search over a discrete compositional space.

    Parameters
    ----------
    evaluator : Evaluator
        Scores terminal configurations (may use benchmark clusters).
    scorer : ScoreFunction
        Tree-policy acquisition function.
    cost_model : CostModel
        Marginal cost estimator (e.g. reuse-aware).
    exploration_constant : float
        c_uct in the UCT formula.
    random_seed : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        scorer: ScoreFunction,
        cost_model: CostModel,
        exploration_constant: float = 1.4,
        random_seed: Optional[int] = 42,
    ) -> None:
        self.evaluator = evaluator
        self.scorer = scorer
        self.cost_model = cost_model
        self.exploration_constant = exploration_constant
        self.random_seed = random_seed

    # -- public API -----------------------------------------------------------

    def cold_start(
        self,
        root: TreeNode,
        context: SearchContext,
        *,
        budget_per_spectrum: int = 10,
        on_iteration: Optional[OnIterationCallback] = None,
    ) -> tuple[float, Optional[SearchState]]:
        """Seed each root child with one random rollout evaluated at budget_per_spectrum.

        Expands all root actions, rolls out each to terminal, evaluates with
        a small batch, and backprops to give initial Q estimates.

        Returns (best_reward, best_state) from cold-start phase.
        """
        rng = context.random
        best_reward = float("-inf")
        best_state: Optional[SearchState] = None

        actions = root.state.available_actions()
        logger.info("Cold-start: %d root children, budget=%d/spectrum",
                    len(actions), budget_per_spectrum)

        for action in actions:
            t0 = time.time()
            keys_before = len(context.materialized_keys)

            child_state = root.state.child(action)
            child_node = root.add_child(action, child_state)

            terminal = random_rollout(child_state, rng)

            context.extra["_cold_start_budget"] = 0  # full pool for cold-start

            try:
                result = self.evaluator.evaluate(terminal, context)
            except Exception as exc:
                _record_evaluator_failure(terminal, exc, 0)
                logger.error("Cold-start FAIL for %s: %s", action, exc)
                result = EvaluationResult(
                    reward=float("nan"), total_cost=0.0, cluster_results=[],
                    metadata={"error": str(exc)},
                )

            context.extra.pop("_cold_start_budget", None)

            for cr in result.cluster_results:
                for key in cr.materialized_keys:
                    context.materialized_keys.add(key)

            # Update cached_qids from evaluation
            drawn_qids = result.metadata.get("drawn_qids_per_spectrum")
            if drawn_qids:
                choices = terminal.state_key()  # type: ignore
                if isinstance(choices, tuple):
                    update_cached_qids(context, choices, drawn_qids)

            context.total_cost += result.total_cost
            context.total_evaluations += 1

            reward = result.reward
            reward_is_finite = isinstance(reward, (int, float)) and not (
                isinstance(reward, float) and math.isnan(reward)
            )

            if reward_is_finite:
                self._backpropagate(child_node, reward)
                if reward > best_reward:
                    best_reward = reward
                    best_state = terminal

            wall_s = time.time() - t0
            logger.info(
                "  Cold-start [%s] -> reward=%.4f, wall=%.1fs",
                action, reward if reward_is_finite else float("nan"), wall_s,
            )

            if on_iteration is not None:
                rec = IterationRecord()
                rec.iteration = 0
                rec.config_key = terminal.pretty() if hasattr(terminal, "pretty") else str(terminal)
                try:
                    key = terminal.state_key()
                    if hasattr(key, "__iter__") and not isinstance(key, str):
                        rec.config_dict = {f"slot_{i}": str(v) for i, v in enumerate(key)}
                except Exception:
                    pass
                rec.total_reward = round(reward, 6) if reward_is_finite else None
                rec.total_cost = round(result.total_cost, 4)
                rec.new_materialized_keys = len(context.materialized_keys) - keys_before
                rec.cumulative_materialized_keys = len(context.materialized_keys)
                rec.selection_path = [str(action)]
                rec.is_expansion = True
                rec.best_reward_so_far = round(best_reward, 6) if best_reward > float("-inf") else None
                rec.wall_time_s = round(wall_s, 3)
                rec.q_values_at_root = {
                    str(a): round(c.q_value, 6) for a, c in root.children.items()
                }
                rec.visit_counts_at_root = {
                    str(a): c.visit_count for a, c in root.children.items()
                }
                rec.drawn_qids_per_spectrum = (
                    result.metadata.get("drawn_qids_per_spectrum")
                    if context.extra.get("_sampling_mode")
                    else None
                )
                try:
                    on_iteration(rec)
                except Exception:
                    pass

        return best_reward, best_state

    def search(
        self,
        root_state: SearchState,
        max_iterations: int = 200,
        max_cost: Optional[float] = None,
        on_iteration: Optional[OnIterationCallback] = None,
        *,
        cold_start_budget: int = 0,
        batch_size: int = 0,
        frozen_samples: Optional[dict[str, list]] = None,
    ) -> SearchResult:
        """Run UCT search from *root_state*.

        Parameters
        ----------
        cold_start_budget : int
            If >0, run cold_start() with this many items/spectrum before main loop.
        batch_size : int
            If >0, draw batch_size items/spectrum at the start of each iteration
            for adaptive cost computation. Passed to evaluator via context.
        frozen_samples : dict
            Full frozen sample pool per spectrum (needed for batch drawing).
        on_iteration
            Optional callback invoked after each iteration.
        """
        rng = random.Random(self.random_seed)
        context = SearchContext(random=rng)

        if batch_size > 0:
            context.extra["_sampling_mode"] = True

        if frozen_samples:
            population_sizes = {sid: len(items) for sid, items in frozen_samples.items()}
            context.extra["population_sizes"] = population_sizes

        root = TreeNode(state=root_state)
        best_state: Optional[SearchState] = None
        best_reward: float = float("-inf")
        cumulative_wall = 0.0

        # --- Cold start ---
        if cold_start_budget > 0:
            cs_reward, cs_state = self.cold_start(
                root, context,
                budget_per_spectrum=cold_start_budget,
                on_iteration=on_iteration,
            )
            if cs_state is not None and cs_reward > best_reward:
                best_reward = cs_reward
                best_state = cs_state

        for iteration in range(1, max_iterations + 1):
            t0 = time.time()
            keys_before = len(context.materialized_keys)

            # --- 0. Draw batch for this iteration (before selection) ---
            if batch_size > 0 and frozen_samples:
                iter_rng = random.Random(self.random_seed + iteration * 7919)
                current_batches = _draw_batches(frozen_samples, batch_size, iter_rng)
                context.extra["current_batches"] = current_batches
            else:
                context.extra.pop("current_batches", None)

            # Store iteration number for lambda decay
            context.extra["iteration"] = iteration

            # --- 1. Selection ---
            node = self._select(root, context, iteration=iteration)

            # Track selection path
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

            # Harvest materialized keys from cluster results -> Path_t
            for cr in result.cluster_results:
                for key in cr.materialized_keys:
                    context.materialized_keys.add(key)

            # --- Update cached_qids for Q_{t+1} ---
            drawn_qids = result.metadata.get("drawn_qids_per_spectrum")
            if drawn_qids:
                choices = eval_state.state_key()  # type: ignore
                if isinstance(choices, tuple):
                    update_cached_qids(context, choices, drawn_qids)

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

    # -- internals ------------------------------------------------------------

    def _select(self, node: TreeNode, context: SearchContext,
                iteration: int = 1) -> TreeNode:
        """Descend the tree using the scorer until a leaf or unexpanded node."""
        while not node.state.is_terminal():
            unexpanded = node.unexpanded_actions()
            if unexpanded:
                return node
            node = self._best_child(node, context, iteration=iteration)
        return node

    @staticmethod
    def _trace_path(root: TreeNode, leaf: TreeNode) -> list[str]:
        """Walk from root to leaf and collect the actions taken."""
        path: list[str] = []
        current = leaf
        while current is not root and current.parent is not None:
            for action, child in current.parent.children.items():
                if child is current:
                    path.append(str(action))
                    break
            current = current.parent
        path.reverse()
        return path

    @staticmethod
    def _build_record(
        *,
        iteration: int,
        eval_state: SearchState,
        result: EvaluationResult,
        root: TreeNode,
        selection_path: list[str],
        is_expansion: bool,
        best_reward: float,
        best_state: Optional[SearchState],
        keys_before: int,
        context: SearchContext,
        wall_s: float,
        cumulative_wall: float,
    ) -> IterationRecord:
        rec = IterationRecord()
        rec.iteration = iteration

        try:
            key = eval_state.state_key()
            rec.config_key = eval_state.pretty()
            if hasattr(key, "__iter__") and not isinstance(key, str):
                rec.config_dict = {f"slot_{i}": str(v) for i, v in enumerate(key)}
        except Exception as exc:
            logger.warning(
                "iter=%d _build_record state_key/pretty failed: %s: %s; "
                "falling back to str(state)",
                iteration, type(exc).__name__, exc,
            )
            rec.config_key = str(eval_state)

        rec.total_reward = round(result.reward, 6)
        rec.group_means = result.metadata.get("group_means")
        if rec.group_means:
            rec.group_means = {k: round(v, 6) for k, v in rec.group_means.items()}

        rec.spectrum_rewards = {
            cr.cluster_id: round(cr.reward, 6) for cr in result.cluster_results
        }
        rec.per_spectrum_cost = {
            cr.cluster_id: round(cr.cost, 4) for cr in result.cluster_results
        }
        rec.total_cost = round(result.total_cost, 4)
        rec.new_materialized_keys = len(context.materialized_keys) - keys_before
        rec.cumulative_materialized_keys = len(context.materialized_keys)
        rec.selection_path = selection_path
        rec.is_expansion = is_expansion
        rec.best_reward_so_far = round(best_reward, 6)
        rec.best_config_so_far = best_state.pretty() if best_state else None
        rec.wall_time_s = round(wall_s, 3)
        rec.cumulative_wall_time_s = round(cumulative_wall, 3)

        rec.q_values_at_root = {}
        rec.visit_counts_at_root = {}
        for action, child in root.children.items():
            a = str(action)
            rec.q_values_at_root[a] = round(child.q_value, 6)
            rec.visit_counts_at_root[a] = child.visit_count

        rec.drawn_qids_per_spectrum = (
            result.metadata.get("drawn_qids_per_spectrum")
            if context.extra.get("_sampling_mode")
            else None
        )

        return rec

    def _best_child(self, node: TreeNode, context: SearchContext,
                    iteration: int = 1) -> TreeNode:
        best_score = float("-inf")
        best_child: Optional[TreeNode] = None
        for action, child in node.children.items():
            mc = self.cost_model.marginal_cost(node.state, action, context)
            try:
                s = self.scorer.score(
                    parent_visits=node.visit_count,
                    child_visits=child.visit_count,
                    child_q=child.q_value,
                    marginal_cost=mc,
                    exploration_constant=self.exploration_constant,
                    iteration=iteration,
                )
            except TypeError:
                s = self.scorer.score(
                    parent_visits=node.visit_count,
                    child_visits=child.visit_count,
                    child_q=child.q_value,
                    marginal_cost=mc,
                    exploration_constant=self.exploration_constant,
                )
            if s > best_score:
                best_score = s
                best_child = child
        assert best_child is not None
        return best_child

    @staticmethod
    def _backpropagate(node: TreeNode, reward: float) -> None:
        current: Optional[TreeNode] = node
        while current is not None:
            current.update(reward)
            current = current.parent
