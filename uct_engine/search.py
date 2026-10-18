"""Backend-agnostic Agent-UCT tree search.

Each iteration follows the complete-feedback boundary:

1. draw one optional initial batch ``B_t``;
2. select using a caller-supplied complete-evaluation forecast;
3. expand and complete a configuration;
4. evaluate with the same batch;
5. merge only evaluator-reported reuse keys;
6. backpropagate the raw reward.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Hashable

from .checkpoint import CheckpointCodec, SearchCheckpoint
from .core import TreeNode
from .interfaces import (
    BatchProvider,
    CostModel,
    Evaluator,
    ScoreFunction,
    SearchContext,
    SearchResult,
    SearchState,
)


@dataclass(frozen=True)
class SelectionCandidateAudit:
    """Exact numeric inputs used to score one existing tree edge."""

    action: Hashable
    child_visits: int
    child_q: float
    marginal_cost: float
    score: float
    selected: bool


@dataclass(frozen=True)
class SelectionStepAudit:
    """Candidate comparison performed at one fully-expanded parent."""

    parent_path: tuple[Hashable, ...]
    parent_visits: int
    parent_q: float
    selected_action: Hashable
    candidates: tuple[SelectionCandidateAudit, ...]


@dataclass(frozen=True)
class RandomChoiceAudit:
    """One uniform random choice over the exact ordered candidate set."""

    state_path: tuple[Hashable, ...]
    candidates: tuple[Hashable, ...]
    chosen_action: Hashable


@dataclass(frozen=True)
class BackpropagationAudit:
    """One tree node's N/Q transition for the raw iteration reward."""

    node_path: tuple[Hashable, ...]
    before_visits: int
    before_q: float
    after_visits: int
    after_q: float


@dataclass(frozen=True)
class CacheCostAudit:
    """Count-only reuse-ledger and cost transition for one evaluation."""

    materialized_keys_at_selection: int
    materialized_keys_before_merge: int
    materialized_keys_reported: int
    materialized_keys_added: int
    materialized_keys_after_merge: int
    total_cost_before_evaluation: float
    evaluation_cost: float
    total_cost_after_evaluation: float
    total_evaluations_before: int
    total_evaluations_after: int


@dataclass(frozen=True)
class IterationAlgorithmAudit:
    """Payload-free trace of selection, rollout, cache, and backpropagation."""

    root_state_path: tuple[Hashable, ...]
    tree_selection_path: tuple[Hashable, ...]
    expansion_candidates: tuple[Hashable, ...]
    expanded_action: Hashable | None
    tree_evaluation_path: tuple[Hashable, ...]
    rollout_actions: tuple[Hashable, ...]
    rollout_choices: tuple[RandomChoiceAudit, ...]
    selection_steps: tuple[SelectionStepAudit, ...]
    backpropagation: tuple[BackpropagationAudit, ...]
    cache_cost: CacheCostAudit
    batch_cluster_sizes: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class IterationRecord:
    """Generic diagnostics emitted after one completed search iteration."""

    iteration: int
    evaluated_state: SearchState
    reward: float
    evaluation_cost: float
    materialized_keys_added: int
    selection_path: tuple[Hashable, ...]
    expanded: bool
    best_state: SearchState
    best_reward: float
    evaluation_metadata: dict[str, Any] = field(default_factory=dict)
    algorithm_audit: IterationAlgorithmAudit | None = None


@dataclass(frozen=True)
class RootChildSeed:
    """One already-evaluated rollout used to seed a root child.

    ``action_path`` is replayed from the search root before any tree statistics
    are initialized.  Omitting it is valid only when the evaluated terminal
    state is the direct child selected by ``action``.
    """

    action: Hashable
    evaluated_state: SearchState
    reward: float
    action_path: tuple[Hashable, ...] | None = None

    def __post_init__(self) -> None:
        path = (
            (self.action,)
            if self.action_path is None
            else tuple(self.action_path)
        )
        if not path:
            raise ValueError("root-child seed action_path must not be empty")
        if path[0] != self.action:
            raise ValueError(
                "root-child seed action_path must start with its root action"
            )
        object.__setattr__(self, "action_path", path)


def _is_finite_non_negative_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(normalized) and normalized >= 0


@dataclass(frozen=True)
class WarmStartBundle:
    """Atomically couple a warmed ledger with its evaluated root rollouts.

    Integrations may perform extra pre-loop bookkeeping, so the context can
    report more evaluations than there are root-child seeds.  It can never
    report fewer: every seeded visit/Q value must have a corresponding
    evaluation in the shared ledger.
    """

    context: SearchContext
    root_child_seeds: tuple[RootChildSeed, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.context, SearchContext):
            raise TypeError("warm-start context must be a SearchContext")
        object.__setattr__(
            self,
            "root_child_seeds",
            tuple(self.root_child_seeds),
        )
        self.validate()

    def validate(self) -> None:
        seed_count = len(self.root_child_seeds)
        if seed_count == 0:
            raise ValueError("warm start requires at least one root-child seed")
        total_cost = self.context.total_cost
        if not _is_finite_non_negative_number(total_cost):
            raise ValueError(
                "warm-start context total_cost must be finite and non-negative"
            )
        total_evaluations = self.context.total_evaluations
        if (
            isinstance(total_evaluations, bool)
            or not isinstance(total_evaluations, int)
            or total_evaluations < 0
        ):
            raise ValueError(
                "warm-start context total_evaluations must be a non-negative integer"
            )
        if total_evaluations < seed_count:
            raise ValueError(
                "warm-start context total_evaluations must be at least "
                f"the root-child seed count ({seed_count})"
            )


OnIterationCallback = Callable[[IterationRecord], None]
OnCheckpointCallback = Callable[[SearchResult], None]


def random_rollout(state: SearchState, rng: random.Random) -> SearchState:
    """Complete ``state`` by uniformly sampling available actions."""
    terminal, _actions, _choices = _random_rollout_with_actions(state, rng)
    return terminal


def _random_rollout_with_actions(
    state: SearchState,
    rng: random.Random,
    *,
    initial_path: tuple[Hashable, ...] = (),
) -> tuple[
    SearchState,
    tuple[Hashable, ...],
    tuple[RandomChoiceAudit, ...],
]:
    """Complete ``state`` and retain only the chosen action identifiers."""

    current = state
    selected_actions: list[Hashable] = []
    choices: list[RandomChoiceAudit] = []
    while not current.is_terminal():
        actions = current.available_actions()
        if not actions:
            raise ValueError(
                "non-terminal SearchState returned no available actions: "
                f"{current.pretty()}"
            )
        action = rng.choice(actions)
        choices.append(
            RandomChoiceAudit(
                state_path=initial_path + tuple(selected_actions),
                candidates=tuple(actions),
                chosen_action=action,
            )
        )
        selected_actions.append(action)
        current = current.child(action)
    return current, tuple(selected_actions), tuple(choices)


class UCTSearchEngine:
    """Run Agent-UCT over a discrete compositional search space."""

    def __init__(
        self,
        evaluator: Evaluator,
        scorer: ScoreFunction,
        cost_model: CostModel,
        exploration_constant: float = 1.4,
        random_seed: int | None = 42,
        batch_provider: BatchProvider | None = None,
    ) -> None:
        if exploration_constant < 0:
            raise ValueError("exploration_constant must be non-negative")
        self.evaluator = evaluator
        self.scorer = scorer
        self.cost_model = cost_model
        self.exploration_constant = float(exploration_constant)
        self.random_seed = random_seed
        self.batch_provider = batch_provider

    def search(
        self,
        root_state: SearchState,
        max_iterations: int = 200,
        max_cost: float | None = None,
        on_iteration: OnIterationCallback | None = None,
        *,
        warm_start: WarmStartBundle | None = None,
        on_checkpoint: OnCheckpointCallback | None = None,
    ) -> SearchResult:
        """Search from ``root_state``, optionally using an atomic warm start.

        ``warm_start`` couples the pre-materialized context to the root-child
        rollouts that produced its initial tree statistics.  Integrations
        cannot seed Q/visits without supplying the matching ledger, or supply
        a warm ledger without the matching seeds.

        ``max_cost`` stops further iterations once already recorded actual cost
        reaches the target. No predicted bill is used to admit an evaluation.
        The crossing evaluation completes and remains in this raw result.
        Reporting a budget requires an integration-owned copy of the last
        complete affordable snapshot; do not use this raw result's later best
        reward, tree, or cache for an earlier reporting cutoff.

        ``on_checkpoint`` is also called for the initialized search boundary
        and after an evaluator exception, because either boundary may contain
        warm-start state or externally realized cost that must survive a crash.
        """
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if max_cost is not None and not _is_finite_non_negative_number(max_cost):
            raise ValueError("max_cost must be finite and non-negative")

        if warm_start is None:
            context = SearchContext(random=random.Random(self.random_seed))
            root_child_seeds: tuple[RootChildSeed, ...] = ()
        else:
            if not isinstance(warm_start, WarmStartBundle):
                raise TypeError("warm_start must be a WarmStartBundle")
            warm_start.validate()
            context = warm_start.context
            root_child_seeds = warm_start.root_child_seeds

        root = TreeNode(state=root_state)
        best_state, best_reward = self._apply_root_child_seeds(
            root, root_child_seeds
        )
        if on_checkpoint is not None:
            on_checkpoint(
                self._search_result(
                    root=root,
                    context=context,
                    best_state=best_state,
                    best_reward=best_reward,
                    iterations=0,
                )
            )
        return self._continue_search(
            root=root,
            context=context,
            best_state=best_state,
            best_reward=best_reward,
            completed_iterations=0,
            max_iterations=max_iterations,
            max_cost=max_cost,
            on_iteration=on_iteration,
            on_checkpoint=on_checkpoint,
        )

    def resume(
        self,
        checkpoint: SearchCheckpoint,
        codec: CheckpointCodec,
        max_iterations: int = 200,
        max_cost: float | None = None,
        on_iteration: OnIterationCallback | None = None,
        *,
        on_checkpoint: OnCheckpointCallback | None = None,
    ) -> SearchResult:
        """Resume a complete checkpoint for ``max_iterations`` more rounds.

        Iteration numbers continue at ``checkpoint.completed_iterations + 1``.
        ``max_cost`` remains an absolute actual-cost stopping target.
        The engine's evaluator, scorer, cost model, and batch provider are used
        for the continuation; the tree, ledger, RNG, and best result come only
        from the checkpoint. ``on_checkpoint`` first receives that restored
        boundary even when no additional iteration can run.
        """
        if not isinstance(checkpoint, SearchCheckpoint):
            raise TypeError("checkpoint must be a SearchCheckpoint")
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if max_cost is not None and not _is_finite_non_negative_number(max_cost):
            raise ValueError("max_cost must be finite and non-negative")
        (
            root,
            context,
            best_state,
            best_reward,
            completed_iterations,
        ) = checkpoint._restore(codec)
        if on_checkpoint is not None:
            on_checkpoint(
                self._search_result(
                    root=root,
                    context=context,
                    best_state=best_state,
                    best_reward=best_reward,
                    iterations=completed_iterations,
                )
            )
        return self._continue_search(
            root=root,
            context=context,
            best_state=best_state,
            best_reward=best_reward,
            completed_iterations=completed_iterations,
            max_iterations=max_iterations,
            max_cost=max_cost,
            on_iteration=on_iteration,
            on_checkpoint=on_checkpoint,
        )

    def _continue_search(
        self,
        *,
        root: TreeNode,
        context: SearchContext,
        best_state: SearchState | None,
        best_reward: float,
        completed_iterations: int,
        max_iterations: int,
        max_cost: float | None,
        on_iteration: OnIterationCallback | None,
        on_checkpoint: OnCheckpointCallback | None,
    ) -> SearchResult:
        rng = context.random
        root_state_key = root.state.state_key()
        root_state_path = (
            tuple(root_state_key) if isinstance(root_state_key, tuple) else ()
        )
        iterations_completed = completed_iterations

        for iteration in range(
            completed_iterations + 1,
            completed_iterations + max_iterations + 1,
        ):
            if max_cost is not None and context.total_cost >= max_cost:
                break

            context.current_batch = (
                self.batch_provider.draw(iteration, context)
                if self.batch_provider is not None
                else None
            )

            materialized_keys_at_selection = len(context.materialized_keys)
            total_cost_before_evaluation = context.total_cost
            total_evaluations_before = context.total_evaluations
            selection_steps: list[SelectionStepAudit] = []
            node = self._select(
                root,
                context,
                iteration,
                audit_steps=selection_steps,
            )
            selection_path = self._trace_path(root, node)

            expanded = False
            expanded_action: Hashable | None = None
            expansion_candidates: tuple[Hashable, ...] = ()
            if not node.state.is_terminal():
                unexpanded = node.unexpanded_actions()
                if unexpanded:
                    expansion_candidates = tuple(unexpanded)
                    action = rng.choice(unexpanded)
                    node = node.add_child(action, node.state.child(action))
                    expanded = True
                    expanded_action = action

            tree_evaluation_path = self._trace_path(root, node)
            if node.state.is_terminal():
                evaluated_state = node.state
                rollout_actions: tuple[Hashable, ...] = ()
                rollout_choices: tuple[RandomChoiceAudit, ...] = ()
            else:
                (
                    evaluated_state,
                    rollout_actions,
                    rollout_choices,
                ) = _random_rollout_with_actions(
                    node.state,
                    rng,
                    initial_path=tree_evaluation_path,
                )
            try:
                result = self.evaluator.evaluate(evaluated_state, context)
            except BaseException:
                # Evaluators may have incurred provider cost or committed an
                # isolated physical-cache shard before reporting failure.  The
                # current tree, RNG, context, and accounting state are therefore
                # a recovery boundary even though this iteration has no reward
                # and is not backpropagated.
                if on_checkpoint is not None:
                    on_checkpoint(
                        self._search_result(
                            root=root,
                            context=context,
                            best_state=best_state,
                            best_reward=best_reward,
                            iterations=iterations_completed,
                        )
                    )
                raise
            reward = float(result.reward)
            evaluation_cost = float(result.total_cost)
            if not math.isfinite(reward):
                raise ValueError(
                    f"evaluator returned a non-finite reward: {result.reward!r}"
                )
            if not math.isfinite(evaluation_cost) or evaluation_cost < 0:
                raise ValueError(
                    "evaluator returned an invalid total_cost: "
                    f"{result.total_cost!r}"
                )

            keys_before = len(context.materialized_keys)
            keys_reported = len(result.materialized_keys)
            context.materialized_keys.update(result.materialized_keys)
            keys_added = len(context.materialized_keys) - keys_before
            context.total_cost += evaluation_cost
            context.total_evaluations += 1
            iterations_completed = iteration

            if reward > best_reward:
                best_reward = reward
                best_state = evaluated_state

            backpropagation_nodes: list[
                tuple[TreeNode, tuple[Hashable, ...], int, float]
            ] = []
            current: TreeNode | None = node
            while current is not None:
                backpropagation_nodes.append(
                    (
                        current,
                        self._node_path(current),
                        current.visit_count,
                        current.q_value,
                    )
                )
                current = current.parent

            self._backpropagate(node, reward)
            backpropagation = tuple(
                BackpropagationAudit(
                    node_path=path,
                    before_visits=before_visits,
                    before_q=before_q,
                    after_visits=tree_node.visit_count,
                    after_q=tree_node.q_value,
                )
                for tree_node, path, before_visits, before_q in backpropagation_nodes
            )
            batch_cluster_sizes = (
                ()
                if context.current_batch is None
                else tuple(
                    (cluster.cluster_id, len(cluster.item_ids))
                    for cluster in context.current_batch.clusters
                )
            )
            algorithm_audit = IterationAlgorithmAudit(
                root_state_path=root_state_path,
                tree_selection_path=selection_path,
                expansion_candidates=expansion_candidates,
                expanded_action=expanded_action,
                tree_evaluation_path=tree_evaluation_path,
                rollout_actions=rollout_actions,
                rollout_choices=rollout_choices,
                selection_steps=tuple(selection_steps),
                backpropagation=backpropagation,
                cache_cost=CacheCostAudit(
                    materialized_keys_at_selection=materialized_keys_at_selection,
                    materialized_keys_before_merge=keys_before,
                    materialized_keys_reported=keys_reported,
                    materialized_keys_added=keys_added,
                    materialized_keys_after_merge=len(context.materialized_keys),
                    total_cost_before_evaluation=total_cost_before_evaluation,
                    evaluation_cost=evaluation_cost,
                    total_cost_after_evaluation=context.total_cost,
                    total_evaluations_before=total_evaluations_before,
                    total_evaluations_after=context.total_evaluations,
                ),
                batch_cluster_sizes=batch_cluster_sizes,
            )

            if on_iteration is not None:
                assert best_state is not None
                on_iteration(
                    IterationRecord(
                        iteration=iteration,
                        evaluated_state=evaluated_state,
                        reward=reward,
                        evaluation_cost=evaluation_cost,
                        materialized_keys_added=keys_added,
                        selection_path=selection_path,
                        expanded=expanded,
                        best_state=best_state,
                        best_reward=best_reward,
                        evaluation_metadata=dict(result.metadata),
                        algorithm_audit=algorithm_audit,
                    )
                )

            if on_checkpoint is not None:
                on_checkpoint(
                    self._search_result(
                        root=root,
                        context=context,
                        best_state=best_state,
                        best_reward=best_reward,
                        iterations=iteration,
                    )
                )

        return self._search_result(
            root=root,
            context=context,
            best_state=best_state,
            best_reward=best_reward,
            iterations=iterations_completed,
        )

    @staticmethod
    def _search_result(
        *,
        root: TreeNode,
        context: SearchContext,
        best_state: SearchState | None,
        best_reward: float,
        iterations: int,
    ) -> SearchResult:
        return SearchResult(
            best_state=best_state,
            best_reward=best_reward,
            root_node=root,
            iterations=iterations,
            total_evaluations=context.total_evaluations,
            total_cost=context.total_cost,
            context=context,
        )

    @classmethod
    def _apply_root_child_seeds(
        cls,
        root: TreeNode,
        seeds: Sequence[RootChildSeed],
    ) -> tuple[SearchState | None, float]:
        """Create seeded root children and backpropagate their raw rewards."""
        if not seeds:
            return None, float("-inf")
        if root.state.is_terminal():
            raise ValueError("cannot seed root children of a terminal state")

        available_actions = root.state.available_actions()
        seen_actions: set[Hashable] = set()
        best_state: SearchState | None = None
        best_reward = float("-inf")
        for seed in seeds:
            if seed.action in seen_actions:
                raise ValueError(
                    f"duplicate root-child seed action: {seed.action!r}"
                )
            if seed.action not in available_actions:
                raise ValueError(
                    f"unknown root-child seed action: {seed.action!r}"
                )
            if not seed.evaluated_state.is_terminal():
                raise ValueError("root-child seed evaluated_state must be terminal")
            replayed_state = root.state
            child_state: SearchState | None = None
            assert seed.action_path is not None
            for index, action in enumerate(seed.action_path):
                if (
                    replayed_state.is_terminal()
                    or action not in replayed_state.available_actions()
                ):
                    raise ValueError(
                        "root-child seed action_path does not reproduce "
                        "evaluated_state from its root edge"
                    )
                replayed_state = replayed_state.child(action)
                if index == 0:
                    child_state = replayed_state
            if (
                not replayed_state.is_terminal()
                or replayed_state.state_key()
                != seed.evaluated_state.state_key()
            ):
                raise ValueError(
                    "root-child seed action_path does not reproduce "
                    "evaluated_state from its root edge"
                )
            assert child_state is not None
            reward = float(seed.reward)
            if not math.isfinite(reward):
                raise ValueError(
                    f"root-child seed reward must be finite: {seed.reward!r}"
                )

            child = root.add_child(
                seed.action,
                child_state,
            )
            cls._backpropagate(child, reward)
            seen_actions.add(seed.action)
            if reward > best_reward:
                best_state = seed.evaluated_state
                best_reward = reward

        return best_state, best_reward

    def _select(
        self,
        node: TreeNode,
        context: SearchContext,
        iteration: int,
        *,
        audit_steps: list[SelectionStepAudit] | None = None,
    ) -> TreeNode:
        while not node.state.is_terminal():
            if node.unexpanded_actions():
                return node
            node = self._best_child(
                node,
                context,
                iteration,
                audit_steps=audit_steps,
            )
        return node

    def _best_child(
        self,
        node: TreeNode,
        context: SearchContext,
        iteration: int,
        *,
        audit_steps: list[SelectionStepAudit] | None = None,
    ) -> TreeNode:
        best_score = float("-inf")
        best_child: TreeNode | None = None
        best_action: Hashable | None = None
        candidates: list[
            tuple[Hashable, TreeNode, float, float]
        ] = []
        for action, child in node.children.items():
            marginal_cost = self.cost_model.marginal_cost(
                node.state,
                action,
                context,
            )
            score = self.scorer.score(
                parent_visits=node.visit_count,
                child_visits=child.visit_count,
                child_q=child.q_value,
                marginal_cost=marginal_cost,
                exploration_constant=self.exploration_constant,
                iteration=iteration,
            )
            candidates.append((action, child, marginal_cost, score))
            if score > best_score:
                best_score = score
                best_child = child
                best_action = action
        if best_child is None:
            raise ValueError(
                "cannot select a child from a fully expanded node with no children"
            )
        if audit_steps is not None:
            assert best_action is not None
            audit_steps.append(
                SelectionStepAudit(
                    parent_path=self._node_path(node),
                    parent_visits=node.visit_count,
                    parent_q=node.q_value,
                    selected_action=best_action,
                    candidates=tuple(
                        SelectionCandidateAudit(
                            action=action,
                            child_visits=child.visit_count,
                            child_q=child.q_value,
                            marginal_cost=float(marginal_cost),
                            score=float(score),
                            selected=action == best_action,
                        )
                        for action, child, marginal_cost, score in candidates
                    ),
                )
            )
        return best_child

    @staticmethod
    def _node_path(node: TreeNode) -> tuple[Hashable, ...]:
        actions: list[Hashable] = []
        current = node
        while current.parent is not None:
            if current.action_from_parent is None:
                raise RuntimeError("tree node is missing action_from_parent")
            actions.append(current.action_from_parent)
            current = current.parent
        actions.reverse()
        return tuple(actions)

    @staticmethod
    def _trace_path(root: TreeNode, leaf: TreeNode) -> tuple[Hashable, ...]:
        actions: list[Hashable] = []
        current = leaf
        while current is not root and current.parent is not None:
            if current.action_from_parent is None:
                raise RuntimeError("tree node is missing action_from_parent")
            actions.append(current.action_from_parent)
            current = current.parent
        actions.reverse()
        return tuple(actions)

    @staticmethod
    def _backpropagate(node: TreeNode, reward: float) -> None:
        current: TreeNode | None = node
        while current is not None:
            current.update(reward)
            current = current.parent
