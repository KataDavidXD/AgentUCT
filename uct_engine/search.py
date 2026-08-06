"""Backend-agnostic Agent-UCT tree search.

Each iteration follows the paper boundary:

1. draw one optional initial batch ``B_t``;
2. select with reuse-aware marginal cost computed from that batch;
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


def random_rollout(state: SearchState, rng: random.Random) -> SearchState:
    """Complete ``state`` by uniformly sampling available actions."""
    current = state
    while not current.is_terminal():
        actions = current.available_actions()
        if not actions:
            raise ValueError(
                "non-terminal SearchState returned no available actions: "
                f"{current.pretty()}"
            )
        current = current.child(rng.choice(actions))
    return current


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
    ) -> SearchResult:
        """Search from ``root_state``, optionally using an atomic warm start.

        ``warm_start`` couples the pre-materialized context to the root-child
        rollouts that produced its initial tree statistics.  Integrations
        cannot seed Q/visits without supplying the matching ledger, or supply
        a warm ledger without the matching seeds.

        ``max_cost`` is checked before a new iteration and after each atomic
        evaluation. An individual evaluation may therefore cross the boundary;
        the core cannot safely interrupt an external evaluator mid-call.
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

        rng = context.random
        root = TreeNode(state=root_state)
        best_state, best_reward = self._apply_root_child_seeds(
            root, root_child_seeds
        )
        iterations_completed = 0

        for iteration in range(1, max_iterations + 1):
            if max_cost is not None and context.total_cost >= max_cost:
                break

            context.current_batch = (
                self.batch_provider.draw(iteration, context)
                if self.batch_provider is not None
                else None
            )

            node = self._select(root, context, iteration)
            selection_path = self._trace_path(root, node)

            expanded = False
            if not node.state.is_terminal():
                unexpanded = node.unexpanded_actions()
                if unexpanded:
                    action = rng.choice(unexpanded)
                    node = node.add_child(action, node.state.child(action))
                    expanded = True

            evaluated_state = (
                node.state
                if node.state.is_terminal()
                else random_rollout(node.state, rng)
            )
            result = self.evaluator.evaluate(evaluated_state, context)
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
            context.materialized_keys.update(result.materialized_keys)
            keys_added = len(context.materialized_keys) - keys_before
            context.total_cost += evaluation_cost
            context.total_evaluations += 1
            iterations_completed = iteration

            if reward > best_reward:
                best_reward = reward
                best_state = evaluated_state

            self._backpropagate(node, reward)

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
                    )
                )

        return SearchResult(
            best_state=best_state,
            best_reward=best_reward,
            root_node=root,
            iterations=iterations_completed,
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
    ) -> TreeNode:
        while not node.state.is_terminal():
            if node.unexpanded_actions():
                return node
            node = self._best_child(node, context, iteration)
        return node

    def _best_child(
        self,
        node: TreeNode,
        context: SearchContext,
        iteration: int,
    ) -> TreeNode:
        best_score = float("-inf")
        best_child: TreeNode | None = None
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
            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None:
            raise ValueError(
                "cannot select a child from a fully expanded node with no children"
            )
        return best_child

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
