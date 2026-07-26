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
from collections.abc import Callable
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
    ) -> SearchResult:
        """Search from ``root_state``.

        ``max_cost`` is checked before a new iteration and after each atomic
        evaluation. An individual evaluation may therefore cross the boundary;
        the core cannot safely interrupt an external evaluator mid-call.
        """
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if max_cost is not None and max_cost < 0:
            raise ValueError("max_cost must be non-negative")

        rng = random.Random(self.random_seed)
        context = SearchContext(random=rng)
        root = TreeNode(state=root_state)
        best_state: SearchState | None = None
        best_reward = float("-inf")
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
