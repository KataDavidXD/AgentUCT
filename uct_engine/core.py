"""Engine-internal tree node used by UCTSearchEngine.

Users interact with SearchState; TreeNode is the bookkeeping wrapper
that the engine maintains during search.
"""
from __future__ import annotations

from typing import Any, Hashable, Optional

from .interfaces import SearchState


class TreeNode:
    """A single node in the UCT search tree.

    Stores visit statistics, value estimates, and parent/child links.
    The *state* field holds the user-provided SearchState; everything
    else is engine bookkeeping.
    """

    __slots__ = (
        "state",
        "parent",
        "action_from_parent",
        "children",
        "visit_count",
        "value_sum",
        "best_value",
        "metadata",
        "_available_actions_cache",
    )

    def __init__(
        self,
        state: SearchState,
        parent: Optional["TreeNode"] = None,
        action_from_parent: Optional[Hashable] = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: dict[Hashable, "TreeNode"] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.best_value: float = float("-inf")
        self.metadata: dict[str, Any] = {}
        self._available_actions_cache: Optional[list[Hashable]] = None

    # -- properties -----------------------------------------------------------

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return len(self.unexpanded_actions()) == 0

    # -- methods --------------------------------------------------------------

    def unexpanded_actions(self) -> list[Hashable]:
        if self._available_actions_cache is None:
            self._available_actions_cache = self.state.available_actions()
        return [a for a in self._available_actions_cache if a not in self.children]

    def add_child(self, action: Hashable, child_state: SearchState) -> "TreeNode":
        child = TreeNode(state=child_state, parent=self, action_from_parent=action)
        self.children[action] = child
        return child

    def update(self, reward: float) -> None:
        self.visit_count += 1
        self.value_sum += reward
        if reward > self.best_value:
            self.best_value = reward

    def __repr__(self) -> str:
        return (
            f"TreeNode(key={self.state.state_key()!r}, "
            f"visits={self.visit_count}, q={self.q_value:.4f})"
        )
