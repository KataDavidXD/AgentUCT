"""Versioned, JSON-safe checkpoints for complete Agent-UCT search sessions.

The checkpoint stores every tree node state and every edge action through an
explicit domain codec.  Restoration therefore never reconstructs a tree by
stringifying actions and replaying them from the root.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Hashable, Protocol, runtime_checkable

from .core import TreeNode
from .interfaces import (
    ClusterBatch,
    EvaluationBatch,
    SearchContext,
    SearchResult,
    SearchState,
)

SCHEMA_VERSION = 1


@runtime_checkable
class CheckpointCodec(Protocol):
    """Encode domain objects into JSON values and decode them losslessly.

    ``encode_state`` handles SearchState instances. ``encode_value`` handles
    actions, materialized keys, batch item identifiers, and ``context.extra``.
    Keeping this boundary explicit prevents implicit ``str(action)`` replay and
    makes provider/model revisions part of the integration's own contract.
    """

    def encode_state(self, state: SearchState) -> Any: ...

    def decode_state(self, payload: Any) -> SearchState: ...

    def encode_value(self, value: Any) -> Any: ...

    def decode_value(self, payload: Any) -> Any: ...


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_finite(value: Any, name: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or (non_negative and normalized < 0):
        qualifier = "finite and non-negative" if non_negative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return normalized


def _json_copy(value: Any, name: str) -> Any:
    try:
        serialized = json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON-serializable: {exc}") from exc
    return json.loads(serialized)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _encode_random_state(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_encode_random_state(item) for item in value]
    if value is None or isinstance(value, (int, float, str)):
        return value
    raise TypeError(f"unsupported random state value: {type(value).__name__}")


def _decode_random_state(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode_random_state(item) for item in value)
    return value


@dataclass(frozen=True)
class _TreeEdgeSnapshot:
    action: Any
    node: "_TreeNodeSnapshot"

    def to_payload(self) -> dict[str, Any]:
        return {
            "action": copy.deepcopy(self.action),
            "node": self.node.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "_TreeEdgeSnapshot":
        if not isinstance(payload, dict) or set(payload) != {"action", "node"}:
            raise ValueError(f"{path} must contain exactly action and node")
        action = _json_copy(payload["action"], f"{path}.action")
        return cls(
            action=action,
            node=_TreeNodeSnapshot.from_payload(payload["node"], f"{path}.node"),
        )


@dataclass(frozen=True)
class _TreeNodeSnapshot:
    state: Any
    visit_count: int
    value_sum: float
    best_value: float | None
    available_actions: tuple[Any, ...] | None
    children: tuple[_TreeEdgeSnapshot, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "state": copy.deepcopy(self.state),
            "visit_count": self.visit_count,
            "value_sum": self.value_sum,
            "best_value": self.best_value,
            "available_actions": (
                None
                if self.available_actions is None
                else copy.deepcopy(list(self.available_actions))
            ),
            "children": [child.to_payload() for child in self.children],
        }

    @classmethod
    def capture(
        cls,
        node: TreeNode,
        codec: CheckpointCodec,
        path: str = "root",
    ) -> "_TreeNodeSnapshot":
        state = _json_copy(codec.encode_state(node.state), f"{path}.state")
        available_actions = (
            None
            if node._available_actions_cache is None
            else tuple(
                _json_copy(codec.encode_value(action), f"{path}.available_actions")
                for action in node._available_actions_cache
            )
        )
        best_value = None if node.best_value == float("-inf") else node.best_value
        child_snapshots = []
        for index, (action, child) in enumerate(node.children.items()):
            if child.parent is not node:
                raise ValueError(
                    f"{path}.children[{index}] has an inconsistent parent link"
                )
            if child.action_from_parent != action:
                raise ValueError(
                    f"{path}.children[{index}] has an inconsistent edge action"
                )
            child_snapshots.append(
                _TreeEdgeSnapshot(
                    action=_json_copy(
                        codec.encode_value(action),
                        f"{path}.children[{index}].action",
                    ),
                    node=cls.capture(
                        child,
                        codec,
                        f"{path}.children[{index}].node",
                    ),
                )
            )
        snapshot = cls(
            state=state,
            visit_count=node.visit_count,
            value_sum=node.value_sum,
            best_value=best_value,
            available_actions=available_actions,
            children=tuple(child_snapshots),
        )
        snapshot.validate(path)
        return snapshot

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "_TreeNodeSnapshot":
        expected = {
            "state",
            "visit_count",
            "value_sum",
            "best_value",
            "available_actions",
            "children",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError(f"{path} has invalid tree-node fields")
        state = _json_copy(payload["state"], f"{path}.state")
        raw_available = payload["available_actions"]
        if raw_available is not None and not isinstance(raw_available, list):
            raise ValueError(f"{path}.available_actions must be a list or null")
        raw_children = payload["children"]
        if not isinstance(raw_children, list):
            raise ValueError(f"{path}.children must be a list")
        snapshot = cls(
            state=state,
            visit_count=payload["visit_count"],
            value_sum=payload["value_sum"],
            best_value=payload["best_value"],
            available_actions=(
                None
                if raw_available is None
                else tuple(
                    _json_copy(item, f"{path}.available_actions[{index}]")
                    for index, item in enumerate(raw_available)
                )
            ),
            children=tuple(
                _TreeEdgeSnapshot.from_payload(item, f"{path}.children[{index}]")
                for index, item in enumerate(raw_children)
            ),
        )
        snapshot.validate(path)
        return snapshot

    def validate(self, path: str) -> None:
        visits = _require_int(self.visit_count, f"{path}.visit_count")
        _require_finite(self.value_sum, f"{path}.value_sum")
        if visits == 0:
            if self.best_value is not None:
                raise ValueError(f"{path}.best_value must be null when unvisited")
        elif self.best_value is None:
            raise ValueError(f"{path}.best_value must be present when visited")
        else:
            _require_finite(self.best_value, f"{path}.best_value")
        encoded_actions = [_canonical_json(edge.action) for edge in self.children]
        if len(set(encoded_actions)) != len(encoded_actions):
            raise ValueError(f"{path}.children contains duplicate encoded actions")

    def restore(
        self,
        codec: CheckpointCodec,
        *,
        parent: TreeNode | None = None,
        action_from_parent: Hashable | None = None,
        path: str = "root",
    ) -> TreeNode:
        state = codec.decode_state(copy.deepcopy(self.state))
        if not isinstance(state, SearchState):
            raise TypeError(f"{path}.state decoder did not return a SearchState")
        node = TreeNode(
            state=state,
            parent=parent,
            action_from_parent=action_from_parent,
        )
        node.visit_count = self.visit_count
        node.value_sum = self.value_sum
        node.best_value = (
            float("-inf") if self.best_value is None else self.best_value
        )
        if self.available_actions is not None:
            decoded_available = [
                codec.decode_value(copy.deepcopy(action))
                for action in self.available_actions
            ]
            for action in decoded_available:
                try:
                    hash(action)
                except TypeError as exc:
                    raise TypeError(
                        f"{path}.available_actions decoder returned an unhashable value"
                    ) from exc
            node._available_actions_cache = decoded_available

        for index, edge in enumerate(self.children):
            action = codec.decode_value(copy.deepcopy(edge.action))
            try:
                hash(action)
            except TypeError as exc:
                raise TypeError(
                    f"{path}.children[{index}].action decoder returned an "
                    "unhashable value"
                ) from exc
            if action in node.children:
                raise ValueError(f"{path}.children decodes to duplicate actions")
            if (
                node._available_actions_cache is not None
                and action not in node._available_actions_cache
            ):
                raise ValueError(
                    f"{path}.children[{index}].action is absent from the saved "
                    "available-action cache"
                )
            child = edge.node.restore(
                codec,
                parent=node,
                action_from_parent=action,
                path=f"{path}.children[{index}].node",
            )
            node.children[action] = child
        return node


@dataclass(frozen=True)
class _ContextSnapshot:
    materialized_keys: tuple[Any, ...]
    current_batch: Any
    total_cost: float
    total_evaluations: int
    random_state: Any
    extra: Any

    @classmethod
    def capture(
        cls,
        context: SearchContext,
        codec: CheckpointCodec,
    ) -> "_ContextSnapshot":
        keys = [
            _json_copy(codec.encode_value(key), "context.materialized_keys")
            for key in context.materialized_keys
        ]
        keys.sort(key=_canonical_json)
        current_batch = None
        if context.current_batch is not None:
            current_batch = {
                "clusters": [
                    {
                        "cluster_id": cluster.cluster_id,
                        "item_ids": [
                            _json_copy(
                                codec.encode_value(item_id),
                                "context.current_batch.item_ids",
                            )
                            for item_id in cluster.item_ids
                        ],
                    }
                    for cluster in context.current_batch.clusters
                ]
            }
        snapshot = cls(
            materialized_keys=tuple(keys),
            current_batch=current_batch,
            total_cost=_require_finite(
                context.total_cost,
                "context.total_cost",
                non_negative=True,
            ),
            total_evaluations=_require_int(
                context.total_evaluations,
                "context.total_evaluations",
            ),
            random_state=_json_copy(
                _encode_random_state(context.random.getstate()),
                "context.random_state",
            ),
            extra=_json_copy(codec.encode_value(context.extra), "context.extra"),
        )
        snapshot.validate()
        return snapshot

    @classmethod
    def from_payload(cls, payload: Any) -> "_ContextSnapshot":
        expected = {
            "materialized_keys",
            "current_batch",
            "total_cost",
            "total_evaluations",
            "random_state",
            "extra",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("context has invalid fields")
        if not isinstance(payload["materialized_keys"], list):
            raise ValueError("context.materialized_keys must be a list")
        snapshot = cls(
            materialized_keys=tuple(
                _json_copy(item, "context.materialized_keys")
                for item in payload["materialized_keys"]
            ),
            current_batch=_json_copy(
                payload["current_batch"],
                "context.current_batch",
            ),
            total_cost=payload["total_cost"],
            total_evaluations=payload["total_evaluations"],
            random_state=_json_copy(payload["random_state"], "context.random_state"),
            extra=_json_copy(payload["extra"], "context.extra"),
        )
        snapshot.validate()
        return snapshot

    def validate(self) -> None:
        _require_finite(self.total_cost, "context.total_cost", non_negative=True)
        _require_int(self.total_evaluations, "context.total_evaluations")
        if self.current_batch is not None:
            if (
                not isinstance(self.current_batch, dict)
                or set(self.current_batch) != {"clusters"}
                or not isinstance(self.current_batch["clusters"], list)
            ):
                raise ValueError("context.current_batch has invalid fields")

    def to_payload(self) -> dict[str, Any]:
        return {
            "materialized_keys": copy.deepcopy(list(self.materialized_keys)),
            "current_batch": copy.deepcopy(self.current_batch),
            "total_cost": self.total_cost,
            "total_evaluations": self.total_evaluations,
            "random_state": copy.deepcopy(self.random_state),
            "extra": copy.deepcopy(self.extra),
        }

    def restore(self, codec: CheckpointCodec) -> SearchContext:
        materialized_keys: set[Hashable] = set()
        for encoded_key in self.materialized_keys:
            key = codec.decode_value(copy.deepcopy(encoded_key))
            try:
                hash(key)
            except TypeError as exc:
                raise TypeError(
                    "context materialized-key decoder returned an unhashable value"
                ) from exc
            materialized_keys.add(key)

        current_batch = None
        if self.current_batch is not None:
            clusters = []
            for index, cluster in enumerate(self.current_batch["clusters"]):
                if (
                    not isinstance(cluster, dict)
                    or set(cluster) != {"cluster_id", "item_ids"}
                    or not isinstance(cluster["cluster_id"], str)
                    or not isinstance(cluster["item_ids"], list)
                ):
                    raise ValueError(
                        f"context.current_batch.clusters[{index}] has invalid fields"
                    )
                item_ids = tuple(
                    codec.decode_value(copy.deepcopy(item))
                    for item in cluster["item_ids"]
                )
                clusters.append(ClusterBatch(cluster["cluster_id"], item_ids))
            current_batch = EvaluationBatch(tuple(clusters))

        extra = codec.decode_value(copy.deepcopy(self.extra))
        if not isinstance(extra, dict):
            raise TypeError("context.extra decoder must return a dict")
        rng = random.Random()
        try:
            rng.setstate(_decode_random_state(copy.deepcopy(self.random_state)))
        except (TypeError, ValueError) as exc:
            raise ValueError("context.random_state is invalid") from exc
        return SearchContext(
            materialized_keys=materialized_keys,
            current_batch=current_batch,
            total_cost=self.total_cost,
            total_evaluations=self.total_evaluations,
            random=rng,
            extra=extra,
        )


@dataclass(frozen=True)
class SearchCheckpoint:
    """A complete, versioned search checkpoint ready for JSON persistence."""

    root: _TreeNodeSnapshot
    context: _ContextSnapshot
    best_state: Any | None
    best_reward: float | None
    completed_iterations: int
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def capture(
        cls,
        result: SearchResult,
        codec: CheckpointCodec,
    ) -> "SearchCheckpoint":
        if not isinstance(codec, CheckpointCodec):
            raise TypeError("codec must implement CheckpointCodec")
        if not isinstance(result.root_node, TreeNode):
            raise TypeError("result.root_node must be a TreeNode")
        if result.root_node.parent is not None:
            raise ValueError("result.root_node must not have a parent")
        if result.root_node.action_from_parent is not None:
            raise ValueError("result.root_node must not have an incoming action")
        if result.total_evaluations != result.context.total_evaluations:
            raise ValueError(
                "result.total_evaluations must match result.context.total_evaluations"
            )
        if result.total_cost != result.context.total_cost:
            raise ValueError("result.total_cost must match result.context.total_cost")
        completed_iterations = _require_int(
            result.iterations,
            "completed_iterations",
        )
        if result.best_state is None:
            if result.best_reward != float("-inf"):
                raise ValueError("an empty best_state requires best_reward=-inf")
            best_state = None
            best_reward = None
        else:
            if not result.best_state.is_terminal():
                raise ValueError("best_state must be terminal")
            best_state = _json_copy(
                codec.encode_state(result.best_state),
                "best_state",
            )
            best_reward = _require_finite(result.best_reward, "best_reward")
        checkpoint = cls(
            root=_TreeNodeSnapshot.capture(result.root_node, codec),
            context=_ContextSnapshot.capture(result.context, codec),
            best_state=best_state,
            best_reward=best_reward,
            completed_iterations=completed_iterations,
        )
        checkpoint.validate()
        return checkpoint

    @classmethod
    def from_payload(cls, payload: Any) -> "SearchCheckpoint":
        expected = {
            "schema_version",
            "root",
            "context",
            "best_state",
            "best_reward",
            "completed_iterations",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("checkpoint has invalid fields")
        if (
            isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], int)
            or payload["schema_version"] != SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported checkpoint schema_version: "
                f"{payload['schema_version']!r}"
            )
        checkpoint = cls(
            root=_TreeNodeSnapshot.from_payload(payload["root"], "root"),
            context=_ContextSnapshot.from_payload(payload["context"]),
            best_state=(
                None
                if payload["best_state"] is None
                else _json_copy(payload["best_state"], "best_state")
            ),
            best_reward=payload["best_reward"],
            completed_iterations=payload["completed_iterations"],
            schema_version=payload["schema_version"],
        )
        checkpoint.validate()
        return checkpoint

    def validate(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported checkpoint schema_version: {self.schema_version!r}"
            )
        _require_int(self.completed_iterations, "completed_iterations")
        if self.completed_iterations > self.context.total_evaluations:
            raise ValueError(
                "completed_iterations cannot exceed context.total_evaluations"
            )
        if self.best_state is None:
            if self.best_reward is not None:
                raise ValueError("best_reward must be null when best_state is null")
        elif self.best_reward is None:
            raise ValueError("best_reward must be present when best_state is present")
        else:
            _require_finite(self.best_reward, "best_reward")

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "root": self.root.to_payload(),
            "context": self.context.to_payload(),
            "best_state": copy.deepcopy(self.best_state),
            "best_reward": self.best_reward,
            "completed_iterations": self.completed_iterations,
        }
        return _json_copy(payload, "checkpoint")

    def _restore(
        self,
        codec: CheckpointCodec,
    ) -> tuple[
        TreeNode,
        SearchContext,
        SearchState | None,
        float,
        int,
    ]:
        if not isinstance(codec, CheckpointCodec):
            raise TypeError("codec must implement CheckpointCodec")
        self.validate()
        root = self.root.restore(codec)
        context = self.context.restore(codec)
        if self.best_state is None:
            best_state = None
            best_reward = float("-inf")
        else:
            best_state = codec.decode_state(copy.deepcopy(self.best_state))
            if not isinstance(best_state, SearchState):
                raise TypeError("best_state decoder did not return a SearchState")
            assert self.best_reward is not None
            best_reward = self.best_reward
        return (
            root,
            context,
            best_state,
            best_reward,
            self.completed_iterations,
        )


__all__ = ["CheckpointCodec", "SearchCheckpoint"]
