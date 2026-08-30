"""Pure-data export of node-local UCT and Agent-UCT search signals."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from typing import Any

from .core import TreeNode
from .interfaces import CostModel, SearchContext, SearchResult
from .scoring import CostAwareUCTScorer

ValueEncoder = Callable[[Hashable], Any]


def _json_safe(value: Any, encoder: ValueEncoder | None, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite for JSON export")
        return value
    if isinstance(value, (tuple, list)):
        return [
            _json_safe(item, encoder, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{path} JSON object keys must be strings")
        return {
            key: _json_safe(item, encoder, f"{path}.{key}")
            for key, item in value.items()
        }
    if encoder is None:
        raise TypeError(
            f"{path} is not JSON-safe; pass encode_value for {type(value).__name__}"
        )
    encoded = encoder(value)
    if encoded is value:
        raise TypeError(f"encode_value did not encode {path} ({type(value).__name__})")
    return _json_safe(encoded, encoder, path)


def _available_actions(node: TreeNode) -> list[Hashable]:
    if node._available_actions_cache is not None:
        return list(node._available_actions_cache)
    return node.state.available_actions()


def _node_path(node: TreeNode) -> tuple[Hashable, ...]:
    path: list[Hashable] = []
    current = node
    while current.parent is not None:
        if current.action_from_parent is None:
            raise RuntimeError("tree node is missing action_from_parent")
        path.append(current.action_from_parent)
        current = current.parent
    path.reverse()
    return tuple(path)


def export_search_signals(
    result: SearchResult,
    *,
    scorer: CostAwareUCTScorer,
    cost_model: CostModel,
    exploration_constant: float,
    iteration: int,
    context: SearchContext | None = None,
    encode_value: ValueEncoder | None = None,
) -> dict[str, Any]:
    """Export a deterministic JSON-safe snapshot of every tree action.

    The snapshot is observational: it does not draw a batch, expand the tree,
    evaluate a state, mutate visits/Q, or backpropagate.  ``context`` defaults
    to ``result.context`` and determines the current logical reuse view.

    This is a projection over the final tree, explicit ``iteration``, and the
    supplied context's current batch/reuse keys.  It is not a historical
    selection trace.  Only the standard :class:`CostAwareUCTScorer` has the
    declared Agent-UCT decomposition.  Custom score functions may implement
    arbitrary math and are intentionally not mislabeled as Agent-UCT terms.
    """
    if not isinstance(result, SearchResult):
        raise TypeError("result must be a SearchResult")
    if not isinstance(result.root_node, TreeNode):
        raise TypeError("result.root_node must be a TreeNode")
    if not isinstance(scorer, CostAwareUCTScorer):
        raise TypeError("search-signal decomposition requires CostAwareUCTScorer")
    if not isinstance(cost_model, CostModel):
        raise TypeError("cost_model must implement CostModel")
    if exploration_constant < 0 or not math.isfinite(exploration_constant):
        raise ValueError("exploration_constant must be finite and non-negative")
    signal_iteration = iteration
    if (
        isinstance(signal_iteration, bool)
        or not isinstance(signal_iteration, int)
        or signal_iteration < 1
    ):
        raise ValueError("iteration must be an integer of at least 1")
    signal_context = result.context if context is None else context
    if not isinstance(signal_context, SearchContext):
        raise TypeError("context must be a SearchContext")

    nodes: list[dict[str, Any]] = []

    def visit(node: TreeNode) -> None:
        available_actions = _available_actions(node)
        if not node.state.is_terminal() and not available_actions:
            raise ValueError(
                "non-terminal SearchState returned no available actions: "
                f"{node.state.pretty()}"
            )
        if available_actions:
            fully_expanded = all(
                action in node.children for action in available_actions
            )
            child_visit_total = sum(
                node.children[action].visit_count
                for action in available_actions
                if action in node.children
            )
            scored_actions: list[tuple[Hashable, dict[str, Any]]] = []
            for action in available_actions:
                child = node.children.get(action)
                child_visits = 0 if child is None else child.visit_count
                child_q = 0.0 if child is None else child.q_value
                marginal_cost = cost_model.marginal_cost(
                    node.state,
                    action,
                    signal_context,
                )
                terms = scorer.decompose(
                    parent_visits=node.visit_count,
                    child_visits=child_visits,
                    child_q=child_q,
                    marginal_cost=marginal_cost,
                    exploration_constant=exploration_constant,
                    iteration=signal_iteration,
                )
                expansion_coefficient = scorer.expansion_cost_coefficient(
                    signal_iteration
                )
                action_path = (*_node_path(node), action)
                scored_actions.append(
                    (
                        action,
                        {
                            "action": _json_safe(action, encode_value, "action"),
                            "child_path": _json_safe(
                                action_path,
                                encode_value,
                                "child_path",
                            ),
                            "expanded": child is not None,
                            "child_visits": child_visits,
                            "child_q_raw": terms.child_q_raw,
                            "exploration_bonus": terms.exploration_bonus,
                            "marginal_cost": terms.marginal_cost,
                            "normalized_marginal_cost": (
                                terms.normalized_marginal_cost
                            ),
                            "cost_scale": terms.cost_scale,
                            "lambda_t": terms.lambda_t,
                            "selection_cost_coefficient": (
                                terms.selection_cost_coefficient
                            ),
                            "expansion_cost_coefficient": expansion_coefficient,
                            "expansion_log_bias": (
                                -expansion_coefficient
                                * terms.normalized_marginal_cost
                            ),
                            "cost_penalty_schedule": terms.cost_schedule,
                            "decayed_cost_penalty": terms.decayed_cost_penalty,
                            "uct_score": terms.uct_score,
                            "agent_uct_selection_score": terms.selection_score,
                            "visit_policy_probability": (
                                None
                                if child_visit_total == 0
                                else child_visits / child_visit_total
                            ),
                            "eligible_for_engine_argmax_if_node_reached": (
                                fully_expanded
                            ),
                            "selected_by_agent_uct_snapshot_argmax": False,
                            "selected_by_uct_only_snapshot_argmax": False,
                        },
                    )
                )

            agent_uct_argmax_action: Hashable | None = None
            uct_only_argmax_action: Hashable | None = None
            if fully_expanded:
                agent_uct_argmax_action = max(
                    scored_actions,
                    key=lambda item: item[1]["agent_uct_selection_score"],
                )[0]
                uct_only_argmax_action = max(
                    scored_actions,
                    key=lambda item: item[1]["uct_score"],
                )[0]
                for action, action_payload in scored_actions:
                    action_payload["selected_by_agent_uct_snapshot_argmax"] = (
                        action == agent_uct_argmax_action
                    )
                    action_payload["selected_by_uct_only_snapshot_argmax"] = (
                        action == uct_only_argmax_action
                    )

            node_path = _node_path(node)
            nodes.append(
                {
                    "path": _json_safe(node_path, encode_value, "node.path"),
                    "state_key": _json_safe(
                        node.state.state_key(),
                        encode_value,
                        "node.state_key",
                    ),
                    "parent_visits": node.visit_count,
                    "parent_q_raw": node.q_value,
                    "selection_mode": (
                        "agent_uct_argmax_if_node_reached"
                        if fully_expanded
                        else (
                            "cost_tilted_expansion"
                            if scorer.expansion_cost_coefficient(signal_iteration)
                            > 0.0
                            else "random_expansion"
                        )
                    ),
                    "agent_uct_snapshot_argmax_action": (
                        None
                        if agent_uct_argmax_action is None
                        else _json_safe(
                            agent_uct_argmax_action,
                            encode_value,
                            "node.agent_uct_snapshot_argmax_action",
                        )
                    ),
                    "uct_only_snapshot_argmax_action": (
                        None
                        if uct_only_argmax_action is None
                        else _json_safe(
                            uct_only_argmax_action,
                            encode_value,
                            "node.uct_only_snapshot_argmax_action",
                        )
                    ),
                    "visit_policy_defined": child_visit_total > 0,
                    "visit_policy_temperature": 1.0,
                    "actions": [payload for _action, payload in scored_actions],
                }
            )

        for child in node.children.values():
            visit(child)

    visit(result.root_node)
    return {
        "schema_version": 1,
        "signal_kind": "uct_search_policy_diagnostic",
        "iteration": signal_iteration,
        "source_search_iterations": result.iterations,
        "total_evaluations": result.total_evaluations,
        "total_cost": result.total_cost,
        "semantics": {
            "signal_scope": "search_snapshot",
            "is_historical_selection_trace": False,
            "reward_basis": "raw_evaluator_reward",
            "is_trajectory_signal": False,
            "is_training_advantage": False,
            "uct_score_is_cost_free_ablation": True,
            "visit_policy_is_advantage": False,
            "reuse_graph_scope": "logical_materialized_keys",
            "batch_basis": "provided_context.current_batch",
            "cost_penalty_schedule": scorer.schedule.value,
            "cost_aware_selection": scorer.cost_aware_selection,
            "cost_aware_expansion": scorer.cost_aware_expansion,
            "cost_scale": scorer.cost_scale,
        },
        "context_current_batch_cluster_sizes": (
            []
            if signal_context.current_batch is None
            else [
                {
                    "cluster_id": cluster.cluster_id,
                    "item_count": len(cluster.item_ids),
                }
                for cluster in signal_context.current_batch.clusters
            ]
        ),
        "nodes": nodes,
    }
