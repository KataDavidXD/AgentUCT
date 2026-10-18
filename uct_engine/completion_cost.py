"""Online prices and sampled complete-feedback costs from arXiv:2607.24162v3.

The estimate is an empirical completion expectation, not a confidence bound.
Execution, successful prefix publication and actual billing belong to the caller.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
from typing import Callable, Hashable, Iterable, Mapping

from .interfaces import CostModel, SearchContext, SearchState

Prefix = tuple[Hashable, ...]
ResourceKey = tuple[Hashable, Prefix]


def _nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class ResourceReceipt:
    """One fresh physical training receipt; restored objects supply no sample."""

    receipt_id: Hashable
    query_id: Hashable
    prefix: Prefix
    actual_cost: float


class OnlineResourcePrices:
    """One prior observation plus capped query-prefix/prefix/action samples.

    Caps constrain predictions only. The caller retains the uncapped physical
    receipts in its actual-cost ledger, including failed attempts and retries.
    """

    def __init__(self, priors: Mapping[Hashable, float], caps: Mapping[Hashable, float]):
        if not priors or set(priors) != set(caps):
            raise ValueError("nonempty priors and caps must have the same actions")
        self.caps = {a: _nonnegative(v, "price cap") for a, v in caps.items()}
        self.priors = {a: min(self.caps[a], _nonnegative(v, "price prior"))
                       for a, v in priors.items()}
        self._query: dict[ResourceKey, tuple[float, int]] = {}
        self._prefix: dict[Prefix, tuple[float, int]] = {}
        self._action: dict[Hashable, tuple[float, int]] = {}
        self._receipts: dict[Hashable, ResourceReceipt] = {}

    def observe(self, receipt: ResourceReceipt) -> None:
        prefix = tuple(receipt.prefix)
        if not prefix or prefix[-1] not in self.priors:
            raise ValueError("receipt needs a nonempty prefix with a priced action")
        actual = _nonnegative(receipt.actual_cost, "physical cost")
        normalized = ResourceReceipt(receipt.receipt_id, receipt.query_id, prefix, actual)
        previous = self._receipts.get(receipt.receipt_id)
        if previous is not None:
            if previous != normalized:
                raise ValueError("conflicting receipt ID")
            return
        value = min(actual, self.caps[prefix[-1]])
        for table, key in ((self._query, (receipt.query_id, prefix)),
                           (self._prefix, prefix), (self._action, prefix[-1])):
            total, count = table.get(key, (0.0, 0))
            table[key] = (total + value, count + 1)
        self._receipts[receipt.receipt_id] = normalized

    def price(self, query_id: Hashable, prefix: Prefix) -> float:
        prefix = tuple(prefix)
        if not prefix:
            raise ValueError("resource prefix must be nonempty")
        total, count = self._query.get((query_id, prefix), self._prefix.get(
            prefix, self._action.get(prefix[-1], (0.0, 0))))
        return (self.priors[prefix[-1]] + total) / (1 + count)

    def frozen_lookup(self) -> Callable[[Hashable, Prefix], float]:
        """Freeze all price tiers for one forecast without copying receipt history."""
        priors = dict(self.priors)
        query, prefix, action = dict(self._query), dict(self._prefix), dict(self._action)

        def price(qid: Hashable, path: Prefix) -> float:
            total, count = query.get((qid, path), prefix.get(
                path, action.get(path[-1], (0.0, 0))))
            return (priors[path[-1]] + total) / (1 + count)

        return price


@dataclass(frozen=True)
class CompletionForecast:
    cost: float
    sampled_paths: tuple[Prefix, ...]
    target: str
    reference: str


class HistoricalCompletionCostModel(CostModel):
    """Sample legal completions and sum each query's entire missing suffix.

    States use tuple-valued ``state_key()`` prefixes and pure ``child()`` previews.
    Cache keys are ``(query_id, full_prefix)``. ``structural_zero`` declares only
    inherently free resources and must not inspect a mutable execution cache.

    ``reference`` is historical, uniform (uniform successive legal actions), or
    static_full (frozen initial prices/law/tape and empty pricing cache).
    StaticFull changes prediction only; it never clears the execution context.
    Dynamic forecasts also read exact known zero-work keys from
    ``context.extra['zero_work_resources']``. Such observations do not imply a
    materialized checkpoint, and StaticFull ignores these later observations.
    """

    def __init__(
        self, query_ids: Iterable[Hashable], priors: Mapping[Hashable, float],
        caps: Mapping[Hashable, float], *, samples: int = 16, seed: int = 11,
        reference: str = "historical", target: str = "complete", max_depth: int = 128,
        structural_zero: Callable[[Hashable, Prefix], bool] | None = None,
    ):
        self.query_ids = tuple(query_ids)
        if not self.query_ids or len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("query IDs must be nonempty and unique")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("samples must be a positive integer")
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if reference not in {"historical", "uniform", "static_full"}:
            raise ValueError("unknown completion reference")
        if target not in {"complete", "next_unpaid"}:
            raise ValueError("target must be complete or next_unpaid")
        if reference == "static_full" and target != "complete":
            raise ValueError("StaticFull prices complete feedback")
        self.prices = OnlineResourcePrices(priors, caps)
        self.samples, self.seed, self.max_depth = samples, seed, max_depth
        self.reference, self.target = reference, target
        self.structural_zero = structural_zero or (lambda _qid, _prefix: False)
        self._initial_prices = dict(self.prices.priors)
        self._transitions: dict[Prefix, Counter] = {}
        self._evaluations: dict[Hashable, Prefix] = {}

    @staticmethod
    def _prefix(state: SearchState) -> Prefix:
        key = state.state_key()
        if not isinstance(key, tuple):
            raise ValueError("completion states need tuple-valued prefix keys")
        return key

    def observe_completed(
        self, evaluation_id: Hashable, path: Prefix,
        receipts: Iterable[ResourceReceipt] = (), *, complete: bool = True,
        scope: str = "training",
    ) -> None:
        """Commit observations only after complete training feedback.

        The caller may keep incomplete/held-out bills in its actual ledger;
        they do not fit the completion model. Repeated IDs are idempotent.
        """
        if not isinstance(complete, bool) or scope not in {"training", "validation", "test"}:
            raise ValueError("declare a valid scope and a boolean completion flag")
        if not complete or scope != "training" or self.reference == "static_full":
            return
        path = tuple(path)
        if not path or any(action not in self.prices.priors for action in path):
            raise ValueError("completed path needs known priced actions")
        if evaluation_id in self._evaluations:
            if self._evaluations[evaluation_id] != path:
                raise ValueError("conflicting evaluation ID")
            return
        # Validate receipt identities and prices before changing any statistics.
        staged = tuple(receipts)
        identities = {}
        for receipt in staged:
            if not isinstance(receipt, ResourceReceipt):
                raise TypeError("training receipts must be ResourceReceipt objects")
            if receipt.query_id not in self.query_ids:
                raise ValueError("training receipt query must belong to the fixed panel")
            prefix = tuple(receipt.prefix)
            if not prefix or prefix[-1] not in self.prices.priors:
                raise ValueError("receipt needs known priced actions")
            actual = _nonnegative(receipt.actual_cost, "physical cost")
            normalized = ResourceReceipt(receipt.receipt_id, receipt.query_id, prefix, actual)
            previous = identities.get(receipt.receipt_id,
                                      self.prices._receipts.get(receipt.receipt_id))
            if previous is not None and previous != normalized:
                raise ValueError("conflicting receipt ID")
            identities[receipt.receipt_id] = normalized
        for receipt in staged:
            self.prices.observe(receipt)
        for depth, action in enumerate(path):
            self._transitions.setdefault(path[:depth], Counter())[action] += 1
        self._evaluations[evaluation_id] = path

    def forecast(
        self, state: SearchState, action: Hashable, context: SearchContext,
        *, target: str | None = None,
    ) -> CompletionForecast:
        target = self.target if target is None else target
        if target not in {"complete", "next_unpaid"}:
            raise ValueError("unknown completion target")
        if self.reference == "static_full" and target != "complete":
            raise ValueError("StaticFull prices complete feedback")
        prefix = self._prefix(state)
        if action not in state.available_actions():
            raise ValueError("forecast action must be legal")
        frozen = self.reference == "static_full"
        cache = frozenset() if frozen else frozenset(context.materialized_keys)
        zero_work = frozenset() if frozen else frozenset(
            context.extra.get("zero_work_resources", ()))
        counts = {} if frozen else {p: dict(n) for p, n in self._transitions.items()}
        price = ((lambda _qid, p: self._initial_prices[p[-1]]) if frozen
                 else self.prices.frozen_lookup())
        revision = 0 if frozen else len(self._evaluations)
        version = 0 if frozen else context.total_evaluations
        # A private forecast RNG does not consume context.random. The target is
        # absent from this seed, so complete and NextC use identical completions.
        rng = random.Random(repr((self.seed, revision, version, prefix, action,
                                  self.reference == "uniform")))
        paths, bills = [], []
        for _ in range(self.samples):
            current = state.child(action)
            if self._prefix(current) != prefix + (action,):
                raise ValueError("child must append exactly its chosen action")
            while not current.is_terminal():
                current_prefix = self._prefix(current)
                if len(current_prefix) >= self.max_depth:
                    raise ValueError("completion exceeds max_depth")
                actions = tuple(current.available_actions())
                if not actions or len(set(actions)) != len(actions):
                    raise ValueError("nonterminal state needs unique legal actions")
                observed = counts.get(current_prefix, {})
                weights = ([1.0] * len(actions) if self.reference == "uniform" else
                           [observed.get(a, 0) + 1 / len(actions) for a in actions])
                chosen = rng.choices(actions, weights=weights)[0]
                child = current.child(chosen)
                if self._prefix(child) != current_prefix + (chosen,):
                    raise ValueError("child must append exactly its chosen action")
                current = child
            path = self._prefix(current)
            paths.append(path)
            bill = 0.0
            for qid in self.query_ids:
                saved_depth = max((d for d in range(1, len(path) + 1)
                                   if (qid, path[:d]) in cache), default=0)
                for depth in range(saved_depth + 1, len(path) + 1):
                    resource = path[:depth]
                    if (qid, resource) in zero_work or self.structural_zero(qid, resource):
                        continue
                    value = price(qid, resource)
                    bill += value
                    if target == "next_unpaid" and value > 0:
                        break
            bills.append(bill)
        return CompletionForecast(math.fsum(bills) / self.samples, tuple(paths),
                                  target, self.reference)

    def marginal_cost(self, state, action, context) -> float:
        return self.forecast(state, action, context).cost
