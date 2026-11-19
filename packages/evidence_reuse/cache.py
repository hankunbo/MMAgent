"""Bounded process-local immutable projection cache. Never cache current authority.

Shadow recomputes, compares, and returns the fresh value. No patient results or
authorization tokens survive here. Enabled mode requires a separate future gate.
"""

import copy
from collections import OrderedDict
from time import perf_counter

from packages.grounding.models import digest


class ShadowCache:
    def __init__(self, *, limit=64):
        if type(limit) is not int or not 1 <= limit <= 1024:
            raise ValueError("reuse_cache_limit")
        self.limit = limit
        self.values = OrderedDict()
        self.events = []

    def project(self, *, namespace, bindings, compute, revalidate):
        before = revalidate()  # Original owner, never a stored Boolean.
        key = digest(dict(namespace=namespace, bindings=bindings, current=before))
        prior = self.values.get(key)
        start = perf_counter()
        fresh = compute()
        elapsed = perf_counter() - start
        if revalidate() != before:
            raise ValueError("reuse_cache_current_changed")
        if prior is not None and digest(prior) != digest(fresh):
            self.values.pop(key)
            raise ValueError("reuse_cache_shadow_mismatch")
        self.values[key] = copy.deepcopy(fresh)
        self.values.move_to_end(key)
        while len(self.values) > self.limit:
            self.values.popitem(last=False)
        event = dict(
            key=key,
            namespace=namespace,
            mode="shadow",
            candidate_hit=prior is not None,
            recomputed=True,
            current_owner_checks=2,
            compute_seconds=elapsed,
            authorization_cached=False,
        )
        self.events.append(event)
        self.events = self.events[-self.limit :]
        return fresh
