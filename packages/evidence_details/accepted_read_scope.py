"""Private single-operation proof reuse; never a dispatch or cross-call cache.

Original owners still do the first proof. Every document actually read and every
current source state is read again before returning the outermost proof. No
global atomicity is claimed: dispatch retains its separate current-owner gates.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy, deepcopy

from packages.grounding.models import digest

_scope = ContextVar("accepted_review_read_scope", default=None)
_facts = ContextVar("accepted_review_fact_scope", default=None)


def _callable_identity(value):
    return id(getattr(value, "__self__", None)), id(getattr(value, "__func__", value))


def _fact_binding(owner, proof):
    return (
        id(owner),
        id(owner.base),
        id(getattr(owner, "owner", None)),
        _callable_identity(owner.read),
        _callable_identity(getattr(owner, "guard", None)),
        _callable_identity(proof),
        owner.target,
        getattr(owner, "execution_key", None),
        owner.binding,
        owner.input_binding,
    )


class _Facts:
    """Shared fact proofs only inside one historical acceptance reconstruction."""

    def __init__(self):
        self.entries, self.active = {}, set()
        self.return_checks = []
        self.sources, self.source_active = {}, set()
        self.medical_owner = None
        self.medical_owner_active = False
        self.observations = {}

    def medical_sources(self, factory):
        if self.medical_owner_active:
            raise ValueError("historical_medical_source_factory_cycle")
        if self.medical_owner is None:
            self.medical_owner_active = True
            try:
                owner = factory()
                self.medical_owner = (
                    factory,
                    owner,
                    owner._history_binding(),
                    digest(owner.initial),
                )
            finally:
                self.medical_owner_active = False
        if self.medical_owner[0] is not factory:
            raise ValueError("historical_medical_source_factory_changed")
        self.check_medical_owner()
        return self.medical_owner[1]

    def check_medical_owner(self):
        if self.medical_owner is not None:
            _, owner, binding, initial = self.medical_owner
            if owner._history_binding() != binding or digest(owner.initial) != initial:
                raise ValueError("historical_medical_source_factory_owner_changed")

    def observation_documents(self, owner, keys):
        binding = _observation_document_binding(owner)
        key = binding, keys
        if key not in self.observations:
            if len(self.observations) >= 64:
                return owner.read_many(keys)
            values = owner.read_many(keys)
            if len(values) != len(keys) or _observation_document_binding(owner) != binding:
                raise ValueError("historical_observation_document_binding")
            self.observations[key] = owner, binding, keys, deepcopy(values)
        return deepcopy(self.observations[key][3])

    def verify_observations(self):
        # Scope is already removed: original Runtime and artifact bytes are fresh.
        for owner, binding, keys, expected in self.observations.values():
            if _observation_document_binding(owner) != binding:
                raise ValueError("historical_observation_owner_changed")
            owner._check_root()
            current = owner.read_many(keys)
            if _observation_document_binding(owner) != binding or current != expected:
                raise ValueError("historical_observation_documents_changed")

    def current(self, owner, proof):
        owner.check_projection()
        key, binding = id(owner), _fact_binding(owner, proof)
        if key in self.active or (key not in self.entries and len(self.entries) >= 64):
            raise ValueError("accepted_review_fact_cycle_or_depth")
        if key in self.entries:
            if self.entries[key][1] != binding:
                raise ValueError("accepted_review_fact_owner_changed")
            return
        self.active.add(key)
        try:
            proof()
            if _fact_binding(owner, proof) != binding:
                raise ValueError("accepted_review_fact_owner_changed")
            self.entries[key] = owner, binding, proof
        finally:
            self.active.remove(key)

    def source(self, owner):
        key = id(owner)
        binding = owner._history_binding()
        if key in self.source_active:
            raise ValueError("historical_source_cycle")
        if key not in self.sources and len(self.sources) >= 64:
            # This bound limits optimization memory, not valid history size.
            # Owners beyond it keep their original fresh-read behavior.
            return owner._read_current()
        if key in self.sources:
            previous, expected, value = self.sources[key]
            if previous is not owner or binding != expected:
                raise ValueError("historical_source_owner_changed")
            return deepcopy(value)
        self.source_active.add(key)
        try:
            value = owner._read_current()
            if owner._history_binding() != binding:
                raise ValueError("historical_source_owner_changed")
            self.sources[key] = owner, binding, deepcopy(value)
            return value
        finally:
            self.source_active.remove(key)

    def verify_sources(self):
        # Context has ended: these are real lifecycle/object reads, not memo hits.
        for owner, binding, value in self.sources.values():
            if owner._history_binding() != binding:
                raise ValueError("historical_source_owner_changed")
            current = owner._read_current()
            if owner._history_binding() != binding or current != value:
                raise ValueError("historical_source_changed")

    def verify(self):
        # Called after removing the memo context. Original 06/source/input and
        # artifact owners read current state again, not a cached authorization.
        self.check_medical_owner()
        for owner, binding, proof in self.entries.values():
            if _fact_binding(owner, proof) != binding:
                raise ValueError("accepted_review_fact_owner_changed")
            proof()
            if _fact_binding(owner, proof) != binding:
                raise ValueError("accepted_review_fact_owner_changed")
        self.verify_sources()
        self.verify_observations()
        self.check_medical_owner()


def _observation_document_binding(owner):
    # Never fingerprint connection strings. Exact reader resources and run/root
    # identity suffice for immutable observation bundles; Control is not cached.
    return (
        id(owner.repo),
        id(owner.store),
        owner.run_id,
        digest(owner.root.model_dump(mode="json")),
        id(getattr(owner.repo, "_target", None)),
        getattr(owner.repo, "_role", None),
        _callable_identity(owner.repo.read_observation_bundles),
        _callable_identity(owner.store.read_bounded),
        _callable_identity(owner.read_many)[1],
        _callable_identity(owner.read_artifact)[1],
        _callable_identity(owner._check_root)[1],
        id(getattr(owner.contexts, "_target", owner.contexts)),
        _callable_identity(owner.contexts.read_research_run)[1],
    )


def historical_observation_documents(owner, result):
    """Only historical observation reads reuse; no Review, journal or write cache.

    The caller still validates the exact phase, acceptance, parent and image.
    Normal execution retains the original ordered single-document reads.
    """
    from packages.research_runtime.agent_records import RuntimeDocuments

    facts = _facts.get()
    if not (
        facts is not None
        and type(owner) is RuntimeDocuments
        and hasattr(owner.repo, "read_observation_bundles")
        and hasattr(owner.root, "model_dump")
    ):
        # Preserve prior failure order outside the private historical scope.
        phase = owner.read("dynamic-phase-result")
        if phase != (result,) or not result["accepted"]:
            return phase, None, None
        return (
            phase,
            owner.read("dynamic-phase-input"),
            owner.read("accepted:" + result["acceptance_key"]),
        )
    keys = ("dynamic-phase-result",)
    if result["accepted"]:
        keys += ("dynamic-phase-input", "accepted:" + result["acceptance_key"])
    values = facts.observation_documents(owner, keys)
    return (*values, None, None)[:3]


def _historical_medical_sources(factory):
    """Reuse the original public-source owner inside one historical read only.

    No patient facts or mutable campaigns are cached. Original first reads and
    final source/object/Runtime checks remain in historical_fact_read_scope.
    Outside that scope this is exactly the original constructor call.
    """
    facts = _facts.get()
    return facts.medical_sources(factory) if facts is not None else factory()


def historical_source(owner):
    """Only original historical read scopes may reuse; dispatch reads remain fresh."""
    facts = _facts.get()
    return facts.source(owner) if facts is not None else owner._read_current()


def current_fact(owner, proof):
    facts = _facts.get()
    return facts.current(owner, proof) if facts is not None else proof()


@contextmanager
def historical_fact_read_scope():
    """One completed-parent reconstruction; never wrap dispatch or persistence.

    Nested immutable-history readers share original component proofs. The
    outermost return repeats each proof without memoization, then re-reads the
    accepted Runtime/document bindings consumed by nested Review readers.
    """
    if _facts.get() is not None:
        yield
        return
    facts = _Facts()
    token = _facts.set(facts)
    try:
        yield
        _facts.reset(token)
        token = None
        facts.verify()
        for check in facts.return_checks:
            check()
    finally:
        if token is not None:
            _facts.reset(token)


class _Documents:
    def __init__(self, owner):
        self.owner, self.reads = owner, {}

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def read(self, key):
        value = self.owner.read(key)
        sha = digest(value)
        if self.reads.setdefault(key, sha) != sha:
            raise ValueError("accepted_review_reuse_document_changed")
        return value

    def _read_many(self, keys):
        from packages.research_runtime.agent_records import RuntimeDocuments

        keys = tuple(keys)
        if not 1 <= len(keys) <= 64 or len(set(keys)) != len(keys):
            raise ValueError("accepted_review_reuse_document_batch")
        if isinstance(self.owner, RuntimeDocuments) and hasattr(
            self.owner.repo, "read_observation_bundles"
        ):
            values = self.owner.read_many(keys)
        else:
            values = tuple(self.owner.read(key) for key in keys)
        if len(values) != len(keys):
            raise ValueError("accepted_review_reuse_document_batch")
        return keys, values

    def read_many(self, keys):
        # Retain the exact original Runtime artifact/scope reader. This wrapper
        # records only proof digests; it never substitutes cached document bytes.
        keys, values = self._read_many(keys)
        for key, value in zip(keys, values, strict=True):
            sha = digest(value)
            if self.reads.setdefault(key, sha) != sha:
                raise ValueError("accepted_review_reuse_document_changed")
        return values

    def verify(self):
        keys = tuple(self.reads)
        for start in range(0, len(keys), 64):
            selected, values = self._read_many(keys[start : start + 64])
            for key, value in zip(selected, values, strict=True):
                if digest(value) != self.reads[key]:
                    raise ValueError("accepted_review_reuse_document_changed")


def _binding(reader):
    return (
        id(reader.documents),
        id(reader.requests),
        id(reader.state_reader),
        id(reader.ancestor_reader),
        reader.acceptance_key,
        tuple(reader.lineage),
        digest(getattr(reader, "config", None)),
        digest(getattr(reader, "saved_proof", None)),
        digest(getattr(reader, "network_configs", None)),
        digest(getattr(reader, "capacity_configs", None)),
        digest(getattr(reader, "client_configs", None)),
    )


def _runtime(reader):
    docs, requests = reader.documents, reader.requests
    root = getattr(docs, "root", None)
    if root is not None and not hasattr(root, "model_dump"):
        root = vars(root)
    return digest(
        dict(
            run_id=docs.run_id,
            root=root,
            control=docs.sessions.read(docs.run_id),
            journal=requests.journal.read(),
            records=tuple(requests.records()),
            durable=requests.journal.durable,
        )
    )


class _Scope:
    def __init__(self):
        self.proofs, self.ancestors, self.active, self.entries = {}, {}, set(), []
        self.originals, self.locals = {}, []

    def original(self, reader):
        return self.originals.get(id(reader), reader)

    def ancestor(self, reader, run_id, lineage):
        reader = self.original(reader)
        key = id(reader), run_id, tuple(lineage)
        if key not in self.ancestors:
            self.ancestors[key] = reader.ancestor_reader(run_id, lineage)
        return self.ancestors[key]

    def current(self, reader, state):
        reader = self.original(reader)
        key = id(reader), digest(state)
        if key in self.active or len(self.entries) + len(self.active) >= 64:
            raise ValueError("accepted_review_reuse_cycle_or_depth")
        if key in self.proofs:
            return self.proofs[key].model_copy(deep=True)
        self.active.add(key)
        try:
            binding, runtime = _binding(reader), _runtime(reader)
            docs = _Documents(reader.documents)
            states = []

            def read_state():
                value = reader.state_reader()
                if states and digest(value) != states[0][0]:
                    raise ValueError("accepted_review_reuse_current_state_changed")
                if not states:
                    states.append((digest(value), value.material.scope))
                return value

            # A private copy, not temporary mutation of shared ports. All reads,
            # reference construction and source checks delegate to original owners.
            local = copy(reader)
            self.locals.append(local)
            local.documents, local.state_reader = docs, read_state
            # Preserve the original permission owner's journal/documents identity
            # check inside this private read operation. Never repair a mismatched
            # original binding: only mirror an already identical installed pair.
            if getattr(reader.requests.journal, "documents", None) is reader.documents:
                local.requests = copy(reader.requests)
                local.requests.journal = copy(reader.requests.journal)
                local.requests.journal.documents = docs
            self.originals[id(local)] = reader
            result = local._current(state)
            self.entries.append((reader, binding, runtime, docs, states[0]))
            self.proofs[key] = result.model_copy(deep=True)
            return result
        finally:
            self.active.remove(key)

    def verify(self):
        for reader, binding, runtime, docs, (state_sha, scope) in self.entries:
            if _binding(reader) != binding:
                raise ValueError("accepted_review_reuse_owner_changed")
            if digest(reader.state_reader()) != state_sha:
                raise ValueError("accepted_review_reuse_current_state_changed")
            reader.documents.validate_scope(scope)
            if not reader.documents.consumed(scope.graph):
                raise ValueError("accepted_review_reuse_receipt_binding")
            docs.verify()
            if _runtime(reader) != runtime:
                raise ValueError("accepted_review_reuse_runtime_changed")

    def verify_durable(self):
        # Fact return checks may read external original owners. Check Runtime and
        # all consumed document bytes once more afterwards; no external dispatch.
        for reader, binding, runtime, docs, _ in self.entries:
            if _binding(reader) != binding:
                raise ValueError("accepted_review_reuse_owner_changed")
            docs.verify()
            if _runtime(reader) != runtime:
                raise ValueError("accepted_review_reuse_runtime_changed")


def current(reader, state):
    scope = _scope.get()
    if scope is not None:
        return scope.current(reader, state)
    scope = _Scope()
    token = _scope.set(scope)
    facts = _facts.get()
    fact_token = None
    if facts is None:
        facts = _Facts()
        fact_token = _facts.set(facts)
    try:
        result = scope.current(reader, state)
        # End the reuse scope before fresh final reads, even if an owner itself
        # performs nested proofs. Nothing survives this public current() call.
        _scope.reset(token)
        token = None
        scope.verify()
        if fact_token is not None:
            _facts.reset(fact_token)
            fact_token = None
            facts.verify()
            if facts.entries or facts.sources or facts.observations:
                scope.verify_durable()
        elif facts.entries or facts.sources or facts.observations:
            facts.return_checks.append(scope.verify_durable)
        return result
    finally:
        if token is not None:
            _scope.reset(token)
        if fact_token is not None:
            _facts.reset(fact_token)


def ancestor(reader, run_id, lineage):
    scope = _scope.get()
    return (
        scope.ancestor(reader, run_id, lineage)
        if scope is not None
        else reader.ancestor_reader(run_id, lineage)
    )


def has_return_recheck():
    return _scope.get() is not None
