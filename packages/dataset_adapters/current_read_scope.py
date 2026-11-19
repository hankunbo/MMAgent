"""Private single-projection input proof reuse, never a dispatch authority cache.

Only synchronous read-only Phase projection/verification and completed-history
reconstruction install this scope. Historical fact/Runtime return checks run
after this input scope has ended, retaining the original fresh final boundary.
Each input is fully read by its original owner at first use AND before returning.
Source/Tool/Control/permission checks are not memoized. No wire or write may run
inside this private scope; all external dispatch retains its separate guards.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

from packages.grounding.models import digest

_scope = ContextVar("current_projection_visible_inputs", default=None)
_history_owners = ContextVar("current_projection_history_owners", default=None)


def _binding(owner):
    return id(owner.repository), id(owner.visible_store)


def _historical_input_owners(factory):
    """One original connection/store/input service per read-only reconstruction.

    Shared ancestors must not create a new input owner on every DAG edge. This
    retains owner objects only, never a campaign, result, source qualification
    or dispatch permission. The normal input scope still checks each actual
    input/kind at first use and again before returning. Outside that private
    scope, the original factory always runs.
    """
    state = _history_owners.get()
    if state is None:
        return factory()
    if not state:
        owners = factory()
        if not isinstance(owners, tuple) or len(owners) != 4:
            raise ValueError("projection_history_owner_factory_shape")
        state.update(factory=factory, owners=owners, binding=_binding(owners[2]))
    if state["factory"] is not factory or _binding(state["owners"][2]) != state["binding"]:
        raise ValueError("projection_history_owner_changed")
    return state["owners"]


def _read_visible(owner, case_code, read, *, kind="visible"):
    if kind not in {"visible", "components"}:
        raise ValueError("projection_input_read_kind")
    entries = _scope.get()
    if entries is None:
        return read()
    # The full multimodal pair and the visible manifest have different shapes;
    # neither may satisfy the other's proof even for the same owner and case.
    key = id(owner), case_code, kind
    if key not in entries:
        if len(entries) >= 16:
            raise ValueError("projection_input_read_bound")
        binding = _binding(owner)
        value = read()
        if _binding(owner) != binding:
            raise ValueError("projection_input_owner_changed")
        entries[key] = owner, binding, read, deepcopy(value), digest(value)
    saved_owner, binding, _, value, _ = entries[key]
    if saved_owner is not owner or _binding(owner) != binding:
        raise ValueError("projection_input_owner_changed")
    return deepcopy(value)


@contextmanager
def _projection_visible_reads():
    """Not a public execution context: never wrap network/model/storage writes."""
    if _scope.get() is not None:
        yield
        return
    entries = {}
    token = _scope.set(entries)
    owners_token = _history_owners.set({})
    try:
        yield
    finally:
        _scope.reset(token)
        _history_owners.reset(owners_token)
    # Exceptions do not return a usable proof; do not obscure them with a new
    # read failure. Successful return requires fresh complete input validation.
    for owner, binding, read, _, expected in entries.values():
        if _binding(owner) != binding:
            raise ValueError("projection_input_owner_changed")
        current = read()
        if _binding(owner) != binding or digest(current) != expected:
            raise ValueError("projection_input_changed_before_return")


@contextmanager
def _independent_projection_visible_reads():
    """One historical method's inputs, independent of the current case projection.

    A bounded Memory search may visit many different cases. Each method proof
    must complete its own first/last owner reads before returning; it must not
    accumulate those cases in the caller's 16-entry patient projection. This
    scope confers no dispatch authority and never encloses network or writes.
    The caller's scope (including its final fresh checks) is restored on error.
    """
    token = _scope.set(None)
    try:
        with _projection_visible_reads():
            yield
    finally:
        _scope.reset(token)
