"""Original acceptance chain: scoped proof reuse, fresh return boundary, no cache authority."""

from copy import deepcopy

import pytest

from agents.provider_agents.batched_review import run_batches
from packages.evidence_details import accepted_read_scope as scoped
from packages.evidence_details.accepted_review_reuse import AcceptedReviewReader
from packages.grounding.models import digest
from tests.agent_provider.test_accepted_review_reuse import (
    accepted_fixture,
    accepted_successor,
    state_from,
)


def chain(monkeypatch, after=None):
    f = accepted_fixture(new_observation=True)
    sub = run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key="new", round_id="A"
    )
    f.acceptance.submit(sub, key="next")
    assert f.acceptance.accept(key="next").status == "accepted"
    g = accepted_successor(f)
    reader = g.owner.confirmed_review_reader
    original = AcceptedReviewReader._validated_requests
    actual = AcceptedReviewReader._current
    calls, resolutions = [], []
    resolve = reader.ancestor_reader

    def ancestor(run, lineage):
        resolutions.append(run)
        return resolve(run, lineage)

    reader.ancestor_reader = ancestor

    def count(self, state):
        calls.append(self.documents.run_id)
        return actual(self, state)

    def check(self, old, submission, receipt, confirmed):
        result = original(self, old, submission, receipt, confirmed)
        if self.documents.run_id == reader.documents.run_id:
            # Recorded proof reconstruction and report checking revisit the same
            # installed ancestor. Each visit still uses the original public API.
            a = scoped.ancestor(
                self, submission.historical.run_id, (*self.lineage, self.documents.run_id)
            )
            first = a.current(old)
            assert a.current(old) == first
            object.__setattr__(first.submission.candidate, "round", "B")
            assert a.current(old).submission.candidate.round == "A"
            if after:
                after(f, reader)
        return result

    monkeypatch.setattr(AcceptedReviewReader, "_validated_requests", check)
    monkeypatch.setattr(AcceptedReviewReader, "_current", count)
    return f, g, reader, calls, resolutions


def test_recursive_proof_once_per_owner_and_target_but_fresh_on_next_operation(monkeypatch):
    f, g, reader, calls, resolutions = chain(monkeypatch)
    before = deepcopy(f.documents.saved), deepcopy(f.old.documents.saved), len(g.budget.records())
    result = reader.current(g.owner.current())
    assert len(calls) == 2 and len(resolutions) == 1
    assert reader.current(g.owner.current()) == result
    assert len(calls) == 4 and len(resolutions) == 2
    assert scoped._scope.get() is None
    assert before == (f.documents.saved, f.old.documents.saved, len(g.budget.records()))


@pytest.mark.parametrize(
    "fault", ["source", "control", "journal", "accepted", "response", "owner", "throw"]
)
def test_mutations_after_reused_proof_reject_at_return_and_clear_scope(monkeypatch, fault):
    def change(f, reader):
        if fault == "source":
            raw = f.old.owner.state.model_dump(mode="json")
            raw["material"]["sources"][0]["text"] += " changed after proof"
            f.old.owner.state = state_from(raw)
        elif fault == "control":
            current = f.old.documents.sessions.read(f.old.documents.run_id)
            f.old.documents.sessions.read = lambda _: current.model_copy(update={"phase": "select"})
        elif fault == "journal":
            f.old.budget.journal.append(dict(phase="reserved", request_id="unknown"))
        elif fault == "accepted":
            f.old.documents.saved["accepted:old-accepted"][3]["records_checksum"] = digest(
                "changed"
            )
        elif fault == "response":
            key = next(k for k in f.old.documents.saved if k.endswith(":response"))
            f.old.documents.saved[key][0]["raw_model_text"] = "changed after proof"
        elif fault == "owner":
            reader.acceptance_key = "uninstalled"
        else:
            raise OSError("injected_after_proof")

    f, g, reader, _, _ = chain(monkeypatch, change)
    error = OSError if fault == "throw" else ValueError
    with pytest.raises(error, match="source_withdrawn|accepted_review_reuse_|injected_after_proof"):
        reader.current(g.owner.current())
    assert scoped._scope.get() is None
    assert not g.budget.records()


def test_exception_does_not_keep_proofs_for_next_current_call(monkeypatch):
    failures = [True]

    def fail_once(f, reader):
        if failures.pop() if failures else False:
            raise OSError("injected_once")

    _, g, reader, calls, _ = chain(monkeypatch, fail_once)
    with pytest.raises(OSError, match="injected_once"):
        reader.current(g.owner.current())
    assert reader.current(g.owner.current()).acceptance_key == "next"
    assert len(calls) == 4 and scoped._scope.get() is None


def test_changed_target_does_not_reuse_previous_semantic_binding(monkeypatch):
    _, g, reader, _, _ = chain(monkeypatch)
    state = g.owner.current()
    first = reader.current(state)
    # Different target policy is not allowed to borrow a prior proof.
    policy = state.policy.model_copy(update={"items": ()})
    with pytest.raises(ValueError):
        reader.current(state.model_copy(update={"policy": policy}))
    assert reader.current(state) == first


def component_chain(tmp_path, monkeypatch, after=None):
    from tests.evaluation_preparation.test_component_observations import installed_fixture

    component = installed_fixture(tmp_path)
    calls = []
    read = component.read

    def counted():
        calls.append(True)
        return read()

    component.read = counted
    f, g, reader, _, _ = chain(monkeypatch, after)

    def install(owner):
        state = owner.state_reader

        def current_state():
            component.check_fact()
            component.check_fact()
            return state()

        owner.state_reader = current_state
        return owner

    resolve = reader.ancestor_reader
    reader.ancestor_reader = lambda run, lineage: install(resolve(run, lineage))
    install(reader)
    return component, calls, g, reader


def test_component_fact_is_checked_initially_and_at_return_not_per_ancestor(tmp_path, monkeypatch):
    _, calls, g, reader = component_chain(tmp_path, monkeypatch)
    reader.current(g.owner.current())
    assert len(calls) == 2
    reader.current(g.owner.current())
    assert len(calls) == 4


def test_historical_reconstruction_scope_checks_fact_again_after_nested_readers(
    tmp_path, monkeypatch
):
    component, calls, g, reader = component_chain(tmp_path, monkeypatch)
    with scoped.historical_fact_read_scope():
        component.check_fact()
        with scoped.historical_fact_read_scope():
            first = reader.current(g.owner.current())
            assert reader.current(g.owner.current()) == first
        assert len(calls) == 1
    assert len(calls) == 2
    assert scoped._facts.get() is None and scoped._scope.get() is None
    component.check_fact()
    assert len(calls) == 3  # A later consumer must prove current eligibility again.


@pytest.mark.parametrize("fail", [False, True])
def test_completed_parent_installs_bounded_fact_scope(tmp_path, monkeypatch, fail):
    import json

    from packages.multimodal_research import campaign_renewal

    component, calls, g, reader = component_chain(tmp_path, monkeypatch)
    (tmp_path / "protocol.json").write_text(json.dumps({"schema_version": "original-test"}))

    def saved(path):
        assert path == tmp_path
        component.check_fact()
        result = reader.current(g.owner.current())
        component.check_fact()
        assert len(calls) == 1
        if fail:
            raise OSError("historical_read_failed")
        return result

    monkeypatch.setattr(campaign_renewal, "saved_campaign", saved)
    if fail:
        with pytest.raises(OSError, match="historical_read_failed"):
            campaign_renewal.completed_parent(tmp_path)
    else:
        assert campaign_renewal.completed_parent(tmp_path)
        assert len(calls) == 2
    assert scoped._facts.get() is None and scoped._scope.get() is None
    before = len(calls)
    component.check_fact()
    assert len(calls) == before + 1
    assert not g.budget.records()


@pytest.mark.parametrize("fault", ["fact", "runtime", "exception"])
def test_historical_scope_final_check_rejects_changes_and_never_leaks(tmp_path, monkeypatch, fault):
    component, calls, g, reader = component_chain(tmp_path, monkeypatch)
    with pytest.raises(OSError if fault == "exception" else ValueError):
        with scoped.historical_fact_read_scope():
            reader.current(g.owner.current())
            if fault == "fact":
                component.value["values"]["vertical_cdr"] = 0.9
            elif fault == "runtime":
                current = reader.documents.sessions.read(reader.documents.run_id)
                reader.documents.sessions.read = lambda _: current.model_copy(
                    update={"phase": "select"}
                )
            else:
                raise OSError("injected_historical_reconstruction_failure")
    assert scoped._facts.get() is None and scoped._scope.get() is None
    assert not g.budget.records()


@pytest.mark.parametrize("fault", ["bytes", "projection", "method", "exception"])
def test_component_fact_reuse_revalidates_on_return_and_clears_context(
    tmp_path, monkeypatch, fault
):
    holder = {}

    def mutate(f, reader):
        component = holder["component"]
        if fault == "bytes":
            holder["changed"] = True
        elif fault == "projection":
            component.value["values"]["vertical_cdr"] = 0.9
        elif fault == "method":
            component.read = lambda: None
        else:
            raise OSError("component_proof_interrupted")

    component, _, g, reader = component_chain(tmp_path, monkeypatch, mutate)
    holder["component"] = component
    read = component.read

    def changed_read():
        values = read()
        if holder.get("changed"):
            values[0]["values"]["vertical_cdr"] = 0.7
        return values

    component.read = changed_read
    with pytest.raises(OSError if fault == "exception" else ValueError):
        reader.current(g.owner.current())
    assert scoped._scope.get() is None
    assert scoped._facts.get() is None
    assert not g.budget.records()


def test_component_final_owner_return_cannot_hide_runtime_change(tmp_path, monkeypatch):
    component, calls, g, reader = component_chain(tmp_path, monkeypatch)
    read = component.read

    def revoke_after_fresh_proof():
        result = read()
        if len(calls) == 2:
            current = reader.documents.sessions.read(reader.documents.run_id)
            reader.documents.sessions.read = lambda _: current.model_copy(
                update={"phase": "select"}
            )
        return result

    component.read = revoke_after_fresh_proof
    with pytest.raises(ValueError, match="runtime_changed"):
        reader.current(g.owner.current())
    assert scoped._facts.get() is None and scoped._scope.get() is None
    assert not g.budget.records()


@pytest.mark.parametrize("wrong_owner", [False, True])
def test_recorded_reasoned_owner_identity_is_preserved_not_repaired(wrong_owner):
    from packages.evidence_details.accepted_review_reuse import RecordedAcceptedReviewReader
    from tests.agent_provider.support import Documents
    from tests.agent_provider.test_recorded_acceptance_reuse import closed, read

    f, permission = closed(reasoned=True)
    proof = read(f)
    reader = RecordedAcceptedReviewReader(
        documents=f.docs,
        requests=f.requests,
        state_reader=f.owner.current,
        acceptance_key=permission.acceptance_key,
        config=f.config,
        saved_proof=proof,
    )
    raw = f.state.model_dump(mode="json")
    raw["material"]["scope"]["run_id"] = "RUN:SCOPED_RECORDED_SUCCESSOR"
    state = state_from(raw)
    if wrong_owner:
        f.requests.journal.documents = Documents()
        with pytest.raises(ValueError, match="recorded_finalization_owner"):
            reader.current(state)
    else:
        result = reader.current(state)
        assert result.acceptance_checksum == proof["acceptance_checksum"]
        assert reader.current(state) == result
        assert f.requests.journal.documents is f.docs
    assert len(f.wire.calls) == 3 and scoped._scope.get() is None


@pytest.mark.parametrize("fault", [None, "input", "owner", "exception"])
def test_completed_history_reuses_input_only_with_fresh_return_check(tmp_path, monkeypatch, fault):
    import json

    from packages.dataset_adapters import current_read_scope as inputs
    from packages.multimodal_research import campaign_renewal
    from tests.dataset_adapters.test_current_read_scope import Manifest
    from tests.dataset_adapters.test_current_read_scope import reader as input_reader

    owner, state = input_reader(monkeypatch)
    (tmp_path / "protocol.json").write_text(json.dumps({"schema_version": "original-test"}))

    def saved(path):
        assert path == tmp_path
        for _ in range(12):
            assert owner.read_visible("CASE:ONE").version == 1
        assert state.calls == 1
        if fault == "input":
            state.value = Manifest(version=2, details={"input": "changed"})
        elif fault == "owner":
            owner.repository = object()
        elif fault == "exception":
            raise OSError("original_history_read_failure")
        return "saved-only"

    monkeypatch.setattr(campaign_renewal, "saved_campaign", saved)
    if fault:
        error, reason = (
            (OSError, "original_history_read_failure")
            if fault == "exception"
            else (ValueError, "projection_input_")
        )
        with pytest.raises(error, match=reason):
            campaign_renewal.completed_parent(tmp_path)
    else:
        assert campaign_renewal.completed_parent(tmp_path) == "saved-only"
        assert state.calls == 2
    assert inputs._scope.get() is None and scoped._facts.get() is None
    before = state.calls
    owner.read_visible("CASE:ONE")
    assert state.calls == before + 1


def test_history_input_scope_ends_before_fact_and_durable_return_checks(tmp_path, monkeypatch):
    import json

    from packages.dataset_adapters import current_read_scope as inputs
    from packages.multimodal_research import campaign_renewal
    from tests.dataset_adapters.test_current_read_scope import reader as input_reader

    owner, state = input_reader(monkeypatch)
    (tmp_path / "protocol.json").write_text(json.dumps({"schema_version": "original-test"}))
    checkpoints = []

    def saved(path):
        owner.read_visible("CASE:ONE")
        owner.read_visible("CASE:ONE")
        assert state.calls == 1
        scoped._facts.get().return_checks.append(durable)
        return "saved-only"

    def verify(facts):
        assert inputs._scope.get() is None
        assert state.calls == 2
        owner.read_visible("CASE:ONE")
        checkpoints.append("fresh_fact")

    def durable():
        assert inputs._scope.get() is None and state.calls == 3
        checkpoints.append("durable_after_fresh_input")

    monkeypatch.setattr(campaign_renewal, "saved_campaign", saved)
    monkeypatch.setattr(scoped._Facts, "verify", verify)
    assert campaign_renewal.completed_parent(tmp_path) == "saved-only"
    assert checkpoints == ["fresh_fact", "durable_after_fresh_input"]
