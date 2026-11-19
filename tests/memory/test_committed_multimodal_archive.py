"""Narrow archive envelope tests; real original-owner readback is separate evidence."""

import copy
from types import SimpleNamespace

import pytest

from agents.expert_agent import committed_review
from packages.grounding.models import digest
from packages.memory.multimodal_archive import _committed_binding
from packages.multimodal_research.finalization import _saved_export
from packages.multimodal_research.report_storage import pack


@pytest.fixture
def fixture(monkeypatch):
    phase = dict(run_id="RUN:ONE", target=None, accepted=False, error="deadline")
    old = dict(phases=[phase], task={"target": "GOAL:ONE"}, new_requests=3)
    parent = dict(config_checksum=digest("config"))
    proof = dict(
        schema_version="committed-review-readback/1.0.0",
        run_id="RUN:ONE",
        event=dict(role="event", code="EVENT:ONE", version="1.0.0", checksum=digest("event")),
        acceptance_key="EVENT:ONE:domain",
        **{
            k: digest(k)
            for k in (
                "parent_checksum",
                "accepted_checksum",
                "report_checksum",
                "intent_checksum",
                "journal_checksum",
            )
        },
    )
    result = dict(
        old,
        phases=[dict(phase, accepted=True, acceptance_key=proof["acceptance_key"])],
        original_result_checksum=digest(old),
        original_requests=3,
        new_requests=0,
        execution_kind="committed_review_export_not_reacceptance",
        finalization_grant_checksum=None,
        local_review_closure=dict(
            original_phase=phase,
            original_phase_checksum=digest(phase),
            confirmation=proof,
        ),
    )
    calls = []

    def original_owner(documents, *, parent, config_checksum):
        assert config_checksum == parent["config_checksum"]
        assert documents.run_id == "RUN:ONE"
        assert not hasattr(documents, "write") and not hasattr(documents, "session")
        assert documents.read("absent") is None
        calls.append("shared-original-owner")
        return proof

    monkeypatch.setattr(committed_review, "read_committed_review_binding", original_owner)

    def read(name, **kwargs):
        if name == "multimodal-case-result":
            return [old]
        if name == "submission:" + proof["acceptance_key"]:
            return None
        assert name == "absent" and kwargs == {"optional": True}
        return None

    def check(value):
        return _committed_binding(
            SimpleNamespace(),
            read,
            {},
            {"root_checksum": digest("root")},
            parent,
            value,
            "CASE:ONE",
            "RUN:ONE",
        )

    return SimpleNamespace(result=result, check=check, calls=calls)


def test_commit_archive_uses_shared_owner_without_new_grant(fixture):
    fixture.check(fixture.result)
    assert fixture.calls == ["shared-original-owner"]


@pytest.mark.parametrize("fault", ["old_hash", "requests", "grant", "key", "task", "proof"])
def test_commit_archive_rejects_changed_bindings(fixture, fault):
    value = copy.deepcopy(fixture.result)
    if fault == "old_hash":
        value["original_result_checksum"] = digest("changed")
    elif fault == "requests":
        value["new_requests"] = 1
    elif fault == "grant":
        value["finalization_grant_checksum"] = digest("invented-grant")
    elif fault == "key":
        value["phases"][-1]["acceptance_key"] = "wrong:domain"
    elif fault == "task":
        value["task"] = {"target": "GOAL:OTHER"}
    else:
        value["local_review_closure"]["confirmation"]["accepted_checksum"] = digest("wrong")
    with pytest.raises(ValueError, match="memory_committed_"):
        fixture.check(value)


def test_saved_export_keeps_exact_html_and_rejects_different_review(tmp_path):
    result = dict(complete=True, new_requests=0)
    review = dict(scores={"root": "original-not-recomputed"})
    report = dict(result=result, original_review=review, scores=review["scores"])
    values = pack(report, "<!doctype html><p>exact saved page</p>", key="report")
    kw = dict(key="report", result=result, review=review, name="closure", directory=tmp_path)
    first = _saved_export(values, **kw)
    page = tmp_path / "Multimodal_closure.html"
    before = page.stat()
    assert _saved_export(values, **kw) == first
    assert page.stat().st_ino == before.st_ino and page.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError, match="saved_report_binding"):
        _saved_export(values, **dict(kw, review={"scores": "changed"}))


def test_archive_routes_installed_tool_outputs_and_visible_objects(tmp_path):
    from packages.multimodal_research.component_observations import ComponentObservations
    from packages.multimodal_research.memory import archive_reader
    from packages.object_store import LocalObjectStore, checksum
    from packages.research_runtime.deployment import CombinedResearchObjects

    visible, output = [LocalObjectStore(tmp_path / name) for name in ("visible", "output")]
    component = object.__new__(ComponentObservations)
    component.owner = SimpleNamespace(
        research=SimpleNamespace(
            compiler=SimpleNamespace(
                admission=SimpleNamespace(objects=CombinedResearchObjects(visible, output)),
            )
        )
    )
    campaign = SimpleNamespace(base=SimpleNamespace(component_observations=component))
    read = archive_reader(campaign, SimpleNamespace())
    for store, content in ((visible, b"original report"), (output, b"confirmed tool output")):
        uri = store.promote(store.stage(content), checksum(content))
        assert read(SimpleNamespace(uri=uri)) == content
    with pytest.raises(ValueError, match="invalid managed"):
        read(SimpleNamespace(uri="file:///etc/passwd"))


def test_archive_uninstalled_component_keeps_original_reader():
    from packages.multimodal_research.memory import archive_reader

    documents = SimpleNamespace(read_artifact=object())
    campaign = SimpleNamespace(base=SimpleNamespace())
    assert archive_reader(campaign, documents) is documents.read_artifact
    campaign.base.component_observations = SimpleNamespace()
    with pytest.raises(ValueError, match="component_owner"):
        archive_reader(campaign, documents)


def test_partition_archive_rebuilds_original_accepted_ancestor(monkeypatch):
    import json

    from db.runtime.repository import planner
    from packages.memory.multimodal_archive import _archive_ancestor_reader
    from packages.research_runtime.agent_records import RuntimeDocuments
    from tests.evaluation_preparation.test_partition_increment import partition_fixture

    f = partition_fixture()
    history = f.submission.historical
    old, state = f.old, f.old.owner.current()
    before = digest(old.documents.saved), digest(old.budget.journal.read())
    monkeypatch.setattr(planner, "PlannerRepository", lambda _: old.documents.sessions)
    monkeypatch.setattr(
        RuntimeDocuments, "reference", lambda self, *a, **kw: old.documents.reference(*a, **kw)
    )
    root = digest("archive-test-root")

    def read(name, *, source_run, optional):
        assert source_run == history.run_id and optional
        if name == "dynamic-phase-input":
            return [state.model_dump(mode="json")]
        if name.startswith("journal:"):
            index = int(name.split(":")[1])
            events = old.budget.journal.read()
            return [events[index]] if index < len(events) else None
        return old.documents.read(name)

    def artifact(case, code):
        assert case == state.material.scope.case_code
        if code == state.material.scope.graph.code:
            ref = state.material.scope.graph
            raw = dict(
                run_id=history.run_id, root_checksum=root, value={"schema_version": ref.version}
            )
        else:
            ref = next(r for r in state.resources if r.code == code)
            raw = {}
        return SimpleNamespace(asset_code=code, checksum=ref.checksum), json.dumps(raw).encode()

    owner = SimpleNamespace(runtime=SimpleNamespace(_target=object()), artifact=artifact)
    reader = _archive_ancestor_reader(
        owner, read, {}, {"root_checksum": root}, state.material.scope.case_code, history
    )
    assert reader.current(f.owner.current()) == history
    assert before == (digest(old.documents.saved), digest(old.budget.journal.read()))
    assert not hasattr(reader.documents, "write") and not hasattr(reader.documents, "session")
    with pytest.raises(ValueError, match="ancestor_version_or_cycle"):
        _archive_ancestor_reader(
            owner,
            read,
            {},
            {"root_checksum": root},
            state.material.scope.case_code,
            history,
            (history.run_id,),
        )
