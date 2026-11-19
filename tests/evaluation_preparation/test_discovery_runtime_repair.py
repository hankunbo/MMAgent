"""Public lookup timers do not repeatedly run expensive owner checks."""

import json
from types import SimpleNamespace

import pytest

from packages.evidence_reuse.definition_sources import LiteratureDefinitionConnector
from packages.llm_provider.provider import ProviderError
from packages.multimodal_research.definition_expansion import read_expansion_policy
from tests.evaluation_preparation.test_definition_expansion import fixture


def resilient(e):
    raw = e.policy.model_dump(mode="json")
    raw.update(schema_version="campaign-definition-expansion/1.2.0", require_target_relation=True)
    e.policy = read_expansion_policy(raw)
    return e


def test_timer_poll_is_constant_local_work_but_send_checks_owner(tmp_path):
    e = fixture(tmp_path)
    calls = []
    old = e.guard

    def check():
        calls.append("owner")
        old()

    e.guard = check
    g = LiteratureDefinitionConnector(None, e, {}).governor()
    for _ in range(200):
        assert g.remaining() > 0
    assert calls == [] and e.operations() == []
    g.before_send("SOURCE:PUBLIC")
    assert calls == ["owner"] and len(e.operations()) == 1
    (tmp_path / "cancel").touch()
    from packages.knowledge_search.policy import SearchStopped

    with pytest.raises(SearchStopped, match="cancelled"):
        g.remaining()


def test_current_db_cancel_before_wire_refuses_reservation(tmp_path):
    e = fixture(tmp_path)
    e.campaign.sessions.read = lambda _: SimpleNamespace(cancelled=True, phase="stopped")
    g = LiteratureDefinitionConnector(None, e, {}).governor()
    assert g.remaining() > 0  # A clock poll does not grant dispatch authority.
    from packages.evidence_reuse.definition_sources import DefinitionAdmissionStopped

    with pytest.raises(DefinitionAdmissionStopped, match="cancelled") as rejected:
        g.before_send("SOURCE:PUBLIC")
    assert rejected.value.original.code == "cancelled"
    assert e.operations() == []


def test_search_failure_is_versioned_technical_backlog_not_empty(tmp_path, monkeypatch):
    from apps import knowledge_literature_tools

    e = resilient(fixture(tmp_path))
    calls = []

    def search(*a, **kw):
        calls.append(kw)
        return dict(
            status="timeout",
            runs=[dict(reference="original-owner", outcome=dict(stop_reason="timeout"))],
            results=[],
        )

    monkeypatch.setattr(knowledge_literature_tools, "search", search)
    result = LiteratureDefinitionConnector(None, e, {"public": "rnfl"}).search(
        query="public", key="one", limit=1
    )
    assert result["status"] == "technical_backlog" and not result["retried"]
    assert result["result"]["runs"][0]["reference"] == "original-owner"
    assert calls[0]["timeout_ms"] == 120000 and len(calls) == 1


def test_previous_failed_query_not_replayed_new_query_continues(tmp_path):
    e = resilient(fixture(tmp_path))
    raw = e.policy.model_dump(mode="json")
    raw["public_queries"] = ["public old", "public new"]
    raw["query_requirements"] = {q: ["missing-check"] for q in raw["public_queries"]}
    e.policy = read_expansion_policy(raw)
    e.campaign.discovery_continuation = SimpleNamespace(
        failed_queries=("public old",), reference={"requests": 1}
    )
    calls = []
    e.sources.search = lambda **kw: calls.append(kw["query"]) or dict(materials=[], backlog=[])
    e.advance()
    assert calls == ["public new"]
    revision = e.documents.read("definition:revision")[0]
    assert revision["rejected"][0]["reason"] == "previous_search_technical_failure_not_replayed"
    assert revision["denominator_before"] == revision["denominator_after"]
    e.advance()
    assert calls == ["public new"]


def test_profile4_does_not_change_medical_inventory():
    from packages.multimodal_research.active_requirements import load

    a = load("configs/research/multimodal_active_requirements_v3.json")
    b = load("configs/research/multimodal_active_requirements_v4.json")
    for key in ("checks", "nodes", "goals"):
        assert a[key] == b[key]
    assert b["definition_expansion"]["search_timeout_ms"] == 120000


def test_parent_identity_requires_stop_and_retains_deadline(tmp_path):
    from packages.multimodal_research.discovery_continuation import load_parent

    spec = dict(
        schema_version="multimodal-live-entry/1.11.0",
        input_checksum="x",
        intent={"checksum": "target"},
        implementation={},
    )
    protocol = dict(
        input="x", task="target", implementation={}, previous_attempts=[], deadline="fixed"
    )
    (tmp_path / "entry.json").write_text(json.dumps(spec))
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "lock").touch()
    with pytest.raises(ValueError, match="parent_binding"):
        load_parent(tmp_path, checksum="x", intent=spec["intent"])
    (tmp_path / "cancel").touch()
    assert load_parent(tmp_path, checksum="x", intent=spec["intent"])[1]["deadline"] == "fixed"
    with pytest.raises(ValueError, match="parent_binding"):
        load_parent(tmp_path, checksum="other", intent=spec["intent"])


def test_expansion_preserves_review_reserve_including_parent_requests(tmp_path):
    e = resilient(fixture(tmp_path))
    e.campaign.previous_attempts = lambda: [dict(requests=28)]
    with pytest.raises(ProviderError, match="definition_review_reserve"):
        e.guard()
    assert not e.operations()


def test_expired_revision_can_be_reported_but_never_dispatches(tmp_path):
    from datetime import UTC, datetime, timedelta

    e = fixture(tmp_path)
    revision = e.advance()
    before = (e.sources.calls, e.budget.used, list(e.operations()))
    e.campaign.protocol["deadline"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    assert e.read_revision() == revision
    with pytest.raises(ProviderError, match="deadline"):
        e.current()
    with pytest.raises(ProviderError, match="deadline"):
        e.advance()
    assert before == (e.sources.calls, e.budget.used, e.operations())
    e.sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="source_changed"):
        e.read_revision()


def test_restore_report_keeps_source_checks_separate_from_dispatch(tmp_path):
    from packages.multimodal_research.campaign import MultimodalCampaign

    e = fixture(tmp_path)
    revision = e.advance()
    c = e.campaign
    c.definition_expansion = e
    (tmp_path / "cancel").touch()
    assert MultimodalCampaign.expand_definitions(c, restore_only=True, report_only=True) == revision
    assert c.base.definition_current() == revision
    with pytest.raises(ProviderError, match="cancelled"):
        MultimodalCampaign.expand_definitions(c, restore_only=True)
    with pytest.raises(ValueError, match="report_requires_restore_only"):
        MultimodalCampaign.expand_definitions(c, report_only=True)


def test_fresh_profile4_installs_exact_status_prompt_with_explicit_entry_version(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from packages.multimodal_research import entry
    from tests.dataset_adapters.test_multimodal import gamma

    def views(command, **kwargs):
        out = Path(command[4])
        out.mkdir(parents=True)
        (out / "views.json").write_text(json.dumps(dict(views=[])))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", views)
    out = tmp_path / "new-entry"
    entry.prepare_entry(
        dataset="gamma",
        root=gamma(tmp_path / "source"),
        sample="0001",
        task="advanced_glaucoma",
        output=out,
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition="configs/research/multimodal_active_requirements_v4.json",
        resilient_vision=True,
        view_profile="detail-v1",
    )
    spec = json.loads((out / "entry.json").read_bytes())
    assert spec["schema_version"] == "multimodal-live-entry/1.13.0"
    role = next(r for r in spec["provider"]["roles"] if r["role"] == "review")
    assert role["prompt_version"] == "review-direction-2.1.0"

    def boundary():
        raise RuntimeError("offline_boundary_no_private_owner")

    monkeypatch.setattr(entry, "owners", boundary)
    with pytest.raises(RuntimeError, match="offline_boundary_no_private_owner"):
        entry.assemble_entry(out)
    spec["schema_version"] = "multimodal-live-entry/1.12.0"
    (out / "entry.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="implementation_changed"):
        entry.assemble_entry(out)
