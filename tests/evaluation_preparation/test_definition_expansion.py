"""New source -> actual definition owner -> revision -> original target scheduler."""

import copy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.llm_provider import DeterministicFakeProvider, MemoryJournal, RequestBudget
from packages.llm_provider.provider import ProviderError
from packages.multimodal_research.campaign import MultimodalCampaign, configuration
from packages.multimodal_research.definition_expansion import DefinitionExpansion, definition_config
from tests.agent_provider.support import Documents
from tests.evaluation_preparation.test_directional_review import fixture as review_fixture


def test_public_fetch_journal_keeps_binary_body_and_detects_drift(tmp_path):
    import json

    from packages.evidence_reuse.definition_sources import restore_fetch, save_fetch
    from packages.grounding.models import canonical
    from packages.object_store import LocalObjectStore
    from tests.database.test_knowledge_archive_04b3 import fetch
    from tests.knowledge_search.support import config

    c = config()
    hit = SimpleNamespace(
        discovery=SimpleNamespace(
            **{k: getattr(c, k) for k in ("provider_code", "provider_version", "adapter")},
            config_checksum=digest("synthetic-public-fetch"),
            provider_record_id="12345",
        ),
        discovered_at=datetime.now(UTC),
    )
    raw = fetch(hit, body=b"%PDF-synthetic\x00\xff\xfe")
    store = LocalObjectStore(tmp_path / "objects")
    saved = json.loads(canonical(save_fetch(raw, store)))
    assert "body" not in saved["payload"]["document"]["response"]
    restored = restore_fetch(saved, store)
    assert restored.document.response.body == raw.document.response.body
    assert restored.model_dump(mode="json") == raw.model_dump(mode="json")
    # Restore a wrong, but well-formed original managed object: checksum gate rejects it.
    from packages.object_store import checksum

    wrong = b"wrong public material"
    saved["payload"]["document"]["response"]["body_object"] = store.promote(
        store.stage(wrong), checksum(wrong)
    )
    with pytest.raises(ValueError, match="byte count/checksum mismatch"):
        restore_fetch(saved, store)


def test_connector_failed_search_is_not_empty_literature(monkeypatch):
    from apps import knowledge_literature_tools
    from packages.evidence_reuse.definition_sources import LiteratureDefinitionConnector

    connector = LiteratureDefinitionConnector(None, None, {"public question": "glaucoma"})
    monkeypatch.setattr(
        knowledge_literature_tools, "search", lambda *a, **k: dict(status="timeout", results=[])
    )
    with pytest.raises(ValueError, match="definition_public_search_not_complete:timeout"):
        connector.search(query="public question", key="one", limit=1)
    monkeypatch.setattr(
        knowledge_literature_tools, "search", lambda *a, **k: dict(status="empty", results=[])
    )
    assert connector.search(query="public question", key="two", limit=1) == []


def test_default_index_scope_is_original_draft_type_not_clinical_authority():
    from db.knowledge import models as m
    from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
    from tests.database.knowledge_support_04a import candidate_records

    record = candidate_records("DEFINITION_MATERIAL")["source_version"]
    writes = []
    repo = SimpleNamespace(
        get=lambda ref: SimpleNamespace(
            record=record, status=m.KnowledgeStatus.DRAFT, withdrawn=False
        ),
        put=writes.append,
    )
    owner = KnowledgeDefinitionSources(repository=repo, store=None)
    refs = owner.index_scope(dict(version_ref=record.reference().model_dump(mode="json")))
    condition = writes[0]
    assert type(condition) is m.ApplicabilityCondition
    assert condition.source_refs == (record.reference(),) and condition.synthetic
    assert (
        condition.population == owner.filters.population == ("research_definition_material_only",)
    )
    assert refs == (condition.reference(),)


def source():
    text = (
        "Synthetic local texture can support or oppose the specified research proposition; "
        "alternative artifact is possible."
    )
    ref = dict(role="source", code="SOURCE:NEW_RESEARCH", version="1.0.0", checksum=digest(text))
    return dict(
        code=ref["code"],
        checksum=ref["checksum"],
        text=text,
        state="available",
        purpose="research_material",
        hit=dict(test_only=True),
        material=dict(
            reference=ref,
            text=text,
            purpose="research_material",
            state="research_only",
            release=None,
            locator="synthetic exact paragraph",
        ),
    )


def draft(target, *, modality="new_image"):
    s = source()
    return dict(
        schema_version="source-definition-draft/2.0.0",
        target=target,
        relation="qualitative",
        examination="Synthetic local texture",
        interpretation=(
            "Texture provides a weak proposition-relative relationship; consider artifacts"
        ),
        applicability="Only the visible region under the supplied synthetic relation",
        input_concepts=["texture"],
        method="live_image_observation",
        unit="qualitative",
        excerpts=[
            dict(
                source_code=s["code"],
                source_checksum=s["checksum"],
                start=0,
                end=len(s["text"]),
                text=s["text"],
            )
        ],
        automatically_accepted=False,
        modalities=[modality],
        acquisition="vision",
        relation_use="directional",
        visible_scope="visible image only",
        extraction_reason="A new source covers an unresolved feature",
        alternatives=["artifact"],
        limitations=["research only"],
        dependencies=[],
    )


class Sources:
    def __init__(self):
        self.value, self.calls = source(), 0

    def read(self, code):
        assert code == self.value["code"]
        return copy.deepcopy(self.value)

    def restore(self, materials):
        pass

    def search(self, **kwargs):
        self.calls += 1
        return dict(materials=[self.read(self.value["code"])], reused=False, backlog=[])


def fixture(tmp_path, *, modality="new_image", documents=None):
    seed = review_fixture().owner.current()
    base = SimpleNamespace(
        ledger=[],
        manifest=SimpleNamespace(
            checksum=digest("synthetic-input"),
            components=[
                SimpleNamespace(id="image-a", modality="new_image", kind="image", status="obtained")
            ],
        ),
        task=SimpleNamespace(target_code=seed.material.goal_code),
        check=lambda: None,
        observations={},
        prepared=SimpleNamespace(payload_bytes=()),
        views=dict(views=[dict(component="image-a", status="obtained")]),
    )
    campaign = SimpleNamespace(
        base=base,
        campaign_id="bounded-synthetic",
        output=tmp_path,
        sessions=SimpleNamespace(read=lambda run: None),
        guard=lambda **kwargs: None,
        protocol=dict(deadline=(datetime.now(UTC) + timedelta(minutes=30)).isoformat()),
        select_targets=MultimodalCampaign.select_targets,
        usage=lambda: 0,
    )
    config = definition_config(configuration(directional=True).model_copy(update=dict(mode="fake")))
    response = dict(
        schema_version="research-definition-proposal/1.0.0",
        definitions=[draft(base.task.target_code, modality=modality)],
        no_addition_reason=None,
    )
    provider = DeterministicFakeProvider(config, [response])
    budget = RequestBudget(
        run_id=seed.material.scope.run_id,
        deadline=datetime.fromisoformat(campaign.protocol["deadline"]),
        max_requests=40,
        journal=MemoryJournal(),
    )
    expansion = DefinitionExpansion(
        campaign,
        documents=documents or Documents(),
        sources=Sources(),
        provider=provider,
        budget=budget,
        scope=seed.material.scope,
        policy=dict(index_keys=[], public_queries=["public texture relationship alternatives"]),
    )
    return expansion


def test_new_qualitative_definition_same_campaign_scheduled_and_restored(tmp_path):
    e = fixture(tmp_path)
    before = digest(e.campaign.protocol)
    revision = e.advance()
    assert revision["denominator_before"] == 0 and revision["denominator_after"] == 1
    assert e.campaign.targets == tuple(revision["added"])
    assert e.campaign.base.ledger[0]["definition"].interpretation_status == "installed"
    assert e.campaign.base.observations == {}
    assert digest(e.campaign.protocol) == before
    assert e.advance() == revision and e.budget.used == 1 and e.sources.calls == 1
    assert e.current() == revision
    e.sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="source_changed"):
        e.current()
    key = next(iter(revision["accepted"]))
    assert e.owner.read(key, historical=True) == revision["accepted"][key]


def test_missing_modality_keeps_requirement_without_observation(tmp_path):
    e = fixture(tmp_path, modality="absent_oct")
    revision = e.advance()
    assert revision["denominator_after"] == 1 and e.campaign.targets == ()
    assert e.campaign.base.ledger[0]["missing_reason"] == "patient_input_missing:absent_oct"
    assert e.campaign.base.observations == {}


def test_pending_unknown_does_not_resend(tmp_path):
    e = fixture(tmp_path)
    with pytest.raises(TimeoutError):
        e.operation("unknown", "search", lambda: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(ValueError, match="unknown_no_repeat"):
        e.operation("unknown", "search", lambda: pytest.fail("repeated remote request"))


def test_cancel_and_review_reserve(tmp_path):
    e = fixture(tmp_path)
    e.campaign.usage = lambda: 28
    assert e.advance()["expansion_stop"] == "definition_review_reserve"
    assert e.budget.used == 0
    e.campaign.usage = lambda: 0
    (tmp_path / "cancel").touch()
    with pytest.raises(ValueError, match="cancelled"):
        e.advance()


@pytest.mark.parametrize("change", ["target", "quote", "unit"])
def test_invalid_definition_not_accepted(tmp_path, change):
    e = fixture(tmp_path)
    value = draft(e.binding["target"])
    if change == "target":
        value["target"] = "GOAL:OTHER"
    elif change == "quote":
        value["excerpts"][0]["text"] = "not in original source"
    else:
        value["unit"] = "invented"
    with pytest.raises(ValueError):
        e.owner.accept("invalid", value)
    assert e.documents.read("definition:invalid") is None


def test_revision_keeps_parent_and_reuses_existing_definition_without_search(tmp_path):
    e = fixture(tmp_path)
    result = e.advance()
    parent = copy.deepcopy(e.campaign.base.ledger)
    following = fixture(tmp_path)
    following.original_rows = parent
    current = following.advance()
    assert not current["added"] and following.budget.used == 0 and following.sources.calls == 0
    assert current["denominator_before"] == current["denominator_after"] == 1
    assert e.current() == result


def test_duplicate_candidates_merge_without_duplicate_vote(tmp_path):
    e = fixture(tmp_path)
    d = draft(e.binding["target"])
    e.provider = DeterministicFakeProvider(
        e.provider.config,
        [
            dict(
                schema_version="research-definition-proposal/1.0.0",
                definitions=[d, d],
                no_addition_reason=None,
            )
        ],
    )
    result = e.advance()
    assert len(result["added"]) == 1 and len(result["merged"]) == 1
    assert len(e.campaign.targets) == 1


@pytest.mark.parametrize("fault", ["quote", "dependencies", "compound"])
def test_unsupported_candidate_is_explicit_backlog_not_whole_case_failure(tmp_path, fault):
    e = fixture(tmp_path)
    d = draft(e.binding["target"])
    if fault == "quote":
        d["excerpts"][0]["text"] = "incorrect exact quotation"
    elif fault == "dependencies":
        d["dependencies"] = ["unavailable-segmentation"]
    else:
        d["modalities"] = ["new_image", "other_modality"]
    e.provider = DeterministicFakeProvider(
        e.provider.config,
        [
            dict(
                schema_version="research-definition-proposal/1.0.0",
                definitions=[d],
                no_addition_reason=None,
            )
        ],
    )
    result = e.advance()
    assert not result["added"] and result["rejected"] and e.campaign.targets == ()
    assert e.campaign.base.observations == {}


def test_cancel_after_confirmed_response_retains_fact_but_no_revision(tmp_path):
    e = fixture(tmp_path)

    def stop_on_return():
        (tmp_path / "cancel").touch()
        return {"fact": "confirmed"}

    with pytest.raises(ProviderError, match="cancelled"):
        e.operation("cancel-response", "search", stop_on_return)
    assert e.documents.read("cancel-response:result")[0] == {"fact": "confirmed"}
    assert e.documents.read("definition:revision") is None


def test_actual_preparer_freezes_directional_provider_and_verifies_before_owner(
    tmp_path, monkeypatch
):
    import json
    from pathlib import Path

    from packages.multimodal_research import entry
    from tests.dataset_adapters.test_multimodal import gamma

    def views(command, **kwargs):
        out = Path(command[4])
        out.mkdir(parents=True)
        (out / "views.json").write_text(json.dumps(dict(views=[])))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", views)
    out = tmp_path / "prepared"
    entry.prepare_entry(
        dataset="gamma",
        root=gamma(tmp_path / "source"),
        sample="0001",
        task="early_glaucoma",
        output=out,
        registry="configs/research/multimodal_propositions_v1.json",
        active_acquisition="configs/research/multimodal_active_requirements_v2.json",
        resilient_vision=True,
        view_profile="detail-v1",
    )
    spec = json.loads((out / "entry.json").read_text())
    assert spec["schema_version"] == "multimodal-live-entry/1.10.0"
    assert spec["provider"]["schema_version"] == "agent-provider-config/8.0.0"

    def stop_before_private_owner():
        raise RuntimeError("offline owner boundary reached")

    monkeypatch.setattr(entry, "owners", stop_before_private_owner)
    with pytest.raises(RuntimeError, match="offline owner boundary reached"):
        entry.assemble_entry(out)


def test_new_profile_uses_current_direction_without_rewriting_old_profile(tmp_path):
    from packages.dataset_adapters.multimodal.adapters import prepare
    from packages.multimodal_research.active_requirements import load, project
    from tests.dataset_adapters.test_multimodal import gamma
    from tests.evaluation_preparation.test_active_acquisition import source_fixture, task

    p = prepare("gamma", gamma(tmp_path), "0001")
    rows, _ = project(
        p.manifest,
        source_fixture(),
        task(),
        load("configs/research/multimodal_active_requirements_v2.json"),
    )
    assert rows and all(
        "无法区分时e=0" not in r["definition"].applicability_conditions for r in rows
    )
    old, _ = project(p.manifest, source_fixture(), task(), load())
    assert any("无法区分时e=0" in r["definition"].applicability_conditions for r in old)


def test_numeric_existing_branch_uses_new_direction_not_old_stage_zero():
    from packages.multimodal_research.active_requirements import load, project
    from tests.evaluation_preparation.test_active_acquisition import source_fixture, task

    manifest = SimpleNamespace(
        adapter_version="synthetic-test/1.0.0",
        checksum=digest("synthetic numeric input"),
        components=[
            SimpleNamespace(
                id="synthetic-age",
                modality="structured_measurement",
                kind="scalar",
                concept="age",
                status="obtained",
                reason=None,
            )
        ],
    )

    def selected(profile):
        rows, _ = project(manifest, source_fixture(), task(), profile)
        return next(r for r in rows if r["component"] == "synthetic-age")

    new = selected(load("configs/research/multimodal_active_requirements_v2.json"))
    old = selected(load())
    assert "无法区分时e=0" not in new["definition"].applicability_conditions
    assert "连续方向" in new["definition"].applicability_conditions
    assert "无法区分时e=0" in old["definition"].applicability_conditions


def test_incomplete_retrieval_slice_is_recorded_not_promoted_or_fatal():
    from packages.evidence_reuse.definition_sources import KnowledgeDefinitionSources
    from packages.knowledge_claims.evidence import Rejected

    hits = [dict(chunk_ref=dict(code=c)) for c in ("slice", "complete")]

    def read(hit, store, **kwargs):
        if hit["chunk_ref"]["code"] == "slice":
            raise Rejected("incomplete_context")
        return dict(code="complete", text="Complete actual paragraph")

    repo = SimpleNamespace(
        query_candidate_index=lambda *a: dict(hits=hits), read_definition_material=read
    )
    owner = KnowledgeDefinitionSources(repository=repo, store=None, index_keys=("index",))
    result = owner.search(
        query="public", key="q", limit=1, operation=lambda *a: pytest.fail("no HTTP")
    )
    assert result["materials"] == [dict(code="complete", text="Complete actual paragraph")]
    assert result["backlog"][0]["reason"] == "source_chunk_incomplete_context"
    assert "slice" not in owner.hits
