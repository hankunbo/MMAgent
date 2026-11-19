"""Offline regressions for full-text identity, cold report recovery and missing interpretation."""

import copy
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest, validate
from packages.multimodal_research.campaign import MultimodalCampaign, configuration
from packages.multimodal_research.definition_expansion import DefinitionExpansion
from tests.agent_provider.support import Documents
from tests.evaluation_preparation.test_definition_expansion import draft, fixture


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("definition repair tests must not use live DNS or sockets")

    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(socket, "socket", denied)


def test_literature_campaign_uses_original_executor_governor(tmp_path, monkeypatch):
    from apps import knowledge_literature_tools as tools
    from packages.evidence_reuse.definition_sources import LiteratureDefinitionConnector
    from packages.knowledge_connectors import ConnectorConfiguration
    from packages.knowledge_search import FederatedSearch
    from tests.knowledge_search.support import config, policy

    cfg = config()
    e = fixture(tmp_path)
    connector = LiteratureDefinitionConnector(None, e, {"public question": "rnfl"})
    instances = []

    def original_search(repo, keys, topic, **kwargs):
        # Exercise real production assembly, not a replacement search stub response.
        with FederatedSearch(
            ConnectorConfiguration(config_version="1.0.0", providers=(cfg,)),
            policy(),
            lambda c, g: tools.client(c, g),
            request_guard=kwargs.get("request_guard"),
        ) as executor:
            client = executor.registry.get(cfg.provider_code).client
            assert client.governor is executor.governor
            assert executor.governor.request_guard is not None
            instances.append(executor)
        return dict(status="empty", results=[])

    from packages.knowledge_connectors import SafeHTTPClient
    from packages.knowledge_connectors.fake import RecordedTransport

    monkeypatch.setattr(tools, "search", original_search)
    monkeypatch.setattr(
        tools, "client", lambda c, g: SafeHTTPClient(c, RecordedTransport(()), governor=g)
    )
    assert connector.search(query="public question", key="one", limit=1) == []
    assert len(instances) == 1 and e.operations() == []


@pytest.mark.parametrize("cancelled", [False, True])
def test_original_search_wire_honours_campaign_budget(tmp_path, cancelled):
    from packages.evidence_reuse.definition_sources import LiteratureDefinitionConnector
    from packages.knowledge_connectors import ConnectorConfiguration, SafeHTTPClient
    from packages.knowledge_connectors.fake import FakeClock, RecordedTransport
    from packages.knowledge_search import FederatedSearch
    from tests.knowledge_search.support import config, exchange, plan, policy

    e = fixture(tmp_path)
    cfg, clock = config(), FakeClock()
    query = plan(cfg)
    transport = RecordedTransport((exchange(cfg, query),))
    guard = LiteratureDefinitionConnector(None, e, {}).governor()
    if cancelled:
        (e.campaign.output / "cancel").touch()
    with FederatedSearch(
        ConnectorConfiguration(config_version="1.0.0", providers=(cfg,)),
        policy(),
        lambda c, g: SafeHTTPClient(
            c, transport, governor=g, clock=clock, resolver=lambda _: ("93.184.216.34",)
        ),
        clock=clock,
        request_guard=guard,
    ) as executor:
        if cancelled:
            assert executor.execute(query).status == "cancelled"
            assert transport.calls == 0 and not e.operations()
        else:
            result = executor.execute(query)
            assert result.status == "succeeded" and transport.calls == 1
            assert [o["kind"] for o in e.operations()] == ["wire"]


@pytest.mark.parametrize("source,identifier", [("MED", "999990001"), ("PMC", "PMC999990001")])
def test_original_search_record_to_original_fulltext_adapter(
    tmp_path, monkeypatch, source, identifier
):
    from apps import knowledge_literature_tools as tools
    from db.knowledge import models as m
    from packages.evidence_reuse.definition_sources import LiteratureDefinitionConnector
    from packages.knowledge_connectors import StructuredQuery, load_fixture
    from packages.knowledge_connectors.connectors import EuropePMCConnector
    from packages.knowledge_connectors.fake import FakeClock, RecordedTransport
    from packages.knowledge_connectors.http import SafeHTTPClient
    from tests.knowledge_search.support import config

    cfg = config()
    exchanges = list(
        load_fixture(Path("tests/knowledge_connectors/fixtures/europe_pmc.json")).exchanges
    )
    body = json.loads(exchanges[0].body_text)
    body["resultList"]["result"][0].update(source=source, id=identifier)
    exchanges[0] = exchanges[0].model_copy(update=dict(body_text=json.dumps(body)))
    transport = RecordedTransport(tuple(exchanges))
    clock = FakeClock()

    def client(c, governor=None):
        return SafeHTTPClient(
            c,
            transport,
            resolver=lambda _: ("93.184.216.34",),
            clock=clock,
            environment=lambda _: None,
            governor=governor,
        )

    record = (
        EuropePMCConnector(cfg, client(cfg))
        .search(StructuredQuery(text="synthetic fixture", page_size=1))
        .records[0]
    )
    assert record.record_id == source + ":" + identifier
    discovery = SimpleNamespace(
        provider_record_id=record.record_id, discovery_identifiers=record.identifiers
    )
    snapshot = SimpleNamespace(
        withdrawn=False,
        status=m.KnowledgeStatus.DRAFT,
        record=SimpleNamespace(discovery=discovery),
    )
    expansion = fixture(tmp_path)
    connector = LiteratureDefinitionConnector(
        SimpleNamespace(get=lambda _: snapshot), expansion, {}
    )
    monkeypatch.setattr(tools, "active_provider", lambda *a: (cfg, None))
    monkeypatch.setattr(tools, "client", client)
    hit = dict(
        reference=dict(
            kind="search_hit", code="SOURCE:SYNTHETIC_HIT", version="1.0.0", checksum=digest("hit")
        )
    )
    fetched = connector.fetch(hit)
    assert fetched.document.locator.record_id == "PMC999990001"
    assert fetched.document.response.body == exchanges[1].body_text.encode()
    assert fetched.metadata_response_checksum == fetched.document.response.response_checksum
    assert transport.calls == 2
    transport.assert_consumed()
    assert [o["kind"] for o in expansion.operations()] == ["wire"]


@pytest.mark.parametrize(
    "identifiers,reason",
    [
        ([], "europepmc_full_text_pmcid_missing"),
        (["PMC111", "PMC222"], "definition_full_text_pmcid_identity"),
    ],
)
def test_missing_or_ambiguous_pmcid_never_sends(tmp_path, monkeypatch, identifiers, reason):
    from apps import knowledge_literature_tools as tools
    from db.knowledge import models as m
    from packages.evidence_reuse.definition_sources import (
        DefinitionMaterialUnavailable,
        KnowledgeDefinitionSources,
        LiteratureDefinitionConnector,
    )

    snapshot = SimpleNamespace(
        withdrawn=False,
        status=m.KnowledgeStatus.DRAFT,
        record=SimpleNamespace(
            discovery=SimpleNamespace(
                provider_record_id="MED:111",
                discovery_identifiers=[
                    SimpleNamespace(identifier_type="PMCID", value=v) for v in identifiers
                ],
            )
        ),
    )
    connector = LiteratureDefinitionConnector(SimpleNamespace(get=lambda _: snapshot), None, {})
    monkeypatch.setattr(tools, "client", lambda *a, **k: pytest.fail("unexpected HTTP client"))
    hit = dict(
        reference=dict(
            kind="search_hit", code="SOURCE:SYNTHETIC_HIT", version="1.0.0", checksum=digest("hit")
        )
    )
    error = ValueError if identifiers else DefinitionMaterialUnavailable
    with pytest.raises(error, match=reason):
        connector.fetch(hit)
    if not identifiers:
        sources = KnowledgeDefinitionSources(
            repository=None,
            store=None,
            search_public=lambda **_: [hit],
            fetch_hit=connector.fetch,
        )
        result = sources.search(
            query="synthetic", key="one", limit=1, operation=lambda k, t, f: f()
        )
        assert result["materials"] == []
        assert result["backlog"][0]["result"]["material_unavailable"] == reason


def missing_interpretation(tmp_path, *, obtained=False):
    from packages.evidence_reuse.definition_acceptance import requirement_rows
    from packages.llm_provider.models import TaskDefinition
    from packages.multimodal_research.material import observation

    e = fixture(tmp_path)
    accepted = e.owner.build(draft(e.binding["target"]))
    row = requirement_rows(accepted, e.campaign.base.manifest)[0]
    raw = row["definition"].model_dump(mode="json")
    raw.update(interpretation=None, interpretation_status="missing")
    raw["reference"].update(
        code="DEF:SYNTHETIC_MISSING",
        checksum=digest({k: v for k, v in raw.items() if k != "reference"}),
    )
    row.update(id="active:existing", parent="EXISTING", definition=validate(TaskDefinition, raw))
    row.pop("definition_receipt_checksum")
    e.campaign.base.ledger = [row]
    if obtained:
        e.campaign.base.observations[row["id"]] = observation(
            row["id"],
            "Synthetic observed texture",
            method="live_image_observation",
            unit="qualitative",
        )
    e.campaign.base.active_owner = SimpleNamespace(
        check=lambda: None,
        native=None,
        plan=dict(
            nodes=[dict(id="EXISTING", parent="root", title="Existing")],
            rows=[
                dict(
                    id=row["id"],
                    parent="EXISTING",
                    state="scheduled",
                )
            ],
        ),
        report=lambda: dict(synthetic=True),
    )
    return DefinitionExpansion(
        e.campaign,
        documents=e.documents,
        sources=e.sources,
        provider=e.provider,
        budget=e.budget,
        scope=e.scope,
        policy=e.policy.model_dump(mode="json"),
    )


@pytest.mark.parametrize("obtained", [False, True])
def test_missing_interpretation_filled_same_id_without_new_vote(tmp_path, obtained):
    from packages.multimodal_research.active_acquisition import structure

    e = missing_interpretation(tmp_path, obtained=obtained)
    parent = copy.deepcopy(e.original_rows)
    observations = copy.deepcopy(e.campaign.base.observations)
    revision = e.advance()
    assert revision["schema_version"] == "campaign-requirement-revision/1.1.0"
    assert not revision["added"] and len(revision["replaced"]) == 1
    assert revision["denominator_before"] == revision["denominator_after"] == 1
    assert revision["replaced"][0]["before"]["interpretation_status"] == "missing"
    assert revision["replaced"][0]["after"]["interpretation_status"] == "installed"
    assert revision["replaced"][0]["review_required"]
    assert e.original_rows == parent and e.campaign.base.observations == observations
    row = e.campaign.base.ledger[0]
    assert row["id"] == "active:existing" and row["parent"] == "EXISTING"
    assert (
        row["category"] == parent[0]["category"]
        and row["evidence_unit"] == parent[0]["evidence_unit"]
    )
    assert e.campaign.targets == (() if obtained else ("active:existing",))
    assert e.advance() == revision and e.budget.used == e.sources.calls == 1
    report = dict(requirements=[dict(id=row["id"])], report_production={})
    structure(report, e.campaign.base)
    assert (
        report["requirements"][0]["source_requirement"]["definition"]["interpretation_status"]
        == "installed"
    )


def test_filled_definition_and_original_observation_reach_review_tasks(tmp_path):
    from packages.evidence_details.review_tasks import project_review_tasks
    from packages.multimodal_research.material import state
    from tests.evaluation_preparation.test_directional_review import fixture as review_fixture

    e = missing_interpretation(tmp_path, obtained=True)
    e.advance()
    seed = review_fixture().owner.current()
    scope = seed.material.scope
    task = SimpleNamespace(
        target_code=e.binding["target"], output_kind="evidence_review", code="TASK:SYNTHETIC"
    )
    projected = state(
        context=SimpleNamespace(run_id=scope.run_id, request=SimpleNamespace(task=task)),
        visible=SimpleNamespace(
            case=SimpleNamespace(case_code=scope.case_code, laterality=seed.laterality),
            checksum=scope.visible_checksum,
            components=SimpleNamespace(checksum=scope.input_checksum),
        ),
        root=SimpleNamespace(asset_code=scope.graph.code, checksum=scope.graph.checksum),
        config=configuration(directional=True),
        medical=e.campaign.base.definition_materials,
        ledger=e.campaign.base.ledger,
        observations=e.campaign.base.observations,
        resources=seed.resources,
    )
    tasks = project_review_tasks(projected).tasks
    assert len(tasks) == len(projected.policy.requirements) == 1
    assert tasks[0].assessment_id == "active:existing"
    assert tasks[0].definition.interpretation_status == "installed"
    assert tasks[0].observation == e.campaign.base.observations["active:existing"].reference
    assert tasks[0].allowed_sources == e.campaign.base.ledger[0]["definition"].source_refs


def test_installed_interpretation_is_not_replaced_when_another_gap_triggers_search(tmp_path):
    from packages.evidence_reuse.definition_acceptance import requirement_rows

    e = fixture(tmp_path)
    value = draft(e.binding["target"])
    value["interpretation"] = "Synthetic already installed interpretation"
    original = requirement_rows(e.owner.build(value), e.campaign.base.manifest)[0]
    e.campaign.base.ledger = [original]
    policy = e.policy.model_dump(mode="json")
    policy["query_requirements"] = {policy["public_queries"][0]: ["another_missing_requirement"]}
    following = DefinitionExpansion(
        e.campaign,
        documents=e.documents,
        sources=e.sources,
        provider=e.provider,
        budget=e.budget,
        scope=e.scope,
        policy=policy,
    )
    revision = following.advance()
    assert not revision["added"] and not revision["replaced"]
    assert following.campaign.base.ledger[0]["definition"] == original["definition"]
    assert (
        revision["merged"][0]["disposition"] == "existing_definition_kept_new_relation_not_applied"
    )


@pytest.mark.parametrize("fault", ["method", "unit", "observed_method", "started_review"])
def test_interpretation_fill_does_not_change_method_or_reuse_started_review(tmp_path, fault):
    e = missing_interpretation(tmp_path, obtained=fault == "observed_method")
    if fault == "method":
        e.original_rows[0]["expected_method"] = "different_measurement"
    elif fault == "unit":
        e.original_rows[0]["expected_unit"] = "ratio"
    elif fault == "observed_method":
        old = e.campaign.base.observations["active:existing"]
        e.campaign.base.observations["active:existing"] = old.model_copy(
            update=dict(method="other")
        )
    else:
        (tmp_path / "review").mkdir()
        (tmp_path / "review" / "request.json").write_text("{}")
        with pytest.raises(ValueError, match="replacement_after_review_started"):
            e.advance()
        assert e.documents.read("definition:revision") is None
        return
    revision = e.advance()
    assert not revision["added"] and not revision["replaced"]
    assert revision["rejected"][0]["reason"] == "interpretation_method_or_unit_mismatch"
    assert e.campaign.base.ledger[0]["definition"].interpretation_status == "missing"


def cold_campaign(tmp_path, monkeypatch, docs, result):
    from packages.multimodal_research import memory

    c = object.__new__(MultimodalCampaign)
    c.output = tmp_path
    c.base = SimpleNamespace(ledger=[], definition_expansion_policy={})
    c.targets = ()
    c.freeze = lambda: None
    c.documents = lambda _: docs
    (tmp_path / "result.json").write_text(json.dumps(result))
    docs.write("multimodal-case-result", (result,))
    monkeypatch.setattr(memory, "archive", lambda *a: dict(status="offline_archive_boundary"))
    return c


@pytest.mark.parametrize("accepted", [True, False])
def test_cold_completed_run_restores_saved_bytes_without_definition_or_model(
    tmp_path, monkeypatch, accepted
):
    from packages.multimodal_research import report
    from packages.multimodal_research.report_storage import save

    docs = Documents()
    result = dict(
        complete=accepted,
        new_requests=3,
        task=dict(target_code="GOAL:SYNTHETIC"),
        phases=[dict(run_id="synthetic", target=None, accepted=accepted, acceptance_key="review")],
    )
    c = cold_campaign(tmp_path, monkeypatch, docs, result)
    scores = dict(synthetic=True) if accepted else None
    detail = dict(records={}, graph={}, receipt={}) if accepted else None
    saved = dict(
        task=result["task"],
        result=result,
        original_review=detail,
        scores=scores,
        requirements=[dict(id="discovered:synthetic")],
        definition_revision=dict(added=["discovered:synthetic"]),
    )
    if accepted:
        docs.write("accepted:review", ({}, {}, dict(scores=scores), {}))
    save(docs, saved, "<html>Saved synthetic tree</html>", key="dynamic-case-report")
    before = copy.deepcopy(docs.saved)
    original = report.restore_export
    monkeypatch.setattr(
        report, "restore_export", lambda campaign, r: original(campaign, r, tmp_path / "exports")
    )
    monkeypatch.setattr(report, "export", lambda *a: pytest.fail("must not rerender saved report"))
    c.expand_definitions = lambda **_: pytest.fail("must not re-enter definition expansion")
    output = c.execute()
    assert json.loads(Path(output["json"]).read_text()) == saved
    assert Path(output["html"]).read_text() == "<html>Saved synthetic tree</html>"
    assert docs.saved == before and not hasattr(c.base, "definition_revision")


@pytest.mark.parametrize("missing_revision", [False, True])
def test_cold_result_without_report_restores_revision_only(tmp_path, monkeypatch, missing_revision):
    from packages.evidence_reuse import definition_sources
    from packages.llm_provider import provider
    from packages.multimodal_research import report
    from packages.research_runtime import agent_records

    e = fixture(tmp_path)
    revision = e.advance()
    if missing_revision:
        e.documents.saved.pop("definition:revision")
    result = dict(phases=[dict(run_id=e.scope.run_id)])
    c = cold_campaign(tmp_path, monkeypatch, e.documents, result)
    # A freshly assembled base has the parent ledger and no in-memory revision.
    c.base = copy.copy(e.campaign.base)
    c.base.ledger = []
    del c.base.definition_revision, c.base.definition_current, c.base.definition_materials
    c.base.definition_expansion_policy = e.policy.model_dump(mode="json")
    c.base.sources = SimpleNamespace(
        reader=SimpleNamespace(original=SimpleNamespace(knowledge=None, knowledge_objects=None))
    )
    c.campaign_id, c.protocol, c.sessions = (
        e.campaign.campaign_id,
        e.campaign.protocol,
        e.campaign.sessions,
    )
    c.config = configuration(directional=True)
    c.phase_class = lambda *a, **k: None
    c.register_phase = lambda *a: (
        SimpleNamespace(run_id=e.scope.run_id),
        SimpleNamespace(material=SimpleNamespace(scope=e.scope)),
    )
    monkeypatch.setattr(definition_sources, "KnowledgeDefinitionSources", lambda **_: e.sources)
    monkeypatch.setattr(agent_records, "RuntimeAgentJournal", lambda _: e.budget.journal)
    monkeypatch.setattr(
        provider, "StructuredProvider", lambda *a, **k: pytest.fail("no provider on restore")
    )
    e.sources.search = lambda **_: pytest.fail("no search on restore")

    def export(campaign, value):
        assert campaign.base.definition_revision == revision
        assert campaign.targets == tuple(revision["added"])
        assert set(campaign.base.definition_materials) == {s["code"] for s in revision["materials"]}
        return dict(restored=True)

    monkeypatch.setattr(report, "export", export)
    if missing_revision:
        with pytest.raises(ValueError, match="definition_revision_binding"):
            c.execute()
    else:
        assert c.execute()["restored"]


def test_replacement_definition_is_in_six_section_report():
    from packages.evaluation_preparation.report_production import produce
    from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture
    from tests.evaluation_preparation.test_report_production import report_for

    f = dynamic_fixture(1)
    report = report_for(f)
    key = report["requirements"][0]["id"]
    accepted = dict(draft=dict(extraction_reason="Synthetic missing interpretation filled"))
    report["definition_revision"] = dict(
        revision=1,
        added=[],
        replaced=[dict(id=key)],
        denominator_before=1,
        denominator_after=1,
        rows=[dict(id=key, definition_receipt_checksum=digest(accepted))],
        accepted={"synthetic": accepted},
    )
    result = produce(report, f.owner.current())
    rationale = result["requirements"][0]["presentation_sections"]["rationale"]
    assert rationale["research_definition"] == accepted
    assert rationale["denominator_before"] == rationale["denominator_after"] == 1
    assert result["scores"] == report["scores"]
