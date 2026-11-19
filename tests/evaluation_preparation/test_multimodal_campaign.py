"""New modality composition via original owners; injected HTTP, never live calls."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.vectorengine_case_materials import ref
from packages.grounding.models import canonical, digest
from packages.llm_provider.models import Source
from packages.multimodal_research import material
from packages.multimodal_research.campaign import configuration
from tests.database.test_runtime_repository_02b import database as database


def sources():
    keys = {k for _, names, _, _ in material.RULES.values() for k in names}
    initial = {
        k: Source(
            reference=ref("source", "SOURCE:" + k.upper().replace("-", "_"), digest(k)),
            locator="Synthetic rule",
            purpose="research_material",
            state="research_only",
            text="Synthetic source supports research interpretation with uncertainty.",
        )
        for k in keys
    }
    return SimpleNamespace(initial=initial, read=lambda: initial)


def test_configuration_new_partial_prompt_keeps_original_transport():
    from apps.vectorengine_single_case import load_config

    old = load_config("configs/agents/vectorengine_generic_evidence_v1.json")
    new = configuration()
    from packages.llm_provider.config import prompt

    assert "OCT" in prompt(new, "vision")[1]
    assert new.transport == old.transport and new.max_requests == 40
    assert (
        next(r for r in new.roles if r.role == "vision").prompt_version
        == "vision-partial-multimodal-1.1.0"
    )
    assert all(a == b for a, b in zip(new.roles, old.roles, strict=True) if a.role != "vision")


def test_shared_request_deadline_can_only_narrow_without_changing_run():
    from datetime import timedelta

    from packages.llm_provider.provider import MemoryJournal, ProviderError, RequestBudget
    from tests.agent_provider.batch_support import fixture, generate

    f = fixture(1)

    class Narrow(RequestBudget):
        @property
        def effective_deadline(self):
            return self.clock() - timedelta(seconds=1)

    budget = Narrow(
        run_id=f.binding.run_id,
        deadline=f.binding.deadline,
        max_requests=40,
        journal=MemoryJournal(),
    )
    f.budget = f.agent.budget = budget
    with pytest.raises(ProviderError, match="deadline"):
        generate(f)
    assert budget.used == 0 and budget.deadline == f.binding.deadline


def test_numeric_zero_negative_no_threshold_or_hidden_label(tmp_path):
    from packages.dataset_adapters.multimodal.adapters import prepare
    from tests.dataset_adapters.test_multimodal import gdp

    p = prepare("harvard_gdp", gdp(tmp_path / "source"), "data_0001.npz")
    values = {c.concept: material.numeric(c, dict(p.payload_bytes)) for c in p.manifest.components}
    assert values["age"]["value"] == 0
    assert values["md"]["value"] == -2.5
    assert values["rnflt"]["unit"] == "source_units_unqualified"
    assert "99123" not in canonical(values).decode()
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture

    seed = dynamic_fixture(1).owner.current()
    task = load_intent("glaucoma_detection")[0]
    context = SimpleNamespace(run_id=seed.material.scope.run_id, request=SimpleNamespace(task=task))
    visible = SimpleNamespace(
        case=SimpleNamespace(case_code=seed.material.scope.case_code, laterality="UNKNOWN"),
        checksum=seed.material.scope.visible_checksum,
        components=SimpleNamespace(checksum=seed.material.scope.input_checksum),
    )
    root = SimpleNamespace(
        asset_code=seed.material.scope.graph.code, checksum=seed.material.scope.graph.checksum
    )
    ledger = material.requirements(p.manifest, sources().initial, task)
    observations = {
        r["id"]: material.observation(
            r["id"],
            values[next(c.concept for c in p.manifest.components if c.id == r["component"])],
        )
        for r in ledger
        if values[next(c.concept for c in p.manifest.components if c.id == r["component"])]
        is not None
    }
    state = material.state(
        context=context,
        visible=visible,
        root=root,
        config=configuration(),
        medical=sources().initial,
        ledger=ledger,
        observations=observations,
        resources=seed.resources,
    )
    from packages.evidence_details.review_tasks import project_review_tasks

    assert len(project_review_tasks(state).tasks) == 4


def test_failed_predecessor_requests_count_against_same_case_cap():
    from datetime import UTC, datetime, timedelta

    from packages.llm_provider.provider import ProviderError
    from packages.multimodal_research.campaign import MultimodalCampaign

    c = object.__new__(MultimodalCampaign)
    c.base = SimpleNamespace(check=lambda: None)
    prior = [dict(requests=1, result_checksum=digest("old failed result"))]
    c.previous_attempts = lambda: prior
    c.protocol = dict(
        previous_attempts=prior, deadline=(datetime.now(UTC) + timedelta(minutes=1)).isoformat()
    )
    c.usage = lambda: 39
    with pytest.raises(ProviderError, match="budget"):
        c.guard()
    c.guard(dispatch=True)  # Existing fortieth reservation may complete.
    c.usage = lambda: 40
    with pytest.raises(ProviderError, match="budget"):
        c.guard(dispatch=True)


def test_same_modality_different_rnfl_sectors_are_not_paired():
    def component(concept, visit, value):
        return SimpleNamespace(
            id=f"visit:{visit}:{concept}",
            concept=concept,
            visit=str(visit),
            value=value,
            kind="scalar",
            status="obtained",
            modality="structured_measurement",
            unit=None,
            semantic_limits=(),
            source_locator="synthetic",
            laterality="OD",
        )

    manifest = SimpleNamespace(
        components=(
            component("rnfl_mean", 0, 80),
            component("rnfl_superior", 0, 100),
            component("rnfl_mean", 1, 78),
            component("rnfl_superior", 1, 99),
        )
    )
    ledger, observations = material.paired_numeric(manifest, {}, sources().initial)
    assert {r["id"] for r in ledger} == {"comparison:rnfl_mean", "comparison:rnfl_superior"}
    assert json.loads(observations["comparison:rnfl_mean"].value)["delta"] == [-2]
    assert json.loads(observations["comparison:rnfl_superior"].value)["delta"] == [-1]


def test_new_attempt_reuses_input_registration_not_new_patient(tmp_path):
    from packages.multimodal_research.entry import registration_key

    old = tmp_path / "old"
    old.mkdir()
    spec = dict(input_checksum=digest("input"), intent=dict(task="same"), previous=None)
    (old / "entry.json").write_text(json.dumps(spec))
    new = dict(spec, previous=str(old))
    assert registration_key(tmp_path / "new", new) == registration_key(old, spec)
    with pytest.raises(ValueError, match="previous_input_identity"):
        registration_key(tmp_path / "new", dict(new, input_checksum=digest("another")))


def test_known_measurements_only_has_no_vision_dispatch_path():
    from packages.multimodal_research.campaign import MultimodalCampaign

    b = SimpleNamespace(known_only=True, ledger=[dict(id="check:oct", component="oct")])
    assert MultimodalCampaign.select_targets(b, {"oct"}) == ()
    b.known_only = False
    assert MultimodalCampaign.select_targets(b, {"oct"}) == ("check:oct",)


@pytest.mark.parametrize("four_hypotheses", [False, True, "local_timeout", "resilient"])
def test_original_registered_task_pipeline_pg(database, tmp_path, monkeypatch, four_hypotheses):
    from db.runtime.repository.case_evidence import CaseEvidenceRepository
    from packages.dataset_adapters.multimodal.adapters import prepare
    from packages.dataset_adapters.multimodal.preparation import persist
    from packages.dataset_adapters.multimodal.tasks import load_intent
    from packages.dataset_adapters.reading import grayscale_image
    from packages.llm_provider.transport import HTTPTransport, Reply
    from packages.multimodal_research.campaign import InputEvidence, MultimodalCampaign
    from packages.object_store import checksum
    from tests.agent_provider.batch_support import batch_reply
    from tests.dataset_adapters.test_multimodal import gdp
    from tests.dataset_adapters.test_registration import service
    from tests.evaluation_preparation.test_partial_observation import detail

    p = prepare("harvard_gdp", gdp(tmp_path / "source"), "data_0001.npz")
    out = tmp_path / "campaign"
    persist(p, out / "input")
    views = out / "views"
    views.mkdir()
    c = next(c for c in p.manifest.components if c.kind == "map")
    raw = grayscale_image((8, 8), bytes(64))
    filename = checksum(c.id.encode())[7:31] + ".png"
    (views / filename).write_bytes(raw)
    (views / "views.json").write_text(
        json.dumps(
            dict(
                input_checksum=p.manifest.checksum,
                views=[
                    dict(
                        component=c.id,
                        status="obtained",
                        file=filename,
                        checksum=checksum(raw),
                        width=8,
                        height=8,
                        modality=c.modality,
                        indices=[0],
                        available_members=1,
                        inspected_fraction=1,
                        transform=dict(kind="synthetic_display"),
                        limits=["synthetic fixture only"],
                    )
                ],
            )
        )
    )
    inputs = service(database, tmp_path)
    registered = inputs.register_multimodal(
        p, key="multimodal-campaign-input:" + str(four_hypotheses)
    )
    base = InputEvidence(
        registered=registered,
        inputs=inputs,
        runtime=CaseEvidenceRepository(database[2]),
        objects=inputs.visible_store,
        prepared=p,
        task=load_intent(
            "glaucoma_four_hypotheses", "configs/research/multimodal_tasks_four_hypotheses_v1.json"
        )[0]
        if four_hypotheses
        else load_intent("glaucoma_detection")[0],
        output=out,
        sources=sources(),
    )
    base.resilient_vision = four_hypotheses == "resilient"
    campaign = MultimodalCampaign(
        base, database[2], campaign_id="test-mm-campaign:" + str(four_hypotheses), synthetic=True
    )
    campaign.targets = ("check:" + c.id,)
    if four_hypotheses == "local_timeout":
        base.detailed = True  # Exercise the explicit new policy, not legacy semantics.
    campaign.freeze()
    calls = []

    def send(self, config, body, **kwargs):
        payload = json.loads(body["messages"][1]["content"][0]["text"])
        calls.append(payload["role"])
        if payload["role"] == "vision":
            if four_hypotheses == "local_timeout":
                return Reply(error="timeout", api_requests=1)
            schema = json.loads(body["messages"][1]["content"][-1]["text"].split("\n", 1)[1])
            raw = dict(
                schema_version="visual-observation-content/1.0.0",
                binding_code=schema["properties"]["binding_code"]["enum"][0],
                status="observed",
                finding="Synthetic partial view",
                partial=detail(),
            )
        else:
            raw = batch_reply(body)
        # Wire-shaped response only; HTTPTransport is replaced, no socket/API is used.
        return Reply(content=canonical(raw), api_requests=1, finish_reason="stop")

    monkeypatch.setattr(HTTPTransport, "send", send)
    observed = campaign.execute_phase(campaign.targets[0])
    if four_hypotheses == "local_timeout":
        assert not observed["accepted"]
        assert campaign.local_acquisition_gap(observed), observed
        assert campaign.execute_phase(campaign.targets[0]) == observed
        assert calls == ["vision"] and campaign.usage() == 1
        return
    assert observed["accepted"], observed
    if four_hypotheses == "resilient":
        from packages.multimodal_research.continuation import Continuation

        spec = dict(
            input_checksum=base.manifest.checksum,
            intent=base.task.model_dump(mode="json"),
            independent_propositions=True,
            provider=campaign.config.model_dump(mode="json"),
        )
        suite = dict(
            case_code=base.binding["case_code"],
            shared_acquisition=[observed],
            trees=[],
            complete=False,
            new_requests=1,
        )
        (out / "entry.json").write_text(json.dumps(spec))
        (out / "suite-result.json").write_text(json.dumps(suite))
        (out / "suite-checksum.json").write_text(json.dumps(dict(checksum=digest(suite))))
        reuse = Continuation(campaign, out)
        assert reuse.check_attempts()[0]["requests"] == 1
        assert reuse.check() == campaign.accepted_observations()
        assert reuse.report()["original_runs_resumed"] is False
        assert calls == ["vision"]
    review = campaign.execute_phase()
    assert review["accepted"], review
    saved = campaign.documents(review["run_id"]).read("accepted:" + review["acceptance_key"])
    assert saved[2]["scores"]["schema_version"] == (
        "case-research-scoring/10.0.0" if four_hypotheses else "case-research-scoring/9.0.0"
    )
    assert saved[2]["scores"]["target"]["code"] == base.task.target_code
    assert campaign.execute_phase() == review
    expected_calls = ["vision"] + ["review"] * (6 if four_hypotheses else 2)
    assert calls == expected_calls
    assert (
        campaign.contexts.read_research_run(review["run_id"]).schema_version == "research-run/2.0.0"
    )
    from packages.multimodal_research.report import export

    result = dict(
        schema_version="multimodal-case-result/1.0.0",
        sample=p.manifest.sample,
        dataset=p.manifest.dataset,
        case_code=base.binding["case_code"],
        task=base.task.model_dump(mode="json"),
        phases=[observed, review],
        complete=True,
        new_requests=campaign.usage(),
        hidden_labels_used=False,
        clinical_acceptance=False,
    )
    if four_hypotheses:
        from packages.multimodal_research.confirmed_observations import ConfirmedObservations

        reader = object.__new__(ConfirmedObservations)
        reader.campaign, reader.config = campaign, campaign.config
        reader.result = dict(phases=[observed])
        reader.values, reader.origins = reader.read()
        reader.binding = digest((reader.values, reader.origins))
        result["schema_version"] = "multimodal-case-result/1.2.0"
        result["confirmed_observations"] = reader.report()
        campaign.confirmed_observations = reader
    docs = campaign.documents(review["run_id"])
    with docs.session():
        docs.write("multimodal-case-result", (result,))
    report = export(campaign, result, directory=tmp_path / "exports")
    assert report["complete"] and not report["task_clinically_validated"]
    exported = json.loads(Path(report["json"]).read_bytes())
    frozen = docs.read("dynamic-phase-input")[0]
    expected_observations = {
        item["id"]: next(
            observation
            for observation in frozen["material"]["observations"]
            if observation["reference"]["code"] == item["observation_code"]
        )
        for item in frozen["policy"]["items"]
    }
    assert {
        row["id"]: row["observation"]
        for row in exported["requirements"]
        if row["observation"] is not None
    } == expected_observations
    assert export(campaign, result, directory=tmp_path / "exports") == report
    assert calls == expected_calls
    from db import manage
    from db.memory.admin import bootstrap_memory_principals
    from db.memory.repository import MemoryRepository
    from packages.memory.archive import ArchiveSources
    from packages.multimodal_research.memory import persist as archive

    bootstrap_memory_principals(
        database[0], *(manage.test_target(c).user for c in manage.MEMORY_CHANNELS)
    )
    source_owner = ArchiveSources(
        runtime=base.runtime,
        read_bytes=docs.read_artifact,
        read_run_case=lambda run: campaign.contexts.read_research_run(run).request.scope.case_code,
        read_view=lambda *_: None,
    )
    memory = MemoryRepository(
        manage.test_target("memory_writer"), capability="writer", sources=source_owner
    )
    stored = archive(memory, source_owner, docs, result)
    assert len(stored["items"]) == 5
    assert archive(memory, source_owner, docs, result) == stored
    assert all(not row["eligible_for_planning"] for row in stored["items"])
    if four_hypotheses:
        # New read-only owner port: real PG receipt/current sources, no new transport.
        from packages.multimodal_research.confirmed_observations import ConfirmedObservations

        assert set(reader.check()) == {observed["target"]}
        assert reader.report()["origins"][0]["run_id"] == observed["run_id"]
        assert calls == expected_calls  # Never replay the old Vision or Review.
        from db.runtime.repository.case_pipeline import identity

        original_objects = source_owner.runtime.read_observation_bundle(
            identity("TASK", observed["run_id"] + ":agent:accepted:" + observed["acceptance_key"])
        )
        archived = memory.read(stored["items"][0]["key"]).record
        assert {a.asset_code for a in original_objects} <= {s.object_code for s in archived.sources}
    base.sources.read = lambda: {}
    if four_hypotheses:
        with pytest.raises(ValueError, match="medical_source_changed"):
            reader.check()
    with pytest.raises(ValueError, match="medical_source_changed"):
        campaign.guard()
