"""Affected current-qualification seam; memory persistence, no API or PG.

This is not a public CLI cold-process or report-I/O fault test. Original
InputEvidence, definition/component owners, Phase and stages.local_current run.
Only stored input/tool/source/document reads and Control are in-memory doubles.
"""

import copy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from db.runtime.repository.case_pipeline import identity
from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.dataset_adapters.service import DatasetPreparationService
from packages.evidence_reuse.definition_acceptance import ResearchDefinitionOwner
from packages.grounding.models import digest, seal, validate
from packages.llm_provider.models import TaskDefinition
from packages.multimodal_research import stages
from packages.multimodal_research.campaign import (
    InputEvidence,
    MultimodalCampaign,
    Phase,
    configuration,
)
from packages.multimodal_research.definition_expansion import DefinitionExpansion
from packages.object_store import LocalObjectStore
from packages.research_runtime.task_context import TaskRunContext, project_request
from tests.agent_provider.support import Documents
from tests.dataset_adapters.test_multimodal import gamma
from tests.evaluation_preparation.test_component_observations import installed_fixture
from tests.evaluation_preparation.test_confirmed_interpretation_definition import candidate
from tests.evaluation_preparation.test_definition_expansion import Sources, draft, fixture
from tests.evaluation_preparation.test_multimodal_campaign import sources


def assembled(tmp_path):
    sample = prepare("gamma", gamma(tmp_path / "source"), "0001")
    inputs = DatasetPreparationService(
        SimpleNamespace(register=lambda key, sha, build: build()),
        LocalObjectStore(tmp_path / "visible"),
        LocalObjectStore(tmp_path / "restricted"),
    )
    registered = inputs.register_multimodal(sample, key="synthetic-only")
    inputs.read_visible = lambda key: registered.visible
    views = tmp_path / "views"
    views.mkdir()
    (views / "views.json").write_text(
        json.dumps(
            dict(
                input_checksum=sample.manifest.checksum,
                views=[dict(component="fundus", status="obtained")],
            )
        )
    )
    task = load_intent("advanced_glaucoma", "configs/research/multimodal_propositions_v1.json")[0]
    base = InputEvidence(
        registered=registered,
        inputs=inputs,
        runtime=None,
        objects=None,
        prepared=sample,
        task=task,
        output=tmp_path,
        sources=sources(),
    )
    source = Sources()
    original = ResearchDefinitionOwner(
        documents=Documents(),
        sources=source,
        campaign_id="synthetic-seam",
        target=task.target_code,
        input_checksum=base.manifest.checksum,
        input_reader=base.read_dependency_input,
        guard=lambda: None,
    )
    previous = original.accept("previous", draft(task.target_code, modality="color_fundus"))
    row = next(r for r in base.ledger if r["component"] == "fundus")
    base.ledger = [
        dict(
            row,
            id="check:geometry",
            acquisition_method="disc_cup_measurement",
            definition=validate(TaskDefinition, previous["definition"]),
        )
    ]
    base.observations = {}
    item = installed_fixture(tmp_path)
    item.base = base
    item.input_row, item.input_resources = copy.deepcopy(base.ledger[0]), base.resources
    item.input_binding = digest((item.input_row, item.input_resources))
    item.install()
    seed = fixture(tmp_path)
    campaign = seed.campaign
    campaign.base = base
    policy = json.loads(Path("configs/research/multimodal_definition_gate_v21.json").read_bytes())
    policy["query_requirements"] = {q: [item.target] for q in policy["public_queries"]}
    expansion = DefinitionExpansion(
        campaign,
        documents=seed.documents,
        sources=source,
        provider=seed.provider,
        budget=seed.budget,
        scope=seed.scope,
        policy=policy,
    )
    value = candidate()
    value.update(
        target=task.target_code,
        modalities=["color_fundus"],
        method="confirmed_fundus_disc_cup_geometry",
    )
    receipt = expansion.owner.accept("interpretation", value)
    projection = expansion.project(receipt)[0]
    changes = {
        k: projection[k]
        for k in (
            "definition",
            "basis_kind",
            "definition_receipt_checksum",
            "dependency_assessment",
            "acquisition_allowed",
            "review_allowed",
            "expected_method",
            "expected_unit",
        )
    }
    replacement = dict(
        id=item.target,
        definition="interpretation",
        before=expansion.original_rows[0]["definition"].model_dump(mode="json"),
        after=receipt["definition"],
        reason="target_relation_rebound",
        review_required=True,
    )
    revision = expansion.commit_revision(
        [dict(expansion.original_rows[0], **changes, definition_receipt_key="interpretation")],
        {source.value["code"]: source.value},
        {"interpretation": receipt},
        [],
        [],
        [],
        replaced=[replacement],
    )
    campaign.definition_expansion = expansion
    campaign.expand_definitions = MethodType(MultimodalCampaign.expand_definitions, campaign)
    campaign.request_key = lambda target: "synthetic-seam:" + str(target)
    rid = identity("RUN", "task-research:" + campaign.request_key(None))
    ref = dict(kind="policy", code="POLICY:SYNTHETIC", version="1.0.0", checksum=digest("policy"))
    request = project_request(
        registered.visible,
        run_id=rid,
        task=task,
        policy_ref=ref,
        catalog_ref=dict(
            kind="catalog", code="CATALOG:SYNTHETIC", version="1.0.0", checksum=digest("catalog")
        ),
    )
    context = seal(
        TaskRunContext, request=request, synthetic=True, scope_checksum=digest(request.scope)
    )
    campaign.contexts = SimpleNamespace(read_research_run=lambda *a, **kw: context)
    base.contexts[rid] = context
    campaign.config, campaign.phase_class = configuration(directional=True), Phase
    campaign.accepted_observations = lambda: dict(base.observations)
    campaign.read_model = lambda: pytest.fail("unexpected model read")
    campaign.phase_path = lambda target: tmp_path / "review"
    campaign.phase_path(None).mkdir()
    phase = Phase(base, target=None, observation_reader=campaign.accepted_observations)
    parent, _ = phase.project(run_id=rid, config=campaign.config)
    docs = Documents()
    docs.write("dynamic-phase-input", (parent,))
    docs.write("dynamic-campaign", (campaign.protocol,))
    review = dict(
        run_id=rid,
        target=None,
        accepted=True,
        evidence_limited=False,
        acceptance_key="synthetic-original-review",
    )
    docs.write("dynamic-phase-result", (review,))
    campaign.documents = lambda run: docs
    control = SimpleNamespace(phase="stopped", pending=None)
    campaign.sessions = SimpleNamespace(read=lambda run: control)
    return campaign, expansion, item, docs, control, parent, revision


@pytest.mark.parametrize("stop", ["cancelled", "expired"])
def test_new_interpretation_stage_current_is_local_and_repeatable(tmp_path, stop):
    campaign, expansion, item, docs, _, parent, revision = assembled(tmp_path)
    before = digest((docs.saved, docs.audit, expansion.documents.saved, expansion.documents.audit))
    if stop == "cancelled":
        (tmp_path / "cancel").touch()

    def no_dispatch(**kwargs):
        raise ValueError(stop)

    campaign.guard = no_dispatch
    campaign.execute_phase = lambda *a, **kw: pytest.fail("phase dispatch")
    for _ in range(2):
        stages.local_current(campaign, None)
        assert expansion.campaign.base.definition_current() == revision
    assert (
        digest((docs.saved, docs.audit, expansion.documents.saved, expansion.documents.audit))
        == before
    )
    assert expansion.budget.used == campaign.usage() == 0
    assert len(parent.policy.items) == len(parent.policy.requirements) == 1
    assert parent.material.observations[0] == item.observation()
    assert parent.policy.items[0].method == item.observation().method
    expansion.sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="definition_.*(source|excerpt)"):
        stages.local_current(campaign, None)
    assert (
        digest((docs.saved, docs.audit, expansion.documents.saved, expansion.documents.audit))
        == before
    )


@pytest.mark.parametrize("change", ["parent", "pending", "running", "fact"])
def test_stage_current_rejects_changed_binding_before_recovery_write(tmp_path, change):
    campaign, expansion, item, docs, control, parent, _ = assembled(tmp_path)
    if change == "parent":
        raw = parent.model_dump(mode="json")
        raw["nodes"] = list(raw["nodes"]) + ["node:different"]
        docs.saved["dynamic-phase-input"] = (raw,)
        expected = "stage_recovery_current_parent_changed"
    elif change == "fact":
        item.value["changed"] = True
        expected = "component_observation_changed"
    else:
        control.pending = object() if change == "pending" else None
        control.phase = "running" if change == "running" else "stopped"
        expected = "stage_recovery_control_unconfirmed"
    before = digest((docs.saved, docs.audit, expansion.documents.saved, expansion.documents.audit))
    with pytest.raises(ValueError, match=expected):
        stages.local_current(campaign, None)
    assert (
        digest((docs.saved, docs.audit, expansion.documents.saved, expansion.documents.audit))
        == before
    )
    assert expansion.budget.used == 0
