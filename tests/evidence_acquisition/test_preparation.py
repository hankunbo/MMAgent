"""No network/PG/patient execution; real projection/Provider/admission owner logic."""

import copy
import json
from types import SimpleNamespace

import pytest

from packages.evidence_acquisition.materials import (
    TaskTargetedReader,
    _rebuild,
    observation_alias_candidate,
    prepare_definition,
    prepare_visual_task,
)
from packages.evidence_acquisition.models import Requirement, SearchProgress, SearchSuggestion
from packages.evidence_acquisition.planning import (
    affected_assessments,
    plan_acquisition,
    search_plan,
)
from packages.grounding.models import digest, validate
from packages.llm_provider.models import ImageBinding, TaskDefinition
from packages.object_store import checksum
from tests.agent_provider.support import definition, materials, ref


def state_with(*, status="obtained", value="0", interpretation=None, image=False):
    import base64

    from packages.llm_provider.config import ProviderConfig, load_provider_config

    config = load_provider_config(environ={})
    if image:
        config = validate(ProviderConfig, config.model_copy(update={"image_capable": True}))
    reader, b = materials(role="vision", config=config)
    reader.acquisition_config = config
    state = reader.state
    d = validate(TaskDefinition, definition(interpretation=interpretation))
    item = state.policy.items[0].model_copy(update={"definition": d})
    obs = state.material.observations[0].model_copy(update={"status": status, "value": value})
    if image:
        raw = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1sAAAAASUVORK5CYII="
        )
        binding = ImageBinding(
            reference=ref("image", "IMAGE:ONE", checksum=checksum(raw)),
            media_type="image/png",
            width=1,
            height=1,
            coordinate_system="pixel_xy_top_left",
            preprocessing=ref("preprocessing", "PRE:ONE"),
            location=(0, 0, 1, 1),
        )
        obs = obs.model_copy(update={"image": binding})
        reader.image_bytes = raw
    reader.state = _rebuild(state, items=(item,), observations=(obs,))
    return reader, b


def requirement(**changes):
    return Requirement(
        assessment_id="item:one",
        purpose="disease_relation",
        acquisition="image",
        required_inputs=("slo_image",),
        public_topics=("rim", "slo"),
        **changes,
    )


def test_known_zero_preserved_missing_rule_not_missing_observation():
    reader, _ = state_with()
    plan = plan_acquisition(reader.state, [requirement()])
    card = plan.cards[0]
    assert card.observation_status == "obtained"
    assert card.blockers == ("interpretation_reference",)
    assert card.next_actions == ("bind_existing_reference", "search_knowledge")
    assert not plan.execution_authorized


@pytest.mark.parametrize("value", ["0", "-0.7"])
def test_ready_measurements_with_installed_rule(value):
    reader, _ = state_with(value=value, interpretation="Synthetic signed comparison")
    card = plan_acquisition(reader.state, [requirement()]).cards[0]
    assert card.disposition == "ready_for_review" and card.next_actions == ("review",)


def test_auxiliary_does_not_require_disease_threshold():
    reader, _ = state_with()
    req = requirement().model_copy(update={"purpose": "measurement_dependency"})
    card = plan_acquisition(reader.state, [req]).cards[0]
    assert card.disposition == "auxiliary_observed"
    assert card.interpretation_status == "not_independent_disease_rule"
    assert not card.next_actions


@pytest.mark.parametrize("status", ["unknown", "missing"])
def test_image_present_routes_to_targeted_acquisition(status):
    reader, _ = state_with(status=status, value=None, image=True)
    card = plan_acquisition(reader.state, [requirement()]).cards[0]
    assert "targeted_vision" in card.next_actions
    assert card.disposition == "needs_acquisition"


def test_absent_external_exam_stops_automatic_search():
    reader, _ = state_with(status="missing", value=None)
    req = requirement().model_copy(
        update={
            "acquisition": "external_exam",
            "required_inputs": ("oct",),
            "unavailable_inputs": ("oct",),
        }
    )
    card = plan_acquisition(reader.state, [req]).cards[0]
    assert card.disposition == "current_input_unavailable" and not card.next_actions
    assert card.missing_inputs == ("oct",)


def test_ungradable_is_not_negative_and_not_repeated_until_positive():
    reader, _ = state_with(status="ungradable", value="Limited visible region", image=True)
    card = plan_acquisition(reader.state, [requirement()]).cards[0]
    assert card.disposition == "quality_limited" and not card.next_actions


def test_method_gap_does_not_erase_numeric_observation():
    reader, _ = state_with(interpretation="Synthetic comparison")
    req = requirement(method_details_required=True)
    card = plan_acquisition(reader.state, [req]).cards[0]
    assert card.next_actions == ("read_method_provenance",)
    assert card.observation_status == "obtained"


@pytest.mark.parametrize("reqs", [[], [requirement(), requirement()]])
def test_inventory_cannot_drop_or_duplicate_tasks(reqs):
    reader, _ = state_with()
    with pytest.raises(ValueError, match="coverage"):
        plan_acquisition(reader.state, reqs)


def test_search_uses_existing_public_query_contract_no_case_data():
    reader, _ = state_with()
    card = plan_acquisition(reader.state, [requirement()]).cards[0]
    suggestion = SearchSuggestion(
        assessment_id="item:one", topics=("rim", "slo"), purpose="measurement_method"
    )
    query = search_plan(card, suggestion, SearchProgress(), ("SOURCE:PROVIDER:EUROPEPMC",))
    assert query.query_plan_version == "5.0.0"
    assert query.providers[0].max_retries == 0 and query.providers[0].max_pages == 1
    assert reader.state.material.scope.case_code not in query.model_dump_json()
    with pytest.raises(ValueError):
        SearchSuggestion.model_validate({**suggestion.model_dump(), "patient_text": "private"})
    with pytest.raises(ValueError, match="scope"):
        search_plan(
            card,
            suggestion.model_copy(update={"topics": ("family_history",)}),
            SearchProgress(),
            ("SOURCE:PROVIDER:EUROPEPMC",),
        )


@pytest.mark.parametrize(
    "stop",
    [
        dict(rounds=2),
        dict(no_progress=2),
        dict(cancelled=True),
        dict(pending_unknown=True),
        dict(deadline_reached=True),
    ],
)
def test_search_bounded_and_unknown_never_retried(stop):
    reader, _ = state_with()
    card = plan_acquisition(reader.state, [requirement()]).cards[0]
    suggestion = SearchSuggestion(assessment_id="item:one", topics=("rim",), purpose="association")
    with pytest.raises(ValueError, match="acquisition_search_"):
        search_plan(card, suggestion, SearchProgress(**stop), ("SOURCE:PROVIDER:EUROPEPMC",))


def test_definition_reads_original_owners_before_and_after_preparation():
    reader, _ = state_with()
    old = reader.state.model_dump_json()
    d = validate(TaskDefinition, definition())
    calls = []

    def read_definition(target, reference):
        calls.append((target, reference))
        return d

    owner = SimpleNamespace(read_definition=read_definition)
    sources = SimpleNamespace(read_source_ref=lambda r: reader.state.material.sources[0])
    prepared = prepare_definition(
        reader.state,
        "item:one",
        definition_reference=d.reference,
        definition_owner=owner,
        source_owner=sources,
    )
    assert (
        len(calls) == 2 and prepared.policy.items[0].definition.interpretation_status == "installed"
    )
    assert prepared.material.scope.policy_checksum != reader.state.material.scope.policy_checksum
    assert prepared.policy.groups == reader.state.policy.groups
    assert reader.state.model_dump_json() == old
    assert prepared.material.scope.run_id == reader.state.material.scope.run_id


def test_source_withdrawal_during_prepare_is_not_accepted():
    reader, _ = state_with()
    d = validate(TaskDefinition, definition())
    calls = []

    def source_read(_):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("source_withdrawn")
        return reader.state.material.sources[0]

    with pytest.raises(ValueError, match="source_withdrawn"):
        prepare_definition(
            reader.state,
            "item:one",
            definition_reference=d.reference,
            definition_owner=SimpleNamespace(read_definition=lambda *_: d),
            source_owner=SimpleNamespace(read_source_ref=source_read),
        )


def test_targeted_vision_goes_through_original_provider_acceptance_and_readback():
    from agents.provider_agents.service import StructuredRoleAgent
    from packages.evidence_details.acceptance import ResearchAcceptance
    from packages.llm_provider import DeterministicFakeProvider
    from packages.llm_provider.models import VisionProposal
    from tests.agent_provider.support import Documents, budget_for, review_from_wire, submission

    reader, b = state_with(status="unknown", value=None, image=True)
    d = validate(TaskDefinition, definition())
    reader.state = prepare_definition(
        reader.state,
        "item:one",
        definition_reference=d.reference,
        definition_owner=SimpleNamespace(read_definition=lambda *_: d),
        source_owner=SimpleNamespace(read_source_ref=lambda _: reader.state.material.sources[0]),
    )
    before = reader.state.model_dump_json()
    img = reader.state.material.observations[0].image
    reader.state = prepare_visual_task(
        reader.state,
        "item:one",
        image=img,
        instruction="Inspect only focal rim continuity; locate any observed discontinuity.",
    )
    assert before != reader.state.model_dump_json()
    target = TaskTargetedReader(reader)
    candidate = VisionProposal(
        scope_checksum=digest(reader.state.material.scope),
        image_checksum=img.reference.checksum,
        preprocessing_checksum=img.preprocessing.checksum,
        location=img.location,
        status="observed",
        finding="Synthetic visible boundary is continuous.",
    )
    config = reader.acquisition_config
    provider = DeterministicFakeProvider(config, [candidate.model_dump(mode="json")])
    budget = budget_for(b, config)
    result = StructuredRoleAgent("vision", provider=provider, materials=target, budget=budget).call(
        reader.state.material.scope, deadline=b.deadline
    )
    wire = provider.transport.calls[0]
    payload = json.loads(wire["messages"][1]["content"][0]["text"])
    assert payload["vision_task"]["assessment_id"] == "item:one"
    assert "rim continuity" in payload["vision_task"]["definition"]["examination"]
    assert payload["schema_version"] == "targeted-vision-input/1.0.0"
    docs = Documents()
    service = ResearchAcceptance(owner=target, documents=docs, requests=budget)
    sub = submission(reader, result, budget)
    service.submit(sub, key="targeted")
    receipt = service.accept(key="targeted")
    assert receipt.status == "accepted"
    saved = service.read(key="targeted")
    assert saved[0]["observations"][0]["value"] == candidate.finding
    n = len(docs.audit)
    assert service.accept(key="targeted") == receipt and len(docs.audit) == n
    assert budget.used == 1 and len(provider.transport.calls) == 1

    # Install only the original domain's successor; never stitch a finding into
    # a review or reuse ratings bound to the pre-acquisition material.
    successor = service.successor_material(key="targeted")
    reader.state = reader.state.model_copy(
        update={
            "material": successor,
            "prompt_checksum": next(r.prompt_checksum for r in config.roles if r.role == "review"),
        }
    )
    review_provider = DeterministicFakeProvider(config, [review_from_wire])
    reviewed = StructuredRoleAgent(
        "review", provider=review_provider, materials=target, budget=budget
    ).call(successor.scope, deadline=b.deadline, round_id="A")
    service.submit(submission(reader, reviewed, budget), key="targeted-review")
    assert service.accept(key="targeted-review").status == "accepted"
    records, graph, scores, _ = service.read(key="targeted-review")
    assert records["review_tasks"]["tasks"][0]["observation"] == successor.observations[
        0
    ].reference.model_dump(mode="json")
    assert scores["scores"]["schema_version"] == "case-research-scoring/6.1.0"
    assert scores["scores"]["root"]["coverage"]["slo_image_research"]["acquisition"] == 1.0
    assert graph["clinical_patient_graph"] is False
    assert budget.used == 2 and len(review_provider.transport.calls) == 1


def test_targeted_vision_does_not_fan_out_to_other_observations():
    reader, _ = state_with(status="unknown", value=None, image=True)
    original = reader.state
    obs = original.material.observations[0]
    another = obs.model_copy(update={"reference": ref("observation", "OBS:TWO"), "image": None})
    policy = original.policy.items[0].model_copy(
        update={"id": "item:two", "observation_code": "OBS:TWO"}
    )
    state = _rebuild(original, items=(*original.policy.items, policy), observations=(obs, another))
    prepared = prepare_visual_task(
        state, "item:two", image=obs.image, instruction="Inspect only task two"
    )
    assert sum(o.image is not None for o in prepared.material.observations) == 1
    assert prepared.material.observations[0].image is None
    assert prepared.material.observations[0].reference != obs.reference
    assert all(o.status == "unknown" for o in prepared.material.observations)
    assert len(prepared.policy.items) == 2


def test_alias_only_proposes_reuse_and_does_not_copy_a_different_finding():
    reader, _ = state_with(status="ungradable", value="poor quality")
    before = reader.state.model_dump_json()
    candidate = observation_alias_candidate(reader.state, "item:one", "item:one")
    assert candidate["status"] == "ungradable" and not candidate["accepted"]
    assert reader.state.model_dump_json() == before


def test_dependency_invalidation_includes_shared_resource_and_unknown_rejected():
    reader, _ = state_with()
    assert affected_assessments(reader.state, resource_codes=("INPUT:ONE",)) == ("item:one",)
    assert affected_assessments(reader.state, source_codes=("SOURCE:ONE",)) == ("item:one",)
    with pytest.raises(ValueError, match="unknown_dependency"):
        affected_assessments(reader.state, observation_codes=("OBS:WRONG",))


def test_real_saved_case_has_full_inventory_and_no_old_ratings_in_vision():
    from apps.evidence_acquisition import audit, load_case
    from apps.vectorengine_round_materials import load_catalog

    state = load_case()
    result = audit(state, load_catalog())
    assert len(result["plan"]["cards"]) == 45
    assert result["counts"]["current_input_unavailable"] == 11
    assert result["real_requests"] == 0 and not result["new_scores"]
    assert len(result["source_gaps"]) == 5
    img = next(o.image for o in state.material.observations if o.image)
    prepared = prepare_visual_task(
        state,
        "criterion:RIM_NOTCH",
        image=img,
        instruction="检查盘沿局灶切迹，说明位置及可见程度。",
    )
    assert len(prepared.policy.items) == len(state.policy.items) == 46
    obs = next(o for o in prepared.material.observations if o.image)
    assert obs.reference.code == "observation:criterion:RIM_NOTCH"
    assert obs.status == "unknown" and obs.value is None


def test_html_escapes_node_text():
    from apps.evidence_acquisition import render_report

    reader, _ = state_with()
    result = dict(plan=plan_acquisition(reader.state, [requirement()]).model_dump(mode="json"))
    result = copy.deepcopy(result)
    result["plan"]["cards"][0]["node"] = "<script>bad()</script>"
    result["plan"]["cards"][0]["assessment_id"] = "<script>bad()</script>"
    html = render_report(result)
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_original_planner_identify_keeps_real_gaps_and_no_new_execution():
    from packages.evidence_acquisition.planner import AcquisitionPlannerOwner
    from packages.planner_control.models import Command, Outcome
    from packages.research_runtime.planner_steps import ResearchPlannerSteps

    reader, binding = state_with(status="unknown", value=None, image=True)
    original = ResearchPlannerSteps(snapshots=SimpleNamespace())
    owner = AcquisitionPlannerOwner(original, reader, lambda state: [requirement()])
    control = SimpleNamespace(binding=binding, graph=reader.state.material.scope.graph)
    pending = SimpleNamespace(
        command=Command(key="identify", expected_revision=0, action="identify")
    )
    result = owner.invoke(control, pending, cancel=None)
    assert isinstance(result, Outcome)
    assert result.started == 0
    assert any(g.target == "item:one" and g.kind == "product_demand" for g in result.gaps)
    assert all(g.capability_code is None for g in result.gaps)
    # No hiding other owner gaps (unavailable original ports remain visible).
    assert any(g.kind == "unavailable_port" for g in result.gaps)
    reader.withdrawn = True
    with pytest.raises(ValueError, match="source_withdrawn"):
        owner.invoke(control, pending, cancel=None)


def test_targeted_provider_bad_payload_rejected_before_request():
    from packages.llm_provider import DeterministicFakeProvider
    from packages.llm_provider.models import VisionProposal
    from packages.llm_provider.provider import ProviderError
    from tests.agent_provider.support import budget_for

    reader, binding = state_with(image=True)
    config = reader.acquisition_config
    provider = DeterministicFakeProvider(config, [])
    budget = budget_for(binding, config)
    with pytest.raises(ProviderError, match="scope"):
        provider.generate_structured(
            role="vision",
            input_payload=None,
            output_schema=VisionProposal,
            scope=reader.state.material.scope,
            budget=budget,
            deadline=binding.deadline,
        )
    assert budget.used == 0 and not provider.transport.calls


def test_definition_does_not_drop_other_sources_to_fit_material_limit():
    reader, _ = state_with()
    source = reader.state.material.sources[0]
    sources = tuple(
        source.model_copy(update={"reference": ref("source", f"SOURCE:EXTRA_{i}")})
        for i in range(15)
    )
    reader.state = _rebuild(reader.state, sources=(source, *sources))
    extra = source.model_copy(update={"reference": ref("source", "SOURCE:NEW")})
    d = validate(TaskDefinition, definition(source="SOURCE:NEW"))
    before = reader.state.model_dump_json()
    with pytest.raises(ValueError, match="acquisition_scoped_material_required"):
        prepare_definition(
            reader.state,
            "item:one",
            definition_reference=d.reference,
            definition_owner=SimpleNamespace(read_definition=lambda *_: d),
            source_owner=SimpleNamespace(read_source_ref=lambda _: extra),
        )
    assert reader.state.model_dump_json() == before
