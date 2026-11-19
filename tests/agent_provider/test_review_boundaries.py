"""Independent synthetic counterexamples; no network, patient, GPU or production writes."""

import copy
import json

import pytest

from agents.provider_agents import StructuredRoleAgent
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_scoring.accepted import build_accepted
from packages.evidence_scoring.continuous_models import ContinuousReview
from packages.grounding.models import digest, validate
from packages.llm_provider import DeterministicFakeProvider
from packages.llm_provider.config import load_provider_config
from packages.llm_provider.models import Material, ReviewProposal
from tests.agent_provider.support import Documents, budget_for, materials, ref, review, submission


def rebind(reader, *, policy=None, observations=None, sources=None):
    state = reader.state
    policy = policy or state.policy
    raw = state.material.model_dump(mode="json")
    if observations is not None:
        raw["observations"] = observations
    if sources is not None:
        raw["sources"] = sources
    raw["scope"]["policy_checksum"] = digest(policy)
    raw["policy_ref"]["checksum"] = digest(policy)
    raw["scope"].pop("material_checksum")
    raw["scope"]["material_checksum"] = digest(raw)
    reader.state = validate(
        DomainState,
        state.model_copy(update={"policy": policy, "material": validate(Material, raw)}),
    )


def round_trip(reader, binding, candidate):
    config = load_provider_config(environ={})
    provider = DeterministicFakeProvider(
        config, [candidate if callable(candidate) else candidate.model_dump(mode="json")]
    )
    budget = budget_for(binding, config)
    result = StructuredRoleAgent("review", provider=provider, materials=reader, budget=budget).call(
        reader.state.material.scope, deadline=binding.deadline, round_id="A"
    )
    service = ResearchAcceptance(owner=reader, documents=Documents(), requests=budget)
    submitted = service.submit(submission(reader, result, budget), key="review-boundary")
    receipt = service.accept(key="review-boundary") if isinstance(submitted, dict) else submitted
    return service, receipt, provider


def test_obtained_assessed_positive_control():
    reader, b = materials()
    service, receipt, _ = round_trip(reader, b, review(reader.state.material.scope))
    assert receipt.status == "accepted"
    coverage = service.read(key="review-boundary")[2]["scores"]["root"]["coverage"]
    assert coverage["slo_image_research"]["acquisition"] == 1.0
    assert coverage["slo_image_research"]["interpretation"] == 1.0


def test_unknown_observation_cannot_fill_acquisition_slot():
    reader, b = materials()
    observations = reader.state.material.model_dump(mode="json")["observations"]
    observations[0].update(status="unknown", value=None)
    rebind(reader, observations=observations)
    raw = review(reader.state.material.scope).model_dump(mode="json")
    raw["items"][0].update(
        status="reference_missing",
        judgment={
            "value": None,
            "lower": None,
            "upper": None,
            "reason": "Synthetic unknown observation, no acquired value.",
            "missing_reason": "No acquired measurement",
            "method": "not_assessed",
        },
    )
    candidate = validate(ReviewProposal, raw)
    service, receipt, _ = round_trip(reader, b, candidate)
    if receipt.status == "rejected":
        return
    scores = service.read(key="review-boundary")[2]["scores"]
    assert scores["root"]["coverage"]["slo_image_research"]["acquisition"] == 0.0, scores["root"]


def test_fixed_model_judgment_bounds_cannot_be_changed_by_reviewer():
    reader, b = materials()
    fixed = build_accepted(reader.state, review(reader.state.material.scope))["reviews"][0]
    fixed.update(role="model", kind="model")
    fixed["judgment"]["method"] = "fixed_model_margin"
    fixed = validate(ContinuousReview, fixed)
    item = reader.state.policy.items[0].model_copy(
        update={
            "role": "model",
            "kind": "model",
            "fixed_observation_ref": reader.state.material.observations[0].reference,
            "fixed_model_row": fixed,
        }
    )
    rebind(reader, policy=reader.state.policy.model_copy(update={"items": (item,)}))
    raw = review(reader.state.material.scope).model_dump(mode="json")
    raw["items"][0]["judgment"].update(method="fixed_model_margin", lower=-0.9, upper=0.9)
    candidate = validate(ReviewProposal, raw)
    with pytest.raises(ValueError, match="fixed_model_mapping_immutable"):
        build_accepted(reader.state, candidate)
    # Fixed-only policy is deliberately not sent to a language model.
    with pytest.raises(ValueError, match="scope"):
        round_trip(reader, b, candidate)


def two_item_wire(swap, reverse=False):
    reader, b = materials()
    observations = reader.state.material.model_dump(mode="json")["observations"]
    observations.append(copy.deepcopy(observations[0]))
    observations[1]["reference"] = ref("observation", "OBS:TWO").model_dump(mode="json")
    observations[1]["value"] = "A different synthetic observation"
    from packages.llm_provider.models import TaskDefinition
    from tests.agent_provider.support import definition

    sources = reader.state.material.model_dump(mode="json")["sources"]
    sources.append(copy.deepcopy(sources[0]))
    sources[1]["reference"] = ref("source", "SOURCE:TWO").model_dump(mode="json")
    sources[1]["text"] = "Second synthetic reference, different permitted comparison."
    items = (
        reader.state.policy.items[0].model_copy(
            update={"id": "item:one", "observation_code": "OBS:TWO" if swap else "OBS:ONE"}
        ),
        reader.state.policy.items[0].model_copy(
            update={
                "id": "item:two",
                "observation_code": "OBS:ONE" if swap else "OBS:TWO",
                "source_codes": ("SOURCE:TWO",),
                "definition": validate(TaskDefinition, definition("SOURCE:TWO")),
            }
        ),
    )
    policy = reader.state.policy.model_copy(
        update={"items": tuple(reversed(items)) if reverse else items}
    )
    rebind(
        reader,
        policy=policy,
        observations=list(reversed(observations)) if reverse else observations,
        sources=sources,
    )

    def answer_from_wire(body):
        payload = json.loads(body["messages"][1]["content"][0]["text"])
        # Ratings are synthetic, but every mapping comes exclusively from actual wire data.
        raw = review(validate(Material, payload["material"]).scope).model_dump(mode="json")
        item = raw["items"][0]
        raw["items"] = []
        for task in payload["review_tasks"]["tasks"]:
            row = copy.deepcopy(item)
            row.update(
                assessment_id=task["assessment_id"],
                observation_citations=[task["observation"]["code"]],
                reference_citations=[r["code"] for r in task["allowed_sources"]],
            )
            raw["items"].append(row)
        return raw

    _, receipt, provider = round_trip(reader, b, answer_from_wire)
    assert receipt.status == "accepted"
    wire = provider.transport.calls[0]
    payload = json.loads(wire["messages"][1]["content"][0]["text"])
    return payload


def without_opaque_hashes(value):
    if isinstance(value, dict):
        return {k: without_opaque_hashes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [without_opaque_hashes(v) for v in value]
    if isinstance(value, str) and value.startswith("sha256:"):
        return "<opaque-checksum>"
    return value


def test_review_wire_exposes_task_to_observation_mapping_not_only_hash():
    first, second = two_item_wire(False), two_item_wire(True)
    assert without_opaque_hashes(first) != without_opaque_hashes(second), (
        "Swapping the installed item-to-observation mapping changes only opaque hashes; "
        "the model receives identical task text, item IDs, observations and sources."
    )
