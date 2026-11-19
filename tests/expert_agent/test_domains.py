import json

import pytest

from agents.expert_agent import ResearchReview
from packages.evidence_details.successor_owner import ResearchSuccessorOwner
from packages.evidence_scoring.accepted import build_accepted
from packages.evidence_scoring.continuous_models import ContinuousReview
from packages.grounding.models import digest, validate
from packages.llm_provider.config import ProviderConfig, load_provider_config
from packages.llm_provider.models import Material, TaskDefinition
from packages.object_store import checksum
from tests.agent_provider.support import definition, ref, review, review_from_wire
from tests.agent_provider.test_review_boundaries import rebind
from tests.agent_provider.test_review_repair import missing_review, observation_state
from tests.expert_agent.support import protocol, run, setup
from tests.tool_admission.support import png


@pytest.mark.parametrize(
    "status,available",
    [("unknown", False), ("missing", False), ("ungradable", False), ("obtained", True)],
)
def test_rounds_unknown_and_missing_reference_keep_original_coverage(status, available):
    f = setup()
    reader = f.reader
    observation_state(
        reader, status, "Synthetic fact" if available or status == "ungradable" else None
    )
    if available:
        p = reader.state.policy.items[0].model_copy(
            update={"definition": validate(TaskDefinition, definition(interpretation=None))}
        )
        rebind(reader, policy=reader.state.policy.model_copy(update={"items": (p,)}))

    def reply(body):
        payload = json.loads(body["messages"][1]["content"][0]["text"])
        result = missing_review(
            reader.state.material.scope,
            "reference_missing"
            if available
            else "observation_ungradable"
            if status == "ungradable"
            else "input_missing",
            available,
        ).model_dump(mode="json")
        result["round"] = payload["round"]
        return result

    f = setup([reply] * 3, reader=reader, binding=f.binding)
    parent = f.reader.state
    result = run(f, role="review", abc=True)
    assert result["status"] == "accepted"
    saved = f.acceptance.read(key=result["acceptance_key"])
    assert saved[2]["scores"]["root"]["coverage"]["slo_image_research"] == dict(
        denominator=1.0, acquisition=float(available), interpretation=0.0
    )
    assert saved[2]["scores"]["reviews"][0]["judgment"]["value"] is None
    assert parent.material.scope != f.reader.state.material.scope


def test_fixed_model_domain_backfill_abc_not_in_any_wire():
    f = setup()
    reader = f.reader
    fixed = build_accepted(reader.state, review(reader.state.material.scope))["reviews"][0]
    fixed.update(id="item:model", kind="model", role="model", title="SECRET_OLD_MODEL_TITLE")
    fixed["judgment"]["method"] = "fixed_model_margin"
    fixed = validate(ContinuousReview, fixed)
    p = reader.state.policy.items[0].model_copy(
        update=dict(
            id="item:model",
            role="model",
            kind="model",
            fixed_model_row=fixed,
            fixed_observation_ref=reader.state.material.observations[0].reference,
        )
    )
    rebind(
        reader,
        policy=reader.state.policy.model_copy(update={"items": (*reader.state.policy.items, p)}),
    )
    f = setup([review_from_wire] * 3, reader=reader, binding=f.binding)
    result = run(f, role="review", abc=True)
    assert result["status"] == "accepted"
    assert "SECRET_OLD_MODEL_TITLE" not in json.dumps(f.provider.transport.calls)
    assert "fixed_model_row" not in json.dumps(f.provider.transport.calls)
    scores = f.acceptance.read(key=result["acceptance_key"])[2]["scores"]
    assert next(r for r in scores["reviews"] if r["id"] == "item:model") == fixed.model_dump(
        mode="json"
    )


def test_source_port_uses_domain_and_preserves_research_purpose():
    def source_reply(body):
        data = json.loads(body["messages"][1]["content"][0]["text"])["material"]
        return dict(
            scope_checksum=digest(data["scope"]),
            source_code=data["sources"][0]["reference"]["code"],
            quote=data["sources"][0]["text"],
            target=data["goal_code"],
            relation="research_association",
            missing=[],
            reason="Synthetic relation",
        )

    f = setup([source_reply], role="source")
    result = run(f, role="source")
    assert result["status"] == "accepted"
    saved = f.acceptance.read(key=result["acceptance_key"])
    assert saved[2]["scores"] is None
    assert saved[1]["accepted_validations"] == []
    assert f.reader.state.material.sources[0].purpose == "research_material"


def test_visual_unknown_install_then_new_review_uses_new_material():
    config = load_provider_config(environ={})
    config = validate(ProviderConfig, config.model_copy(update={"image_capable": True}))
    f = setup(config=config, role="vision")
    data = png()
    f.reader.image_bytes = data
    raw = f.reader.state.material.model_dump(mode="json")
    raw["observations"][0]["image"] = dict(
        reference=ref("image", "ASSET:IMAGE", checksum=checksum(data)).model_dump(mode="json"),
        media_type="image/png",
        width=3,
        height=2,
        coordinate_system="pixel_xy_top_left",
        preprocessing=ref("preprocessing", "POLICY:PIXEL").model_dump(mode="json"),
        location=[0, 0, 3, 2],
    )
    raw["scope"].pop("material_checksum")
    raw["scope"]["material_checksum"] = digest(raw)
    f.reader.state = f.reader.state.model_copy(update={"material": validate(Material, raw)})

    def vision(body):
        payload = json.loads(body["messages"][1]["content"][0]["text"])
        image = payload["material"]["observations"][0]["image"]
        return dict(
            scope_checksum=digest(payload["material"]["scope"]),
            image_checksum=image["reference"]["checksum"],
            preprocessing_checksum=image["preprocessing"]["checksum"],
            location=image["location"],
            status="unknown",
            finding="Synthetic unknown visual observation",
        )

    f = setup([vision], config=config, role="vision", reader=f.reader, binding=f.binding)
    old = f.reader.state
    result = run(f, role="vision")
    assert result["status"] == "accepted"
    saved = f.acceptance.read(key=result["acceptance_key"])
    assert saved[0]["invalidated_assessment_ids"] == ["item:one"]
    assert f.reader.state.material.observations[0].status == "unknown"
    assert old.material.observations[0].status == "obtained"
    assert saved[2]["scores"] is None
    assert (
        f.reader.review_template(f.reader.state.material.scope).tasks[0].observation
        == f.reader.state.material.observations[0].reference
    )
    from agents.expert_agent.actions import ResearchActionSource
    from packages.planner_control.models import Control, Phase, RankingPolicy

    f.reader.state = f.reader.state.model_copy(
        update={
            "prompt_checksum": next(r.prompt_checksum for r in config.roles if r.role == "review")
        }
    )
    control = Control(
        binding=f.binding,
        graph=f.reader.state.material.scope.graph,
        phase=Phase.REVIEW,
        candidates=f.source.choices,
    )
    frame = f.frame.model_copy(
        update=dict(
            event=ref("review_control", "review:second"),
            scope=f.reader.state.material.scope,
            actions=ResearchActionSource(f.source, f.reader).read(control, RankingPolicy()),
        )
    )
    f.frame = f.context.frame = frame
    f.provider.transport.replies.append(review_from_wire)
    f.context.checkpoints.clear()
    next_result = run(f, role="review")
    assert next_result["status"] == "accepted"
    assert f.budget.used == 2
    scores = f.acceptance.read(key=next_result["acceptance_key"])[2]["scores"]
    assert scores["root"]["coverage"]["slo_image_research"]["acquisition"] == 0.0
    assert scores["reviews"][0]["judgment"]["value"] is None
    assert f.acceptance.read(key=result["acceptance_key"]) == saved


def test_persistent_owner_install_and_reconstruction_verifies_dependencies():
    f = setup([review_from_wire])
    initial = f.reader.state
    withdrawn = []

    def verify(state):
        if withdrawn or state.material.sources != initial.material.sources:
            raise ValueError("withdrawn")

    owner = ResearchSuccessorOwner(
        initial=initial,
        documents=f.documents,
        verify_dependencies=verify,
        read_image=lambda _: b"",
        config=f.provider.config,
        role="review",
    )
    f.acceptance.owner = owner
    f.context.reader = owner
    f.context.install = owner.install
    for agent in (f.ports.trace, f.ports.expert, f.ports.source, f.ports.vision, f.ports.review):
        agent.materials = owner
    f.service = ResearchReview(
        acceptance=f.acceptance, ports=f.ports, context=f.context, protocol=protocol()
    )
    result = run(f, role="review")
    assert result["status"] == "accepted"
    restarted = ResearchSuccessorOwner(
        initial=initial,
        documents=f.documents,
        verify_dependencies=verify,
        read_image=lambda _: b"",
        config=f.provider.config,
        role="review",
    )
    assert restarted.current() == owner.current()
    assert restarted.current().material.scope.graph.code == result["graph"]["code"]
    withdrawn.append(True)
    with pytest.raises(ValueError, match="withdrawn"):
        restarted.current()
