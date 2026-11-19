"""Structured survey through original transport and acceptance; synthetic image only."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.provider_agents.service import StructuredRoleAgent
from packages.evaluation_preparation.dynamic_stages import TargetReader
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import DomainState, Submission, read_records
from packages.evidence_details.input_scope import ScopeSubmission
from packages.evidence_details.review_batches import read_submission
from packages.grounding.models import digest, validate
from packages.llm_provider import DeterministicFakeProvider
from packages.llm_provider.config import ProviderConfig, load_provider_config
from packages.llm_provider.input_scope import (
    PROMPT,
    RegionScope,
    ScopeTask,
    ScopeVisionPayload,
    ScopeVisionProposal,
)
from packages.object_store import checksum
from tests.agent_provider.support import Documents, budget_for, materials, ref


def fixture(*, response_change=None, binding=None):
    config = load_provider_config(environ={})
    raw = config.model_dump(mode="json")
    raw["image_capable"] = True
    role = next(r for r in raw["roles"] if r["role"] == "vision")
    role.update(
        prompt_version=PROMPT,
        prompt_checksum=digest(Path("configs/prompts/agents", PROMPT + ".txt").read_text()),
    )
    config = validate(ProviderConfig, raw)
    from agents.provider_agents.service import provider_locks
    from packages.planner_control.models import RankingPolicy
    from tests.planner.test_actions import ranked_binding

    binding = binding or ranked_binding(RankingPolicy())
    locks = {r.role: r for r in provider_locks(config)}
    binding = binding.model_copy(
        update={"locks": tuple(locks.get(r.role, r) for r in binding.locks)}
    )
    reader, binding = materials(binding, config=config, role="vision")
    data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nS8AAAAASUVORK5CYII="
    )
    reader.image_bytes = data
    raw = reader.state.model_dump(mode="json")
    raw["policy"]["items"][0]["method"] = "input_scope_survey"
    m = raw["material"]
    m["observations"][0].update(
        method="input_scope_survey",
        status="unknown",
        value=None,
        image=dict(
            reference=ref("image", "ASSET:SURVEY", checksum=checksum(data)).model_dump(mode="json"),
            media_type="image/png",
            width=1,
            height=1,
            coordinate_system="pixel_xy_top_left",
            preprocessing=ref("preprocessing", "POLICY:PIXEL").model_dump(mode="json"),
            location=[0, 0, 1, 1],
        ),
    )
    m["scope"]["policy_checksum"] = digest(raw["policy"])
    m["policy_ref"]["checksum"] = digest(raw["policy"])
    m["scope"].pop("material_checksum")
    m["scope"]["material_checksum"] = digest(m)
    reader.state = validate(DomainState, raw)
    scope = reader.state.material.scope
    task = ScopeTask(
        component_id="component:synthetic",
        component_checksum=digest("component"),
        input_checksum=scope.input_checksum,
        modality="color_fundus",
        representation="image",
        requested_regions=("optic_disc", "macula"),
        sampling_limits="Synthetic image only; no clinical observations.",
    )
    owner = TargetReader(reader)
    owner.input_scope_task = lambda current: task
    budget = budget_for(binding, config)

    def reply(wire):
        p = json.loads(wire["messages"][1]["content"][0]["text"])
        assert p["schema_version"] == "input-scope-vision-input/1.0.0"
        image = p["material"]["observations"][0]["image"]
        value = dict(
            schema_version="visual-input-scope-candidate/1.0.0",
            scope_checksum=digest(scope),
            image_checksum=image["reference"]["checksum"],
            preprocessing_checksum=image["preprocessing"]["checksum"],
            location=image["location"],
            status="observed",
            finding="Synthetic scope fixture only.",
            regions=[
                dict(
                    region="optic_disc",
                    visibility="partly_visible",
                    coverage=0.5,
                    clarity=0.7,
                    finding="Synthetic visible arc",
                    limitation="Synthetic lower arc obscured",
                ),
                dict(
                    region="macula",
                    visibility="not_identified",
                    coverage=0.0,
                    clarity=0.0,
                    finding="Not identified",
                    limitation="Not a claim of absence",
                ),
            ],
        )
        if response_change:
            response_change(value)
        return value

    provider = DeterministicFakeProvider(config, [reply])
    agent = StructuredRoleAgent("vision", provider=provider, materials=owner, budget=budget)
    docs = Documents()
    acceptor = ResearchAcceptance(owner=owner, documents=docs, requests=budget)
    return SimpleNamespace(**locals())


def generate(f):
    c = f.agent.call(f.scope, deadline=f.binding.deadline)
    r = f.budget.records()[-1]
    state = f.reader.state
    return ScopeSubmission(
        scope=f.scope,
        actor="PLANNER",
        correlation="SOURCE:AGENT_RESEARCH",
        request_id=r.request_id,
        prompt_checksum=r.prompt_checksum,
        config_checksum=r.config_checksum,
        observation_checksums=tuple(o.reference for o in state.material.observations),
        source_checksums=tuple(s.reference for s in state.material.sources),
        candidate=c,
    )


def test_original_provider_and_acceptance_preserve_scope_without_a_disease_score():
    f = fixture()
    sub = generate(f)
    f.acceptor.submit(sub, key="scope")
    receipt = f.acceptor.accept(key="scope")
    assert receipt.status == "accepted" and receipt.scores_checksum is None
    record, graph, scored, saved_receipt = f.acceptor.read(key="scope")
    assert saved_receipt == receipt.model_dump(mode="json")
    saved = read_records(record)
    assert saved.schema_version == "case-input-scope-observation/1.0.0"
    assert saved.scope_task == f.task and not saved.interpretations
    assert not saved.invalidated_assessment_ids
    assert graph["scores_checksum"] is None
    assert scored["scores"] is None
    observation = json.loads(saved.observations[0].value)
    assert observation["regions"][1]["visibility"] == "not_identified"
    assert observation["disease_direction"] == "not_assessed"
    assert observation["physical_scale"] == "not_established_by_visual_survey"
    n = len(f.docs.audit)
    assert f.acceptor.accept(key="scope") == receipt
    assert len(f.docs.audit) == n and len(f.budget.records()) == 1
    assert read_submission(sub) == sub
    with pytest.raises(ValueError):
        validate(Submission, sub.model_dump(mode="json"))


@pytest.mark.parametrize("change", ["owner", "input", "component", "source", "image"])
def test_current_owner_change_rejects_before_acceptance_and_never_resends(change):
    f = fixture()
    sub = generate(f)
    if change == "owner":
        del f.owner.input_scope_task
    elif change in {"input", "component"}:
        field = "input_checksum" if change == "input" else "component_checksum"
        changed = f.task.model_copy(update={field: digest("changed")})
        f.owner.input_scope_task = lambda _: changed
    elif change == "image":
        f.reader.image_bytes = b"changed"
    else:
        f.reader.withdrawn = True
    if change == "source":
        with pytest.raises(ValueError, match="source_withdrawn"):
            f.acceptor.submit(sub, key="changed")
    else:
        result = f.acceptor.submit(sub, key="changed")
        assert result.status == "rejected"
    assert len(f.budget.records()) == 1
    assert not f.docs.read("accepted:changed")


def test_scope_response_region_inventory_is_bound_and_not_inferred_from_words():
    from packages.llm_provider.provider import ProviderError

    f = fixture(response_change=lambda v: v["regions"][0].update(region="other"))
    with pytest.raises(ProviderError):
        generate(f)
    assert len(f.provider.transport.calls) == 1
    assert not f.docs.saved


def test_visibility_constraints_and_original_length_bound():
    raw = dict(
        region="x",
        visibility="not_identified",
        coverage=0.8,
        clarity=0.5,
        finding="x",
        limitation="x",
    )
    with pytest.raises(ValueError, match="input_scope_unidentified_coverage"):
        validate(RegionScope, raw)
    raw.update(visibility="visible", coverage=0)
    with pytest.raises(ValueError, match="input_scope_visible_requires_coverage"):
        validate(RegionScope, raw)
    raw.update(coverage=float("nan"))
    with pytest.raises(ValueError):
        validate(RegionScope, raw)


def test_new_schema_is_normalizable_and_old_submission_schema_does_not_change():
    from packages.llm_provider.transport import strict_schema
    from packages.llm_provider.wire_compatibility import normalize

    normalize(strict_schema(ScopeVisionProposal.model_json_schema()))
    assert "ScopeVisionProposal" not in Submission.model_json_schema()["$defs"]
    f = fixture()
    from packages.evidence_details.input_scope import payload

    p = payload(f.owner, f.reader.state)
    assert isinstance(p, ScopeVisionPayload)


def test_original_research_review_orchestrates_scope_and_expert_then_exact_recovery():
    from agents.expert_agent.reporting import read_research_report, render_research_html
    from tests.expert_agent.support import expert_reply, run, setup

    f = fixture()
    f.reader.input_scope_task = f.owner.input_scope_task
    f.reader.vision_task = f.owner.vision_task
    g = setup(
        [f.reply, expert_reply()],
        reader=f.reader,
        binding=f.binding,
        config=f.config,
        role="vision",
        facts={"required": True},
    )
    parent = g.reader.current()
    result = run(g, role="vision")
    assert result["status"] == "accepted"
    report = read_research_report(g.acceptance, key=result["acceptance_key"], parent=parent)
    assert report["scores"] is None
    assert report["records"]["schema_version"] == "case-input-scope-observation/1.0.0"
    assert "optic_disc" in render_research_html(report)
    assert run(g, role="vision") == result and g.budget.used == 2
    wire = g.provider.transport.calls[-1]
    payload_ = json.loads(wire["messages"][1]["content"][0]["text"])
    assert payload_["schema_version"] == "input-scope-expert-input/1.0.0"
    assert payload_["expert_review"]["candidate"]["regions"][0]["region"] == "optic_disc"
