"""New candidate admission and old-record preservation; no medical review claim."""

import pytest

from packages.evidence_details.acceptance import ResearchAcceptance
from packages.grounding.models import digest, validate
from tests.agent_provider.support import Documents, generated, ref


def setup():
    reader, b, budget, sub = generated()
    docs = Documents()
    return ResearchAcceptance(owner=reader, documents=docs, requests=budget), reader, docs, sub


def test_submit_does_not_accept_new_bundle_exact_retry_and_old_unchanged():
    service, reader, docs, sub = setup()
    original = reader.state.model_dump_json()
    r = service.submit(sub, key="one")
    assert r["status"] == "submitted_not_accepted" and service.read(key="one") is None
    result = service.accept(key="one")
    assert result.status == "accepted" and result.graph.version == "case-evidence-graph/2.0.0"
    saved = service.read(key="one")
    assert saved[0]["schema_version"] == "case-observation-interpretation/2.1.0"
    assert saved[2]["scores"]["schema_version"] == "case-research-scoring/6.1.0"
    assert saved[2]["scores"]["root"]["summary"]["confidence"]["value"] is not None
    assert reader.state.model_dump_json() == original
    n = len(docs.audit)
    assert service.accept(key="one") == result and service.submit(sub, key="one") == r
    assert len(docs.audit) == n
    assert service.submit(sub, key="two").status == "rejected"


@pytest.mark.parametrize(
    "change",
    [
        "case",
        "run",
        "graph",
        "observations",
        "sources",
        "policy",
        "prompt",
        "request",
        "config",
        "extra",
        "accepted",
    ],
)
def test_scope_versions_and_injected_accept_rejected(change):
    service, reader, docs, sub = setup()
    raw = sub.model_dump(mode="json")
    if change == "case":
        raw["scope"]["case_code"] = "CASE:OTHER"
    if change == "run":
        raw["scope"]["run_id"] = "RUN:OTHER"
    if change == "graph":
        raw["scope"]["graph"]["checksum"] = digest("old")
    if change == "observations":
        raw["observation_checksums"][0]["checksum"] = digest("old")
    if change == "sources":
        raw["source_checksums"][0]["checksum"] = digest("old")
    if change == "policy":
        raw["scope"]["policy_checksum"] = digest("old")
    if change == "prompt":
        raw["prompt_checksum"] = digest("old")
    if change == "config":
        raw["config_checksum"] = digest("old")
    if change == "request":
        raw["request_id"] = "request:other"
    if change == "extra":
        raw["GraphDelta"] = {"execute": "bad"}
    if change == "accepted":
        raw["status"] = "accepted"
    r = service.submit(raw, key=change)
    assert r.status == "rejected" and docs.audit
    with pytest.raises(ValueError, match="submitted"):
        service.accept(key=change)


def test_same_key_different_content_audited_and_pending_stale_review():
    service, reader, docs, sub = setup()
    service.submit(sub, key="one")
    other = sub.model_copy(update={"prompt_checksum": digest("changed")})
    assert service.submit(other, key="one").reason == "idempotency_conflict"
    reader.state = reader.state.model_copy(update={"laterality": "LEFT"})
    assert service.accept(key="one").status == "rejected"
    assert service.read(key="one")[-1]["status"] == "rejected"


@pytest.mark.parametrize("failure", ["object", "transaction", "lost_reply"])
def test_failures_and_original_key_recovery(failure):
    service, reader, docs, sub = setup()
    service.submit(sub, key="one")
    docs.fail = failure
    with pytest.raises((OSError, ValueError)):
        service.accept(key="one")
    docs.fail = None
    assert service.accept(key="one").status == "accepted"
    assert docs.audit.count("accepted:one") == 1


@pytest.mark.parametrize(
    "change",
    ["method", "unit", "laterality", "applicability", "dependency", "correlation", "model"],
)
def test_domain_owner_policy_and_observation_guards(change):
    # Submission checks the installed current policy and request locks, not candidate self-report.
    service, reader, docs, sub = setup()
    service.submit(sub, key="one")
    state = reader.state
    if change in {"method", "unit"}:
        obs = state.material.observations[0].model_copy(update={change: "changed"})
        reader.state = state.model_copy(
            update={"material": state.material.model_copy(update={"observations": (obs,)})}
        )
    elif change == "laterality":
        reader.state = state.model_copy(update={"laterality": "RIGHT"})
    elif change == "dependency":
        reader.state = state.model_copy(update={"resources": (ref("input", "INPUT:OTHER"),)})
    else:
        p = state.policy.items[0].model_copy(
            update={"applicable": False}
            if change == "applicability"
            else {"correlation_group": "different"}
            if change == "correlation"
            else {"role": "model"}
        )
        reader.state = state.model_copy(
            update={"policy": state.policy.model_copy(update={"items": (p,)})}
        )
    try:
        result = service.accept(key="one")
    except ValueError:
        pass  # Owner refuses the invalid state before anything can be saved.
    else:
        assert result.status == "rejected"
    assert (
        docs.read("accepted:one") is None or docs.read("accepted:one")[-1]["status"] == "rejected"
    )


def test_source_role_passes_04_03_then_07_without_formal_graph():
    from agents.provider_agents import StructuredRoleAgent
    from packages.knowledge_claims.research_candidates import submit_research_candidate
    from packages.llm_provider import DeterministicFakeProvider
    from packages.llm_provider.config import load_provider_config
    from packages.llm_provider.models import SourceProposal
    from tests.agent_provider.support import budget_for, materials, submission

    config = load_provider_config(environ={})
    reader, b = materials(role="source", config=config)
    c = SourceProposal(
        scope_checksum=digest(reader.state.material.scope),
        source_code="SOURCE:ONE",
        quote=reader.state.material.sources[0].text,
        target="GOAL:SYNTHETIC",
        relation="requires",
        missing=(),
        reason="Synthetic requirement",
    )
    p = DeterministicFakeProvider(config, [c.model_dump(mode="json")])
    budget = budget_for(b, config)
    result = StructuredRoleAgent("source", provider=p, materials=reader, budget=budget).call(
        reader.state.material.scope, deadline=b.deadline
    )
    s = ResearchAcceptance(owner=reader, documents=Documents(), requests=budget)
    assert (
        submit_research_candidate(s, submission(reader, result, budget), key="source")["status"]
        == "submitted_not_accepted"
    )
    receipt = s.accept(key="source")
    assert receipt.status == "accepted" and receipt.scores_checksum is None
    assert s.read(key="source")[1]["accepted_validations"] == []
    assert s.read(key="source")[1]["clinical_patient_graph"] is False


def test_fixed_model_mapping_not_changed_by_review():
    from packages.evidence_details.acceptance_models import DomainState
    from packages.evidence_scoring.accepted import build_accepted
    from packages.evidence_scoring.continuous_models import ContinuousReview
    from packages.llm_provider.models import Material
    from tests.agent_provider.support import materials, review

    reader, _ = materials()
    state = reader.state
    c = review(state.material.scope)
    row = build_accepted(state, c)["reviews"][0]
    row.update(role="model", kind="model")
    row["judgment"]["method"] = "fixed_model_margin"
    fixed = ContinuousReview.model_validate(row)
    policy = state.policy.model_copy(
        update={
            "items": (
                state.policy.items[0].model_copy(
                    update={
                        "role": "model",
                        "kind": "model",
                        "fixed_observation_ref": reader.state.material.observations[0].reference,
                        "fixed_model_row": fixed,
                    }
                ),
            )
        }
    )
    m = state.material.model_dump(mode="json")
    m["scope"]["policy_checksum"] = digest(policy)
    m["policy_ref"]["checksum"] = digest(policy)
    m["scope"].pop("material_checksum")
    m["scope"]["material_checksum"] = digest(m)
    state = validate(
        DomainState, state.model_copy(update={"policy": policy, "material": validate(Material, m)})
    )
    c = c.model_copy(
        update={
            "scope_checksum": digest(state.material.scope),
            "items": (c.items[0].model_copy(update={"judgment": fixed.judgment}),),
        }
    )
    assert build_accepted(state, c)["reviews"][0]["judgment"] == fixed.judgment.model_dump(
        mode="json"
    )
    changed = c.model_copy(
        update={
            "items": (
                c.items[0].model_copy(
                    update={"judgment": fixed.judgment.model_copy(update={"value": 0.4})}
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="fixed_model_mapping"):
        build_accepted(state, changed)


def test_score_v6_explicit_dispatch_and_unknown_input():
    from packages.evidence_scoring.api import build
    from packages.evidence_scoring.api import validate as validate_scores
    from tests.agent_provider.support import materials, review

    reader, _ = materials()
    state = reader.state
    c = review(state.material.scope)
    base = dict(state=state, candidate=c)
    result = build(base, version="6.1.0")
    assert validate_scores(base, result) == result
    with pytest.raises(ValueError, match="explicit"):
        build({}, version="6.1.0")
    # A missing observation cannot acquire a disease direction from an LLM answer.
    observation = state.material.observations[0].model_copy(
        update={"status": "missing", "value": None}
    )
    changed = state.model_copy(
        update={"material": state.material.model_copy(update={"observations": (observation,)})}
    )
    from packages.evidence_scoring.accepted import build_accepted

    with pytest.raises(ValueError, match="material_checksum"):
        build_accepted(changed, c)


def test_vision_acceptance_new_observation_invalidates_old_review():
    from agents.provider_agents import StructuredRoleAgent
    from packages.llm_provider import DeterministicFakeProvider
    from packages.llm_provider.config import ProviderConfig, load_provider_config
    from packages.llm_provider.models import Material, VisionProposal
    from packages.object_store import checksum
    from tests.agent_provider.support import budget_for, materials, submission
    from tests.tool_admission.support import png

    config = load_provider_config(environ={})
    config = validate(ProviderConfig, config.model_copy(update={"image_capable": True}))
    reader, b = materials(role="vision", config=config)
    data = png()
    reader.image_bytes = data
    raw = reader.state.material.model_dump(mode="json")
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
    reader.state = reader.state.model_copy(update={"material": validate(Material, raw)})
    m = reader.state.material
    image = m.observations[0].image
    candidate = VisionProposal(
        scope_checksum=digest(m.scope),
        image_checksum=checksum(data),
        preprocessing_checksum=image.preprocessing.checksum,
        location=(0, 0, 3, 2),
        status="unknown",
        finding="Unknown synthetic image observation",
    )
    provider = DeterministicFakeProvider(config, [candidate.model_dump(mode="json")])
    budget = budget_for(b, config)
    result = StructuredRoleAgent("vision", provider=provider, materials=reader, budget=budget).call(
        m.scope, deadline=b.deadline
    )
    service = ResearchAcceptance(owner=reader, documents=Documents(), requests=budget)
    sub = submission(reader, result, budget)
    service.submit(sub, key="vision")
    receipt = service.accept(key="vision")
    saved = service.read(key="vision")
    assert receipt.status == "accepted" and receipt.scores_checksum is None
    assert (
        saved[0]["observations"][0]["reference"]["checksum"] != m.observations[0].reference.checksum
    )
    assert saved[0]["observations"][0]["status"] == "unknown"
    assert saved[0]["interpretations"] == [] and saved[0]["invalidated_assessment_ids"] == [
        "item:one"
    ]
    assert m.observations[0].status == "obtained"
    with pytest.raises(ValueError, match="consumed"):
        service.revalidate(key="vision")

    # Explicit test installation of saved successors, not 08D business orchestration.
    from tests.agent_provider.support import review_from_wire

    history = service.read(key="vision")
    successor = service.successor_material(key="vision")
    reader.state = reader.state.model_copy(
        update={
            "material": successor,
            "prompt_checksum": next(r.prompt_checksum for r in config.roles if r.role == "review"),
        }
    )
    provider = DeterministicFakeProvider(config, [review_from_wire])
    result = StructuredRoleAgent("review", provider=provider, materials=reader, budget=budget).call(
        successor.scope, deadline=b.deadline, round_id="A"
    )
    assert (
        service.submit(submission(reader, result, budget), key="unknown")["status"]
        == "submitted_not_accepted"
    )
    assert service.accept(key="unknown").status == "accepted"
    unknown_saved = service.read(key="unknown")
    assert (
        unknown_saved[2]["scores"]["root"]["coverage"]["slo_image_research"]["acquisition"] == 0.0
    )
    successor = service.successor_material(key="unknown")
    reader.state = reader.state.model_copy(
        update={
            "material": successor,
            "prompt_checksum": next(r.prompt_checksum for r in config.roles if r.role == "vision"),
        }
    )
    candidate = candidate.model_copy(
        update={
            "scope_checksum": digest(successor.scope),
            "status": "observed",
            "finding": "New synthetic acquired feature",
        }
    )
    provider = DeterministicFakeProvider(config, [candidate.model_dump(mode="json")])
    result = StructuredRoleAgent("vision", provider=provider, materials=reader, budget=budget).call(
        successor.scope, deadline=b.deadline
    )
    assert (
        service.submit(submission(reader, result, budget), key="obtained")["status"]
        == "submitted_not_accepted"
    )
    assert service.accept(key="obtained").status == "accepted"
    successor = service.successor_material(key="obtained")
    reader.state = reader.state.model_copy(
        update={
            "material": successor,
            "prompt_checksum": next(r.prompt_checksum for r in config.roles if r.role == "review"),
        }
    )
    provider = DeterministicFakeProvider(config, [review_from_wire])
    result = StructuredRoleAgent("review", provider=provider, materials=reader, budget=budget).call(
        successor.scope, deadline=b.deadline, round_id="B"
    )
    # Wire fixture echoes the requested round rather than an answer outside the request.
    assert (
        service.submit(submission(reader, result, budget), key="assessed")["status"]
        == "submitted_not_accepted"
    )
    assert service.accept(key="assessed").status == "accepted"
    assert (
        service.read(key="assessed")[2]["scores"]["root"]["coverage"]["slo_image_research"][
            "acquisition"
        ]
        == 1.0
    )
    assert service.read(key="vision") == history
    assert service.read(key="unknown") == unknown_saved


def test_rejected_retry_is_exact_and_never_authorizes_acceptance():
    service, reader, docs, sub = setup()
    bad = sub.model_copy(update={"config_checksum": digest("different")})
    first = service.submit(bad, key="bad")
    n = len(docs.audit)
    assert service.submit(bad, key="bad") == first and len(docs.audit) == n
    assert first.status == "rejected"
