"""Case graph identity/entry contracts; synthetic and saved metadata, never live."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.vectorengine_single_case import load_config
from tests.agent_provider.test_round_preparation import synthetic_projection

CONFIG = "configs/agents/vectorengine_batch_citations.json"


def ready_entry_state(state):
    """Complete legacy synthetic entry fixtures; never used to admit production inputs."""
    from packages.evidence_details.acceptance_models import DomainState
    from packages.grounding.models import digest, validate

    raw = state.model_dump(mode="json")
    image = with_graph().project(run_id="RUN:SYNTHETIC", config=load_config())[0]
    binding = next(o.image for o in image.material.observations if o.image)
    raw["material"]["observations"][0]["image"] = binding.model_dump(mode="json")
    raw["material"]["scope"].pop("material_checksum", None)
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    return validate(DomainState, raw)


def with_graph():
    evidence = synthetic_projection()
    evidence.view["nodes"] = [
        dict(key="ROOT", node_code="GOAL:TEST:ROOT", node_type="GOAL", parent=None),
        dict(
            key="EVIDENCE:SYNTHETIC", node_code="REQ:TEST:ONE", node_type="EVIDENCE", parent="ROOT"
        ),
    ]
    return evidence


def test_projection_binds_real_graph_root_not_a_free_goal_string():
    evidence = with_graph()
    state, manifest = evidence.project(run_id="RUN:SYNTHETIC", config=load_config())
    assert state.material.goal_code == "GOAL:TEST:ROOT"
    assert state.material.goal_code in state.nodes
    assert {item.node for item in state.policy.items} <= set(state.nodes)
    assert manifest["graph_target_binding"]["root_code"] == "GOAL:TEST:ROOT"


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("no_graph", "node_inventory"),
        ("no_root", "unique_goal"),
        ("two_roots", "unique_goal"),
        ("duplicate_key", "identity_ambiguous"),
        ("duplicate_code", "identity_ambiguous"),
        ("missing_policy", "policy_node_missing"),
        ("cycle", "node_not_rooted"),
        ("foreign_parent", "node_not_rooted"),
    ],
)
def test_graph_binding_rejects_unsupported_invented_or_ambiguous_nodes(mutation, error):
    e = with_graph()
    nodes = e.view["nodes"]
    if mutation == "no_graph":
        e.view.pop("nodes")
    elif mutation == "no_root":
        nodes[0]["parent"] = nodes[1]["key"]
    elif mutation == "two_roots":
        nodes[1]["parent"] = None
    elif mutation == "duplicate_key":
        nodes.append(dict(nodes[0]))
    elif mutation == "duplicate_code":
        nodes[1]["node_code"] = nodes[0]["node_code"]
    elif mutation == "missing_policy":
        nodes[1]["key"] = "OTHER"
    elif mutation == "cycle":
        nodes[1]["parent"] = nodes[1]["key"]
    elif mutation == "foreign_parent":
        nodes[1]["parent"] = "OTHER"
    with pytest.raises(ValueError, match=error):
        e.project(run_id="RUN:SYNTHETIC", config=load_config(CONFIG))


def test_saved_real_mismatch_still_rejected_and_real_graph_mapping_is_defined():
    from apps.vectorengine_case_preflight import check_case_admissibility, graph_target_binding
    from packages.evidence_details.acceptance_models import DomainState

    evidence = Path("docs/acceptance/12-completion-cde-citations-case-v1")
    diagnostic = json.loads((evidence / "source-target-diagnostic.json").read_text())
    original = Path(diagnostic["input_file"]).read_bytes()
    state = DomainState.model_validate_json(original)
    with pytest.raises(ValueError, match="research_relation_scope"):
        check_case_admissibility(state, load_config(CONFIG))
    # Inspect graph identities only; never project historical ratings into an LLM input.
    view = json.loads(Path("docs/exports/青光眼证据图.json").read_text())
    binding = graph_target_binding(
        view, state.policy, state.material.scope.graph.model_dump(mode="json")
    )
    assert binding["root_code"] == "GOAL:GLAUCOMA:ROOT"
    assert len(binding["policy_nodes"]) == 31
    assert Path(diagnostic["input_file"]).read_bytes() == original


def test_joint_preflight_uses_owner_tasks_and_does_not_write_or_send(monkeypatch):
    import socket

    from apps.vectorengine_case_preflight import check_case_admissibility

    config = load_config(CONFIG)
    state, _ = with_graph().project(run_id="RUN:SYNTHETIC", config=config)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network"))
    result = check_case_admissibility(state, config)
    assert result["review_tasks"] == 4 and result["review_batches"] == 2
    assert result["network_requests"] == 0 and not result["creation_authorized"]
    for change, error in (
        ({"nodes": ("EVIDENCE:SYNTHETIC",)}, "research_relation_scope"),
        ({"config_checksum": "sha256:" + "0" * 64}, "config_binding"),
        ({"prompt_checksum": "sha256:" + "0" * 64}, "prompt_binding"),
    ):
        with pytest.raises(ValueError, match=error):
            check_case_admissibility(state.model_copy(update=change), config)


def test_original_owner_target_and_quote_checks_are_not_relaxed():
    from packages.graph_engine.research_candidates import validate_research_relation
    from packages.grounding.models import digest
    from packages.knowledge_claims.research_candidates import validate_research_source
    from packages.llm_provider.models import SourceProposal

    state, _ = with_graph().project(run_id="RUN:SYNTHETIC", config=load_config(CONFIG))
    source = state.material.sources[0]
    candidate = SourceProposal(
        scope_checksum=digest(state.material.scope),
        source_code=source.reference.code,
        quote=source.text,
        target=state.material.goal_code,
        relation="organizes",
        missing=(),
        reason="Synthetic contract only",
    )
    validate_research_source(candidate, state.material)
    validate_research_relation(candidate, scope=state.material.scope, node_codes=state.nodes)
    with pytest.raises(ValueError, match="research_source_or_target_mismatch"):
        validate_research_source(
            candidate.model_copy(update={"target": "EVIDENCE:SYNTHETIC"}), state.material
        )
    with pytest.raises(ValueError, match="research_source_or_target_mismatch"):
        validate_research_source(
            candidate.model_copy(update={"quote": "Invented quotation"}), state.material
        )
    with pytest.raises(ValueError, match="research_relation_scope"):
        validate_research_relation(
            candidate.model_copy(update={"target": "GOAL:FOREIGN"}),
            scope=state.material.scope,
            node_codes=state.nodes,
        )


def test_current_first_case_owner_rejects_forged_target_before_reading_sources():
    e = with_graph()
    state, _ = e.project(run_id="RUN:SYNTHETIC", config=load_config(CONFIG))
    bad = state.model_copy(
        update={"material": state.material.model_copy(update={"goal_code": "GOAL:FORGED"})}
    )
    with pytest.raises(ValueError, match="case_graph_target_changed"):
        e.verify(bad)


@pytest.mark.parametrize("entry", ["prepare_round", "prepare", "execute"])
def test_bad_target_is_refused_before_output_or_db_constructor(monkeypatch, tmp_path, entry):
    from apps import vectorengine_live_assembly, vectorengine_round, vectorengine_single_case

    config = load_config(CONFIG)
    state, _ = with_graph().project(run_id="RUN:SYNTHETIC", config=config)
    state = state.model_copy(update={"nodes": ("EVIDENCE:SYNTHETIC",)})
    evidence = SimpleNamespace(project=lambda **kw: (state, {}))

    def forbidden(*args, **kwargs):
        pytest.fail("must refuse before deployed owner/gate/requests")

    monkeypatch.setattr("db.runtime.repository.planner.PlannerRepository", forbidden)
    monkeypatch.setattr("db.runtime.repository.research.ResearchRunRepository", forbidden)
    monkeypatch.setattr(vectorengine_live_assembly, "check_regression_gate", forbidden)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="research_relation_scope"):
        if entry == "prepare_round":
            r = vectorengine_round.load_round(
                "configs/research/vectorengine_batch_citations_case.json"
            )
            vectorengine_round.prepare_round(
                r.model_copy(update={"output": str(output)}), config, evidence
            )
        elif entry == "prepare":
            vectorengine_single_case.prepare(evidence, config, output)
        else:
            vectorengine_live_assembly.execute(
                evidence=evidence, initial=state, config=config, target=None, output=output
            )
    assert not output.exists()


def test_source_vision_batched_review_accept_successors_score_and_html():
    from agents.expert_agent.reporting import read_research_report, render_research_html
    from agents.provider_agents import StructuredRoleAgent
    from agents.provider_agents.batched_review import run_batches
    from packages.evidence_details.acceptance import ResearchAcceptance
    from packages.grounding.models import digest
    from packages.llm_provider import DeterministicFakeProvider
    from packages.llm_provider.models import SourceProposal, VisionProposal
    from tests.agent_provider.support import Documents, Reader, budget_for, submission
    from tests.agent_provider.test_review_citation_contract import response
    from tests.planner.support import binding

    config = load_config(CONFIG).model_copy(update={"mode": "fake"})
    b = binding()
    e = with_graph()
    state, _ = e.project(run_id=b.run_id, config=config)
    original = state.model_dump_json()
    reader = Reader(state)
    reader.image_bytes = e.image(None)
    docs = Documents()
    budget = budget_for(b, config)
    service = ResearchAcceptance(owner=reader, documents=docs, requests=budget)
    for role in ("source", "vision"):
        reader.state = reader.state.model_copy(
            update={
                "prompt_checksum": next(r.prompt_checksum for r in config.roles if r.role == role)
            }
        )
        state = reader.current()
        if role == "source":
            source = state.material.sources[0]
            candidate = SourceProposal(
                scope_checksum=digest(state.material.scope),
                source_code=source.reference.code,
                quote=source.text,
                target=state.material.goal_code,
                relation="organizes",
                missing=(),
                reason="Synthetic only",
            )
        else:
            image = next(o.image for o in state.material.observations if o.image)
            candidate = VisionProposal(
                scope_checksum=digest(state.material.scope),
                image_checksum=image.reference.checksum,
                preprocessing_checksum=image.preprocessing.checksum,
                location=image.location,
                status="unknown",
                finding="Synthetic image: indeterminate",
            )
        provider = DeterministicFakeProvider(config, [candidate.model_dump(mode="json")])
        agent = StructuredRoleAgent(role, provider=provider, materials=reader, budget=budget)
        result = agent.call(state.material.scope, deadline=b.deadline)
        sub = submission(reader, result, budget)
        assert service.submit(sub, key=role)["status"] == "submitted_not_accepted"
        assert service.accept(key=role).status == "accepted"
        successor = service.successor_material(key=role)
        reader.state = reader.state.model_copy(update={"material": successor})
    reader.state = reader.state.model_copy(
        update={
            "prompt_checksum": next(r.prompt_checksum for r in config.roles if r.role == "review")
        }
    )
    parent = reader.current()
    agent = StructuredRoleAgent(
        "review",
        provider=DeterministicFakeProvider(config, [response] * 2),
        materials=reader,
        budget=budget,
    )
    sub = run_batches(agent, state=parent, documents=docs, key="whole", round_id="A")
    assert len(sub.batches) == 2 and budget.used == 4
    assert service.submit(sub, key="review")["status"] == "submitted_not_accepted"
    receipt = service.accept(key="review")
    assert receipt.status == "accepted"
    report = read_research_report(service, key="review", parent=parent)
    assert report["scores"]["schema_version"] == "case-research-scoring/6.1.0"
    assert "非临床结论" in render_research_html(report)
    assert all(r["judgment"]["value"] is None for r in report["scores"]["reviews"])
    audit = copy.deepcopy(docs.audit)
    assert run_batches(agent, state=parent, documents=docs, key="whole", round_id="A") == sub
    assert service.accept(key="review") == receipt
    assert budget.used == 4 and audit == docs.audit
    assert e.project(run_id=b.run_id, config=config)[0].model_dump_json() == original
