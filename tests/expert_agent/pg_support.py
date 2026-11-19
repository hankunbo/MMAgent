"""Original unified owned-PG fixture setup, synthetic case/run only."""

from datetime import timedelta
from types import SimpleNamespace


def runtime_fixture(tmp_path):
    from db import manage
    from db.runtime.repository.case_evidence import CaseEvidenceRepository
    from db.runtime.repository.planner import PlannerRepository
    from db.runtime.repository.research import ResearchRunRepository
    from packages.contracts import Artifact, CaseBundle
    from packages.grounding.models import canonical, digest, seal
    from packages.llm_provider import RequestBudget
    from packages.llm_provider.config import load_provider_config
    from packages.object_store import checksum
    from packages.planner_control.models import Graph, RankingPolicy
    from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments
    from packages.research_runtime.models import ResearchRunContext
    from tests.agent_provider.support import materials
    from tests.disease_packs.support import anonymous_case, code
    from tests.planner.test_actions import ranked_binding
    from tests.tool_admission.support import fixture
    from tests.tool_admission.test_research import research

    f = fixture(tmp_path)
    req = research(f, tmp_path)
    target = manage.test_target("runtime")
    contexts = ResearchRunRepository(target)
    repo = CaseEvidenceRepository(target)
    store = f.objects
    case = anonymous_case("GOAL:SYNTHETIC:REVIEW")
    case = CaseBundle.model_validate_json(
        case.model_copy(
            update=dict(
                case_code=req.scope.case_code,
                scope=case.scope.model_copy(update=dict(case_code=req.scope.case_code)),
                encounter_code=req.scope.encounter_code,
                patient_hash=req.scope.patient_hash,
                laterality=req.scope.laterality,
                artifacts=tuple(r.artifact for r in f.owners.records.values()),
            )
        ).model_dump_json()
    )
    contexts.create_case_bundle(case, code("TASK"))
    context = seal(
        ResearchRunContext, request=req, synthetic=True, scope_checksum=digest(req.scope)
    )
    contexts.create_research_run(context, "agent-context")
    sessions = PlannerRepository(target)
    started, sha = sessions.origin(context.run_id)
    raw = canonical(dict(schema_version="case-evidence-graph/1.19.0", synthetic=True))
    original = case.artifacts[0]
    a = original.model_dump(mode="json")
    a.update(
        asset_code="ASSET:OID_AGENT_ROOT",
        artifact_type_code="ARTIFACT:CASE_EVIDENCE_GRAPH",
        uri=store.promote(store.stage(raw), checksum(raw)),
        checksum=checksum(raw),
        media_type="application/json",
        derived_from=[original.asset_code],
        dimensions=[],
        measurements=[],
        research_only=True,
    )
    a["provenance"].update(input_checksums=[original.checksum], source_codes=[original.asset_code])
    root = Artifact.model_validate(a)
    repo.register_observation_bundle((root,), "TASK:OID_AGENT_ROOT")
    b = ranked_binding(
        RankingPolicy(),
        case_code=case.case_code,
        run_id=context.run_id,
        run_context_checksum=sha,
        started_at=started,
        deadline=started + timedelta(seconds=300),
        graph=Graph(
            role="graph",
            code=root.asset_code,
            version="case-evidence-graph/1.19.0",
            checksum=root.checksum,
            kind="research_artifact",
        ),
    )
    config = load_provider_config(environ={})
    from agents.provider_agents import provider_locks

    locks = {r.role: r for r in provider_locks(config)}
    b = b.model_copy(update={"locks": tuple(locks.get(r.role, r) for r in b.locks)})
    reader, _ = materials(b, config=config)
    documents = RuntimeDocuments(
        repo=repo, contexts=contexts, sessions=sessions, store=store, run_id=b.run_id, root=root
    )
    journal = RuntimeAgentJournal(documents)
    budget = RequestBudget(
        run_id=b.run_id, deadline=b.deadline, max_requests=config.max_requests, journal=journal
    )
    return SimpleNamespace(
        binding=b,
        reader=reader,
        documents=documents,
        budget=budget,
        repo=repo,
        store=store,
        root=root,
        raw=raw,
        target=target,
        config=config,
    )
