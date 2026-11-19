"""Second anonymous case/run prepared only through original domain repositories."""

from datetime import timedelta
from types import SimpleNamespace

from packages.contracts import Artifact, CaseBundle
from packages.grounding.models import canonical, digest, seal, validate
from packages.llm_provider import RequestBudget
from packages.planner_control.models import Graph
from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments
from packages.research_runtime.models import ResearchRunContext
from packages.tool_admission.models import ResearchRequest
from tests.agent_provider.support import materials


def second_case(pg, *, same_case=False):
    tag = "MEMORYRERUN" if same_case else "MEMORYSECOND"
    documents = pg.documents
    context = documents.contexts.read_research_run(pg.binding.run_id)
    old_scope = context.request.scope
    old_case = pg.repo.get_case_bundle(old_scope.case_code)
    input_codes = {a.asset_code for a in context.request.artifacts}
    originals = [a for a in old_case.artifacts if a.asset_code in input_codes]
    mapping = {
        old_scope.case_code: "CASE:OID_" + tag,
        context.run_id: "RUN:OID_" + tag,
        old_scope.encounter_code: "ENCOUNTER:OID_" + tag,
        old_scope.patient_hash: digest("second anonymous synthetic person"),
        **{a.asset_code: "ASSET:OID_" + tag + str(n) for n, a in enumerate(originals)},
    }

    if same_case:
        mapping = {context.run_id: "RUN:OID_" + tag}

    def rewrite(v):
        if isinstance(v, str):
            return mapping.get(v, v)
        if isinstance(v, list):
            return [rewrite(i) for i in v]
        if isinstance(v, dict):
            return {k: rewrite(i) for k, i in v.items()}
        return v

    artifacts = tuple(validate(Artifact, rewrite(a.model_dump(mode="json"))) for a in originals)
    raw_case = rewrite(old_case.model_dump(mode="json"))
    raw_case["artifacts"] = [a.model_dump(mode="json") for a in artifacts]
    case = validate(CaseBundle, raw_case)
    if not same_case:
        documents.contexts.create_case_bundle(case, "TASK:OID_" + tag)
    raw_request = rewrite(context.request.model_dump(mode="json", exclude={"checksum"}))
    request = seal(ResearchRequest, **raw_request)
    context = seal(
        ResearchRunContext, request=request, synthetic=True, scope_checksum=digest(request.scope)
    )
    documents.contexts.create_research_run(
        context, "memory-rerun-original-run" if same_case else "memory-second-original-run"
    )
    start, sha = documents.sessions.origin(context.run_id)
    root_data = rewrite(pg.root.model_dump(mode="json"))
    root_data["asset_code"] = "ASSET:OID_" + tag + "ROOT"
    root = validate(Artifact, root_data)
    pg.repo.register_observation_bundle((root,), "TASK:OID_" + tag + "ROOT")
    binding = pg.binding.model_copy(
        update=dict(
            case_code=case.case_code,
            run_id=context.run_id,
            run_context_checksum=sha,
            started_at=start,
            deadline=start + timedelta(seconds=pg.binding.profile.seconds),
            graph=Graph(
                role="graph",
                code=root.asset_code,
                version="case-evidence-graph/1.19.0",
                checksum=root.checksum,
                kind="research_artifact",
            ),
        )
    )
    reader, _ = materials(binding, config=pg.config)
    docs = RuntimeDocuments(
        repo=pg.repo,
        contexts=documents.contexts,
        sessions=documents.sessions,
        store=pg.store,
        run_id=context.run_id,
        root=root,
    )
    budget = RequestBudget(
        run_id=context.run_id,
        deadline=binding.deadline,
        max_requests=pg.config.max_requests,
        journal=RuntimeAgentJournal(docs),
    )
    assert (canonical(context.request.scope) == canonical(old_scope)) is same_case
    return SimpleNamespace(
        binding=binding,
        reader=reader,
        documents=docs,
        budget=budget,
        config=pg.config,
        repo=pg.repo,
        store=pg.store,
    )
