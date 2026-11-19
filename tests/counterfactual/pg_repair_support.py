"""Second synthetic accepted run in the SAME owned PG case, original repositories."""

from datetime import timedelta
from types import SimpleNamespace

from packages.grounding.models import digest, seal
from packages.llm_provider import RequestBudget
from packages.research_runtime.agent_records import RuntimeAgentJournal, RuntimeDocuments
from packages.research_runtime.models import ResearchRunContext
from packages.tool_admission.models import ResearchRequest
from tests.agent_provider.support import materials


def alternative_pg(pg):
    documents = pg.documents
    context = documents.contexts.read_research_run(pg.binding.run_id)
    req = context.request.model_dump(mode="json", exclude={"checksum"})
    req["lock"]["run_id"] = "RUN:OID_CF_ALTERNATIVE"
    request = seal(ResearchRequest, **req)
    context = seal(
        ResearchRunContext, request=request, synthetic=True, scope_checksum=digest(request.scope)
    )
    documents.contexts.create_research_run(context, "synthetic-alternative")
    start, sha = documents.sessions.origin(context.run_id)
    binding = pg.binding.model_copy(
        update=dict(
            run_id=context.run_id,
            run_context_checksum=sha,
            started_at=start,
            deadline=start + timedelta(seconds=pg.binding.profile.seconds),
        )
    )
    reader, _ = materials(binding, config=pg.config)
    documents = RuntimeDocuments(
        repo=pg.repo,
        contexts=documents.contexts,
        sessions=documents.sessions,
        store=pg.store,
        run_id=context.run_id,
        root=pg.root,
    )
    budget = RequestBudget(
        run_id=context.run_id,
        deadline=binding.deadline,
        max_requests=pg.config.max_requests,
        journal=RuntimeAgentJournal(documents),
    )
    return SimpleNamespace(
        binding=binding,
        reader=reader,
        documents=documents,
        budget=budget,
        config=pg.config,
        repo=pg.repo,
        store=pg.store,
    )
