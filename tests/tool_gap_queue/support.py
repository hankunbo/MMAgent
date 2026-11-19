"""Real 06 producer with Fake metadata owners; no tools are executed."""

from types import SimpleNamespace

from packages.grounding.models import digest, seal
from packages.research_runtime.models import ResearchRunContext
from packages.tool_gap_queue import GapQueue, QueuePolicy
from tests.tool_admission.support import fixture, reseal
from tests.tool_admission.test_research import research


def setup(tmp_path, *, purpose="shadow", missing=False, blocked=False, policy=None):
    f = fixture(tmp_path)
    request = research(f, tmp_path, purpose=purpose)
    if missing:
        request = reseal(
            request, inputs=[dict(port=i.port, asset_codes=[]) for i in request.inputs]
        )
        f.owners.requests[(request.code, request.version, request.checksum)] = request
    context = seal(
        ResearchRunContext, request=request, synthetic=True, scope_checksum=digest(request.scope)
    )
    for k, probe in list(f.owners.probes.items()):
        f.owners.probes[k] = reseal(probe, resources="unavailable")
    if blocked:
        f.service.catalogs.disable(request.candidate.implementation_ref)
    contexts = SimpleNamespace(
        read_research_run=lambda rid: context if rid == context.run_id else None
    )
    service = GapQueue(
        repository=None,
        admission_owner=f.service,
        contexts=contexts,
        policy=policy or QueuePolicy(),
    )
    return SimpleNamespace(f=f, context=context, service=service, request=request)


def install_pg(h, database):
    from db import manage
    from db.runtime.repository.research import ResearchRunRepository
    from db.runtime.repository.tool_gap_queue import GapQueueRepository
    from packages.contracts import CaseBundle
    from tests.disease_packs.support import anonymous_case, code

    contexts = ResearchRunRepository(manage.test_target("runtime"))
    scope = h.request.scope
    case = anonymous_case("GOAL:SYNTHETIC:QUEUE")
    case = CaseBundle.model_validate_json(
        case.model_copy(
            update=dict(
                case_code=scope.case_code,
                scope=case.scope.model_copy(update=dict(case_code=scope.case_code)),
                encounter_code=scope.encounter_code,
                patient_hash=scope.patient_hash,
                laterality=scope.laterality,
                artifacts=tuple(r.artifact for r in h.f.owners.records.values()),
            )
        ).model_dump_json()
    )
    contexts.create_case_bundle(case, code("TASK"))
    contexts.create_research_run(h.context, "11a-context")
    h.service.contexts = contexts
    h.service.repository = GapQueueRepository(manage.test_target("runtime"))
    return h.service
