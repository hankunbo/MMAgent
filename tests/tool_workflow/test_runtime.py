"""06C through actual PG Runtime locks and Published Grounding; no workflow writes."""

from types import SimpleNamespace

import pytest

from db import manage
from db.runtime.repository.grounding import RuntimeGroundingLocks
from db.runtime.repository.grounding_query import GroundingQueryService
from packages.integrations.medgraph.resources import LocalResourcePolicy
from packages.object_store import LocalObjectStore
from packages.tool_admission import AdmissionSelection, ToolAdmissionService
from packages.tool_workflow import FormalContext, ToolWorkflowCompiler
from tests.grounding.query_support import NOW
from tests.grounding.test_database import database as database
from tests.grounding.test_query_runtime import counts, query_request, scoped_mapping
from tests.grounding.test_runtime import create_run
from tests.grounding.test_runtime import integrated as integrated
from tests.tool_admission.support import Owners, register_deployments, register_inputs
from tests.tool_workflow.support import choose, commands, config


def test_compile_with_actual_runtime_lock_and_no_writes(database, integrated, tmp_path):
    data, mapping, writer, catalogs = scoped_mapping(integrated)
    locks = RuntimeGroundingLocks(manage.test_target("runtime"))
    lock = locks.bind(**create_run(database, data, mapping, writer))
    grounding = GroundingQueryService(locks, writer)
    owners, objects = Owners(), LocalObjectStore(tmp_path / "objects")
    request = register_inputs(query_request(database, mapping, writer, lock), owners, objects)
    catalog = catalogs.read_snapshot(lock.catalog_ref)
    register_deployments(request, catalog, owners, tmp_path)
    cfg = config()
    reader, invocations = commands(
        SimpleNamespace(owners=owners), tmp_path, cfg, catalog_ref=lock.catalog_ref
    )
    result = grounding.query(request)
    selection = AdmissionSelection(
        candidate_refs=tuple(c.identity for c in result.candidates), invocations=invocations
    )
    service = ToolAdmissionService(
        grounding=grounding,
        catalogs=catalogs,
        manifests=owners,
        profiles=owners,
        probes=owners,
        inputs=owners,
        objects=objects,
        resources=LocalResourcePolicy(allowed_roots=(str(tmp_path),)),
        clock=lambda: NOW,
    )
    before = counts(database)
    admission = service.admit(request, result, selection)
    context = FormalContext(
        request=request, query_result=result, selection=selection, admission=admission
    )
    compiler = ToolWorkflowCompiler(service, reader, cfg)
    plan = compiler.compile(context, choose(admission))
    assert plan.document.steps and not plan.document.execution_authorized
    assert compiler.validate_plan(plan) == plan
    assert counts(database) == before and locks.read(lock.run_id) == lock
    catalogs.disable(plan.document.steps[0].implementation_ref)
    with pytest.raises(ValueError):
        compiler.validate_plan(plan)
    assert counts(database) == before and locks.read(lock.run_id) == lock
