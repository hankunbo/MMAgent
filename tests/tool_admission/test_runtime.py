"""Actual Runtime/Published Knowledge service with isolated PG; no admission writes."""

import pytest

from db import manage
from db.runtime.repository.grounding import RuntimeGroundingLocks
from db.runtime.repository.grounding_query import GroundingQueryService
from packages.grounding.models import digest, validate
from packages.integrations.medgraph.resources import LocalResourcePolicy
from packages.object_store import LocalObjectStore
from packages.tool_admission import AdmissionSelection, ToolAdmissionService
from tests.grounding.query_support import NOW
from tests.grounding.test_database import database as database
from tests.grounding.test_query_runtime import counts, query_request, scoped_mapping
from tests.grounding.test_runtime import create_run
from tests.grounding.test_runtime import integrated as integrated
from tests.tool_admission.support import Owners, register_deployments, register_inputs, reseal


def test_real_runtime_lock_old_run_extension_and_forged_results(database, integrated, tmp_path):
    data, mapping, writer, catalogs = scoped_mapping(integrated)
    locks = RuntimeGroundingLocks(manage.test_target("runtime"))
    lock = locks.bind(**create_run(database, data, mapping, writer))
    grounding = GroundingQueryService(locks, writer)
    owners, objects = Owners(), LocalObjectStore(tmp_path / "objects")
    q = register_inputs(query_request(database, mapping, writer, lock), owners, objects)
    catalog = catalogs.read_snapshot(lock.catalog_ref)
    invocations = register_deployments(q, catalog, owners, tmp_path)
    result = grounding.query(q)
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
    admitted = service.admit(q, result, selection)
    assert admitted.status == "preparation_only" and counts(database) == before
    assert service.validate_result(q, result, selection, admitted) == admitted
    # Adding a future catalog cannot alter this run or change its execution inputs.
    replacement = catalog.tools[0].model_dump(mode="json")
    replacement["implementation_ref"].update(
        code="synthetic_new_implementation",
        version="99.0.0",
        checksum=digest("new validated implementation"),
    )
    future = reseal(catalog, version="99.0.0", tools=[*catalog.tools, replacement])
    catalogs.add(future)
    assert len(catalogs.read_snapshot(future.reference()).tools) == len(catalog.tools) + 1
    assert all(
        t.implementation_ref
        != validate(type(catalog.tools[0].implementation_ref), replacement["implementation_ref"])
        for t in catalogs.read_snapshot(lock.catalog_ref).tools
    )
    assert service.admit(q, result, selection) == admitted
    assert locks.read(lock.run_id) == lock and counts(database) == before
    for changes in (
        {"execution_authorized": True},
        {"request_checksum": digest("other")},
        {"scope": {**admitted.scope.model_dump(mode="json"), "patient_hash": digest("other")}},
    ):
        with pytest.raises(ValueError):
            service.validate_result(q, result, selection, admitted.model_copy(update=changes))
    catalogs.disable(catalog.tools[0].implementation_ref)
    with pytest.raises(ValueError, match="stale_or_forged_query_result"):
        service.admit(q, result, selection)
    assert locks.read(lock.run_id) == lock and counts(database) == before
