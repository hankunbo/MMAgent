"""Output target regression on an independent existing module-scoped PG fixture."""

import pytest

from db import manage
from db.runtime.repository.grounding import RuntimeGroundingLocks
from db.runtime.repository.grounding_query import GroundingQueryService
from packages.grounding.models import validate
from packages.integrations.medgraph.resources import LocalResourcePolicy
from packages.object_store import LocalObjectStore
from packages.tool_admission import AdmissionSelection, ToolAdmissionService
from tests.grounding.query_support import NOW
from tests.grounding.test_database import database as database
from tests.grounding.test_query_runtime import counts, query_request, scoped_mapping
from tests.grounding.test_runtime import create_run
from tests.grounding.test_runtime import integrated as integrated
from tests.tool_admission.support import Owners, register_deployments, register_inputs


def test_output_target_rejection_preserves_runtime_audit(database, integrated, tmp_path):
    from types import SimpleNamespace

    from tests.tool_admission.test_output_targets import (
        assert_output_rejection,
        legacy_wrong_result,
    )

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
    f = SimpleNamespace(service=service, q=q, result=result, selection=selection, owners=owners)
    changed_selection, wrong = legacy_wrong_result(f)
    admitted = service.admit(q, result, changed_selection)
    assert counts(database) == before and locks.read(lock.run_id) == lock
    assert_output_rejection(admitted, catalog.tools[0].implementation_ref)
    assert service.validate_result(q, result, changed_selection, admitted) == admitted
    for value in (
        wrong,
        validate(type(admitted), wrong).model_copy(),
        type(admitted).model_construct(**wrong),
    ):
        with pytest.raises(ValueError, match="stale_or_forged_admission_result"):
            service.validate_result(q, result, changed_selection, value)
    assert counts(database) == before and locks.read(lock.run_id) == lock

    from packages.grounding.models import seal
    from packages.tool_admission.models import ResearchRequest

    service.research = owners
    candidate = result.candidates[0].candidate
    invocation = next(
        i
        for i in changed_selection.invocations
        if i.implementation_ref == candidate.implementation_ref
    )
    for purpose in ("debug", "shadow"):
        request = seal(
            ResearchRequest,
            code="RESEARCH:SYNTHETIC:OUTPUT_TARGET",
            version="1.0.0",
            purpose=purpose,
            compatibility_policy=q.lock.policy,
            lock=dict(run_id=lock.run_id, catalog_ref=lock.catalog_ref),
            scope=q.scope,
            window_start=q.window_start,
            window_end=q.window_end,
            snapshot_at=q.snapshot_at,
            satisfied_terms=q.satisfied_terms,
            artifacts=q.artifacts,
            candidate=candidate,
            inputs=[
                dict(
                    port=i.port,
                    asset_codes=[
                        a.asset_code
                        for a in q.artifacts
                        if a.artifact_ref == i.constraint.artifact_ref
                    ],
                )
                for i in candidate.inputs
            ],
            invocation=invocation,
        )
        owners.requests[(request.code, request.version, request.checksum)] = request
        checked = service.admit_research(request)
        assert_output_rejection(checked, candidate.implementation_ref)
        assert service.validate_research_result(request, checked) == checked
    assert counts(database) == before and locks.read(lock.run_id) == lock
