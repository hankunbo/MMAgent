"""Real PG service regression for intermediate attribution, no query writes."""

import json

import pytest

from db import manage
from db.runtime.repository.grounding import RuntimeGroundingLocks
from db.runtime.repository.grounding_query import GroundingQueryService
from packages.grounding import GroundingError, digest, validate
from packages.grounding.attribution import derive_attribution
from packages.grounding.query_models import Finding, QueryResult, RejectedCandidate
from tests.disease_packs import support
from tests.grounding.attribution_support import DISC_CAPABILITY, add_producer
from tests.grounding.query_support import metadata, updated
from tests.grounding.test_database import database as database
from tests.grounding.test_database import key
from tests.grounding.test_models import reseal
from tests.grounding.test_query_runtime import counts, query_request, scoped_mapping
from tests.grounding.test_runtime import create_run
from tests.grounding.test_runtime import integrated as integrated


def test_official_service_intermediate_target_and_forgery_no_writes(database, integrated, tmp_path):
    data, old, writer, catalogs = scoped_mapping(integrated)
    mapping, catalog = add_producer(old, catalogs.read_snapshot(old.catalog_ref))
    catalogs.add(catalog)
    writer.new_version(old.reference(), mapping, support.context(), request_key=key())
    writer.machine_review(mapping.reference(), support.context(), request_key=key())
    locks = RuntimeGroundingLocks(manage.test_target("runtime"))
    lock = locks.bind(**create_run(database, data, mapping, writer))
    service = GroundingQueryService(locks, writer)
    q = query_request(database, mapping, writer, lock)
    q = updated(
        q,
        patient_data_types=[],
        artifacts=[
            a for a in q.artifacts if a.artifact_ref != mapping.candidates[-1].output.artifact_ref
        ]
        + [metadata(q, mapping.candidates[-1].inputs[0].constraint, "IMAGE001")],
        environment=[
            e
            for e in q.environment
            if e.implementation_ref != mapping.candidates[-1].implementation_ref
        ],
    )
    before = counts(database)
    result = service.query(q)
    assert counts(database) == before and locks.read(lock.run_id) == lock
    assert {s.capability_gap.capability_code for s in result.suggestions} == {DISC_CAPABILITY}
    assert service.validate_result(q, result) == result
    for mutate in ("target", "findings", "binding", "coordinated_target"):
        fields = result.model_dump(mode="json")
        suggestion = fields["suggestions"][0]
        if mutate in {"target", "coordinated_target"}:
            suggestion["capability_gap"]["capability_code"] = mapping.candidates[
                0
            ].capability_ref.code
            if mutate == "coordinated_target":
                suggestion["target_capability_ref"] = mapping.candidates[
                    0
                ].capability_ref.model_dump(mode="json")
        elif mutate == "findings":
            suggestion["finding_checksums"] = suggestion["finding_checksums"][:1]
        else:
            suggestion["policy_ref"]["version"] = "9.0.0"
        fields.pop("checksum")
        fields["checksum"] = digest(fields)
        with pytest.raises(ValueError):
            service.validate_result(q, fields)
        assert counts(database) == before and locks.read(lock.run_id) == lock
    # Even internally consistent, fully resealed source/target substitutions are not authority.
    disc = mapping.candidates[-1].capability_ref.model_dump(mode="json")
    ratio = mapping.candidates[0].capability_ref.model_dump(mode="json")

    def substitute(value):
        if isinstance(value, dict):
            return ratio if value == disc else {k: substitute(v) for k, v in value.items()}
        if isinstance(value, list):
            return [substitute(v) for v in value]
        return value

    fields = substitute(result.model_dump(mode="json"))
    gaps = tuple(validate(Finding, f) for f in fields["gaps"])
    rejected = tuple(validate(RejectedCandidate, r) for r in fields["rejected"])
    demands, suggestions = derive_attribution(result.request, result.candidates, rejected, gaps)
    forged = reseal(QueryResult, {**fields, "product_demands": demands, "suggestions": suggestions})
    with pytest.raises(GroundingError, match="stale_or_forged_query_result"):
        service.validate_result(q, forged)
    assert counts(database) == before and locks.read(lock.run_id) == lock
    (tmp_path / "attribution-runtime.synthetic.json").write_text(
        json.dumps(
            dict(
                synthetic=True,
                result=result.model_dump(mode="json"),
                counts_before=before,
                counts_after=counts(database),
                lock_unchanged=locks.read(lock.run_id) == lock,
            ),
            indent=2,
        )
        + "\n"
    )

    # An unknown environment cannot repair an explicitly revoked implementation.
    catalogs.disable(mapping.candidates[-1].implementation_ref)
    revoked = service.query(q)
    assert counts(database) == before and locks.read(lock.run_id) == lock
    assert {"tool_safety_revoked", "environment_unknown", "required_condition_unknown"} <= {
        f.reason for f in revoked.gaps
    }
    assert revoked.product_demands and revoked.status == "gaps"
    assert not revoked.suggestions
    assert service.validate_result(q, revoked) == revoked
    with pytest.raises(GroundingError, match="stale_or_forged_query_result"):
        service.validate_result(q, result)
    assert counts(database) == before and locks.read(lock.run_id) == lock
    (tmp_path / "safety-priority-runtime.synthetic.json").write_text(
        json.dumps(
            dict(
                synthetic=True,
                before=result.model_dump(mode="json"),
                revoked=revoked.model_dump(mode="json"),
                counts_before=before,
                counts_after=counts(database),
                lock_unchanged=locks.read(lock.run_id) == lock,
            ),
            indent=2,
        )
        + "\n"
    )
