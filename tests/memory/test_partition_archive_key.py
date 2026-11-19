"""Archive locator regressions; medical acceptance is not simulated as validation."""

import json
from types import SimpleNamespace as NS

import pytest

from db.runtime.repository.case_pipeline import identity
from packages.evidence_details import recorded_finalization as recorded
from packages.grounding.models import digest
from packages.memory import multimodal_archive as archive


@pytest.mark.parametrize("compatibility", [None, "origin", "lineage"])
def test_partition_prior_receipt_does_not_replace_archive_key(monkeypatch, compatibility):
    from packages.evidence_details import acceptance_models, review_batches
    from packages.grounding import models

    old = dict(new_requests=6, task={"target_code": "GOAL:TEST"})
    grant = NS(
        previous_local_grant_checksum=digest({"old": "grant"}),
        previous_local_receipt_checksum=digest(({"status": "rejected"},)),
        failed_result_checksum=digest(old),
        state_checksum=None,
        aggregate_checksum=digest({"candidate": "same"}),
    )
    # Preserve the exact grant-type dispatch while isolating this key regression.
    monkeypatch.setattr(recorded, "RecordedPartitionGrant", type(grant))
    if compatibility == "origin":
        grant.origin_compatibility_checksum = digest("exact-owner-proof")
        monkeypatch.setattr(recorded, "RecordedOriginCompatibilityGrant", type(grant))

        def original_proof(documents):
            assert documents.run_id == "RUN:TEST"
            assert documents.read("multimodal-case-result") == (old,)
            return grant.origin_compatibility_checksum

        monkeypatch.setattr(recorded, "origin_compatibility_proof", original_proof)
    if compatibility == "lineage":
        grant.lineage_capacity_checksum = digest("exact-lineage-owner-proof")
        monkeypatch.setattr(recorded, "RecordedLineageCapacityGrant", type(grant))

        def lineage_proof(documents):
            assert documents.run_id == "RUN:TEST"
            assert documents.read("multimodal-case-result") == (old,)
            return grant.lineage_capacity_checksum

        monkeypatch.setattr(recorded, "lineage_capacity_proof", lineage_proof)
    monkeypatch.setattr(models, "validate", lambda cls, value: grant)
    grant_digest = digest({"grant": "current"})
    real_digest = archive.digest
    monkeypatch.setattr(
        archive, "digest", lambda value: grant_digest if value is grant else real_digest(value)
    )
    parent = dict(material={}, resources=[], policy=dict(requirements=[]))
    grant.state_checksum = digest(parent)
    stored = [
        {"records": "same"},
        {},
        {"scores": {}},
        {"status": "accepted", "permission_checksum": grant_digest},
    ]
    monkeypatch.setattr(
        acceptance_models,
        "read_records",
        lambda value: NS(permission_checksum=grant_digest, candidate="same"),
    )
    monkeypatch.setattr(review_batches, "read_submission", lambda value: NS(candidate="same"))
    # The binding checks digest the parsed submission, not its transport dictionary.
    submission = NS(candidate="same")
    monkeypatch.setattr(review_batches, "read_submission", lambda value: submission)
    patched_digest = archive.digest
    monkeypatch.setattr(
        archive,
        "digest",
        lambda value: grant.aggregate_checksum if value is submission else patched_digest(value),
    )
    result = dict(
        schema_version="multimodal-recorded-finalization-result/1.1.0",
        case_code="CASE:TEST",
        phases=[dict(run_id="RUN:TEST", accepted=True, target=None, acceptance_key="partition")],
        task=old["task"],
        original_requests=6,
        new_requests=0,
        original_result_checksum=digest(old),
        finalization_grant_checksum=grant_digest,
        finalization_policy=recorded.PARTITION_POLICY,
        execution_kind="complete_recorded_fact_finalization_not_run_resume",
    )
    report = dict(
        schema_version="dynamic-case-report/1.0.0",
        result=result,
        scores={},
        original_review=dict(material={}, records=stored[0], graph={}, receipt=stored[3]),
    )
    values = {
        "dynamic-phase-input": [parent],
        "accepted:partition": stored,
        recorded.RESULT_KEY: [result],
        recorded.PARTITION_GRANT_KEY: [
            {
                "schema_version": "recorded-review-finalization/1.4.0"
                if compatibility == "lineage"
                else "recorded-review-finalization/1.2.0"
                if compatibility == "origin"
                else "recorded-review-finalization/1.1.0"
            }
        ],
        recorded.PARTITION_SUBMISSION_KEY: [{}],
        "multimodal-case-result": [old],
        recorded.GRANT_KEY: [{"old": "grant"}],
        "accepted:" + recorded.ACCEPTANCE_KEY: [{"status": "rejected"}],
    }
    bundles, objects = {}, {}
    for name, items in values.items():
        artifacts = []
        for index, value in enumerate(items):
            code = name + ":" + str(index)
            artifact = NS(asset_code=code)
            raw = json.dumps(
                dict(
                    run_id="RUN:TEST",
                    root_checksum=digest("root"),
                    key_checksum=digest(name),
                    index=index,
                    value=value,
                )
            ).encode()
            objects[code] = (artifact, raw)
            artifacts.append(artifact)
        bundles[identity("TASK", "RUN:TEST:agent:" + name)] = artifacts
    owner = NS(
        read_run_case=lambda run: "CASE:TEST",
        runtime=NS(read_observation_bundle=bundles.get),
        artifact=lambda case, code: objects[code],
        reference=lambda **kw: kw,
    )
    # Constructor capture tests projection identity without claiming clinical validity.
    monkeypatch.setattr(archive, "archive_record_class", lambda sources: lambda **kw: NS(**kw))
    record = archive.prepare(
        owner,
        key="mm-memory:expected",
        case_code="CASE:TEST",
        run_id="RUN:TEST",
        origin=NS(asset_code="ASSET:REPORT"),
        envelope=dict(
            run_id="RUN:TEST",
            root_checksum=digest("root"),
            key_checksum=digest(recorded.REPORT_KEY),
            index=0,
            value=report,
        ),
        kind="graph_snapshot",
        target="GOAL:TEST",
    )
    assert record.key == "mm-memory:expected"


@pytest.mark.parametrize(
    "version,cls",
    [
        ("1.0.0", recorded.RecordedReviewGrant),
        ("1.1.0", recorded.RecordedPartitionGrant),
        ("1.2.0", recorded.RecordedOriginCompatibilityGrant),
        ("1.3.0", recorded.RecordedAllReusedGrant),
        ("1.4.0", recorded.RecordedLineageCapacityGrant),
    ],
)
def test_original_and_archive_grant_dispatch_share_strict_version(version, cls):
    assert (
        recorded.recorded_grant_type({"schema_version": "recorded-review-finalization/" + version})
        is cls
    )


@pytest.mark.parametrize("raw", [{}, {"schema_version": "recorded-review-finalization/99.0.0"}])
def test_archive_grant_dispatch_does_not_guess_unknown_version(raw):
    with pytest.raises(ValueError, match="^recorded_finalization_grant_version$"):
        recorded.recorded_grant_type(raw)


def test_persist_rejects_wrong_prepared_key_before_any_write(monkeypatch):
    from packages.multimodal_research import memory, report_storage

    monkeypatch.setattr(report_storage, "unpack", lambda *a, **kw: ({"requirements": []}, {}))
    docs = NS(
        run_id="RUN:TEST",
        read=lambda key: (),
        key=lambda key: key,
        repo=NS(read_observation_bundle=lambda key: [NS(asset_code="ASSET:REPORT")]),
    )
    repo = NS(create=lambda value: pytest.fail("wrong locator must never be written"))
    sources = NS(prepare=lambda **kw: NS(key="accepted:old-receipt"))
    with pytest.raises(ValueError, match="multimodal_memory_prepared_key_binding"):
        memory.persist(
            repo, sources, docs, dict(case_code="CASE:TEST", task=dict(target_code="GOAL:TEST"))
        )
