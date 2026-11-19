"""Engineering review fixtures are explicitly isolated; no real approval is generated."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from packages.grounding.models import seal, validate
from packages.knowledge_review.engineering_stop import (
    EngineeringBasis,
    EngineeringReview,
    EngineeringStopReader,
    MedicalStopDependency,
    StopBasisBinding,
    dependencies,
    engineering_stop_receipt,
    read_stop_basis,
)
from packages.object_store import checksum
from packages.tool_execution.trusted_models import ReviewedFile


def file(path):
    return ReviewedFile(
        path=str(path.resolve()),
        checksum=checksum(path.read_bytes()),
        byte_size=path.stat().st_size,
    )


def fixture(tmp_path, *, runtime_config=None, **review_changes):
    source = Path("packages/deterministic_pipeline/service.py")
    config = tmp_path / "synthetic-policy.json"
    config.write_text(
        '{"synthetic":true,"max_seconds":1}'
        if runtime_config is None
        else runtime_config.model_dump_json()
    )
    basis = seal(
        EngineeringBasis,
        owner="OWNER:SYNTHETIC_07_FIXTURE",
        trigger="time_budget",
        reason_code="REASON:TIME_BUDGET",
        predicate="now >= original started_at + policy.max_seconds",
        source=file(source),
        symbol="DeterministicCasePipeline._deadline_reached",
        engine_version="deterministic-pipeline/1.0.0",
        configuration=file(config),
        configuration_version="synthetic-policy/1.0.0",
        runtime_config_checksum=(
            checksum(config.read_bytes()) if runtime_config is None else runtime_config.checksum
        ),
        rule_version="1.0.0",
    )
    review = seal(
        EngineeringReview,
        **dict(
            subject_checksum=basis.checksum,
            dependency_checksum=dependencies(basis),
            reviewer="REVIEWER:SYNTHETIC_FIXTURE",
            owner=basis.owner,
            decision="APPROVED",
            reviewed_at=datetime.now(UTC),
            synthetic=True,
        )
        | review_changes,
    )
    r = tmp_path / "synthetic-review.json"
    r.write_text(review.model_dump_json())
    binding = seal(
        StopBasisBinding,
        kind="engineering",
        engineering=(basis,),
        review_files=(file(r),),
        synthetic=True,
    )
    record = tmp_path / "synthetic-binding.json"
    record.write_text(binding.model_dump_json())
    current = (
        basis.engine_version,
        basis.configuration_version,
        basis.rule_version,
        basis.source.checksum,
        basis.configuration.checksum,
        basis.symbol,
        basis.reason_code,
        basis.predicate,
        basis.runtime_config_checksum,
    )
    reader = EngineeringStopReader(
        records={binding.checksum: file(record)},
        owners=(basis.owner,),
        reviewers=(review.reviewer,),
        current_versions=lambda *args: current,
        synthetic_test_mode=True,
    )
    return binding, reader, r, config


def test_engineering_binding_no_fake_medical_claim_and_explicit_consumer(tmp_path):
    binding, reader, _, _ = fixture(tmp_path)
    result = engineering_stop_receipt(
        reader,
        binding.checksum,
        trigger="time_budget",
        reason_code="REASON:TIME_BUDGET",
        runtime_config_checksum=binding.engineering[0].runtime_config_checksum,
    )
    assert not result["grants_medical_authority"] and result["synthetic"]
    assert not binding.medical
    with pytest.raises(ValueError, match="trigger_mismatch"):
        engineering_stop_receipt(
            reader,
            binding.checksum,
            trigger="time_budget",
            reason_code="REASON:TIME_BUDGET",
            runtime_config_checksum="sha256:" + "0" * 64,
        )
    assert read_stop_basis(binding.model_dump_json()) == binding
    with pytest.raises(ValueError, match="trigger_mismatch"):
        engineering_stop_receipt(
            reader,
            binding.checksum,
            trigger="paths_exhausted",
            reason_code="REASON:TIME_BUDGET",
            runtime_config_checksum=binding.engineering[0].runtime_config_checksum,
        )
    with pytest.raises(ValueError):
        validate(
            StopBasisBinding, binding.model_copy(update={"clinical_publication_authority": True})
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(decision="REJECTED"),
        dict(decision="WITHDRAWN"),
        dict(owner="OWNER:OTHER"),
        dict(synthetic=False),
        dict(dependency_checksum="sha256:" + "0" * 64),
        dict(subject_checksum="sha256:" + "0" * 64),
    ],
)
def test_review_refusals(tmp_path, change):
    binding, reader, _, _ = fixture(tmp_path, **change)
    with pytest.raises(ValueError, match="review_rejected"):
        reader.read(binding.checksum)


@pytest.mark.parametrize(
    "what", ["review_bytes", "config_bytes", "version", "reviewer", "production"]
)
def test_readback_drift_and_no_fixture_authority_in_production(tmp_path, what):
    binding, reader, review, config = fixture(tmp_path)
    if what == "review_bytes":
        review.write_text("{}")
    elif what == "config_bytes":
        config.write_text("{}")
    elif what == "version":
        reader.current_versions = lambda *args: ()
    elif what == "reviewer":
        reader.reviewers = frozenset()
    else:
        reader.synthetic = False
    with pytest.raises(ValueError):
        reader.read(binding.checksum)


def test_missing_review_and_mixed_medical_dependency_refused(tmp_path):
    binding, reader, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="review_required"):
        validate(StopBasisBinding, binding.model_copy(update=dict(review_files=())))
    with pytest.raises(ValueError, match="kind_dependencies"):
        validate(StopBasisBinding, binding.model_copy(update=dict(kind="mixed")))
    ref = dict(
        kind="stop_rule", code="RULE:SYNTHETIC", version="1.0.0", checksum="sha256:" + "1" * 64
    )
    policy = dict(ref, kind="release_policy", code="POLICY:SYNTHETIC")
    medical = MedicalStopDependency(stop_rule=ref, policy=policy)
    mixed = seal(
        StopBasisBinding,
        **{
            **binding.model_dump(mode="json"),
            "kind": "mixed",
            "medical": [medical.model_dump(mode="json")],
        },
    )
    path = tmp_path / "mixed.json"
    path.write_text(mixed.model_dump_json())
    reader.records[mixed.checksum] = file(path)
    with pytest.raises(ValueError, match="medical_stop_reader_required"):
        reader.read(mixed.checksum)


def test_legacy_stop_rule_replays_exact_bytes_and_rejects_engineering_payload():
    from db.knowledge.models import KnowledgeReference, StopRule
    from db.knowledge.models import seal as knowledge_seal

    claim = validate(
        KnowledgeReference,
        dict(
            kind="clinical_claim",
            code="CLAIM:SYNTHETIC_FIXTURE",
            version="1.0.0",
            checksum="sha256:" + "1" * 64,
        ),
    )
    old = knowledge_seal(
        StopRule,
        code="RULE:SYNTHETIC_FIXTURE",
        version="1.0.0",
        reason_code="RULE:SYNTHETIC_FIXTURE",
        synthetic=True,
        claim_refs=(claim,),
    )
    raw = old.model_dump_json()
    assert read_stop_basis(raw).model_dump_json() == raw
    with pytest.raises(ValueError):
        knowledge_seal(
            StopRule, code=old.code, version=old.version, reason_code=old.reason_code, claim_refs=()
        )


@pytest.mark.parametrize(
    "trigger",
    ["technical_exception", "03_budget_exhausted", "03_no_progress", "03_transition_blocked"],
)
def test_unimplemented_trigger_aliases_are_not_native_stop_events(tmp_path, trigger):
    binding, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        seal(
            EngineeringBasis,
            **{**binding.engineering[0].model_dump(mode="json"), "trigger": trigger},
        )
