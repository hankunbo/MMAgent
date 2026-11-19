"""Owner-installed records cannot be replaced by a caller's recomputed digest."""

from pathlib import Path

import pytest

from packages.grounding.models import canonical, validate
from packages.object_store import checksum
from packages.tool_admission.models import InputRecord
from packages.tool_benchmark.deployment_schema_tools import schemas
from packages.tool_benchmark.registration import InstalledRecords
from packages.tool_execution.trusted_models import ReviewedFile
from tests.tool_admission.support import fixture


def test_versioned_deployment_schema_snapshots():
    root = Path("packages/tool_benchmark/schemas/deployment-v1")
    for name, content in schemas().items():
        assert (root / name).read_text() == content


def test_installed_profile_is_exact_and_missing_or_mutated_record_fails(tmp_path):
    f = fixture(tmp_path)
    manifest = next(iter(f.owners.manifests.values()))
    profile = f.owners.profiles[manifest.checksum]
    p = tmp_path / "profile.json"
    p.write_bytes(canonical(profile))
    pinned = ReviewedFile(
        path=str(p), checksum=checksum(p.read_bytes()), byte_size=p.stat().st_size
    )
    owner = InstalledRecords(profiles=(pinned,))
    assert owner.read_profile(manifest) == profile
    with pytest.raises(KeyError):
        InstalledRecords().read_profile(manifest)
    p.write_bytes(p.read_bytes().replace(b"1.0.0", b"1.0.1", 1))
    with pytest.raises(ValueError):
        owner.read_profile(manifest)
    p.unlink()
    with pytest.raises(OSError):
        owner.read_profile(manifest)


def test_input_scope_is_checked_at_owner_port(tmp_path):
    f = fixture(tmp_path)
    old = next(iter(f.owners.records.values()))
    record = validate(
        InputRecord,
        {
            **old.model_dump(mode="json"),
            "artifact": {
                **old.artifact.model_dump(mode="json"),
                "scope": {
                    "scope_type": "RUN",
                    "case_code": f.q.scope.case_code,
                    "run_id": f.q.lock.run_id,
                },
            },
            "metadata": {**old.metadata.model_dump(mode="json"), "run_id": f.q.lock.run_id},
        },
    )
    p = tmp_path / "input.json"
    p.write_bytes(canonical(record))
    owner = InstalledRecords(
        inputs=(
            ReviewedFile(
                path=str(p), checksum=checksum(p.read_bytes()), byte_size=p.stat().st_size
            ),
        )
    )
    assert owner.read_input(f.q.scope, f.q.lock.run_id, record.artifact.asset_code) == record
    with pytest.raises(KeyError):
        owner.read_input(f.q.scope, "RUN:OID_UNREGISTERED", record.artifact.asset_code)
    with pytest.raises(KeyError):
        owner.read_input(
            f.q.scope.model_copy(update={"patient_hash": checksum(b"wrong")}),
            f.q.lock.run_id,
            record.artifact.asset_code,
        )


def test_json_output_rule_survives_real_admission_projection(tmp_path):
    from packages.grounding.models import digest
    from packages.tool_admission.models import AdmissionResult, JSONOutputRule
    from tests.tool_admission.support import reseal
    from tests.tool_admission.test_research import research

    f = fixture(tmp_path)
    old_request = research(f, tmp_path)
    profile = f.owners.profiles[old_request.invocation.manifest_checksum]
    rule = validate(
        JSONOutputRule,
        {
            **profile.output_rules[0].model_dump(mode="json"),
            "media_types": ["application/json"],
            "dimensions_min": [1],
            "dimensions_max": [1],
        },
    )
    updated = reseal(profile, output_rules=[rule])
    f.owners.profiles[updated.manifest_checksum] = updated
    probe = f.owners.probes[profile.checksum]
    f.owners.probes[updated.checksum] = reseal(
        probe, profile_checksum=updated.checksum, runtime_checksum=digest(updated.runtime)
    )
    request = reseal(
        old_request,
        invocation={
            **old_request.invocation.model_dump(mode="json"),
            "profile_checksum": updated.checksum,
        },
    )
    f.owners.requests[(request.code, request.version, request.checksum)] = request
    result = f.service.admit_research(request)
    assert result.status == "preparation_only", result.model_dump_json()
    assert isinstance(result.retained[0].output_rule, JSONOutputRule)
    assert validate(AdmissionResult, result.model_dump(mode="json")) == result
    with pytest.raises(ValueError, match="json_rule_is_output_only"):
        validate(
            JSONOutputRule,
            {**rule.model_dump(mode="json"), "port": old_request.candidate.inputs[0].port},
        )
