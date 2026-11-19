"""Contract and Fake evidence boundaries, with original 06 complete inputs."""

from datetime import UTC, datetime, timedelta

import pytest

from packages.grounding.models import digest, validate
from packages.tool_adapter import AdapterManifest, AdapterResult, FakeProtocol, SyntheticAuthority
from packages.tool_adapter.models import AdapterPlan, AlgorithmConfiguration
from packages.tool_discovery.matching import match

from .support import plan


@pytest.mark.parametrize(
    "case",
    [
        "ok",
        "safety",
        "interface",
        "empty_mask",
        "performance",
        "repeatability",
        "resource",
        "technical_failure",
        "unknown",
        "not_installed",
    ],
)
def test_fake_evidence_never_grants_medical_or_tool_authority(tmp_path, case):
    p, f = plan(tmp_path, case=case)
    result = f.evaluate(p)
    expected = {
        "ok": "missing",
        "technical_failure": "technical_failure",
        "unknown": "unknown",
        "not_installed": "not_executed",
    }.get(case, "failed")
    assert result.outcome == expected
    assert result.validation_status == "unvalidated"
    assert (
        not result.available_next_run
        and not result.catalog_written
        and not result.formal_job_created
    )
    assert result.gates[1].evidence["executed"] is False
    assert all(g.outcome == "not_executed" for g in result.gates[2:])
    assert all(
        g.policy_checksum == digest(p.policy) and g.input_checksum == digest(p)
        for g in result.gates
    )
    if expected == "failed":
        assert not result.gates[1].evidence["checks"][case]
        assert result.catalog_status == "quarantined" and result.extension_job_status == "FAILED"


@pytest.mark.parametrize(
    "changes",
    [
        dict(shell=True),
        dict(argv=["python", "remote.py"]),
        dict(network="enabled"),
        dict(gpu_count=1),
        dict(execution_entry="arbitrary.py"),
        dict(coordinate_conversion="guess"),
        dict(laterality_conversion="flip"),
        dict(validated=True),
    ],
)
def test_manifest_rejects_execution_and_implicit_semantic_conversion(tmp_path, changes):
    p, _ = plan(tmp_path)
    value = p.manifest.model_dump(mode="json") | changes
    with pytest.raises(ValueError):
        validate(AdapterManifest, value)


def test_algorithm_template_is_distinct_and_versioned(tmp_path):
    p, _ = plan(tmp_path)
    a = validate(
        AdapterManifest,
        p.manifest.model_dump(mode="json")
        | dict(
            kind="algorithm",
            model=None,
            algorithm=AlgorithmConfiguration(
                formula_version="1.0.0", units="fixture", numeric_range=(0.0, 1.0)
            ),
        ),
    )
    assert a.algorithm.deterministic and a.model is None
    assert digest(a) != digest(p.manifest)


@pytest.mark.parametrize(
    "change", ["input", "preprocessing", "shape", "dependency", "version", "policy"]
)
def test_full_contract_configuration_and_policy_change_break_grant(tmp_path, change):
    p, _ = plan(tmp_path)
    value = p.model_dump(mode="json")
    m = value["manifest"]
    if change == "input":
        port = m["candidate"]["semantics"]["inputs"][0]
        port["cardinality_min"] = port["cardinality_max"] = 2
    elif change == "preprocessing":
        m["candidate"]["preprocessing"] = "different preprocessing"
    elif change == "shape":
        m["candidate"]["output_shape"] = ["different"]
    elif change == "dependency":
        m["candidate"]["dependencies"] = ["different==1"]
    elif change == "version":
        m["version"] = "1.0.1"
    else:
        value["policy"]["version"] = "1.0.1"
    with pytest.raises(ValueError, match="adapter_plan_binding"):
        validate(AdapterPlan, value)


def test_stage_grants_are_current_expiring_and_owner_bound(tmp_path):
    p, _ = plan(tmp_path)
    owner = SyntheticAuthority([p.grant])
    g = p.grant
    assert owner.current(g.run_id, g.task_id, g.stage_key, g.worker) == g
    with pytest.raises(ValueError, match="stage_identity"):
        owner.current(g.run_id, g.task_id, "new-key", g.worker)
    owner.revoke(g.run_id, g.task_id)
    with pytest.raises(ValueError, match="current_grant"):
        owner.current(g.run_id, g.task_id, g.stage_key, g.worker)
    owner = SyntheticAuthority(
        [g.model_copy(update=dict(expires_at=datetime.now(UTC) - timedelta(seconds=1)))]
    )
    with pytest.raises(ValueError, match="grant_expired"):
        owner.current(g.run_id, g.task_id, g.stage_key, g.worker)


def test_machine_result_refuses_promoted_states_and_changed_evidence(tmp_path):
    p, f = plan(tmp_path)
    r = f.evaluate(p).model_dump(mode="json")
    for field, value in [
        ("validation_status", "benchmark_passed"),
        ("intended_use", "clinical_deployment"),
        ("available_next_run", True),
        ("catalog_status", "active"),
        ("outcome", "passed"),
    ]:
        with pytest.raises(ValueError):
            validate(AdapterResult, r | {field: value})
    r["gates"][1]["evidence"]["executed"] = True
    with pytest.raises(ValueError, match="evidence_binding"):
        validate(AdapterResult, r)


def test_11c_reuses_complete_quantity_and_port_semantics(tmp_path):
    p, _ = plan(tmp_path)
    r = p.manifest.candidate.model_dump(mode="json")
    i = r["semantics"]["inputs"][0]
    i["cardinality_min"] = i["cardinality_max"] = 2
    from packages.tool_discovery.models_v11 import Provenance

    assert match(validate(Provenance, r), p.query).status == "incompatible"
    i["cardinality_min"], i["cardinality_max"], i["required"] = 0, 1, False
    assert match(validate(Provenance, r), p.query).status == "compatible"
    i["port"]["port_name"] = "different"
    assert match(validate(Provenance, r), p.query).status == "missing"


def test_fake_configuration_is_bound_to_stage_plan(tmp_path):
    p, _ = plan(tmp_path)
    with pytest.raises(ValueError, match="adapter_fake_changed"):
        FakeProtocol(case="resource").evaluate(p)
