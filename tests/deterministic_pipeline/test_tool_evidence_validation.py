"""09A consumer boundary: exact synthetic return facts, never real model calls."""

from types import SimpleNamespace

import pytest

from packages.deterministic_pipeline.validation import DeterministicValidator
from packages.grounding.models import validate
from packages.tool_execution.models import StepResult
from tests.deterministic_pipeline.test_reporting import facts as facts


def validator():
    return DeterministicValidator(None, None, None)


def eligible(f, step=None):
    return validator().eligible(
        SimpleNamespace(synthetic=True),
        SimpleNamespace(pack=SimpleNamespace(definition=SimpleNamespace(synthetic=True))),
        SimpleNamespace(synthetic=True, expression_code="RULE:SYNTHETIC_ASSERTION"),
        step or f.step,
        f.plan,
    )


def changed(f, **fields):
    raw = f.step.model_dump(mode="json")
    raw["result"].update(fields)
    return validate(StepResult, raw)


def test_smoke_tested_synthetic_is_observation_only(facts):
    step = changed(facts, validation_status="smoke_tested")
    assert eligible(facts, step) == (False, "tool_validation_level_not_decisive")


def test_wrong_qc_policy_is_not_a_validation_source(facts):
    raw = facts.step.model_dump(mode="json")
    raw["qc_records"][0]["qc_policy_ref"]["code"] = "POLICY:OTHER"
    step = validate(StepResult, raw)
    with pytest.raises(ValueError, match="tool_output_qc_source_mismatch"):
        eligible(facts, step)


def test_foreign_artifact_cannot_be_interpreted(facts):
    artifact = facts.step.result.artifacts[0].model_copy(update={"asset_code": "ASSET:OTHER"})
    with pytest.raises(ValueError, match="evidence_artifact_not_in_tool_result"):
        validator().evaluate(facts.graph, facts.target, artifact, None, facts.step)


def test_qualified_tool_is_only_candidate_and_real_input_remains_closed(facts):
    from packages.deterministic_pipeline.tool_evidence import inspect_tool_evidence

    check = inspect_tool_evidence(facts.step, facts.plan)
    assert check.technical_pass and check.evidence_candidate and check.clinical_effect == "NONE"
    assert eligible(facts) == (True, "synthetic_assertion_only")
    assert validator().eligible(
        SimpleNamespace(synthetic=False), None, None, facts.step, facts.plan
    ) == (False, "nondecisive_research_observation")


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("plan_checksum", "sha256:" + "a" * 64, "tool_result_plan_source_mismatch"),
        ("step_id", "step_" + "a" * 64, "tool_result_plan_source_mismatch"),
        ("laterality", "LEFT", "tool_result_plan_source_mismatch"),
        ("producer", "IMPLEMENTATION:OID_" + "A" * 32, "tool_result_producer_mismatch"),
        ("qc_uri", "plan-output://other", "tool_output_qc_source_mismatch"),
        ("artifact_eye", "LEFT", "tool_output_applicability_mismatch"),
        (
            "artifact_producer",
            "IMPLEMENTATION:OID_" + "A" * 32,
            "tool_output_applicability_mismatch",
        ),
        ("dimensions", [99999, 3], "tool_output_format_or_range_mismatch"),
        ("media_type", "image/jpeg", "tool_output_format_or_range_mismatch"),
    ],
)
def test_tool_return_source_and_applicability(facts, field, value, error):
    raw = facts.step.model_dump(mode="json")
    if field in {"plan_checksum", "step_id"}:
        raw[field] = value
    elif field == "laterality":
        raw["result"][field] = value
    elif field == "producer":
        raw["result"]["provenance"]["tool_versions"][0]["code"] = value
    elif field == "qc_uri":
        raw["qc_records"][0]["logical_uri"] = value
    elif field == "artifact_eye":
        raw["result"]["artifacts"][0]["laterality"] = value
    elif field == "artifact_producer":
        raw["result"]["artifacts"][0]["provenance"]["tool_versions"][0]["code"] = value
    else:
        raw["result"]["artifacts"][0][field] = value
        raw["qc_records"][0][field] = value
    with pytest.raises(ValueError, match=error):
        eligible(facts, validate(StepResult, raw))


@pytest.mark.parametrize("qc", ["FAIL", "PENDING", "UNASSESSED"])
def test_execution_success_with_bad_or_unassessed_qc_is_not_evidence(facts, qc):
    raw = facts.step.model_dump(mode="json")
    raw["result"]["qc_status"] = qc
    raw["result"]["artifacts"][0]["qc_status"] = qc
    raw["qc_records"][0]["status"] = qc
    step = validate(StepResult, raw)
    assert eligible(facts, step) == (False, "qc_failure")
    with pytest.raises(ValueError, match="evidence_requires_qualified_tool_observation"):
        validator().evaluate(facts.graph, facts.target, step.result.artifacts[0], None, step)


def test_unqualified_rule_never_opens_an_interpreter(facts):
    for rule in [
        SimpleNamespace(synthetic=False, expression_code="RULE:OTHER"),
        SimpleNamespace(synthetic=True, expression_code="RULE:OTHER"),
    ]:
        with pytest.raises(ValueError, match="reviewed_interpreter_missing"):
            validator().evaluate(
                facts.graph, facts.target, facts.step.result.artifacts[0], rule, facts.step
            )


def interpretation_fixture(f, pixel):
    """Create new anonymous object bytes; do not read deleted historical object URIs."""
    import struct
    import zlib

    from packages.graph_engine import canonical_checksum
    from packages.object_store import checksum

    def chunk(kind, body):
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        )

    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 8, 0, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress((b"\x00" + bytes([pixel]) * 3) * 2)) + chunk(b"IEND", b"")
    raw = f.step.model_dump(mode="json")
    raw["result"]["artifacts"][0]["checksum"] = checksum(data)
    raw["qc_records"][0]["content_checksum"] = checksum(data)
    step = validate(StepResult, raw)
    graph = f.snapshots[f.validations[0]["reference"]["source_snapshot"]["graph_version"]]
    artifact = step.result.artifacts[0]
    rule = SimpleNamespace(
        synthetic=True,
        expression_code="RULE:SYNTHETIC_ASSERTION",
        reference=lambda: SimpleNamespace(
            to_versioned_reference=lambda: f.validations[0]["item"].rule_refs[0]
        ),
    )
    reads = []

    def read(uri, limit):
        reads.append(uri)
        assert uri == artifact.uri and len(data) < limit
        return data

    v = DeterministicValidator(None, SimpleNamespace(read_bounded=read), None)
    before = canonical_checksum(graph)
    return v, graph, artifact, rule, step, reads, before


@pytest.mark.parametrize("pixel,effect", [(1, "SUPPORTS"), (0, "REFUTES")])
def test_synthetic_positive_and_negative_use_original_03_contract(facts, pixel, effect):
    from packages.graph_engine import canonical_checksum, plan_graph_delta

    v, graph, artifact, rule, step, reads, before = interpretation_fixture(facts, pixel)
    result = v.evaluate(graph, facts.target, artifact, rule, step)
    assert result.item.clinical_effect.value == effect and result.item.validity.value == "VALID"
    assert result.item.graph_delta_candidate is not None
    plan_graph_delta(graph, result.item.graph_delta_candidate, validation_result=result.item)
    assert canonical_checksum(graph) == before and reads == [artifact.uri]


@pytest.mark.parametrize("failure", ["checksum", "format", "object_unknown"])
def test_bad_or_unknown_object_is_never_a_medical_negative(facts, failure):
    v, graph, artifact, rule, step, _, _ = interpretation_fixture(facts, 1)
    if failure == "object_unknown":

        def unavailable(*args):
            raise OSError("owned_test_object_unavailable")

        v.objects.read_bounded = unavailable
        with pytest.raises(OSError, match="owned_test_object_unavailable"):
            v.evaluate(graph, facts.target, artifact, rule, step)
        return
    data = b"not png"
    v.objects.read_bounded = lambda *_: data
    if failure == "format":
        from packages.object_store import checksum

        raw = step.model_dump(mode="json")
        raw["result"]["artifacts"][0]["checksum"] = checksum(data)
        raw["qc_records"][0]["content_checksum"] = checksum(data)
        step = validate(StepResult, raw)
        artifact = step.result.artifacts[0]
    with pytest.raises(ValueError, match="checksum_mismatch|output_format_invalid"):
        v.evaluate(graph, facts.target, artifact, rule, step)


@pytest.mark.parametrize("status", ["benchmark_passed"])
def test_other_validation_levels_do_not_become_decisive(facts, status):
    assert eligible(facts, changed(facts, validation_status=status)) == (
        False,
        "tool_validation_level_not_decisive",
    )


def test_unqualified_intended_use_is_rejected_by_original_contract(facts):
    with pytest.raises(ValueError, match="research_only intended_use"):
        changed(facts, intended_use="clinical_deployment")


def test_forged_validation_upgrade_against_retained_admission(facts):
    import json

    from packages.grounding.models import digest, seal
    from packages.tool_admission.models import AdmissionResult
    from packages.tool_workflow import ExecutionPlan
    from packages.tool_workflow.models import FormalContext, PlanDocument

    raw = json.loads(facts.plan.to_bundle_json())
    admission = raw["context"]["admission"]
    admission["retained"][0]["candidate"]["validation_required"] = "smoke_tested"
    # Retained identity itself binds the candidate, so even a fabricated context must
    # satisfy the original model before reaching 09A. Do not bypass those guards.
    with pytest.raises(ValueError):
        validate(AdmissionResult, admission)
    # Missing retained producer remains a plan-level refusal, even if all outer seals
    # have been recomputed by a caller. Re-sealing is never current admission.
    admission["retained"] = []
    admission["retained_roots"] = []
    admission["status"] = "rejected"
    admission.pop("checksum")
    raw["context"]["admission"] = seal(AdmissionResult, **admission).model_dump(mode="json")
    context = validate(FormalContext, raw["context"])
    document = raw["plan"]
    document.pop("checksum")
    document["context_checksum"] = digest(context)
    document["admission_checksum"] = context.admission.checksum
    plan = seal(PlanDocument, **document)
    bundle = ExecutionPlan.from_bundle_json(
        json.dumps(dict(plan=plan.model_dump(mode="json"), context=context.model_dump(mode="json")))
    )
    step = facts.step.model_copy(update={"plan_checksum": plan.checksum})
    with pytest.raises(ValueError, match="tool_qualification_source_mismatch"):
        eligible(SimpleNamespace(step=step, plan=bundle))


def test_unvalidated_tool_is_rejected_by_original_contract(facts):
    with pytest.raises(ValueError, match="unvalidated tools cannot execute"):
        changed(facts, validation_status="unvalidated")


def test_inconclusive_maps_to_existing_contract_none_without_delta(facts):
    from packages.contracts import ValidationResult

    raw = facts.validations[0]["item"].model_dump(mode="json")
    raw.update(
        observation="INCONCLUSIVE",
        validity="INCONCLUSIVE",
        clinical_effect="NONE",
        graph_delta_candidate=None,
    )
    value = ValidationResult.model_validate_json(__import__("json").dumps(raw))
    assert value.clinical_effect.value == "NONE" and value.graph_delta_candidate is None
    raw["clinical_effect"] = "INCONCLUSIVE"
    with pytest.raises(ValueError):
        ValidationResult.model_validate_json(__import__("json").dumps(raw))


def test_old_validator_lock_is_not_silently_migrated():
    from pathlib import Path

    from packages.contracts import VersionedReference
    from packages.deterministic_pipeline.models import PipelineConfig
    from packages.deterministic_pipeline.validation_ports import assemble_evidence_validator
    from packages.graph_engine import canonical_checksum
    from packages.grounding.models import seal

    cfg = PipelineConfig.model_validate_json(
        Path("configs/deterministic_pipeline/fairvlmed_gap.json").read_bytes()
    )
    old = VersionedReference(
        code="POLICY:DETERMINISTIC_VALIDATION",
        version="1.0.0",
        checksum=canonical_checksum(
            dict(synthetic="RULE:SYNTHETIC_ASSERTION:all_pixels_one", medical_interpreters=[])
        ),
    )
    raw = cfg.model_dump(mode="json", exclude={"checksum"})
    raw["service_refs"] = [old.model_dump(mode="json")]
    with pytest.raises(ValueError, match="validator_version_not_locked"):
        assemble_evidence_validator(
            DeterministicValidator, None, None, None, (seal(PipelineConfig, **raw),)
        )
