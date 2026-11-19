"""Original input diagnostic consumes installed method qualification, not caller flags."""

import pytest

from packages.tool_admission.models import AdmissionError
from tests.research_runtime.test_method_metadata import installed


@pytest.mark.parametrize(
    "eye,time",
    [
        ("required", "required"),
        ("required", "not_consumed"),
        ("not_consumed", "required"),
        ("not_consumed", "not_consumed"),
    ],
)
def test_gap_diagnostic_uses_original_method_eye_time_requirements(tmp_path, eye, time):
    a = installed(tmp_path, laterality=eye, acquisition_time=time)
    before = a.request.model_dump_json()
    result = a.admission.inspect_research_gap_inputs(a.request)
    reasons = {r["reason"] for r in result["missing"]}
    assert ("required_condition_unknown" in reasons) == (eye == "required")
    assert ("time_unknown" in reasons) == (time == "required")
    assert len(result["inputs"]) == 1
    assert a.request.model_dump_json() == before
    assert a.request.artifacts[0].laterality.value == "UNKNOWN"
    assert a.request.artifacts[0].acquired_at is None
    if eye == time == "not_consumed":
        assert result["missing"] == []
        assert a.admission.admit_research(a.request).retained_roots


def test_gap_current_method_evidence_drift_rejected(tmp_path):
    a = installed(tmp_path)
    from pathlib import Path

    Path(a.ready.metadata.evidence.path).write_text("changed current method review")
    with pytest.raises(ValueError, match="trusted_resource_size_drift"):
        a.admission.inspect_research_gap_inputs(a.request)


@pytest.mark.parametrize(
    "owner,method",
    [("catalogs", "read_snapshot"), ("manifests", "read_manifest"), ("inputs", "read_input")],
)
def test_gap_technical_error_not_missing_input(tmp_path, monkeypatch, owner, method):
    a = installed(tmp_path)

    def unavailable(*args, **kwargs):
        raise OSError("diagnostic_owner_unavailable")

    monkeypatch.setattr(getattr(a.admission, owner), method, unavailable)
    with pytest.raises(OSError, match="diagnostic_owner_unavailable"):
        a.admission.inspect_research_gap_inputs(a.request)


def test_missing_manifest_diagnostic_keeps_strict_legacy_input_contract(tmp_path, monkeypatch):
    a = installed(tmp_path)

    def missing(*args, **kwargs):
        raise KeyError("not installed")

    monkeypatch.setattr(a.admission.manifests, "read_manifest", missing)
    reasons = {r["reason"] for r in a.admission.inspect_research_gap_inputs(a.request)["missing"]}
    assert {"required_condition_unknown", "time_unknown"} <= reasons


def test_gap_missing_original_registration_owner_rejected(tmp_path):
    a = installed(tmp_path)
    a.admission.research = None
    with pytest.raises(AdmissionError, match="research_registration_required"):
        a.admission.inspect_research_gap_inputs(a.request)
