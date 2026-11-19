"""Fair terminal status uses sealed original requests and never dispatches."""

import json

import pytest

from packages.evaluation_preparation import cohort_progress as module
from packages.grounding.models import canonical, digest
from tests.tool_admission.support import fixture
from tests.tool_admission.test_research import research


def setup(tmp_path):
    request = research(fixture(tmp_path), tmp_path)
    out = tmp_path / "case-input"
    evidence = out / "evidence"
    evidence.mkdir(parents=True)
    value = dict(key=out.name)
    target = "criterion:LOCAL_TEST"
    plan = [dict(id=target, action="observe_partial")]
    image = request.artifacts[0]
    (out / "input.json").write_bytes(canonical(value))
    (out / "binding.json").write_bytes(
        canonical(
            dict(
                case_code=request.scope.case_code,
                image=dict(asset_code=image.asset_code, checksum=image.checksum),
            )
        )
    )
    (evidence / "task-plan.json").write_bytes(canonical(plan))
    (evidence / "protocol.json").write_bytes(
        canonical(
            dict(
                schema_version="generic-evidence-campaign/1.0.0",
                input=digest(value),
                case_code=request.scope.case_code,
                observations=dict(plan=digest(plan)),
                deadline="2026-10-20T14:00:00+00:00",
            )
        )
    )
    phase = evidence / ("vision-" + digest(target)[7:19])
    phase.mkdir()
    (phase / "request.json").write_bytes(canonical(request))
    result = dict(
        schema_version="dynamic-phase-result/1.0.0",
        run_id=request.lock.run_id,
        target=target,
        accepted=True,
        error=None,
        acceptance_key="synthetic-acceptance",
        requests=dict(
            run_id=request.lock.run_id,
            api_requests_observed=1,
            api_requests_unknown=0,
            reservations=1,
            completed=1,
        ),
    )
    (phase / "result.json").write_bytes(canonical(result))
    return out, phase, result


def test_saved_observation_is_not_whole_case_success(tmp_path):
    out, phase, _ = setup(tmp_path)
    before = {p: p.read_bytes() for p in out.rglob("*.json")}
    result = module.fair(out)
    assert result["saved_accepted_visual_phases"] == 1
    assert result["saved_accepted_review_phases"] == 0
    assert result["phase_api_requests_lower_bound"] == 1
    assert result["completion"]["all_obtained_reviewed"] is None
    assert result["completion"]["pending_review_ids"] is None
    assert result["authorization"] is False and result["errors"] == []
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.parametrize("mode", ["run", "target", "count", "seal", "scope"])
def test_bad_mirror_not_counted(tmp_path, mode):
    out, phase, result = setup(tmp_path)
    if mode == "run":
        result["run_id"] = "RUN:OTHER"
    elif mode == "target":
        result["target"] = "criterion:OTHER"
    elif mode == "count":
        result["requests"]["api_requests_observed"] = True
    elif mode == "seal":
        (phase / "request.json").write_text("{}")
    else:
        value = json.loads((out / "binding.json").read_bytes())
        value["image"]["checksum"] = digest("other-image")
        (out / "binding.json").write_bytes(canonical(value))
    (phase / "result.json").write_bytes(canonical(result))
    value = module.fair(out)
    assert value["saved_accepted_visual_phases"] == 0
    assert value["phase_api_requests_lower_bound"] == 0
    assert len(value["errors"]) == 1


def test_inflight_and_changed_plan_never_authorize_replay(tmp_path):
    out, phase, _ = setup(tmp_path)
    (phase / "result.json").unlink()
    value = module.fair(out)
    assert value["phases"][0]["status"] == "prepared_or_inflight_not_confirmed"
    assert value["automatic_replay"] is False
    (out / "evidence/task-plan.json").write_text("[]")
    with pytest.raises(ValueError, match="protocol_binding"):
        module.fair(out)


def final_fixture(tmp_path, monkeypatch):
    out, visual, phase = setup(tmp_path)
    review = out / "evidence" / "review"
    review.mkdir()
    request = json.loads((visual / "request.json").read_bytes())
    # Only a final phase is needed here; preserve the original sealed request.
    (visual / "result.json").unlink()
    (review / "request.json").write_bytes(canonical(request))
    final = dict(phase, target=None)
    (review / "result.json").write_bytes(canonical(final))
    binding = json.loads((out / "binding.json").read_bytes())
    result = dict(case_code=binding["case_code"], phases=[final])
    (out / "evidence" / "result.json").write_bytes(canonical(result))
    report = dict(
        result=result,
        completion=dict(
            report_saved=True,
            domain_accepted=True,
            all_obtained_reviewed=True,
            pending_review_ids=[],
        ),
        counts=dict(requirements=2, acquired=1, reviewed=1, directed=0),
    )
    exports = tmp_path / "exports"
    exports.mkdir()
    monkeypatch.setattr(module, "EXPORT_ROOT", exports)
    path = exports / ("FairVLMed_" + out.name + ".json")
    path.write_bytes(canonical(report))
    return out, path, report


def test_final_coverage_does_not_claim_every_item_has_direction(tmp_path, monkeypatch):
    out, path, _ = final_fixture(tmp_path, monkeypatch)
    raw = path.read_bytes()
    completion = module.fair(out)["completion"]
    assert completion["status"] == "saved_report_mirror"
    assert completion["all_obtained_reviewed"] is True
    assert completion["counts"]["directed"] == 0
    assert completion["whole_case_success_not_inferred"] is True
    assert path.read_bytes() == raw


@pytest.mark.parametrize("mode", ["case", "phase", "pending", "count", "accepted"])
def test_invalid_final_report_keeps_completion_unknown(tmp_path, monkeypatch, mode):
    out, path, report = final_fixture(tmp_path, monkeypatch)
    if mode == "case":
        report["result"]["case_code"] = "CASE:OTHER"
    elif mode == "phase":
        report["result"]["phases"][0]["acceptance_key"] = "other"
    elif mode == "pending":
        report["completion"]["pending_review_ids"] = ["still-pending"]
    elif mode == "count":
        report["counts"]["directed"] = True
    else:
        report["completion"]["domain_accepted"] = False
    path.write_bytes(canonical(report))
    value = module.fair(out)["completion"]
    assert value["status"] == "final_report_not_readable"
    assert value["all_obtained_reviewed"] is None


def test_intermediate_html_displays_unknown_not_zero(tmp_path):
    from packages.evaluation_preparation.terminal_cohort import progress_text

    out, _, _ = setup(tmp_path)
    text = progress_text(module.fair(out))
    assert "待评阅 未知" in text
    assert "观察阶段已接纳 1" in text
    assert "Review阶段已接纳 0" in text
