"""Original request seal and precise phase identity guard local status projections."""

import json

import pytest

from packages.evaluation_preparation import cohort_progress as progress
from packages.grounding.models import canonical, digest, seal
from packages.research_runtime.task_context import TaskRunRequest
from tests.research_runtime.test_task_context import intent


def fixture(tmp_path):
    out = tmp_path / "case"
    out.mkdir()
    task = intent()
    checksum = digest("input")

    def ref(kind):
        return dict(kind=kind, code=kind.upper() + ":TEST", version="1.0.0", checksum=digest(kind))

    request = seal(
        TaskRunRequest,
        scope=dict(
            case_code="CASE:ANON_ABCDEF",
            patient_hash=digest("patient"),
            encounter_code="ENCOUNTER:ANON_ABCDEF",
            laterality="RIGHT",
        ),
        lock=dict(run_id="RUN:ANON_ABCDEF", catalog_ref=ref("catalog")),
        task=task,
        policy_ref=ref("policy"),
        input_checksum=checksum,
        visible_checksum=digest("visible"),
        artifacts=[
            dict(
                asset_code="ASSET:ANON_ABCDEF",
                checksum=digest("artifact"),
                artifact_type_code="ARTIFACT:TEST",
                media_type="application/json",
            )
        ],
        selected_components=["fundus"],
    )
    (out / "entry.json").write_bytes(canonical(dict(input_checksum=checksum, intent=task)))
    (out / "protocol.json").write_bytes(
        canonical(dict(campaign_id=out.name, input=checksum, deadline="2026-01-01T00:00:00+00:00"))
    )
    folder = out / ("vision-" + digest("active:test")[7:19])
    folder.mkdir()
    (folder / "request.json").write_bytes(canonical(request))
    result = dict(
        schema_version="dynamic-phase-result/1.0.0",
        run_id=request.lock.run_id,
        target="active:test",
        accepted=True,
        acceptance_key="test-domain",
        error=None,
        requests=dict(
            run_id=request.lock.run_id,
            api_requests_observed=1,
            api_requests_unknown=0,
            reservations=1,
            completed=1,
        ),
    )
    (folder / "result.json").write_bytes(canonical(result))
    return out, folder, result


def test_saved_observation_is_not_review_or_complete_total(tmp_path):
    out, folder, result = fixture(tmp_path)
    raw = (folder / "result.json").read_bytes()
    value = progress.multimodal(out)
    assert value["saved_accepted_visual_phases"] == 1
    assert value["saved_accepted_review_phases"] == 0
    assert value["phase_api_requests_lower_bound"] == 1
    assert value["source_or_unfinished_requests_not_inferred"] is True
    assert value["authorization"] is False and value["automatic_replay"] is False
    assert value["errors"] == [] and (folder / "result.json").read_bytes() == raw


@pytest.mark.parametrize(
    "mode",
    [
        "wrong_run",
        "wrong_target",
        "negative_count",
        "boolean_count",
        "bad_json",
        "unsealed_request",
    ],
)
def test_corrupt_or_cross_case_mirror_is_diagnostic_not_success(tmp_path, mode):
    out, folder, result = fixture(tmp_path)
    if mode == "wrong_run":
        result["run_id"] = "RUN:ANON_OTHERX"
    elif mode == "wrong_target":
        result["target"] = "active:elsewhere"
    elif mode == "negative_count":
        result["requests"]["api_requests_observed"] = -1
    elif mode == "boolean_count":
        result["requests"]["api_requests_observed"] = True
    elif mode == "bad_json":
        (folder / "result.json").write_bytes(b"{")
    else:
        (folder / "request.json").write_bytes(b"{}")
    if mode not in {"bad_json", "unsealed_request"}:
        (folder / "result.json").write_bytes(canonical(result))
    value = progress.multimodal(out)
    assert value["saved_accepted_visual_phases"] == 0
    assert value["phase_api_requests_lower_bound"] == 0
    assert len(value["errors"]) == 1


def test_unconfirmed_request_and_symlink_are_not_success(tmp_path):
    out, folder, result = fixture(tmp_path)
    (folder / "result.json").unlink()
    value = progress.multimodal(out)
    assert value["phases"][0]["status"] == "prepared_or_inflight_not_confirmed"
    assert value["saved_accepted_visual_phases"] == 0
    link = tmp_path / "alias"
    link.symlink_to(out, target_is_directory=True)
    with pytest.raises(ValueError, match="cohort_progress_symlink"):
        progress.multimodal(link)


def final_fixture(tmp_path, monkeypatch):
    out, folder, phase = fixture(tmp_path)
    # A local completed record can still have an unconfirmed remote response.
    phase["transport"] = {"remote_response_unknown": 1}
    (folder / "result.json").write_bytes(canonical(phase))
    entry = json.loads((out / "entry.json").read_bytes())
    result = dict(complete=True, task=entry["intent"], phases=[phase])
    (out / "result.json").write_bytes(canonical(result))
    exports = tmp_path / "exports"
    exports.mkdir()
    monkeypatch.setattr(progress, "EXPORT_ROOT", exports)
    path = exports / ("Multimodal_" + out.name + ".json")
    report = dict(
        result=result,
        task=entry["intent"],
        completion=dict(
            report_saved=True,
            domain_accepted=True,
            all_obtained_reviewed=False,
            selected_increment_reviewed=False,
            pending_review_ids=["check:b", "check:c"],
        ),
        counts=dict(requirements=23, obtained_observations=3, reviewed=1),
    )
    path.write_bytes(canonical(report))
    return out, path, report


def test_accepted_partial_result_true_never_implies_completed_case(tmp_path, monkeypatch):
    out, path, report = final_fixture(tmp_path, monkeypatch)
    original = path.read_bytes()
    value = progress.multimodal(out)["completion"]
    assert value["reported_result_complete"] is True
    assert value["status"] == "accepted_partial_review"
    assert value["pending_review_ids"] == ["check:b", "check:c"]
    assert value["remote_response_unknown_lower_bound"] == 1
    assert value["counts"]["reviewed"] == 1
    assert value["whole_case_success_not_inferred"] is True
    assert value["authorization"] is False
    assert path.read_bytes() == original
    from packages.evaluation_preparation.terminal_cohort import progress_text

    text = progress_text(progress.multimodal(out))
    assert "待评阅 2" in text and "远端响应未知至少 1" in text


@pytest.mark.parametrize(
    "mode",
    ["cross_task", "changed_result", "missing_phase", "pending_conflict", "bad_count"],
)
def test_invalid_final_mirror_does_not_report_success(tmp_path, monkeypatch, mode):
    out, path, report = final_fixture(tmp_path, monkeypatch)
    if mode == "cross_task":
        report["task"] = {}
    elif mode == "changed_result":
        report["result"]["complete"] = False
    elif mode == "missing_phase":
        for folder in out.glob("vision-*"):
            (folder / "result.json").unlink()
    elif mode == "pending_conflict":
        report["completion"]["all_obtained_reviewed"] = True
    else:
        report["counts"]["reviewed"] = True
    path.write_bytes(canonical(report))
    value = progress.multimodal(out)["completion"]
    assert value["status"] == "completion_mirror_not_readable"
    assert value["authorization"] is False


def test_legacy_report_and_absent_report_remain_unknown(tmp_path, monkeypatch):
    out, path, report = final_fixture(tmp_path, monkeypatch)
    del report["completion"]
    path.write_bytes(canonical(report))
    assert progress.multimodal(out)["completion"]["status"] == "legacy_completion_unknown"
    path.unlink()
    assert progress.multimodal(out)["completion"]["status"] == "final_report_not_saved"


def test_selected_review_finished_does_not_assert_full_coverage(tmp_path, monkeypatch):
    out, path, report = final_fixture(tmp_path, monkeypatch)
    report["completion"].update(selected_increment_reviewed=True, pending_review_ids=[])
    report["result"]["phases"][0]["transport"]["remote_response_unknown"] = 0
    for folder in out.glob("vision-*"):
        (folder / "result.json").write_bytes(canonical(report["result"]["phases"][0]))
    (out / "result.json").write_bytes(canonical(report["result"]))
    path.write_bytes(canonical(report))
    value = progress.multimodal(out)["completion"]
    assert value["status"] == "accepted_review_saved"
    assert value["all_obtained_reviewed"] is False
    assert value["whole_case_success_not_inferred"] is True


@pytest.mark.parametrize(
    "version",
    ["1.35.0", "1.36.0", "1.38.0", "1.39.0", "1.40.0", "1.42.0", "1.43.0", "1.45.0", "1.46.0"],
)
def test_text_successor_mirror_uses_sealed_request_not_new_entry(tmp_path, monkeypatch, version):
    out, path, report = final_fixture(tmp_path, monkeypatch)
    folder = next(out.glob("vision-*"))
    folder.rename(out / "review")
    phase = report["result"]["phases"][0]
    phase["target"] = None
    (out / "review/result.json").write_bytes(canonical(phase))
    (out / "result.json").write_bytes(canonical(report["result"]))
    path.write_bytes(canonical(report))
    request = json.loads((out / "review/request.json").read_bytes())
    protocol = dict(
        schema_version="multimodal-campaign-renewal/" + version,
        mode="review",
        input=request["input_checksum"],
        task=request["task"]["checksum"],
        deadline="2026-01-01T00:00:00+00:00",
    )
    (out / "entry.json").unlink()
    (out / "protocol.json").write_bytes(canonical(protocol))
    value = progress.multimodal(out)
    assert value["completion"]["status"] == "accepted_partial_review"
    assert value["identity_source"] == "renewal_sealed_review_request_not_ancestor_authorization"
    protocol["task"] = digest("wrong-task")
    (out / "protocol.json").write_bytes(canonical(protocol))
    with pytest.raises(ValueError, match="cohort_progress_renewal_binding"):
        progress.multimodal(out)


@pytest.mark.parametrize("version", ["1.40.0", "1.43.0", "1.45.0", "1.46.0"])
def test_method_successor_missing_request_is_unknown_not_zero_dispatch(tmp_path, version):
    (tmp_path / "protocol.json").write_bytes(
        canonical(
            dict(
                schema_version="multimodal-campaign-renewal/" + version,
                mode="review",
            )
        )
    )
    value = progress.multimodal(tmp_path)
    assert value["status"] == "renewal_request_mirror_not_available"
    assert value["request_state"] == "unknown"
    assert value["authorization"] is value["automatic_replay"] is False
    assert value["diagnostic_api_requests"] == 0
    assert "api_requests_observed" not in value


def retained_fixture(tmp_path):
    out, folder, result = fixture(tmp_path)
    request = json.loads((folder / "request.json").read_bytes())
    protocol = dict(
        schema_version="multimodal-campaign-renewal/1.37.0",
        mode="acquisition",
        input=request["input_checksum"],
        task=request["task"]["checksum"],
        acquisition_targets=["active:test"],
        deadline="2026-01-01T00:00:00+00:00",
    )
    (out / "entry.json").unlink()
    (out / "protocol.json").write_bytes(canonical(protocol))
    return out, folder, result, protocol


def test_retained_visual_before_review_has_readonly_progress(tmp_path):
    out, folder, _, _ = retained_fixture(tmp_path)
    raw = (folder / "result.json").read_bytes()
    value = progress.multimodal(out)
    assert value["saved_accepted_visual_phases"] == 1
    assert value["saved_accepted_review_phases"] == 0
    assert value["automatic_replay"] is value["authorization"] is False
    assert value["diagnostic_api_requests"] == 0 and value["errors"] == []
    assert (folder / "result.json").read_bytes() == raw


@pytest.mark.parametrize("field", ["mode", "input", "task", "acquisition_targets"])
def test_wrong_retained_binding_rejected(tmp_path, field):
    out, _, _, protocol = retained_fixture(tmp_path)
    replacements = dict(
        mode="review", acquisition_targets=[], input=digest("other"), task=digest("other")
    )
    protocol[field] = replacements[field]
    (out / "protocol.json").write_bytes(canonical(protocol))
    with pytest.raises(ValueError, match="cohort_progress_renewal_binding"):
        progress.multimodal(out)


def test_retained_without_request_is_unknown_not_replayable(tmp_path):
    out, folder, _, _ = retained_fixture(tmp_path)
    (folder / "request.json").unlink()
    value = progress.multimodal(out)
    assert value["status"] == "renewal_request_mirror_not_available"
    assert value["request_state"] == "unknown"
    assert value["authorization"] is value["automatic_replay"] is False
    assert "api_requests_observed" not in value


def test_retained_final_review_mirror_and_outside_target(tmp_path):
    out, folder, phase, protocol = retained_fixture(tmp_path)
    folder.rename(out / "review")
    phase["target"] = None
    (out / "review/result.json").write_bytes(canonical(phase))
    value = progress.multimodal(out)
    assert value["saved_accepted_review_phases"] == 1
    assert (
        value["identity_source"] == "renewal_sealed_acquisition_request_not_ancestor_authorization"
    )
    outside = out / ("vision-" + digest("active:outside")[7:19])
    outside.mkdir()
    value = progress.multimodal(out)
    assert len(value["errors"]) == 1
    assert value["saved_accepted_review_phases"] == 1
    assert value["authorization"] is False


def test_terminal_inspection_does_not_run_or_initialize(tmp_path, monkeypatch, capsys):
    import sys

    from apps import case_cohort

    out, path, report = final_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["case_cohort", "inspect-multimodal", "--input", str(out)])

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only inspection cannot dispatch")

    monkeypatch.setattr(case_cohort.owner, "run", forbidden)
    monkeypatch.setattr(case_cohort.owner, "initialize", forbidden)
    assert case_cohort.main() == 0
    value = json.loads(capsys.readouterr().out)
    assert value["completion"]["status"] == "accepted_partial_review"
    assert value["diagnostic_api_requests"] == 0


@pytest.mark.parametrize("accepted", [False, True])
def test_original_dynamic_report_acquired_count_is_read_without_rewriting(
    tmp_path, monkeypatch, accepted
):
    out, path, report = final_fixture(tmp_path, monkeypatch)
    report["schema_version"] = "dynamic-case-report/1.0.0"
    report["counts"]["acquired"] = report["counts"].pop("obtained_observations")
    report["completion"]["domain_accepted"] = accepted
    report["result"]["complete"] = accepted
    report["result"]["phases"][0]["accepted"] = accepted
    report["result"]["phases"][0]["acceptance_key"] = "test-domain" if accepted else None
    if not accepted:
        report["counts"]["reviewed"] = 0
    for folder in out.glob("vision-*"):
        (folder / "result.json").write_bytes(canonical(report["result"]["phases"][0]))
    (out / "result.json").write_bytes(canonical(report["result"]))
    path.write_bytes(canonical(report))
    original = path.read_bytes()
    value = progress.multimodal(out)["completion"]
    assert value["status"] == (
        "accepted_partial_review" if accepted else "saved_without_domain_acceptance"
    )
    assert value["counts"]["obtained_observations"] == 3
    assert value["counts"]["reviewed"] == (1 if accepted else 0)
    assert value["remote_response_unknown_lower_bound"] == 1
    assert value["authorization"] is False
    assert path.read_bytes() == original


@pytest.mark.parametrize("mode", ["unknown_version", "conflicting_alias", "boolean", "negative"])
def test_original_acquired_alias_remains_versioned_and_strict(tmp_path, monkeypatch, mode):
    out, path, report = final_fixture(tmp_path, monkeypatch)
    report["schema_version"] = "dynamic-case-report/1.0.0"
    report["counts"]["acquired"] = report["counts"].pop("obtained_observations")
    if mode == "unknown_version":
        report["schema_version"] = "unrecognized/99.0"
    elif mode == "conflicting_alias":
        report["counts"]["obtained_observations"] = 9
    else:
        report["counts"]["acquired"] = True if mode == "boolean" else -1
    path.write_bytes(canonical(report))
    value = progress.multimodal(out)["completion"]
    assert value["status"] == "completion_mirror_not_readable"
    assert value["authorization"] is False


def test_retained_total_and_review_scope_are_not_conflicting_aliases(tmp_path, monkeypatch):
    out, path, report = final_fixture(tmp_path, monkeypatch)
    report["schema_version"] = "dynamic-case-report/1.0.0"
    report["counts"].update(
        acquired=4,
        obtained_observations=3,
        review_scope_observations=3,
        retained_unscored_observations=1,
    )
    report["completion"]["retained_context_not_newly_reviewed_ids"] = ["active:retained"]
    path.write_bytes(canonical(report))
    original = path.read_bytes()
    value = progress.multimodal(out)["completion"]
    assert value["status"] == "accepted_partial_review"
    assert value["counts"]["obtained_observations"] == 4
    assert value["counts"]["review_scope_observations"] == 3
    assert value["all_obtained_reviewed"] is False
    assert value["authorization"] is False
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "mode", ["missing_ids", "wrong_count", "bool_count", "duplicated_ids", "false_complete"]
)
def test_invalid_retained_split_is_not_normalized(tmp_path, monkeypatch, mode):
    out, path, report = final_fixture(tmp_path, monkeypatch)
    report["schema_version"] = "dynamic-case-report/1.0.0"
    report["counts"].update(
        acquired=4,
        obtained_observations=3,
        review_scope_observations=3,
        retained_unscored_observations=1,
    )
    report["completion"]["retained_context_not_newly_reviewed_ids"] = ["active:retained"]
    if mode == "missing_ids":
        del report["completion"]["retained_context_not_newly_reviewed_ids"]
    elif mode == "wrong_count":
        report["counts"]["retained_unscored_observations"] = 2
    elif mode == "bool_count":
        report["counts"]["retained_unscored_observations"] = True
    elif mode == "duplicated_ids":
        report["completion"]["retained_context_not_newly_reviewed_ids"] *= 2
    else:
        report["completion"].update(all_obtained_reviewed=True, pending_review_ids=[])
    path.write_bytes(canonical(report))
    assert progress.multimodal(out)["completion"]["status"] == "completion_mirror_not_readable"
