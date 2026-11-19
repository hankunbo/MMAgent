"""Method results are not silently coerced into Memory or re-reviewed to export."""

import json
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import memory, report_recovery


@pytest.mark.parametrize("version", ["1.35.0", "1.36.0"])
def test_unimplemented_memory_contract_is_explicit_before_credentials(monkeypatch, version):
    from apps import case_memory_store

    def forbidden():
        pytest.fail("unsupported method must not load Memory credentials")

    monkeypatch.setattr(case_memory_store, "configuration", forbidden)
    campaign = SimpleNamespace(
        protocol=dict(schema_version="multimodal-campaign-renewal/" + version)
    )
    result = dict(
        complete=True,
        phases=[dict(accepted=True)],
        schema_version="multimodal-method-"
        + ("reviewed" if version == "1.35.0" else "continued")
        + "-case-result/1.0.0",
    )
    value = memory.archive(campaign, result)
    assert value["status"] == "deferred_unsupported_method_result_contract"
    assert value["candidate_archived"] is value["eligible_for_planning"] is False
    with pytest.raises(ValueError, match="multimodal_memory_method_result_binding"):
        memory.archive(campaign, dict(result, schema_version="multimodal-case-result/1.0.0"))
    assert (
        memory.archive(campaign, dict(result, complete=False))["status"]
        == "not_archived_incomplete"
    )


def fixture(tmp_path, monkeypatch):
    from packages.multimodal_research import campaign_renewal

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "case"
    path.mkdir()
    (path / "lock").touch()
    protocol = dict(schema_version="multimodal-campaign-renewal/1.36.0", mode="review")
    final = dict(run_id="RUN:METHOD", target=None, accepted=True, acceptance_key="accepted")
    task = dict(target_code="GOAL:PROPOSITION")
    result = dict(
        schema_version="multimodal-method-continued-case-result/1.0.0",
        complete=True,
        phases=[final],
        case_code="CASE:METHOD",
        task=task,
        dataset="synthetic",
        new_requests=1,
    )
    scores = dict(reviews=[dict(id="test", judgment=dict(value=0.04))])
    detail = dict(records={"original": True}, graph={"graph": True}, receipt={"status": "accepted"})
    report = dict(result=result, task=task, original_review=detail, scores=scores)
    page = "<!doctype html><p>Saved exact report &amp; score</p>"
    failure = dict(stage="export", display_completed=False, score_status="saved")
    values = {
        "dynamic-phase-result": (final,),
        "multimodal-case-result": (result,),
        "dynamic-case-report": (report, dict(html=page, report_checksum=digest(report))),
        "accepted:accepted": (
            detail["records"],
            detail["graph"],
            dict(scores=scores),
            detail["receipt"],
        ),
        "multimodal-stage-failure": (failure,),
    }
    docs = SimpleNamespace(run_id=final["run_id"], read=values.get)
    control = SimpleNamespace(phase="stopped", pending=None, cancelled=True)
    base = SimpleNamespace(
        binding=dict(case_code=result["case_code"]),
        task=SimpleNamespace(model_dump=lambda **_: task),
        manifest=SimpleNamespace(dataset="synthetic"),
        check=lambda: None,
    )
    campaign = SimpleNamespace(
        output=path,
        protocol=protocol,
        base=base,
        documents=lambda _: docs,
        sessions=SimpleNamespace(read=lambda _: control),
    )
    monkeypatch.setattr(campaign_renewal, "assemble", lambda _: campaign)
    (path / "protocol.json").write_text(json.dumps(protocol))
    (path / "result.json").write_text(json.dumps(result))
    (path / "stage-failure.json").write_text(json.dumps(failure))
    return path, campaign, values, page


def test_saved_method_restore_exact_repeat_preserves_failure(tmp_path, monkeypatch):
    path, campaign, values, page = fixture(tmp_path, monkeypatch)
    from packages.multimodal_research import report

    monkeypatch.setattr(report, "export", lambda *a, **k: pytest.fail("must not render or score"))
    before = (path / "stage-failure.json").read_bytes()
    first = report_recovery.restore(path)
    record = (path / "report-restoration.json").read_bytes()
    assert first == report_recovery.restore(path)
    assert (path / "report-restoration.json").read_bytes() == record
    assert (path / "stage-failure.json").read_bytes() == before
    assert first["new_api_requests"] == first["database_writes"] == first["new_model_runs"] == 0
    assert first["memory"]["candidate_archived"] is False
    assert (tmp_path / first["html"]).read_text() == page
    assert campaign.sessions.read("RUN:METHOD").cancelled is True
    assert values["multimodal-case-result"][0]["new_requests"] == 1


@pytest.mark.parametrize(
    "fault", ["pending", "active", "phase", "result", "report", "source", "failure"]
)
def test_bad_owner_or_source_never_creates_report(tmp_path, monkeypatch, fault):
    path, campaign, values, _ = fixture(tmp_path, monkeypatch)
    if fault == "pending":
        campaign.sessions.read("").pending = object()
    elif fault == "active":
        campaign.sessions.read("").phase = "select"
    elif fault == "phase":
        values["dynamic-phase-result"] = (dict(other=True),)
    elif fault == "result":
        values["multimodal-case-result"] = (dict(other=True),)
    elif fault == "report":
        del values["dynamic-case-report"]
    elif fault == "source":

        def changed():
            raise ValueError("original_source_changed")

        campaign.base.check = changed
    else:
        (path / "stage-failure.json").write_text("{}")
    with pytest.raises(ValueError):
        report_recovery.restore(path)
    assert not (tmp_path / "docs/exports").exists()
    assert not (path / "report-restoration.json").exists()


def test_explicit_legacy_route_and_unknown_method_result_reject(tmp_path, monkeypatch):
    path, _, values, _ = fixture(tmp_path, monkeypatch)
    raw = json.loads((path / "result.json").read_bytes())
    raw["schema_version"] = "multimodal-case-result/1.0.0"
    (path / "result.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="multimodal_memory_method_result_binding"):
        report_recovery.restore(path)
    assert values["multimodal-case-result"][0]["schema_version"].startswith("multimodal-method-")
