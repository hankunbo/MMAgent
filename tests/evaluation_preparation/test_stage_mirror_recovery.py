"""Original bound records, not damaged local mirrors, establish historical progress."""

import json
from types import SimpleNamespace

import pytest

from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import canonical, digest
from packages.multimodal_research import stage_failure, stages


@pytest.mark.parametrize("raw", [b'{"run_id":', b"\xe8\xa7", b"[]", b'{"wrong":1}'])
def test_status_reports_unreadable_receipt_without_execution(tmp_path, raw):
    (tmp_path / "protocol.json").write_text('{"deadline":"2030-01-01T00:00:00+00:00"}')
    (tmp_path / "stages").mkdir()
    path = tmp_path / "stages" / "export.json"
    path.write_bytes(raw)
    result = stages.inspect_local(tmp_path)
    assert result["schema_version"] == "multimodal-stage-status/1.1.0"
    assert result["next_stage"] is None and result["stages"]["export"] is None
    assert result["execution_authorized"] is False
    assert result["automatic_replay_authorized"] is False
    assert result["local_mirror_errors"][0]["bytes"] == len(raw)
    assert path.read_bytes() == raw


def fixture(tmp_path):
    protocol = {"input": "frozen", "task": "one"}
    rid = identity("RUN", "task-research:review")
    phase = dict(run_id=rid, target=None, accepted=True, acceptance_key="one:domain")
    failure = dict(
        run_id=rid, campaign_id="campaign", protocol_checksum=digest(protocol), stage="review"
    )
    records = {
        "dynamic-campaign": (protocol,),
        "dynamic-phase-result": (phase,),
        stage_failure.KEY: (failure,),
    }
    campaign = SimpleNamespace(
        protocol=protocol,
        output=tmp_path,
        campaign_id="campaign",
        targets=(),
        request_key=lambda target: "review" if target is None else target,
        contexts=SimpleNamespace(
            read_research_run=lambda run, **kw: object() if run == rid else None
        ),
        documents=lambda run: SimpleNamespace(read=lambda key: records.get(key)),
        phase_path=lambda target: tmp_path,
    )
    return campaign, phase, failure, records


@pytest.mark.parametrize("which", ["phase", "failure"])
def test_bound_owner_reconstructs_only_exact_partial_mirror(tmp_path, which):
    campaign, phase, failure, records = fixture(tmp_path)
    name, value, read = (
        ("result.json", phase, lambda: stages.confirmed_phase(campaign, None))
        if which == "phase"
        else ("stage-failure.json", failure, lambda: stage_failure.read_saved_failure(campaign))
    )
    path = tmp_path / name
    raw = canonical(value)[:8]
    path.write_bytes(raw)
    assert read() == value and path.read_bytes() == raw
    path.write_text(json.dumps(value, indent=3))
    assert read() == value
    path.write_text('{"foreign":1}')
    with pytest.raises(ValueError):
        read()
    path.write_bytes(b'{"foreign":')
    with pytest.raises(ValueError):
        read()


@pytest.mark.parametrize("fault", ["missing", "run", "campaign", "protocol", "multiple"])
def test_failure_fragment_cannot_create_owner_authority(tmp_path, fault):
    campaign, phase, failure, records = fixture(tmp_path)
    path = tmp_path / "stage-failure.json"
    path.write_bytes(b"{")
    if fault == "missing":
        records.pop(stage_failure.KEY)
    elif fault == "multiple":
        records[stage_failure.KEY] = (failure, failure)
    else:
        field = {"run": "run_id", "campaign": "campaign_id", "protocol": "protocol_checksum"}[fault]
        records[stage_failure.KEY] = (dict(failure, **{field: "wrong"}),)
    with pytest.raises(ValueError, match="stage_failure_mirror_unconfirmed"):
        stage_failure.read_saved_failure(campaign)
    assert path.read_bytes() == b"{"
