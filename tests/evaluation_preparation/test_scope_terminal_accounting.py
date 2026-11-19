"""Survey phases are separate from disease phases, but share the request budget."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal


@pytest.mark.parametrize(
    "fault",
    [None, "missing", "foreign", "pending", "protocol", "duplicate", "count", "fact"],
)
def test_terminal_budget_counts_new_scope_not_inherited(tmp_path, monkeypatch, fault):
    target = "input-scope:new"
    rid = identity("RUN", "task-research:mm-live:" + digest((tmp_path.name, target))[7:55])
    phase = dict(run_id=rid, target=target, accepted=True)
    review = dict(run_id="review", target=None, accepted=False)
    protocol = dict(
        schema_version="multimodal-campaign-renewal/1.83.0",
        input="input",
        task="task",
        previous_attempts=[dict(requests=34)],
        input_scope=dict(targets=["input-scope:old", target], dispatch_targets=[target]),
    )
    result = dict(
        case_code="case",
        task={"target": "test"},
        phases=[review],
        input_scopes=[dict(phase=dict(run_id="old", target="input-scope:old")), dict(phase=phase)],
        new_requests=1,
        previous_attempts=protocol["previous_attempts"],
    )
    records = {
        rid: {"dynamic-phase-result": (deepcopy(phase),), "dynamic-campaign": (protocol,)},
        "review": {
            "dynamic-phase-result": (review,),
            "dynamic-campaign": (protocol,),
            "multimodal-case-result": (result,),
        },
    }
    controls = {r: SimpleNamespace(phase="stopped", pending=None) for r in records}
    reads = []

    def documents(run):
        reads.append(run)
        return SimpleNamespace(run_id=run, read=lambda key: records[run].get(key))

    parent = SimpleNamespace(
        documents=documents,
        sessions=SimpleNamespace(read=lambda r: controls[r]),
        base=SimpleNamespace(
            binding={"case_code": "case"},
            manifest=SimpleNamespace(checksum="input"),
            task=SimpleNamespace(checksum="task", model_dump=lambda **_: result["task"]),
        ),
    )
    monkeypatch.setattr(
        renewal,
        "RuntimeAgentJournal",
        lambda d: SimpleNamespace(
            read=lambda: [dict(phase="reserved", request_id="one")] if d.run_id == rid else []
        ),
    )
    if fault == "missing":
        records[rid].pop("dynamic-phase-result")
    elif fault == "foreign":
        records[rid]["dynamic-phase-result"][0]["run_id"] = "old"
    elif fault == "pending":
        controls[rid].pending = object()
    elif fault == "protocol":
        records[rid]["dynamic-campaign"] = ({"wrong": True},)
    elif fault == "duplicate":
        result["phases"].insert(0, phase)
    elif fault == "count":
        result["new_requests"] = 0
    elif fault == "fact":
        result["input_scopes"][1]["phase"] = dict(phase, accepted=False)
    for name, value in [("protocol", protocol), ("result", result)]:
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    if fault:
        with pytest.raises(ValueError, match="input_scope_accounting_|renewal_tip_"):
            renewal.stopped_tip(parent, tmp_path)
    else:
        first = renewal.stopped_tip(parent, tmp_path)
        assert first == renewal.stopped_tip(parent, tmp_path)
        assert sum(r["requests"] for r in first["previous_attempts"]) == 35
        assert first["run_id"] == "review"
        assert "old" not in reads
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
