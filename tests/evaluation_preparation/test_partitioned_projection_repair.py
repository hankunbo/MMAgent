"""A projection amendment is never authority to replay a started Review."""

from types import SimpleNamespace as NS

import pytest

from db.runtime.repository.case_pipeline import identity
from packages.multimodal_research import pending_review as p


@pytest.mark.parametrize(
    "fault", [None, "control", "journal", "intent", "request", "run", "case", "input"]
)
def test_projection_repair_requires_exact_unstarted_original_owner(monkeypatch, tmp_path, fault):
    rid = identity("RUN", "task-research:test-key")
    request = NS(
        input_checksum="input",
        task=NS(checksum="task"),
        visible_checksum="visible",
        scope=NS(case_code="case"),
    )
    context = NS(run_id=rid, request=request)
    docs = NS(
        run_id=rid,
        read=lambda key: ({},) if fault == "intent" and key == "live-stage-intent:review" else None,
    )
    monkeypatch.setattr(
        "packages.research_runtime.agent_records.RuntimeAgentJournal",
        lambda _: NS(read=lambda: [{}] if fault == "journal" else []),
    )
    c = NS(
        request_key=lambda _: "test-key",
        contexts=NS(read_research_run=lambda _: context),
        documents=lambda _: docs,
        sessions=NS(read=lambda _: {} if fault == "control" else None),
        protocol=dict(input="input", task="task", visible="visible"),
        base=NS(binding=dict(case_code="case"), visible_value=NS(checksum="visible")),
        phase_path=lambda _: tmp_path,
    )
    if fault == "request":
        (tmp_path / "request.json").write_text("{}")
    if fault == "run":
        context.run_id = "another-run"
    if fault == "case":
        request.scope.case_code = "another-case"
    if fault == "input":
        request.input_checksum = "changed"
    if fault is None:
        assert p.partition_projection_unstarted(c) == (docs, context)
    else:
        with pytest.raises(ValueError, match="started_or_changed"):
            p.partition_projection_unstarted(c)


def test_old_versions_do_not_consume_partition_amendment(tmp_path):
    assert not p.partition_projection_repair_current(
        NS(output=tmp_path, protocol={"schema_version": p.REASSESSMENT_WINDOW_VERSION})
    )
