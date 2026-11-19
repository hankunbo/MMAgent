"""Archive adds no publication or fabricated execution, and rejects source tampering."""

import json
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.memory.aggregation_archive import Documents, prepare


@pytest.mark.parametrize("fault", ["case", "run", "root", "key", "index"])
def test_archive_read_adapter_rejects_changed_binding(fault):
    value = dict(
        run_id="RUN:ONE",
        root_checksum=digest("root"),
        key_checksum=digest("entry"),
        index=0,
        value={},
    )
    if fault != "case":
        field = dict(run="run_id", root="root_checksum", key="key_checksum", index="index")[fault]
        value[field] = "changed"
    asset = SimpleNamespace(asset_code="ASSET:ONE")
    owner = SimpleNamespace(
        read_run_case=lambda _: "CASE:WRONG" if fault == "case" else "CASE:ONE",
        runtime=SimpleNamespace(read_observation_bundle=lambda _: (asset,)),
        artifact=lambda *_: (asset, json.dumps(value).encode()),
    )
    docs = Documents(owner, "CASE:ONE", "RUN:ONE", digest("root"), {}, None, "accepted")
    with pytest.raises(ValueError, match="memory_aggregation_"):
        docs.read("entry")
    assert not hasattr(docs, "write") and not hasattr(docs, "session")


def test_aggregation_cannot_invent_tool_workflow():
    owner = SimpleNamespace(read_run_case=lambda _: "CASE:ONE")
    with pytest.raises(ValueError, match="snapshot_binding"):
        prepare(
            owner,
            key="k",
            case_code="CASE:ONE",
            run_id="RUN:ONE",
            origin=None,
            envelope={},
            kind="tool_experience",
            target="GOAL:ONE",
        )


def test_missing_original_state_is_explicit_rejection():
    owner = SimpleNamespace(
        read_run_case=lambda _: "CASE:ONE",
        runtime=SimpleNamespace(read_observation_bundle=lambda _: None),
    )
    docs = Documents(owner, "CASE:ONE", "RUN:ONE", digest("root"), {}, None, "accepted")
    with pytest.raises(ValueError, match="source_state_missing"):
        docs.state()


def test_orphan_report_artifact_is_not_an_original_origin():
    owner = SimpleNamespace(
        read_run_case=lambda _: "CASE:ONE",
        runtime=SimpleNamespace(read_observation_bundle=lambda _: (object(),)),
    )
    with pytest.raises(ValueError, match="origin_binding"):
        prepare(
            owner,
            key="k",
            case_code="CASE:ONE",
            run_id="RUN:ONE",
            origin=object(),
            envelope=dict(
                run_id="RUN:ONE", key_checksum=digest("accepted-aggregation-report"), index=0
            ),
            kind="graph_snapshot",
            target="GOAL:ONE",
        )


def test_snapshot_cli_requires_explicit_commit(monkeypatch, capsys):
    from apps.case_cohort import main

    calls = []

    def archive(directory, *, commit):
        calls.append((directory, commit))
        return dict(status="not_archived_source_capacity", committed=False, new_api_requests=0)

    monkeypatch.setattr("packages.multimodal_research.memory.archive_saved_snapshot", archive)
    monkeypatch.setattr("sys.argv", ["case_cohort", "archive-result", "--input", "prior"])
    assert main() == 2 and calls == [("prior", False)]
    assert json.loads(capsys.readouterr().out)["committed"] is False


def test_incomplete_snapshot_refused_before_database(tmp_path, monkeypatch):
    from packages.multimodal_research.memory import archive_saved_snapshot

    (tmp_path / "result.json").write_text(
        json.dumps(dict(schema_version="multimodal-case-result/1.0.0", complete=False))
    )

    def forbidden(*a, **k):
        raise AssertionError("no database assembly")

    monkeypatch.setattr("apps.case_memory_store.assembly", forbidden)
    with pytest.raises(ValueError, match="completed_result_required"):
        archive_saved_snapshot(tmp_path, commit=True)


@pytest.mark.parametrize("invalid", [False, True])
def test_reasoned_archive_plan_dispatch_is_explicit_and_validated(invalid):
    """Plan parsing only; the real origin/Memory/PG path is covered separately."""
    from packages.memory.aggregation_archive import original_origins

    plan = dict(
        schema_version="accepted-review-aggregation-plan/1.4.0",
        facts="new",
        origins=[dict(campaign="old"), dict(campaign="new")],
        reuse_kind="unsupported" if invalid else "same_method_partition_only",
    )
    empty = SimpleNamespace(origins=(), candidate=SimpleNamespace(items=()))
    if invalid:
        with pytest.raises(ValueError, match="reuse_kind"):
            original_origins(None, None, empty, plan=plan)
    else:
        assert original_origins(None, None, empty, plan=plan) is None
