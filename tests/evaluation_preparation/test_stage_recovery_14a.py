"""14A uses synthetic sources only; historical failed replies remain invalid."""

import copy
import json
import ssl

import pytest

from packages.grounding.models import digest
from tests.evaluation_preparation.test_definition_expansion import Sources, draft


def quoted():
    value = draft("GOAL:TEST")
    value["schema_version"] = "source-definition-draft/2.1.0"
    for e in value["excerpts"]:
        e.pop("start")
        e.pop("end")
        e.update(prefix=None, suffix=None)
    return value


def test_unique_quote_owner_uses_unicode_local_offsets_and_keeps_candidate():
    from packages.evidence_reuse.definitions import locate_draft

    sources = Sources()
    value = quoted()
    original = copy.deepcopy(value)
    resolved = locate_draft(value, target="GOAL:TEST", source_owner=sources)
    assert resolved.excerpts[0].start == 0
    assert resolved.excerpts[0].end == len(sources.value["text"])
    assert value == original


def test_ambiguous_quote_requires_exact_adjacent_context():
    from packages.evidence_reuse.definitions import locate_draft

    sources = Sources()
    sources.value["text"] = "α观察。β观察。"
    sources.value["checksum"] = digest(sources.value["text"])
    value = quoted()
    value["excerpts"] = [
        dict(
            source_code=sources.value["code"],
            source_checksum=sources.value["checksum"],
            text="观察",
            prefix=None,
            suffix=None,
        )
    ]
    with pytest.raises(ValueError, match="definition_quote_ambiguous"):
        locate_draft(value, target="GOAL:TEST", source_owner=sources)
    value["excerpts"][0]["prefix"] = "β"
    resolved = locate_draft(value, target="GOAL:TEST", source_owner=sources)
    assert (resolved.excerpts[0].start, resolved.excerpts[0].end) == (5, 7)


def test_safe_tls_diagnostic_does_not_echo_exception_message():
    from packages.llm_provider.worker import safe_transport_error

    error = ssl.SSLError(1, "PRIVATE note Authorization Bearer SECRET")
    value = safe_transport_error(error, "response_body")
    assert value["errno"] == 1 and value["remote_execution"] == "unknown"
    assert "SECRET" not in json.dumps(value) and value["automatic_replay_authorized"] is False


def test_stage_status_is_local_and_no_implicit_assembly(tmp_path, monkeypatch):
    from packages.multimodal_research import entry
    from packages.multimodal_research.stages import inspect_local

    monkeypatch.setattr(entry, "assemble_entry", lambda *a: pytest.fail("assembled run"))
    (tmp_path / "protocol.json").write_text(
        json.dumps(dict(campaign_id="SYNTHETIC", deadline="2000-01-01T00:00:00+00:00"))
    )
    result = inspect_local(tmp_path)
    assert result["read_only"] and result["execution_authorized"] is False
    assert result["control"] == "not_checked_local_view"


@pytest.mark.parametrize(
    "change,code",
    [
        ("quote", "definition_quote_not_found"),
        ("checksum", "definition_draft_exact_excerpt"),
        ("withdraw", "definition_draft_exact_excerpt"),
        ("target", "definition_draft_target"),
    ],
)
def test_quote_refusals_keep_original_candidate(change, code):
    from packages.evidence_reuse.definitions import locate_draft

    sources, value = Sources(), quoted()
    if change == "quote":
        value["excerpts"][0]["text"] = "not present in source"
    elif change == "checksum":
        value["excerpts"][0]["source_checksum"] = digest("different version")
    elif change == "withdraw":
        sources.value["state"] = "withdrawn"
    else:
        value["target"] = "GOAL:OTHER"
    original = copy.deepcopy(value)
    with pytest.raises(ValueError, match=code):
        locate_draft(value, target="GOAL:TEST", source_owner=sources)
    assert value == original


def test_legacy_offsets_are_not_silently_corrected():
    from packages.evidence_reuse.definitions import locate_draft

    value = draft("GOAL:TEST")
    assert locate_draft(value, target="GOAL:TEST", source_owner=Sources()).schema_version.endswith(
        "2.0.0"
    )
    value["excerpts"][0]["start"] = 5550
    with pytest.raises(ValueError, match="definition_draft_exact_excerpt"):
        locate_draft(value, target="GOAL:TEST", source_owner=Sources())


def test_owner_current_rebuild_and_historical_original_bytes():
    from packages.evidence_reuse.definition_acceptance import ResearchDefinitionOwner
    from tests.evaluation_preparation.test_definition_expansion import Documents

    sources = Sources()
    owner = ResearchDefinitionOwner(
        documents=Documents(),
        sources=sources,
        campaign_id="synthetic",
        target="GOAL:TEST",
        input_checksum=digest("input"),
        guard=lambda: None,
    )
    result = owner.accept("one", quoted())
    assert result["schema_version"] == "accepted-research-definition/1.1.0"
    assert owner.accept("one", quoted()) == result
    assert "start" not in result["draft"]["excerpts"][0]
    assert result["resolved_draft"]["excerpts"][0]["start"] == 0
    sources.value["state"] = "withdrawn"
    with pytest.raises(ValueError, match="definition_draft_exact_excerpt"):
        owner.read("one")
    assert owner.read("one", historical=True) == result


def test_quote_changed_on_return_is_not_current_qualified():
    from packages.evidence_reuse.definitions import locate_draft

    sources = Sources()
    original = sources.read
    n = 0

    def read(code):
        nonlocal n
        n += 1
        value = original(code)
        if n >= 3:
            value["checksum"] = digest("changed during validation")
        return value

    sources.read = read
    with pytest.raises(ValueError, match="definition_draft_source_changed"):
        locate_draft(quoted(), target="GOAL:TEST", source_owner=sources)


@pytest.mark.parametrize("operation", ["stage-status", "stage-check", "stage-preview"])
def test_cli_read_only_never_assembles_or_loads_secrets(tmp_path, monkeypatch, operation, capsys):
    import sys

    from apps.multimodal_case import main
    from packages.multimodal_research import stages

    (tmp_path / "protocol.json").write_text(json.dumps(dict(deadline="2000-01-01T00:00:00+00:00")))
    monkeypatch.setattr(stages, "assemble_for_stage", lambda *a: pytest.fail("must not assemble"))
    monkeypatch.setattr(sys, "argv", ["multimodal_case", operation, "--input", str(tmp_path)])
    main()
    value = json.loads(capsys.readouterr().out)
    assert value["read_only"] and not value["execution_authorized"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["protocol.json"]


@pytest.mark.parametrize("reason", ["expired", "cancelled", "drift"])
def test_stage_plan_does_not_reopen_old_or_changed_window(tmp_path, reason):
    from datetime import UTC, datetime, timedelta

    from packages.multimodal_research.implementation import freeze
    from packages.multimodal_research.stages import prepare

    value = dict(
        deadline=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        implementation=freeze(),
        max_new_requests=3,
    )
    if reason == "expired":
        value["deadline"] = "2000-01-01T00:00:00+00:00"
    if reason == "cancelled":
        (tmp_path / "cancel").touch()
    if reason == "drift":
        value["implementation"] = {}
    (tmp_path / "protocol.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        prepare(tmp_path)
    assert not (tmp_path / "stage-plan.json").exists()


@pytest.mark.parametrize("command", ["stage-run", "stage-resume"])
def test_cli_execution_routes_only_to_original_stage_composition(tmp_path, monkeypatch, command):
    from types import SimpleNamespace

    from packages.multimodal_research import stages

    original = object()
    calls = []
    monkeypatch.setattr(stages, "assemble_for_stage", lambda path: original)
    monkeypatch.setattr(
        stages, "run", lambda campaign, stage: calls.append((campaign, stage)) or {"stage": stage}
    )
    assert stages.cli(
        SimpleNamespace(command=command, input=str(tmp_path), stage="definitions")
    ) == {"stage": "definitions"}
    assert calls == [(original, "definitions")]


def test_new_source_prompt_config_has_no_automatic_transport_retry():
    from pathlib import Path

    from packages.multimodal_research.campaign import configuration
    from packages.multimodal_research.definition_expansion import (
        definition_config,
        read_expansion_policy,
    )

    policy = read_expansion_policy(
        json.loads(Path("configs/research/multimodal_definition_gate_v5.json").read_bytes())
    )
    config = definition_config(
        configuration(directional=True), prompt_version=policy.prompt_version
    )
    assert config.transport.max_retries == 0
    assert (
        next(r for r in config.roles if r.role == "source").prompt_version
        == "definition-draft-2.3.0"
    )
    legacy = definition_config(
        configuration(directional=True), prompt_version="definition-draft-2.2.0"
    )
    assert legacy.transport.max_retries == 1


def test_stage_prepare_round_trips_strict_json_contract_without_restarting_clock(tmp_path):
    from datetime import UTC, datetime, timedelta

    from packages.multimodal_research.implementation import freeze
    from packages.multimodal_research.stages import inspect_local, prepare

    protocol = dict(
        implementation=freeze(),
        deadline=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        max_new_requests=3,
    )
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    value = prepare(tmp_path)
    assert prepare(tmp_path) == value
    assert value["deadline"] == protocol["deadline"]
    assert inspect_local(tmp_path)["plan_matches"] is True


def test_definition_stage_checkpoint_does_not_create_revision_or_send_model(tmp_path):
    from tests.evaluation_preparation.test_definition_expansion import fixture

    expansion = fixture(tmp_path)
    sourced = expansion.advance(until="sources")
    assert sourced["stage"] == "sources" and expansion.budget.used == 0
    assert expansion.documents.read("definition:revision") is None
    defined = expansion.advance(until="definitions")
    assert defined["accepted"] and expansion.budget.used == 1
    assert expansion.documents.read("definition:revision") is None
    revised = expansion.advance(until="requirements")
    assert revised["added"] and expansion.budget.used == 1


@pytest.mark.parametrize("condition", ["cancelled", "deadline", "budget"])
def test_stage_definition_boundary_blocks_before_provider(tmp_path, condition):
    from datetime import UTC, datetime, timedelta

    from packages.llm_provider.provider import ProviderError
    from tests.evaluation_preparation.test_definition_expansion import fixture

    expansion = fixture(tmp_path)
    if condition == "cancelled":
        (tmp_path / "cancel").touch()
    elif condition == "deadline":
        expansion.campaign.protocol["deadline"] = (
            datetime.now(UTC) - timedelta(seconds=1)
        ).isoformat()
    else:

        def denied(**kwargs):
            raise ProviderError("budget")

        expansion.campaign.guard = denied
    with pytest.raises(ProviderError, match=condition):
        expansion.advance(until="sources")
    assert expansion.budget.used == 0 and expansion.sources.calls == 0


def test_unconfirmed_definition_operation_is_not_sent_again(tmp_path):
    from packages.llm_provider.provider import ProviderError
    from tests.evaluation_preparation.test_definition_expansion import fixture

    expansion = fixture(tmp_path)
    calls = []

    def lost():
        calls.append(1)
        raise OSError("synthetic connection disappeared")

    with pytest.raises(OSError):
        expansion.operation("definition:proposal", "llm", lost)
    with pytest.raises(ProviderError):
        expansion.operation("definition:proposal", "llm", lost)
    assert calls == [1]


def test_technical_report_escapes_untrusted_saved_fields_without_a_score():
    from packages.multimodal_research.stage_failure import render

    page = render(dict(stage="<script>alert(1)</script>", scores=None, control=dict(exists=False)))
    assert "<script>" not in page and "未评分" in page
    assert "null" in page and "false" in page


def test_missing_stage_run_is_optional_but_owner_errors_are_not_swallowed():
    from types import SimpleNamespace

    from packages.multimodal_research.stages import durable_receipt

    seen = []

    def read(run, *, missing_ok=False):
        seen.append(missing_ok)
        return None

    campaign = SimpleNamespace(
        request_key=lambda target: "synthetic-key", contexts=SimpleNamespace(read_research_run=read)
    )
    assert durable_receipt(campaign, "sources") is None and seen == [True]

    def broken(*args, **kwargs):
        raise OSError("synthetic database unavailable")

    campaign.contexts.read_research_run = broken
    with pytest.raises(OSError):
        durable_receipt(campaign, "sources")
