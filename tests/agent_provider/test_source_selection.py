"""Saved PDF split-word failure and the bounded exact-source selection path."""

import json
from pathlib import Path

import pytest

from apps.vectorengine_single_case import load_config
from packages.grounding.models import canonical, validate
from packages.knowledge_claims.research_candidates import validate_research_source
from packages.llm_provider.models import Material
from packages.llm_provider.source_snippets import choices
from packages.llm_provider.transport import HTTPTransport, Reply
from tests.agent_provider.batch_support import batch_reply
from tests.expert_agent.batch_case_support import source_reply, vision_reply
from tests.expert_agent.test_batch_assembly import assembly, stages

CONFIG = "configs/agents/vectorengine_batch_integrated_r01.json"


def test_failed_original_quote_is_not_silently_word_joined():
    base = Path("docs/acceptance/12-completion-integrated-v1/case-readback")
    actual = json.loads((base / "actual-requests.json").read_text())[0]
    candidate = json.loads(actual["raw_response"][0]["raw_model_text"])
    material = validate(
        Material,
        json.loads(
            Path(
                ".local/vectorengine-batch-integrated-case/prepared-01/visible-material.json"
            ).read_text()
        ),
    )
    with pytest.raises(ValueError, match="source_or_target_mismatch"):
        validate_research_source(
            candidate, material, quote_policy="research-quote-policy/layout-whitespace/1.0.0"
        )
    rows = choices(material)
    assert rows and all("Becau\nse" not in row["quote"] for row in rows)
    for row in rows:
        source = next(s for s in material.sources if s.reference.code == row["source_code"])
        assert source.text[row["start"] : row["end"]] == row["quote"]
        selected = dict(candidate, source_code=row["source_code"], quote=row["quote"])
        validate_research_source(selected, material)


def bad_quote(body):
    value = source_reply(body)
    value["quote"] = "Not a literal quote from this source."
    return Reply(
        content=canonical(value), finish_reason="stop", response_confirmed=True, api_requests=1
    )


def menu_quote(body):
    value = source_reply(body)
    menus = [
        json.loads(c["text"])
        for c in body["messages"][1]["content"]
        if c["type"] == "text" and c["text"].startswith('{"SOURCE_LITERAL_CHOICES"')
    ]
    assert len(menus) == 1
    row = menus[0]["SOURCE_LITERAL_CHOICES"][0]
    value.update(source_code=row["source_code"], quote=row["quote"])
    return value


def test_source_quote_feedback_then_original_three_stage_acceptance(monkeypatch, tmp_path):
    f = assembly(
        monkeypatch,
        config_path=CONFIG,
        replies=[bad_quote, menu_quote, vision_reply, batch_reply, batch_reply, batch_reply],
    )
    monkeypatch.setattr(
        HTTPTransport,
        "send",
        lambda self, config, body, **kw: self._observed(
            f.wire.send(config, body, **kw), {"http_status": 200}, body, "chat_completions"
        ),
    )
    result, roles = stages(f, tmp_path)
    assert roles == ("source", "vision", "review") and all(
        r["status"] == "accepted" for r in result
    )
    assert f.ports.budget.used == len(f.wire.calls) == 6
    records = f.ports.budget.records()
    assert records[0].error == "scope" and records[1].error is None
    assert records[1].retry_index == 1
    from packages.llm_provider.review_correction import verify_correction

    verify_correction(records[1], f.ports.budget)
    assert (tmp_path / "review-report.html").exists()
    assert stages(f, tmp_path)[0] == result and f.ports.budget.used == 6


def test_public_source_and_vision_proof_unchanged():
    from packages.llm_provider.public_batches import source_vision_reuse

    assert len(source_vision_reuse(load_config(CONFIG))) == 2


def test_wrong_source_or_target_not_authorized_for_quote_retry():
    from types import SimpleNamespace

    from packages.llm_provider.review_correction import eligible
    from tests.planner.support import binding

    config = load_config(CONFIG)
    scope = SimpleNamespace(case_code=binding().case_code)
    reply = Reply(content=b"{}", finish_reason="stop", response_confirmed=True)
    for fields in ([{"path": ["target"], "type": "quote_not_verbatim"}], []):
        assert not eligible(
            config, scope, "source", 0, "scope", reply, {"http_status": 200}, fields
        )


def test_source_quote_second_failure_stops_without_third_attempt(monkeypatch, tmp_path):
    f = assembly(monkeypatch, config_path=CONFIG, replies=[bad_quote, bad_quote, menu_quote])
    monkeypatch.setattr(
        HTTPTransport,
        "send",
        lambda self, config, body, **kw: self._observed(
            f.wire.send(config, body, **kw), {"http_status": 200}, body, "chat_completions"
        ),
    )
    completed, _ = stages(f, tmp_path)
    assert len(completed) == 1 and completed[0]["status"] != "accepted"
    assert len(f.wire.calls) == f.ports.budget.used == 2
    assert all(r.error == "scope" for r in f.ports.budget.records())
    assert not any(k.startswith("accepted:") for k in f.documents.saved)


@pytest.mark.parametrize("stage", ["public", "case"])
def test_source_selection_entry_has_five_terminal_histories(stage):
    from apps.vectorengine_batch_prepare import preview

    value = preview(stage, "configs/research/vectorengine_batch_integrated_r01_" + stage + ".json")
    assert value["round"]["provider_config"] == CONFIG
    assert len(value["round"]["prior_round_configs"]) == 5
    assert not value["key_loaded"] and not value["database_connected"]
