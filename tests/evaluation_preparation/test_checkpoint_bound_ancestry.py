"""Reuse an assembled fact owner, never a prior checkpoint qualification."""

import json
from types import SimpleNamespace as S

import pytest

from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import source_checkpoint as checkpoint


def fixture(tmp_path, monkeypatch):
    facts = tmp_path / "facts"
    origin = tmp_path / "source"
    origin.mkdir()
    plan = {"facts": str(facts)}
    (origin / "protocol.json").write_text(json.dumps(plan))
    material = S(manifest=S(checksum="input"), task="target")
    ancestor = S(output=facts, base=material)
    parent = S(output=tmp_path / "child", parent=ancestor, base=material)
    previous = S(base=material)
    calls = []

    def bound(owner, path, value):
        assert owner is ancestor and path == origin and value == plan
        calls.append("original_claim_constructor")
        return previous

    attempt = {"directory": str(origin), "requests": 0, "checksum": "original"}

    def inspect(owner):
        assert owner is previous
        calls.append("current_checkpoint")
        return {"previous_attempts": [attempt]}, None

    monkeypatch.setattr(renewal, "assemble_bound_parent", bound)
    monkeypatch.setattr(renewal, "assemble", lambda p: pytest.fail("duplicate ancestry"))
    monkeypatch.setattr(checkpoint, "inspect_checkpoint", inspect)
    return parent, previous, attempt, calls


def test_matching_fact_owner_still_checks_claim_and_checkpoint_every_time(tmp_path, monkeypatch):
    parent, _, attempt, calls = fixture(tmp_path, monkeypatch)
    checkpoint.verify_history(parent, attempt)
    checkpoint.verify_history(parent, attempt)
    assert calls == ["original_claim_constructor", "current_checkpoint"] * 2


def test_no_matching_fact_owner_keeps_original_cold_assembly(tmp_path, monkeypatch):
    parent, previous, attempt, calls = fixture(tmp_path, monkeypatch)
    parent.parent = None
    monkeypatch.setattr(renewal, "assemble", lambda path: previous)
    checkpoint.verify_history(parent, attempt)
    assert calls == ["current_checkpoint"]


def test_claim_failure_does_not_fall_back_or_accept(tmp_path, monkeypatch):
    parent, _, attempt, calls = fixture(tmp_path, monkeypatch)

    def refused(*args):
        raise ValueError("renewal_claim_binding")

    monkeypatch.setattr(renewal, "assemble_bound_parent", refused)
    with pytest.raises(ValueError, match="renewal_claim_binding"):
        checkpoint.verify_history(parent, attempt)
    assert not calls


def test_cycle_is_not_unbounded_reconstruction(tmp_path, monkeypatch):
    parent, _, attempt, _ = fixture(tmp_path, monkeypatch)
    parent.parent = parent
    with pytest.raises(ValueError, match="history_ancestry_cycle"):
        checkpoint.verify_history(parent, attempt)


@pytest.mark.parametrize("field", ["manifest", "task"])
def test_checkpoint_still_rejects_other_input_or_target(tmp_path, monkeypatch, field):
    parent, previous, attempt, _ = fixture(tmp_path, monkeypatch)
    previous.base = S(manifest=S(checksum="input"), task="target")
    setattr(previous.base, field, S(checksum="different") if field == "manifest" else "different")
    with pytest.raises(ValueError, match="source_checkpoint_history_changed"):
        checkpoint.verify_history(parent, attempt)
