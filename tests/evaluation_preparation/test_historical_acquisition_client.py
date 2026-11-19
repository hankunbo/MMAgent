"""Client-only ancestor proof: no implicit old upgrade or medical material drift."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal
from packages.multimodal_research import unknown_acquisition as owner
from tests.evaluation_preparation.test_recovery_http_client import pair


@pytest.mark.parametrize(
    "fault", [None, "binding", "provider", "material", "policy", "prompt", "same"]
)
def test_ancestor_client_and_current_material_are_separately_checked(tmp_path, monkeypatch, fault):
    old_config, current_config, plan = pair()
    old_state = SimpleNamespace(material={"input": "old"}, policy="policy", prompt_checksum="p")
    state = deepcopy(old_state)
    calls = []
    docs = SimpleNamespace(run_id="run", read=lambda _: (old_state,))
    parent = SimpleNamespace(
        config=current_config,
        protocol=plan,
        documents=lambda _: docs,
        base=SimpleNamespace(contexts={}),
        contexts=SimpleNamespace(read_research_run=lambda _: "original context"),
    )
    (tmp_path / "entry.json").write_text(
        json.dumps({"provider": old_config.model_dump(mode="json")})
    )
    (tmp_path / "protocol.json").write_text(json.dumps({"provider": digest(old_config)}))
    if fault == "binding":
        plan["recovery_http_client"]["client_checksum"] = digest("changed")
    elif fault == "provider":
        (tmp_path / "protocol.json").write_text(json.dumps({"provider": digest("changed")}))
    elif fault == "same":
        parent.config = old_config
    elif fault:
        setattr(state, "prompt_checksum" if fault == "prompt" else fault, "changed")

    def original(p, phase, request):
        assert p.config == old_config
        calls.append("original")
        return {"request_id": request}

    class Projection:
        def __init__(self, base, target):
            assert base.contexts == {"run": "original context"}
            assert parent.base.contexts == {}

        def project(self, *, run_id, config):
            assert config == current_config and run_id == "run"
            return state, {}

        def verify(self, value):
            assert value is state
            calls.append("current")

    monkeypatch.setattr(owner, "original_attempt", original)
    monkeypatch.setattr(owner, "validate", lambda _, value: value)
    monkeypatch.setattr("packages.multimodal_research.campaign.Phase", Projection)
    # Only test fixture states bypass Pydantic; production original owner and
    # actual current projection are exercised by the read-only real-case probe.
    monkeypatch.setattr(
        owner, "digest", lambda v: digest(vars(v)) if isinstance(v, SimpleNamespace) else digest(v)
    )
    if fault:
        with pytest.raises(ValueError):
            owner.historical_client_attempt(
                parent, tmp_path, {"run_id": "run", "target": "check"}, "request"
            )
    else:
        proof = owner.historical_client_attempt(
            parent, tmp_path, {"run_id": "run", "target": "check"}, "request"
        )
        assert proof["historical_client_transition"]["exact_material_policy_and_vision_prompt"]
        assert calls == ["original", "current"]
    assert parent.base.contexts == {} and parent.config == (
        old_config if fault == "same" else current_config
    )


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "review"},
        {"exact_request_budget": False},
        {"request_limit": 3},
        {"acquisition_target": []},
        {"acknowledge_unknown_acquisition_request": []},
        {"previous": "another"},
        {"recovery_http_client": "another"},
        {"acknowledge_confirmed_invalid_acquisition_request": ["old"]},
    ],
)
def test_explicit_historical_scope_before_owner_or_writes(change):
    args = dict(
        facts="missing",
        previous="missing",
        output="never-created",
        mode="acquisition",
        authorization="test",
        request_limit=2,
        acquisition_target=["check"],
        exact_request_budget=True,
        acknowledge_unknown_acquisition_request=["request"],
        historical_acquisition_origin="ancestor",
    )
    args.update(change)
    with pytest.raises(ValueError, match="historical_acquisition_explicit_scope"):
        renewal.prepare(**args)


def test_historical_origin_requires_exact_single_unknown():
    for kwargs in ({"confirmed_invalid": True}, {"local_request_ids": ["old"]}):
        with pytest.raises(ValueError, match="historical_acquisition_single_unknown_required"):
            owner.selection(None, [], ["old"], {}, historical_origin="ancestor", **kwargs)


def test_new_version_is_an_exact_bounded_acquisition():
    assert owner.HISTORICAL_CLIENT_VERSION in renewal.EXACT_ACQUISITION_VERSIONS
    assert owner.HISTORICAL_CLIENT_VERSION in renewal.RETAINED_ACQUISITION_VERSIONS
    from packages.evaluation_preparation import cohort_progress

    assert owner.HISTORICAL_CLIENT_VERSION in cohort_progress.RETAINED_ACQUISITION_VERSIONS
