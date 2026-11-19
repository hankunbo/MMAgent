"""A new route changes the new window only; historic configurations remain exact."""

import json
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_capability_successor as module
from packages.grounding.models import digest
from packages.llm_provider.config import review_format_upgrade, with_network_route
from packages.llm_provider.network_route import prepare_route
from tests.agent_provider.test_accepted_review_reuse import fixture


def routed_config():
    return with_network_route(
        review_format_upgrade(fixture().config),
        prepare_route(environ={"HTTPS_PROXY": "http://localhost:8888"}),
    )


@pytest.mark.parametrize("family", ["7.1", "7.2", "9.1", "9.2"])
def test_explicit_route_preserves_all_other_fields_and_legacy_branch(monkeypatch, family):
    from pathlib import Path

    from packages.llm_provider import network_route
    from packages.llm_provider.config import read_provider_config, with_http_client

    original = (
        read_provider_config(
            json.loads(
                Path("configs/agents/fairvlmed_relay_environment_gpt6_review_v1.json").read_bytes()
            )
        )
        if family.startswith("7")
        else routed_config()
    )
    if family.endswith("2"):
        original = with_http_client(
            original, json.loads(Path("configs/agents/system_curl_client_v1.json").read_bytes())
        )
    assert original.schema_version == "agent-provider-config/" + family + ".0"
    before = original.model_dump(mode="json")
    route = prepare_route(environ={"HTTPS_PROXY": "http://localhost:9999"})
    monkeypatch.setattr(network_route, "load_route", lambda path: route)
    selected, proof = module.route_configuration(original, "configs/test-route.json")
    assert original.model_dump(mode="json") == before
    expected = dict(before, network_route=route.model_dump(mode="json"))
    assert selected.model_dump(mode="json") == expected
    assert proof["origin_checksum"] == digest(original)
    assert proof["provider_checksum"] == digest(selected)
    assert module.route_configuration(original, None) == (original, None)


def test_route_is_explicit_in_selection_without_changing_medical_selection(tmp_path, monkeypatch):
    original = routed_config()
    checked = SimpleNamespace(
        campaign=SimpleNamespace(config=original, spec={"key": "old"}),
        history={"reservations": 10},
        owner=SimpleNamespace(snapshot={"requirements": []}, policy={}),
        projection={"dispatch_targets": ["new"]},
        limits={"max_new_requests": 12},
    )
    monkeypatch.setattr(module, "directory", lambda key: tmp_path / key)
    before = module.selection_for(checked, "new")
    checked.execution_config = original
    checked.route_transition = {"path": "configs/test-route.json"}
    after = module.selection_for(checked, "new")
    assert before["schema_version"] == module.VERSION
    assert after["schema_version"] == module.ROUTED_VERSION
    assert {k: v for k, v in after.items() if k not in {"schema_version", "route_transition"}} == {
        k: v for k, v in before.items() if k != "schema_version"
    }


@pytest.mark.parametrize("routed,has_binding", [(True, False), (False, True)])
def test_restore_rejects_route_with_wrong_version(tmp_path, monkeypatch, routed, has_binding):
    from packages.evaluation_preparation import generic_capability_execution as executor
    from tests.evaluation_preparation.test_generic_capability_execution import setup_files

    value, calls = setup_files(tmp_path, monkeypatch)
    value["schema_version"] = module.ROUTED_VERSION if routed else module.VERSION
    if has_binding:
        value["route_transition"] = {"path": "configs/test-route.json"}
    (tmp_path / "new/input.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="selection_route_version"):
        executor.load_selection("new")
    assert calls == []


def test_bad_route_preflight_has_no_dispatch_claim(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from db.runtime.repository.case_pipeline import CasePipelineRepository
    from packages.evaluation_preparation import generic_execution_lock
    from packages.multimodal_research import unknown_acquisition

    @contextmanager
    def lease(*args):
        yield lambda: None

    monkeypatch.setattr(CasePipelineRepository, "__init__", lambda *args: None)
    monkeypatch.setattr(CasePipelineRepository, "guarded_session", lease)
    monkeypatch.setattr(generic_execution_lock, "target", lambda: object())
    monkeypatch.setattr(module, "directory", lambda key: tmp_path / key)
    checked = SimpleNamespace(campaign=SimpleNamespace(config=object()))
    monkeypatch.setattr(module, "origin", lambda key, **kw: checked)

    def reject(config):
        raise ValueError("network_proxy_endpoint_changed")

    monkeypatch.setattr(unknown_acquisition, "check_network", reject)
    with pytest.raises(ValueError, match="network_proxy_endpoint_changed"):
        module.preflight("old", "new", network_route="configs/test-route.json")
    assert list(tmp_path.iterdir()) == []
