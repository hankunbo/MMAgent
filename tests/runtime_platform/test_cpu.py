import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from apps.api.errors import Fault
from apps.platform.__main__ import bind_local, load_factory
from apps.platform.runtime import Installation, Profile
from tests.api_expert.support import configured


def test_operational_snapshot_scope_and_unknown_facets():
    s = configured()
    installation = Installation(s.dependencies)
    actor = s.dependencies.identities.authenticate("Bearer researcher")
    value = installation.snapshot(actor, s.f.binding.run_id)
    assert value["case_code"] == s.f.binding.case_code
    assert value["phase"] == "review" and value["graph"]["checksum"]
    assert value["tool_started"] == 0
    assert all(v["measurements"] is None for v in value["facets"].values())
    with pytest.raises(Fault) as exc:
        installation.snapshot(replace(actor, case_codes=frozenset()), s.f.binding.run_id)
    assert str(exc.value) == "SCOPE_FORBIDDEN"


def test_http_log_no_raw_url_headers_or_body():
    s = configured()
    events = []
    wrapper = s.dependencies
    app = Installation(wrapper).app()
    app.sink = events.append

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://local",
        ) as c:
            response = await c.get(
                "/platform/runs/" + s.f.binding.run_id + "/operations",
                headers={"Authorization": "Bearer researcher"},
            )
            assert response.status_code == 200, response.text
            html = await c.get(
                "/platform/runs/" + s.f.binding.run_id + "/operations.html",
                headers={"Authorization": "Bearer researcher"},
            )
            assert html.status_code == 200 and html.headers["x-source-checksum"]
            bad = await c.get(
                "/SYNTHETIC_SECRET?key=SYNTHETIC_SECRET",
                headers={"Authorization": "Bearer SYNTHETIC_SECRET"},
            )
            assert bad.status_code == 404

    asyncio.run(exercise())
    assert len(events) == 3 and "SYNTHETIC_SECRET" not in json.dumps(events)
    assert all(e["trace_id"] and e["elapsed_seconds"] >= 0 for e in events)


def test_socket_private_directory_and_fencing(tmp_path):
    import os

    path = tmp_path / "api.sock"
    tmp_path.chmod(0o755)
    with pytest.raises(ValueError, match="private_directory"):
        bind_local(path)
    tmp_path.chmod(0o700)
    sock, fd, _ = bind_local(path)
    try:
        assert path.stat().st_mode & 0o777 == 0o600
        with pytest.raises(BlockingIOError):
            bind_local(path)
    finally:
        sock.close()
        os.close(fd)
        path.unlink()


def test_missing_owner_and_profile_boundary(monkeypatch):
    monkeypatch.delenv("MMAGENT_PLATFORM_TOKEN", raising=False)
    with pytest.raises(ValueError, match="identity_required"):
        load_factory("apps.platform.local:build")
    with pytest.raises(ValueError):
        Profile(live_enabled=True)
    with pytest.raises(ValueError):
        Profile(max_ticks=100)
    monkeypatch.setenv("MMAGENT_PLATFORM_TOKEN", "synthetic-local-only")
    installed = load_factory("apps.platform.local:build")
    assert installed.profile == Profile()
    assert installed.dependencies.dispatch is None


def test_original_budget_metrics_and_metric_scope():
    from apps.platform.runtime import BudgetMetricsReader, MetricBlock
    from packages.llm_provider.config import load_provider_config
    from tests.agent_provider.support import budget_for, materials

    config = load_provider_config(environ={})
    _, binding = materials(config=config)
    budget = budget_for(binding, config)
    reader = BudgetMetricsReader(budget, binding.case_code)
    result = reader.read(binding.run_id)
    assert result.measurements["dispatched_observed"] == 0
    assert result.measurements["cost"] is None
    assert result.source.checksum
    with pytest.raises(ValueError, match="metric_scope"):
        reader.read("RUN:OTHER")
    with pytest.raises(ValueError, match="metric_names"):
        MetricBlock.model_validate({**result.model_dump(), "measurements": {"raw_prompt": 1}})
    with pytest.raises(ValueError):
        MetricBlock.model_validate({**result.model_dump(), "measurements": {"cost": float("nan")}})


def test_backup_rejects_nested_output_before_database(tmp_path):
    from apps.platform.backup import backup

    with pytest.raises(ValueError, match="backup_paths"):
        backup(None, tmp_path, tmp_path / "nested")
