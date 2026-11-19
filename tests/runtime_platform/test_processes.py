"""Real API/Worker processes and original 06/Planner/Runtime/Audit on owned PG."""

import json
import multiprocessing
import os
import sys
import time
from dataclasses import replace
from uuid import uuid4

import httpx
import pytest

from apps.platform.runtime import Installation, Profile
from apps.worker.service import Worker
from db import manage
from db.runtime.repository.dispatch import DispatchRepository
from packages.grounding.models import digest
from packages.planner_control.models import Phase
from packages.task_runtime.models import Profile as DispatchProfile
from tests.api.support import dependencies
from tests.database.knowledge_support_04a import sql
from tests.expert_agent.test_pg_boundaries import database as database
from tests.task_runtime.test_postgres import assembly, counts


def child(argv, output):
    from apps.platform.__main__ import main

    with open(output, "w") as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        sys.argv = ["platform", *argv, "--factory", "tests.runtime_platform.process_fixture:build"]
        try:
            code = main()
        except Exception:
            print(json.dumps({"error": "synthetic_child_failed"}), flush=True)
            code = 1
        raise SystemExit(code)


def start(argv, path):
    p = multiprocessing.get_context("fork").Process(target=child, args=(argv, str(path)))
    p.start()
    return p


def finish(p, *, stop=False):
    if stop and p.is_alive():
        p.terminate()
    p.join(25)
    if p.is_alive():
        p.kill()
        p.join(5)
        pytest.fail("owned_platform_process_did_not_stop")
    return p.exitcode


def wait_http(client, process):
    until = time.monotonic() + 20
    while time.monotonic() < until:
        if not process.is_alive():
            pytest.fail("api_process_exited_before_ready")
        try:
            response = client.get("/capabilities")
            if response.status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.05)
    pytest.fail("api_readiness_timeout")


def test_process_api_worker_crash_recovery_and_backup(database, tmp_path):
    from apps.platform.backup import backup, restore_to_empty
    from tests.runtime_platform import process_fixture

    repo, planner, domain, driver, req = assembly(
        tmp_path, original_tool=True, profile=DispatchProfile(lease_seconds=1)
    )
    original_request = driver.research.read(req.run_id).request
    deps = dependencies(driver, original_request, planners={req.run_id: planner}, dispatch=repo)
    flag = {"crash": False}

    def build():
        fresh = DispatchRepository(
            manage.test_target("runtime"), owner=repo.owner, profile=repo.profile
        )
        if flag["crash"]:
            original = fresh.command

            def crash(run, action, key, **kw):
                if action == "confirm" and planner.read(run).phase == Phase.VERIFY:
                    os._exit(23)  # real process death AFTER original 06 and Planner confirmation
                return original(run, action, key, **kw)

            fresh.command = crash
        return Installation(replace(deps, dispatch=fresh), Profile(max_ticks=16))

    process_fixture.factory = build
    private = tmp_path / "socket"
    private.mkdir(mode=0o700)
    sock = private / "api.sock"
    api = start(["serve", "--socket", str(sock)], tmp_path / "api.log")
    processes = [api]
    transport = httpx.HTTPTransport(uds=str(sock))
    client = httpx.Client(
        transport=transport,
        base_url="http://local",
        headers={"Authorization": "Bearer researcher"},
        timeout=5,
    )
    try:
        wait_http(client, api)
        body = dict(api_version="research-http/1.0.0", **req.model_dump(exclude={"run_id"}))
        assert client.post(f"/runs/{req.run_id}/start", json=body).status_code == 200
        initial = client.get(f"/platform/runs/{req.run_id}/operations")
        assert initial.status_code == 200, initial.text
        assert initial.json()["claims"] == 0
        for index in range(6):
            Worker(repo, worker_id="setup").tick(req.run_id, claim_key=f"setup-{index}")
        assert planner.read(req.run_id).phase == Phase.EXECUTE
        flag["crash"] = True
        broken = start(
            ["worker", "--run-id", req.run_id, "--batch-key", "first"], tmp_path / "crash.log"
        )
        processes.append(broken)
        assert finish(broken) == 23
        assert planner.read(req.run_id).phase == Phase.VERIFY
        assert repo.read(req.run_id).command.action == "execute"
        assert counts(database, req.run_id)[2] == 1
        original_command = repo.read(req.run_id).command
        flag["crash"] = False
        time.sleep(1.1)  # only this test's one-second lease, not production retry timing
        recovered = start(
            ["worker", "--run-id", req.run_id, "--batch-key", "recovery"],
            tmp_path / "recovered.log",
        )
        processes.append(recovered)
        assert finish(recovered) == 0, (tmp_path / "recovered.log").read_text()
        state = planner.read(req.run_id)
        assert state.phase == Phase.STOPPED
        assert repo.read(req.run_id).state == "completed"
        assert sum(e.command == original_command for e in state.events) == 1
        assert counts(database, req.run_id)[2] == 1
        assert (state.reserved, state.started, state.confirmed) == (1, 1, 1)
        summary = client.get(f"/platform/runs/{req.run_id}/operations").json()
        assert summary["tool_started"] == 1 and summary["dispatch_state"] == "completed"
        html = client.get(f"/platform/runs/{req.run_id}/operations.html")
        assert html.status_code == 200 and html.headers["x-source-checksum"] == digest(summary)
        assert finish(api, stop=True) == 0 and not sock.exists()
        api2 = start(["serve", "--socket", str(sock)], tmp_path / "api-restarted.log")
        processes.append(api2)
        client.close()
        client = httpx.Client(
            transport=httpx.HTTPTransport(uds=str(sock)),
            base_url="http://local",
            headers={"Authorization": "Bearer researcher"},
            timeout=5,
        )
        wait_http(client, api2)
        assert client.get(f"/platform/runs/{req.run_id}/operations").json() == summary
        assert finish(api2, stop=True) == 0
        before = counts(database, req.run_id)
        plan = driver.research.compile(req.run_id, driver.choice)
        objects = driver.bind_plan(plan, "dispatch")[0]._objects
        bundle = tmp_path / "backup"
        backup(database.admin, objects.root, bundle)
        name = "mmagent_02d_restore_" + uuid4().hex[:16]
        restore = manage.create_test_restore_database(name)
        try:
            restore_to_empty(restore, bundle, tmp_path / "restored_objects")
            assert sql(
                restore,
                "SELECT payload FROM runtime.research_dispatch WHERE run_id=%s",
                (req.run_id,),
            )[0][0] == repo.read(req.run_id).model_dump(mode="json")
            assert sql(
                restore, "SELECT count(*) FROM runtime.tool_attempt WHERE run_id=%s", (req.run_id,)
            ) == [(1,)]
            with pytest.raises(ValueError, match="empty_database"):
                restore_to_empty(restore, bundle, tmp_path / "refused_objects")
        finally:
            manage.drop_test_restore_database(name)
        assert counts(database, req.run_id) == before
    finally:
        client.close()
        for proc in processes:
            if proc.is_alive():
                finish(proc, stop=True)
        process_fixture.factory = None


def test_process_cancel_before_start(database, tmp_path):
    from tests.runtime_platform import process_fixture

    repo, planner, _, driver, req = assembly(tmp_path)
    repo.submit(req)
    repo.command(req.run_id, "cancel", "cancel", expected_revision=repo.read(req.run_id).revision)
    deps = dependencies(
        driver,
        driver.research.read(req.run_id).request,
        planners={req.run_id: planner},
        dispatch=repo,
    )
    process_fixture.factory = lambda: Installation(deps)
    proc = start(
        ["worker", "--run-id", req.run_id, "--batch-key", "cancel-close"], tmp_path / "cancel.log"
    )
    try:
        assert finish(proc) == 0
        assert repo.read(req.run_id).state == "cancelled"
        assert planner.read(req.run_id).cancelled and counts(database, req.run_id)[2] == 0
    finally:
        if proc.is_alive():
            finish(proc, stop=True)
        process_fixture.factory = None
