"""11A's two independent counterexamples and bounded regressions on one owned PG."""

import itertools
import time
from types import SimpleNamespace

import psycopg
import pytest

from db import manage
from db.runtime.repository.postgres import PostgresRuntimeRepository
from db.runtime.repository.tool_gap_queue import GapQueueRepository
from packages.tool_gap_queue import GapQueue, QueuePolicy
from tests.database import test_runtime_repository_02b as formal
from tests.database.knowledge_support_04a import free_port, prepare_database, sql
from tests.tool_gap_queue.support import install_pg, setup
from tests.tool_gap_queue.test_postgres import counts, export


@pytest.fixture(scope="module")
def host(database_lease, tmp_path_factory):
    with database_lease.cluster(free_port()):
        database = prepare_database()
        manage.bootstrap_runtime_principal(database.admin, manage.test_target("runtime").user)
        yield SimpleNamespace(
            h=None,
            db=database,
            service=None,
            counter=itertools.count(86410),
            tmp_path=tmp_path_factory.mktemp("11a-repair"),
        )


@pytest.fixture
def service(host, monkeypatch):
    if host.h is None:
        host.h = setup(host.tmp_path)
        host.service = install_pg(host.h, host.db)
    catalog = host.h.f.service.catalogs
    monkeypatch.setattr(catalog, "_disabled", set(catalog._disabled))
    monkeypatch.setattr(catalog, "_revision", catalog._revision)
    return GapQueue(
        repository=host.service.repository,
        admission_owner=host.h.f.service,
        contexts=host.service.contexts,
        policy=QueuePolicy(lifetime_seconds=next(host.counter)),
    )


def application(host, service, key="request"):
    return service.prepare_research(
        host.h.context.run_id, key=key + ":" + str(service.policy.lifetime_seconds)
    )


def claim(service, a, key="claim", worker="consumer"):
    return service.claim(a.run_id, a.key, command_key=key, worker=worker)


def command_rows(host):
    return sql(
        host.db.admin,
        "SELECT to_jsonb(c) FROM runtime.tool_gap_command c ORDER BY task_id,command_key",
    )


def test_original_formal_task_contracts_before_research_run(host):
    formal_db = (
        host.db.admin,
        host.db.migration,
        manage.test_target("runtime"),
        PostgresRuntimeRepository(manage.test_target("runtime")),
    )
    formal.test_contract_round_trips_idempotency_and_optimistic_graph_version(formal_db)
    formal.test_run_task_channels_contract_outputs_and_offline_review_are_separate(formal_db)


def test_claim_replay_must_not_present_cancelled_lease_as_current(host, service):
    a = application(host, service, "replay")
    origin = sql(
        host.db.admin, "SELECT to_jsonb(r) FROM runtime.run r WHERE run_id=%s", (a.run_id,)
    )
    service.register(a)
    old = claim(service, a)
    assert old.state == "leased"
    closed = service.cancel(a.run_id, a.key, command_key="cancel")
    before, history = counts(host.db), command_rows(host)
    assert claim(service, a) is None
    assert service.read(a.run_id, a.key) == closed and closed.state == "cancelled"
    assert (
        sql(host.db.admin, "SELECT to_jsonb(r) FROM runtime.run r WHERE run_id=%s", (a.run_id,))
        == origin
    )
    assert sql(host.db.admin, "SELECT count(*) FROM runtime.tool_attempt")[0][0] == 0
    assert service.read_command(a.run_id, a.key, "claim") == old
    assert counts(host.db) == before and command_rows(host) == history
    export(
        "cancelled-current-vs-history",
        dict(
            synthetic=True,
            current=closed.model_dump(mode="json"),
            historical_claim=old.model_dump(mode="json"),
            current_claim=None,
            history_is_not_authorization=True,
        ),
    )


def test_repository_must_not_admit_owner_revoked_application(host, service):
    a = application(host, service, "revoked")
    assert a.queueable
    host.h.f.service.catalogs.disable(host.h.request.candidate.implementation_ref)
    assert not service.prepare_research(a.run_id, key="current").queueable
    before = counts(host.db)
    for write in (service.register, service.repository.register):
        with pytest.raises(ValueError, match="^gap_owner_material_changed$"):
            write(a)
    assert counts(host.db) == before


@pytest.mark.parametrize("end", ["failed", "expired_lease", "deadline", "replaced", "acknowledged"])
def test_old_claim_has_no_current_lease_after_transition(host, service, end):
    service.policy = service.policy.model_copy(
        update=dict(
            lease_seconds=1 if end in {"expired_lease", "replaced", "deadline"} else 60,
            lifetime_seconds=2 if end == "deadline" else service.policy.lifetime_seconds,
        )
    )
    a = application(host, service, end)
    receipt = service.register(a)
    old = claim(service, a)
    if end == "failed":
        service.fail(
            a.run_id,
            a.key,
            command_key="fail",
            worker=old.worker,
            lease_token=old.lease_token,
            reason="simulated_consumer_failure",
        )
    elif end == "acknowledged":
        service.acknowledge(
            a.run_id, a.key, command_key="ack", worker=old.worker, lease_token=old.lease_token
        )
    else:
        time.sleep(2.1 if end == "deadline" else 1.1)
        if end == "replaced":
            new = claim(service, a, "replacement", "other")
            assert new.lease_token != old.lease_token and new.claims == 2
            with pytest.raises(psycopg.errors.CheckViolation, match="gap_lease_not_current"):
                service.acknowledge(
                    a.run_id,
                    a.key,
                    command_key="stale-ack",
                    worker=old.worker,
                    lease_token=old.lease_token,
                )
    before, history = counts(host.db), command_rows(host)
    assert service.repository.command(receipt, "claim", "claim", worker="consumer") is None
    assert service.read_command(a.run_id, a.key, "claim") == old
    assert counts(host.db) == before and command_rows(host) == history
    assert service.read(a.run_id, a.key).claims == (2 if end == "replaced" else 1)


def test_valid_replay_conflicts_and_ack_loss_are_exact_history(host, service, monkeypatch):
    a = application(host, service)
    registered = service.repository.register(a)
    assert service.register(a) == registered
    old = claim(service, a)
    before = counts(host.db)
    assert claim(service, a) == old
    assert counts(host.db) == before
    with pytest.raises(psycopg.errors.UniqueViolation, match="gap_command_key_conflict"):
        claim(service, a, worker="other")
    for worker, token in (("other", old.lease_token), (old.worker, "lease:wrong")):
        with pytest.raises(psycopg.errors.CheckViolation, match="gap_lease_not_current"):
            service.acknowledge(
                a.run_id, a.key, command_key="bad-ack", worker=worker, lease_token=token
            )
    assert counts(host.db) == before
    operation = service.repository.command
    arguments = dict(command_key="ack", lease_token=old.lease_token, worker=old.worker)
    with monkeypatch.context() as patch:

        def lost(*args, **kwargs):
            operation(*args, **kwargs)
            raise ConnectionError("synthetic_ack_reply_lost")

        patch.setattr(service.repository, "command", lost)
        with pytest.raises(ConnectionError, match="synthetic_ack_reply_lost"):
            service.acknowledge(a.run_id, a.key, **arguments)
    before, history = counts(host.db), command_rows(host)
    ack = service.acknowledge(a.run_id, a.key, **arguments)
    assert ack.state == "acknowledged" and not ack.completed
    assert service.read_command(a.run_id, a.key, "ack") == ack
    assert claim(service, a) is None
    assert counts(host.db) == before and command_rows(host) == history
    host.h.f.service.catalogs.disable(host.h.request.candidate.implementation_ref)
    with pytest.raises(ValueError, match="^gap_owner_material_changed$"):
        service.acknowledge(a.run_id, a.key, **arguments)
    assert service.read_command(a.run_id, a.key, "ack") == ack
    assert service.report(a.run_id, a.key)["historical_read_not_authorization"]
    assert counts(host.db) == before
    export(
        "ack-history-after-withdrawal",
        dict(
            synthetic=True,
            historical_ack=ack.model_dump(mode="json"),
            current_consumption_refused=True,
            tool_attempts=0,
        ),
    )


def test_public_repository_requires_owner_and_preserves_close_and_read(host, service):
    a = application(host, service)
    registered = service.register(a)
    old = claim(service, a)
    unbound = GapQueueRepository(manage.test_target("runtime"))
    incomplete = GapQueue(repository=unbound).repository
    before = counts(host.db)
    for repo in (unbound, incomplete):
        with pytest.raises(ValueError, match="^gap_owner_required$"):
            repo.register(a)
        for action, params in (
            ("claim", dict(worker="consumer")),
            ("ack", dict(worker=old.worker, lease_token=old.lease_token)),
        ):
            with pytest.raises(ValueError, match="^gap_owner_required$"):
                repo.command(registered, action, "direct", **params)
    assert counts(host.db) == before
    host.h.f.service.catalogs.disable(host.h.request.candidate.implementation_ref)
    for action, params in (
        ("claim", dict(worker="consumer")),
        ("ack", dict(worker=old.worker, lease_token=old.lease_token)),
    ):
        with pytest.raises(ValueError, match="^gap_owner_material_changed$"):
            service.repository.command(registered, action, "withdrawn", **params)
    assert counts(host.db) == before
    failed = unbound.command(
        old,
        "fail",
        "close",
        worker=old.worker,
        lease_token=old.lease_token,
        reason="source_withdrawn",
    )
    assert failed.state == "failed" and unbound.read(a.run_id, a.key) == failed
    assert unbound.read_command(a.run_id, a.key, "claim") == old


def test_direct_entry_rechecks_content_purpose_and_technical_failures_once(
    host, service, monkeypatch
):
    a = application(host, service)
    before = counts(host.db)
    for update in (
        dict(purpose="debug"),
        dict(requirement={"code": "RESEARCH:OTHER"}),
        dict(findings=()),
        dict(case_code="CASE:OTHER"),
    ):
        with pytest.raises(ValueError, match="^gap_owner_material_changed$"):
            service.repository.register(a.model_copy(update=update))
    assert counts(host.db) == before
    with monkeypatch.context() as patch:

        def unavailable(*args, **kwargs):
            raise OSError("synthetic_catalog_unavailable")

        patch.setattr(host.h.f.service.catalogs, "read_snapshot", unavailable)
        with pytest.raises(OSError, match="synthetic_catalog_unavailable"):
            service.repository.register(a)
    assert counts(host.db) == before
    original = service.prepare_research
    calls = []

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "prepare_research", count)
    registered = service.register(a)
    assert len(calls) == 1
    lease = service.repository.command(registered, "claim", "direct", worker="consumer")
    assert len(calls) == 2
    ack = service.acknowledge(
        a.run_id,
        a.key,
        command_key="direct-ack",
        worker=lease.worker,
        lease_token=lease.lease_token,
    )
    assert len(calls) == 3 and ack.state == "acknowledged"


def test_cancel_after_withdrawal_and_rebinding_does_not_mutate_other_owner(host, service):
    a = application(host, service)
    registered = service.register(a)
    other = GapQueue(
        repository=service.repository,
        admission_owner=host.h.f.service,
        contexts=service.contexts,
        policy=QueuePolicy(max_claims=2),
    )
    assert other.repository is not service.repository
    with pytest.raises(ValueError, match="^gap_policy_changed$"):
        other.repository.register(a)
    assert service.register(a) == registered
    host.h.f.service.catalogs.disable(host.h.request.candidate.implementation_ref)
    closed = service.repository.command(registered, "cancel", "close", reason="source_withdrawn")
    assert closed.state == "cancelled"
    assert service.report(a.run_id, a.key)["receipt"]["state"] == "cancelled"


def test_repeated_migration_keeps_history_and_formal_task_contracts(host):
    history, before = command_rows(host), counts(host.db)
    versions = manage.recorded_migrations(host.db.migration, "runtime")
    manage.migrate(host.db.migration)
    assert manage.recorded_migrations(host.db.migration, "runtime") == versions
    assert command_rows(host) == history and counts(host.db) == before
    assert sql(host.db.admin, "SELECT count(*) FROM runtime.tool_attempt")[0][0] == 0
    assert command_rows(host) == history


def test_original_control_cancel_blocks_even_cached_claim(host, service):
    from datetime import timedelta

    from agents.planner.service import Planner
    from db.runtime.repository.planner import PlannerRepository
    from packages.planner_control.models import Command
    from tests.planner.support import FakeOwner, binding

    a = application(host, service, "control")
    service.register(a)
    old = claim(service, a)
    runtime = PlannerRepository(manage.test_target("runtime"))
    start, checksum = runtime.origin(a.run_id)
    b = binding(
        case_code=a.case_code,
        run_id=a.run_id,
        run_context_checksum=checksum,
        started_at=start,
        deadline=start + timedelta(seconds=300),
    )
    planner = Planner(runtime=runtime, owner=FakeOwner(b))
    state = planner.begin(b)
    planner.step(
        a.run_id, Command(key="cancel-original", expected_revision=state.revision, action="cancel")
    )
    assert planner.read(a.run_id).cancelled
    before, history = counts(host.db), command_rows(host)
    with pytest.raises(
        psycopg.errors.CheckViolation, match="gap_original_execution_unconfirmed_or_cancelled"
    ):
        claim(service, a)
    assert counts(host.db) == before and command_rows(host) == history
    assert service.read_command(a.run_id, a.key, "claim") == old
    service.cancel(a.run_id, a.key, command_key="close")
