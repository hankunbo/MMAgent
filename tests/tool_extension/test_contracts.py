"""CPU boundary tests for independently installed Fake catalog and fresh 05/06."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest, validate
from packages.tool_extension import (
    CatalogFixture,
    ExtensionAuthority,
    ExtensionGrant,
    ExtensionPlan,
    ExtensionPolicy,
    SyntheticCatalogOwner,
    SyntheticFutureRunOwner,
)
from tests.grounding.query_support import updated
from tests.tool_adapter.support import plan as adapter_plan
from tests.tool_admission.support import fixture, reseal


def setup(tmp_path):
    path = tmp_path / "origin"
    path.mkdir()
    p, fake = adapter_plan(path)
    record = dict(
        task_id=p.task_id,
        status="saved",
        plan=p.model_dump(mode="json"),
        result=fake.evaluate(p).model_dump(mode="json"),
    )
    path = tmp_path / "future"
    path.mkdir()
    f = fixture(path)
    # Original registered research catalog is 9.0.0; target formal Fake catalog is 1.0.0.
    f.q = updated(f.q, lock=reseal(f.q.lock, run_id="RUN:OID_11DFORMAL"))
    f.locks.q = f.q
    ticket = CatalogFixture(
        adapter_checksum=digest(record),
        candidate_identity=p.candidate.identity,
        catalog_ref=f.catalog.reference(),
        implementation_ref=f.catalog.tools[0].implementation_ref,
        target_output_port=f.catalog.tools[0].output_ports[0],
        fixture_policy=f.policy,
        fixture_policy_checksum=digest(f.policy),
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    owner = SyntheticCatalogOwner(reader=f.catalogs, fixture=ticket)
    future = SyntheticFutureRunOwner(
        catalog=owner,
        admission=f.service,
        run_id=f.q.lock.run_id,
        query_reader=lambda rid: f.q,
        selection_reader=lambda rid: f.selection,
    )
    policy = ExtensionPolicy()
    g = ExtensionGrant(
        run_id=p.application.run_id,
        task_id=p.task_id,
        stage_key="extension-1",
        worker="offline-d",
        adapter_checksum=digest(record),
        policy_checksum=digest(policy),
        catalog_fixture_checksum=digest(ticket),
        future_binding_checksum=digest(future.binding(p)),
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    ep = ExtensionPlan(
        task_id=p.task_id,
        adapter_record=record,
        adapter_plan=p,
        adapter_result=fake.evaluate(p),
        policy=policy,
        grant=g,
        catalog_fixture=ticket,
        catalog_observation=owner.current(record),
        future_binding=future.binding(p),
        runtime_context={},
        lease_token="extension:fixture",
    )
    return SimpleNamespace(f=f, owner=owner, future=future, plan=ep)


def test_fresh_original_05_06_checks_are_not_cached_or_activation(tmp_path, monkeypatch):
    s = setup(tmp_path)
    calls = []
    original = s.f.grounding.query

    def query(q):
        calls.append(q.lock.run_id)
        return original(q)

    monkeypatch.setattr(s.f.grounding, "query", query)
    result = s.future.evaluate(s.plan)
    assert calls and set(calls) == {s.future.run_id}
    assert result["grounding_status"] == "recomputed"
    assert result["admission"]["status"] == "preparation_only"
    assert result["execution"] == "not_executed" and not result["available_next_run"]
    assert result["original_run_id"] != result["new_run_id"]
    s.f.catalogs.disable(s.owner.fixture.implementation_ref)
    with pytest.raises(ValueError, match="unavailable_or_terminal"):
        s.future.evaluate(s.plan)


@pytest.mark.parametrize(
    "change", ["same_run", "old_catalog", "scope", "version", "expired", "unregistered", "contract"]
)
def test_new_run_requires_current_owner_and_exact_new_locks(tmp_path, change):
    s = setup(tmp_path)
    if change == "same_run":
        s.future.run_id = s.plan.grant.run_id
    elif change == "old_catalog":
        s.owner.fixture = s.owner.fixture.model_copy(
            update=dict(
                catalog_ref=s.plan.adapter_plan.application.material["request"]["lock"][
                    "catalog_ref"
                ]
            )
        )
    elif change == "scope":
        s.f.q = s.f.q.model_copy(
            update=dict(scope=s.f.q.scope.model_copy(update=dict(case_code="CASE:OID_OTHER000")))
        )
    elif change == "version":
        s.f.q = updated(s.f.q, lock=reseal(s.f.q.lock, graph_version=2))
    elif change == "expired":
        s.owner.fixture = s.owner.fixture.model_copy(
            update=dict(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
    elif change == "unregistered":
        s.f.locks.q = updated(s.f.q, lock=reseal(s.f.q.lock, graph_version=2))
    else:
        s.owner.fixture = s.owner.fixture.model_copy(
            update=dict(target_output_port=s.f.catalog.tools[1].output_ports[0])
        )
    with pytest.raises((ValueError, AttributeError)):
        s.future.evaluate(s.plan)


@pytest.mark.parametrize("status", ["candidate", "quarantined", "failed", "deprecated"])
def test_catalog_terminal_or_unqualified_never_preview(tmp_path, status):
    s = setup(tmp_path)
    from packages.grounding.synthetic import SyntheticCatalogReader

    tools = [t.model_copy(update=dict(catalog_status=status)) for t in s.f.catalog.tools]
    snapshot = reseal(s.f.catalog, version="3.0.0", tools=tools)
    s.owner.reader = SyntheticCatalogReader([snapshot])
    s.owner.fixture = s.owner.fixture.model_copy(update=dict(catalog_ref=snapshot.reference()))
    with pytest.raises(ValueError, match="unavailable_or_terminal"):
        s.owner.current(s.plan.adapter_record)


def test_no_formal_or_real_owner_can_be_enabled_by_fake_flag(tmp_path):
    s = setup(tmp_path)
    with pytest.raises(ValueError, match="synthetic_catalog_owner_required"):
        SyntheticCatalogOwner(reader=SimpleNamespace(), fixture=s.owner.fixture)
    with pytest.raises(ValueError):
        validate(CatalogFixture, {**s.owner.fixture.model_dump(mode="json"), "synthetic": False})
    with pytest.raises(ValueError):
        validate(
            ExtensionGrant, {**s.plan.grant.model_dump(mode="json"), "purpose": "real_activation"}
        )
    for field in [
        "adapter_checksum",
        "policy_checksum",
        "catalog_fixture_checksum",
        "future_binding_checksum",
    ]:
        grant = s.plan.grant.model_copy(update={field: digest("other")})
        with pytest.raises(ValueError, match="extension_plan_binding"):
            validate(ExtensionPlan, s.plan.model_copy(update=dict(grant=grant)))


def test_grant_is_separate_bounded_expiring_and_revocable(tmp_path):
    s = setup(tmp_path)
    g = s.plan.grant
    authority = ExtensionAuthority([g])
    assert authority.current(g.run_id, g.task_id, g.stage_key, g.worker) == g
    with pytest.raises(ValueError, match="identity_conflict"):
        authority.current(g.run_id, g.task_id, "escape", g.worker)
    authority.revoke(g.run_id, g.task_id)
    with pytest.raises(ValueError, match="current_grant_required"):
        authority.current(g.run_id, g.task_id, g.stage_key, g.worker)
    with pytest.raises(ValueError):
        ExtensionAuthority([g, g])
    expired = g.model_copy(update=dict(expires_at=datetime.now(UTC) - timedelta(seconds=1)))
    with pytest.raises(ValueError, match="grant_expired"):
        ExtensionAuthority([expired]).current(g.run_id, g.task_id, g.stage_key, g.worker)


@pytest.mark.parametrize("blocked", [(), ("B",)])
def test_original_main_channel_finishes_without_extension_consumer(tmp_path, blocked):
    # A prepared D value is entirely independent of the original Planner continuation.
    s = setup(tmp_path)
    from tests.validation_recovery.support import fixture as recovery_fixture

    h = recovery_fixture(category="capability_gap", blocked=blocked)
    before = h.planner.read(h.run_id)
    h.service.resolve(h.run_id, key="11d-main")
    after = h.planner.read(h.run_id)
    assert after.phase == ("stopped" if blocked else "select")
    assert after.graph == before.graph and after.binding == before.binding
    assert all(r in after.result_refs for r in before.result_refs) and after.pending is None
    assert (
        h.f.budget.used == int(bool(blocked))
        and s.plan.adapter_result.validation_status == "unvalidated"
    )


def test_future_owner_failure_is_technical_not_no_candidate(tmp_path, monkeypatch):
    s = setup(tmp_path)

    def unavailable(*args, **kwargs):
        raise OSError("original_05_unavailable")

    monkeypatch.setattr(s.f.grounding, "query", unavailable)
    with pytest.raises(OSError, match="original_05_unavailable"):
        s.future.evaluate(s.plan)


def test_future_fixture_policy_is_explicit_not_fake_pass(tmp_path):
    s = setup(tmp_path)
    s.owner.fixture = s.owner.fixture.model_copy(
        update=dict(fixture_policy_checksum=digest("other"))
    )
    with pytest.raises(ValueError, match="fixture_policy_mismatch"):
        s.future.evaluate(s.plan)
