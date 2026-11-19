"""Independently installed fixture authority; never synthesized by consume."""

from datetime import UTC, datetime, timedelta

from db import manage
from db.runtime.repository.tool_extension import ExtensionRepository
from packages.grounding.models import digest
from packages.tool_extension import (
    ExtensionAuthority,
    ExtensionGrant,
    ExtensionPolicy,
    OfflineExtension,
)


def assemble(s, *, catalog=None, future=None, **changes):
    record = s.c.read(s.a.run_id, s.a.key)
    policy = ExtensionPolicy()
    p = s.c.repository.read(s.a.run_id, s.a.key)["plan"]
    from packages.grounding.models import validate
    from packages.tool_adapter.models import AdapterPlan

    binding = future.binding(validate(AdapterPlan, p)) if future else None
    values = dict(
        run_id=s.a.run_id,
        task_id=record["task_id"],
        stage_key="extension-1",
        worker="offline-d",
        adapter_checksum=digest(record),
        policy_checksum=digest(policy),
        catalog_fixture_checksum=digest(catalog.fixture if catalog else None),
        future_binding_checksum=digest(binding),
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    values.update(changes)
    return OfflineExtension(
        adapter=s.c,
        repository=ExtensionRepository(manage.test_target("runtime")),
        authority=ExtensionAuthority([ExtensionGrant(**values)]),
        policy=policy,
        catalog=catalog,
        future=future,
    )


def consume(e, s):
    return e.consume(s.a.run_id, s.a.key, stage_key="extension-1", worker="offline-d")
