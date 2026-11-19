"""Fixture-only stage grants, manifests and original-owner query material."""

import time
from datetime import UTC, datetime, timedelta

from packages.grounding.models import PortRef, digest
from packages.tool_adapter import (
    AdapterManifest,
    AdapterPolicy,
    FakeProtocol,
    SyntheticGrant,
)
from packages.tool_adapter.models import AdapterPlan, ModelConfiguration
from packages.tool_discovery import Discovery
from packages.tool_discovery.discovery import query
from tests.tool_discovery.support import record, source
from tests.tool_gap_queue.support import setup


def manifest(candidate, **changes):
    return AdapterManifest(
        adapter_id="synthetic-adapter",
        version="1.0.0",
        kind="model",
        candidate=candidate,
        output_port=PortRef(
            tool_ref=candidate.semantics.inputs[0].port.tool_ref,
            direction="output",
            port_name="synthetic_output",
        ),
        model=ModelConfiguration(
            normalization="identity", resize_crop="identity", class_semantics=("fixture",)
        ),
        **changes,
    )


def grant(application, result, m, policy, **changes):
    values = dict(
        run_id=application.run_id,
        task_id=result.task_id,
        stage_key="adapter-1",
        worker="offline-c",
        application_checksum=digest(application),
        discovery_checksum=digest(result),
        candidate_identity=result.selected[0],
        manifest_checksum=digest(m),
        policy_checksum=digest(policy),
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    values.update(changes)
    return SyntheticGrant(**values)


def plan(tmp_path, *, case="ok", candidate=None):
    h = setup(tmp_path)
    a = h.service.prepare_research(h.context.run_id, key="adapter-original")
    p = candidate or record(h)
    d = Discovery(gap_queue=h.service, repository=None, sources=(source([p]),))
    q = query(a, d.policy)
    discovered = d._evaluate(
        "TASK:SYNTHETIC_ADAPTER",
        dict(
            query=q.model_dump(mode="json"),
            policy=d.policy.model_dump(mode="json"),
            sources=list(d.bindings()),
        ),
        deadline=time.monotonic() + 10,
    )
    policy = AdapterPolicy()
    m = manifest(p)
    f = FakeProtocol(case=case)
    g = grant(a, discovered, m, policy)
    return AdapterPlan(
        task_id=g.task_id,
        application=a,
        discovery=discovered,
        query=q,
        candidate=discovered.candidates[0],
        manifest=m,
        policy=policy,
        grant=g,
        fake_protocol_checksum=digest(f),
        lease_token="adapter:fixture",
        runtime_context={},
    ), f
