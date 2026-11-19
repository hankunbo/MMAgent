"""Compact wire preserves owner binding; timing never exports free stderr."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest, validate
from packages.llm_provider.models import ImageBinding, Material
from packages.llm_provider.transport import HTTPTransport
from packages.llm_provider.transport_timing import read
from packages.llm_provider.vision_projection import project
from packages.multimodal_research.campaign import configuration
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture


def payload():
    state = dynamic_fixture(1).owner.current()
    task = project_review_tasks(state).tasks[0]
    material = state.material.model_dump(mode="json")
    material["observations"][0]["image"] = ImageBinding(
        reference=task.observation.model_copy(update=dict(role="image")),
        preprocessing=task.definition.reference.model_copy(update=dict(role="preprocessing")),
        media_type="image/png",
        width=8,
        height=8,
        location=(0, 0, 8, 8),
        coordinate_system="pixel_xy_top_left",
    ).model_dump(mode="json")
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    validate(Material, material)
    return dict(
        schema_version="targeted-vision-input/1.0.0",
        role="vision",
        material=material,
        vision_task=task.model_dump(mode="json"),
    )


def test_projection_only_current_source_and_no_scope_or_scores():
    p = payload()
    before = digest(p)
    wire = project(p)
    assert digest(p) == before
    assert wire["examination"] == p["vision_task"]["definition"]["examination"]
    assert {r["code"] for r in wire["relevant_sources"]} == {
        r["code"] for r in p["vision_task"]["allowed_sources"]
    }
    assert "scope" not in wire and "material" not in wire and "goal" not in wire
    assert "e" not in wire and wire["image"]["width"] == 8
    assert len(json.dumps(wire)) < len(json.dumps(p))


@pytest.mark.parametrize("field", ["observation", "allowed_sources", "definition"])
def test_projection_does_not_invent_missing_binding(field):
    p = payload()
    if field == "observation":
        p["vision_task"][field]["code"] = "OBS:OTHER"
    elif field == "allowed_sources":
        p["vision_task"][field][0]["checksum"] = digest("wrong")
    else:
        p["vision_task"][field]["source_refs"] = []
    with pytest.raises(ValueError):
        project(p)


def test_config_legacy_unchanged_compact_opt_in():
    old, new = configuration(), configuration(resilient_vision=True)
    assert old.transport == new.transport and old.max_requests == new.max_requests == 40
    assert [r for r in old.roles if r.role != "vision"] == [
        r for r in new.roles if r.role != "vision"
    ]
    assert next(r for r in new.roles if r.role == "vision").max_output_tokens == 2048


def test_timing_discards_secrets_arbitrary_fields_and_nonfinite():
    assert read(b'secret stderr\nTIMING {"phase":"secret","seconds":1}\n') == []
    assert read(b'TIMING {"phase":"response_headers","seconds":NaN}') == []
    assert read(b'TIMING {"phase":"response_headers","seconds":2,"key":"secret"}') == [
        dict(phase="response_headers", seconds=2)
    ]


def test_timeout_outer_retains_safe_phase_without_retry(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        raise subprocess.TimeoutExpired(
            "synthetic",
            1,
            output=b"DISPATCH\n",
            stderr=b'TIMING {"phase":"response_headers","seconds":0.5}\nprivate stderr',
        )

    monkeypatch.setattr(subprocess, "run", run)
    reply = HTTPTransport().send(
        configuration().transport, {}, timeout=1, max_bytes=100, timing_events=True
    )
    assert reply.error == "timeout" and reply.api_requests == len(calls) == 1
    assert reply.diagnostic["transport_timing"]["timeout_origin"] == "outer_worker_deadline"
    assert "private" not in json.dumps(reply.diagnostic)


def test_success_timing_keeps_http_observation(monkeypatch):
    value = dict(
        http_status=200,
        body=dict(
            model="synthetic",
            choices=[dict(finish_reason="stop", message=dict(content='{"ok":true}'))],
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0,
            stdout=b"DISPATCH\n" + json.dumps(value).encode(),
            stderr=b'TIMING {"phase":"response_body","seconds":1}',
        ),
    )
    transport = HTTPTransport()
    reply = transport.send(
        configuration().transport, {}, timeout=1, max_bytes=100, timing_events=True
    )
    assert reply.error is None and transport.observation(reply)["http_status"] == 200
    assert reply.diagnostic["transport_timing"]["phases"][-1]["phase"] == "response_body"
