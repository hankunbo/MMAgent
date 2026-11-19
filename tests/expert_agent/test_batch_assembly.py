"""Actual original Planner/acceptance/installation code, synthetic memory persistence."""

from threading import Event
from types import SimpleNamespace

import pytest

from apps.vectorengine_live_assembly import assemble, make_binding, run_case_stages
from apps.vectorengine_single_case import load_config
from packages.grounding.models import digest
from packages.planner_control.models import Command
from tests.agent_provider.batch_support import CONFIG
from tests.agent_provider.support import Documents
from tests.agent_provider.test_bounded_live import SyntheticWire
from tests.expert_agent.batch_case_support import REPLIES, initial_state
from tests.planner.support import MemoryRuntime, binding


class LiveDocuments(Documents):
    def consumed(self, graph):
        return any(
            v.get("schema_version")
            in {
                "research-acceptance/1.0.0",
                "research-acceptance/2.0.0",
                "research-acceptance/3.0.0",
            }
            and v["status"] == "accepted"
            and v["parent_graph"]["code"] == graph.code
            for values in self.saved.values()
            for v in values
        )


def assembly(
    monkeypatch,
    count=7,
    *,
    replies=REPLIES,
    source_quote_policy="research-quote-policy/literal/1.0.0",
    config_path=CONFIG,
):
    b = binding()
    config = load_config(config_path)
    initial, raw_image = initial_state(b, config, count)
    b, ranking = make_binding(
        initial=initial,
        config=config,
        started=b.started_at,
        context_checksum=b.run_context_checksum,
        catalog=next(r for r in b.locks if r.role == "catalog"),
    )
    docs = LiveDocuments()
    docs.sessions = MemoryRuntime()
    docs.sessions.origin = lambda _: (b.started_at, b.run_context_checksum)
    docs.run_id, docs.root = b.run_id, SimpleNamespace(checksum=b.graph.checksum)

    def verify(state):
        assert state.policy == initial.policy and state.material.sources == initial.material.sources

    evidence = SimpleNamespace(verify=verify, image=lambda _: raw_image)
    wire = SyntheticWire(replies)
    monkeypatch.setattr(
        "packages.llm_provider.transport.HTTPTransport.send",
        lambda self, config, body, **kw: wire.send(config, body, **kw),
    )
    planner, owner, ports, review = assemble(
        evidence=evidence,
        initial=initial,
        config=config,
        documents=docs,
        binding=b,
        ranking=ranking,
        source_quote_policy=source_quote_policy,
    )
    planner.begin(b)
    return SimpleNamespace(
        planner=planner,
        owner=owner,
        ports=ports,
        review=review,
        documents=docs,
        binding=b,
        config=config,
        wire=wire,
        initial=initial,
        ranking=ranking,
        evidence=evidence,
    )


def stages(f, tmp_path):
    return run_case_stages(
        planner=f.planner,
        owner=f.owner,
        ports=f.ports,
        review=f.review,
        documents=f.documents,
        binding=f.binding,
        config=f.config,
        output=tmp_path,
        signal_cancel=Event(),
    )


def test_continuous_source_vision_then_freeze_all_batches_one_review_acceptance(
    monkeypatch, tmp_path
):
    f = assembly(monkeypatch)
    completed, roles = stages(f, tmp_path)
    assert roles == ("source", "vision", "review")
    assert [r["status"] for r in completed] == ["accepted"] * 3
    assert f.ports.budget.used == len(f.wire.calls) == 5
    assert len([k for k in f.documents.saved if k.startswith("accepted:")]) == 3
    assert len([k for k in f.documents.saved if k.startswith("research-owner-install:")]) == 3
    result = completed[-1]
    records = f.review.acceptance.read(key=result["acceptance_key"])[0]
    assert records["scope"]["graph"]["code"] == completed[1]["graph"]["code"]
    assert len(records["batch_results"]) == 3
    html = (tmp_path / "review-report.html").read_text()
    assert "计算依赖与判读背景" in html and "<details>" in html
    assert f.planner.read(f.binding.run_id).steps == 13
    audit = list(f.documents.audit)
    assert stages(f, tmp_path)[0] == completed
    assert f.documents.audit == audit and f.ports.budget.used == 5
    state = f.planner.read(f.binding.run_id)
    f.planner.step(
        f.binding.run_id, Command(key="final-stop", expected_revision=state.revision, action="stop")
    )
    assert f.planner.read(f.binding.run_id).steps == 14


@pytest.mark.parametrize(
    "boundary", ["accepted:", "research-owner-install:", "checkpoint", "live-stage:"]
)
def test_original_save_install_confirmation_loss_same_event_recovery(
    monkeypatch, tmp_path, boundary
):
    f = assembly(monkeypatch)
    original = f.documents.write
    fail = [True]

    def write(key, values):
        original(key, values)
        if fail[0] and boundary in key:
            fail[0] = False
            raise OSError("synthetic_saved_ack_lost")

    f.documents.write = write
    if boundary == "checkpoint":
        checkpoint = f.review.context.checkpoint

        def lost_checkpoint(*args, **kwargs):
            checkpoint(*args, **kwargs)
            if fail[0]:
                fail[0] = False
                raise OSError("synthetic_saved_ack_lost")

        f.review.context.checkpoint = lost_checkpoint
    with pytest.raises(OSError, match="ack_lost"):
        stages(f, tmp_path)
    f.documents.write = original
    completed, _ = stages(f, tmp_path)
    assert len(completed) == 3 and all(r["status"] == "accepted" for r in completed)
    assert len(f.wire.calls) == 5
    assert len([k for k in f.documents.saved if k.startswith("research-owner-install:")]) == 3
    assert digest(completed) == digest(stages(f, tmp_path)[0])
