"""09B outer I/O cancellation uses the original event and monotone Control."""

from threading import Event

import pytest

from packages.grounding.models import digest
from tests.expert_agent.support import expert_reply
from tests.path_graph_validation.test_expert import ready


@pytest.mark.parametrize("point", ["summary", "binding", "decision", "report", "report_read"])
@pytest.mark.parametrize("failure", [True, False])
def test_outer_io_observes_cancel_and_retries_without_requests(monkeypatch, point, failure):
    f, h, service, summary, args = ready([expert_reply()], {"required": True})
    signal = Event()
    event = args["event"].code
    read, write = f.documents.read, f.documents.write
    fired = []
    before = h.planner.read(f.binding.run_id)
    original_facts = {k: v for k, v in f.documents.saved.items() if k.startswith("accepted:")}

    def interrupt():
        fired.append(1)
        signal.set()
        if failure:
            raise OSError("synthetic_outer_io_failure")

    def reading(key):
        value = read(key)
        hit = (
            (point == "summary" and key == "path-graph:summary")
            or (
                point == "decision"
                and key == "review:" + event + ":decision"
                and read("review:" + event + ":result") is not None
            )
            or (point == "report_read" and key == "path-graph-review:" + event)
        )
        if hit and not fired:
            interrupt()
        return value

    def writing(key, values):
        # Committed acknowledgement loss for binding/final report; preserve exact bytes.
        write(key, values)
        if not fired and (
            (point == "binding" and key == "review:" + event + ":path-graph-binding")
            or (point == "report" and key == "path-graph-review:" + event)
        ):
            interrupt()

    with monkeypatch.context() as m:
        m.setattr(f.documents, "read", reading)
        m.setattr(f.documents, "write", writing)
        if failure:
            with pytest.raises(OSError, match="synthetic_outer_io_failure"):
                service.review(**args, cancel=signal)
        else:
            service.review(**args, cancel=signal)
    assert fired and h.planner.read(f.binding.run_id).cancelled
    assert any(s.cancelled for s in h.planner.runtime.audit)
    saved = read("path-graph-review:" + event)
    requests = f.budget.used
    signal.clear()
    result = service.review(**args)
    assert service.review(**args) == result
    assert f.budget.used == requests == (1 if point in {"summary", "binding"} else 2)
    state = h.planner.replay(f.binding.run_id)
    assert state.cancelled and state.phase == "stopped" and state.pending is None
    assert (state.graph, state.reserved, state.started) == (
        before.graph,
        before.reserved,
        before.started,
    )
    assert all(f.documents.saved[k] == v for k, v in original_facts.items())
    assert read("accepted:" + event + ":domain") is None
    assert service.read(key="summary", historical=True) == summary
    if saved:
        assert read("path-graph-review:" + event) == saved


@pytest.mark.parametrize("field", ["checksum", "code", "role", "version"])
def test_wrong_event_cannot_cancel_or_write_another_run(monkeypatch, field):
    f, h, service, _, args = ready([expert_reply()], {"required": True})
    signal = Event()
    signal.set()
    state = h.planner.read(f.binding.run_id)
    history = digest(f.documents.saved)
    updates = {
        "checksum": digest("another_run"),
        "code": "review:wrong",
        "role": "wrong",
        "version": "9.0.0",
    }
    args["event"] = args["event"].model_copy(update={field: updates[field]})
    if field == "checksum":
        args["event"] = args["event"].model_copy(
            update={"code": "review:" + args["event"].checksum[7:]}
        )
    with pytest.raises(ValueError, match="review_control_binding"):
        service.review(**args, cancel=signal)
    assert h.planner.read(f.binding.run_id) == state
    assert digest(f.documents.saved) == history and f.budget.used == 1


def test_io_failure_kept_when_cancel_persistence_is_unconfirmed(monkeypatch):
    f, h, service, _, args = ready([expert_reply()], {"required": True})
    signal = Event()
    write, save = f.documents.write, h.planner.runtime.save

    def broken(key, values):
        if key.endswith(":path-graph-binding"):
            signal.set()
            raise OSError("binding_io_unknown")
        return write(key, values)

    def unavailable(old, new):
        if new.cancelled:
            raise OSError("control_commit_unconfirmed")
        return save(old, new)

    with monkeypatch.context() as m:
        m.setattr(f.documents, "write", broken)
        m.setattr(h.planner.runtime, "save", unavailable)
        with pytest.raises(OSError, match="binding_io_unknown") as error:
            service.review(**args, cancel=signal)
    assert str(error.value.__cause__) == "control_commit_unconfirmed"
    state = h.planner.read(f.binding.run_id)
    assert not state.cancelled and state.pending is not None
    assert f.documents.read("review:" + args["event"].code + ":result") is None
    # No false confirmation: retain the still-live signal on recovery.
    result = service.review(**args, cancel=signal)
    assert result["result"]["status"] == "safe_stop" and f.budget.used == 1
    assert h.planner.replay(f.binding.run_id).cancelled
