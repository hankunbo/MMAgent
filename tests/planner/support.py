"""Anonymous values and failure-injection ports; no database/model substitution claims."""

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from agents.planner.service import Planner
from packages.grounding.models import digest, validate
from packages.planner_control.models import (
    Binding,
    Candidate,
    Control,
    Graph,
    Outcome,
    Profile,
    Reference,
)


def binding(**changes):
    now = datetime.now(UTC)
    profile = changes.pop("profile", Profile())
    values = dict(
        case_code="CASE:ANON",
        run_id="RUN:ANON",
        run_context_checksum=digest("context"),
        input_checksum=digest("visible"),
        visible_checksum=digest("manifest"),
        graph=Graph(
            role="graph",
            code="ASSET:ANON",
            version="case-evidence-graph/1.19.0",
            checksum=digest("graph"),
            kind="research_artifact",
        ),
        locks=tuple(
            Reference(role=r, code="POLICY:" + r.upper(), version="1.0.0", checksum=digest(r))
            for r in (
                "knowledge",
                "template",
                "rule",
                "catalog",
                "mapping",
                "execution",
                "ranking",
                "prompt",
                "model",
                "sources",
            )
        ),
        profile=profile,
        started_at=now,
        deadline=now + timedelta(seconds=profile.seconds),
    )
    values.update(changes)
    return Binding(**values)


class MemoryRuntime:
    def __init__(self):
        self.state = None
        self.audit = []
        self.fail = None

    @contextmanager
    def session(self, run_id):
        yield

    def begin(self, value):
        if self.state:
            if self.state.binding != value:
                raise ValueError("planner_binding_conflict")
            return self.state
        return self.save(None, Control(binding=value, graph=value.graph))

    def read(self, run_id):
        return self.state

    def save(self, old, new):
        from packages.planner_control.transitions import validate_checkpoint

        validate_checkpoint(old, new)
        assert self.state == old
        if self.fail == "rollback":
            self.fail = None
            raise RuntimeError("rollback")
        self.state = validate(Control, new)
        self.audit.append(self.state)
        if self.fail == "lost_reply":
            self.fail = None
            raise RuntimeError("lost_reply")
        return self.state


class FakeOwner:
    def __init__(self, value):
        self.graph = value.graph
        self.calls = []
        self.recoveries = 0
        self.failure = None
        self.on_call = None
        self.choices = (Candidate(id="A", branch="A", operation="observe", and_ready=True),)

    def current(self, binding):
        return self.graph

    def invoke(self, state, pending, *, cancel):
        action = pending.command.action
        self.calls.append(action)
        if self.on_call:
            self.on_call(state, pending)
        if action == self.failure:
            raise RuntimeError("injected")
        if action == "identify":
            return Outcome(candidates=self.choices)
        if action == "execute":
            return Outcome(started=1, confirmed=1)
        return Outcome()

    def recover(self, state, pending):
        self.recoveries += 1
        return (
            Outcome(status="unknown", reason="unknown_job")
            if self.failure
            else Outcome(started=1, confirmed=1)
        )


def setup(**changes):
    b = binding(**changes)
    runtime, owner = MemoryRuntime(), FakeOwner(b)
    planner = Planner(runtime=runtime, owner=owner)
    planner.begin(b)
    return planner, runtime, owner


def command(planner, action, key=None):
    from packages.planner_control.models import Command

    state = planner.runtime.state
    return Command(
        key=key or f"step-{state.steps}-{action}", expected_revision=state.revision, action=action
    )


def advance(planner, until="execute"):
    for action in (
        "load",
        "retrieve",
        "instantiate",
        "identify",
        "select",
        "prepare",
        "execute",
        "verify",
        "review",
    ):
        if action == until:
            return
        planner.step(planner.runtime.state.binding.run_id, command(planner, action))
