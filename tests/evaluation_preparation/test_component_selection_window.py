"""Real new owner/read path with synthetic ports; no DB, model or Provider."""

import json
from types import SimpleNamespace as NS

import pytest

from apps.vectorengine_case_materials import ref
from packages.grounding.models import Value, digest
from packages.multimodal_research import component_consumer, component_observations
from packages.research_runtime import component_steps
from tests.evaluation_preparation.test_component_consumer import config


class Asset(Value):
    asset_code: str = "ASSET:MASK"
    checksum: str = digest("mask")
    uri: str = "object://mask"


class Result(Value):
    artifacts: tuple[Asset, ...] = (Asset(),)


class Step(Value):
    result: Result = Result()


class Workflow(Value):
    status: str = "SUCCEEDED"
    plan_checksum: str = digest("plan")
    steps: tuple[Step, ...] = (Step(),)


@pytest.mark.parametrize("changed", [False, True])
def test_selection_changed_during_last_qualification_is_rejected(tmp_path, monkeypatch, changed):
    path, selection = config(tmp_path)
    monkeypatch.setattr(component_steps, "validate", lambda cls, value: value)
    monkeypatch.setattr(
        component_observations,
        "project_fundus",
        lambda *a, **kw: ({"values": {"vCDR": 0.5}}, {"quality": 0.7, "status": "obtained"}),
    )
    task = NS(checksum=digest("task"))
    manifest = NS(checksum=digest("input"))
    parent = NS(
        run_id="RUN:PARENT",
        request=NS(
            scope=NS(case_code="CASE:ORIGINAL"),
            input_checksum=manifest.checksum,
            visible_checksum=digest("visible"),
            task=task,
        ),
    )
    state = NS(
        binding=NS(run_id=parent.run_id),
        pending=None,
        confirmed=1,
        graph="graph",
        result_refs=("result",),
    )
    registered = NS(artifacts=(NS(checksum=digest("image")),), lock=NS(run_id="RUN:CHILD"))
    owner = object.__new__(component_steps.ComponentPlannerSteps)
    owner.requests = NS(
        parent=parent, component_id="fundus", request_key="request", for_key=lambda key: registered
    )
    owner.sessions = NS(read=lambda run: state)
    owner._saved = lambda value: ("plan", NS(run_id="RUN:CHILD"))
    workflow = Workflow()
    owner.executor_for = lambda plan: NS(
        recover=lambda *args: {"result": workflow}, confirmed_result=lambda *args: workflow
    )
    owner._result = lambda value: NS(refs=("result",))
    owner.current = lambda binding: "graph"
    owner.research = NS(compiler=NS(admission=NS(objects=NS(read_bounded=lambda *a: b"raw"))))
    calls = []

    def requirements(*a, **kw):
        calls.append("qc")
        if changed and calls.count("qc") == 2:
            body = json.loads(path.read_bytes())
            body["request_key"] = "changed-selection"
            path.write_text(json.dumps(body))

    owner._consumption_requirements = requirements

    def guard():
        calls.append("selection")
        component_consumer.load(path, expected=digest(selection))

    base = NS(
        binding={"case_code": "CASE:ORIGINAL"},
        task=task,
        visible_value=NS(checksum=parent.request.visible_checksum),
        read_dependency_input=lambda: manifest,
        ledger=[
            dict(id="check:geometry", component="fundus", acquisition_method="disc_cup_measurement")
        ],
        resources=(ref("resource", "ASSET:ROOT", digest("root")),),
    )

    def read():
        return component_observations.ComponentObservations(
            base=base, owner=owner, execution_key="key", target="check:geometry", guard=guard
        )

    if changed:
        with pytest.raises(ValueError, match="component_consumer_selection_changed"):
            read()
    else:
        assert read().value["values"]["vCDR"] == 0.5
    assert calls.count("qc") == 2
