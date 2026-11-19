from pathlib import Path

from packages.tool_workflow.schema_tools import schemas


def test_schema_snapshots():
    root = Path(__file__).resolve().parents[2] / "packages/tool_workflow/schemas"
    assert {p.name for p in root.glob("*.json")} == set(schemas())
    for name, content in schemas().items():
        assert (root / name).read_text() == content


def test_synthetic_examples_and_retained_context():
    import json

    from packages.tool_workflow import ExecutionPlan

    root = Path(__file__).resolve().parents[2] / "docs/acceptance/06c/examples-final"
    assert len(list(root.glob("*.bundle.json"))) == 4
    for p in root.glob("*.bundle.json"):
        plan = ExecutionPlan.from_bundle_json(p.read_text())
        assert plan.dry_run() == json.loads(
            p.with_name(p.name.replace(".bundle.", ".plan.")).read_text()
        )
        assert not plan.document.execution_authorized
        assert all(s.deployment.allowlist.synthetic_only for s in plan.document.steps)
