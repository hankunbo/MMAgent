"""CLI/exclusion/closure orchestration units; original commit integration needs PG."""

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_definition_execution as m
from packages.grounding.models import digest


def setup(monkeypatch, tmp_path):
    from db.runtime.repository import case_pipeline
    from packages.evaluation_preparation import generic_control, generic_execution_lock, generic_run
    from packages.multimodal_research import unknown_acquisition

    parent = tmp_path / "original"
    parent.mkdir()
    (parent / "binding.json").write_text('{"case_code":"CASE:ONE"}')
    child = tmp_path / "next"
    policy = Path("configs/research/multimodal_active_requirements_v1.json").resolve()
    calls = []
    selection = dict(schema_version=m.VERSION, key="next", parent_key="original")
    checked = SimpleNamespace(campaign=SimpleNamespace(config=object()))

    def origin(key, value, *, own_claim=None):
        calls.append(("origin", key, own_claim))
        return checked

    class Lease:
        def __init__(self, target):
            pass

        @contextmanager
        def guarded_session(self, case, key):
            calls.append(("lease", case, key))

            def guard():
                calls.append(("guard",))

            yield guard

    class Campaign:
        def __init__(self, c, key, *, sources):
            assert c is checked and key == "next"

        def execute(self):
            self._parent_execution_guard()
            calls.append(("execute",))
            return {"complete": True}

    monkeypatch.setattr(m, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(m, "origin", origin)
    monkeypatch.setattr(m, "selection_for", lambda c, k: selection)
    monkeypatch.setattr(m, "source_owner", lambda c: object())
    monkeypatch.setattr(m, "DefinitionReview", Campaign)
    monkeypatch.setattr(case_pipeline, "CasePipelineRepository", Lease)
    monkeypatch.setattr(generic_execution_lock, "target", lambda: None)
    monkeypatch.setattr(unknown_acquisition, "check_network", lambda c: calls.append(("route",)))
    monkeypatch.setattr(generic_control, "start", lambda v: calls.append(("start",)))
    monkeypatch.setattr(
        generic_run, "persist_export", lambda *a: calls.append(("persist",)) or {"saved": True}
    )
    monkeypatch.setattr(
        generic_run, "finish_export", lambda key: calls.append(("export", key)) or {"saved": True}
    )
    return SimpleNamespace(
        parent=parent, child=child, policy=policy, calls=calls, selection=selection
    )


def test_actual_cli_dispatches_explicit_definition_execution_or_local_export(monkeypatch):
    import sys

    from apps.fairvlmed_case import main

    calls = []
    monkeypatch.setattr(m, "run", lambda *a: calls.append(a) or {"result": {"complete": True}})
    monkeypatch.setattr(m, "export_saved", lambda key: calls.append((key,)) or {"saved": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fairvlmed_case",
            "definition-review",
            "--key",
            "old",
            "--review-key",
            "new",
            "--definition-policy",
            "public.json",
        ],
    )
    assert main() == 0 and calls == [("old", "new", "public.json")]
    monkeypatch.setattr(sys, "argv", ["fairvlmed_case", "definition-export", "--key", "new"])
    assert main() == 0 and calls[-1] == ("new",)


def test_new_window_claim_is_saved_before_original_executor(monkeypatch, tmp_path):
    f = setup(monkeypatch, tmp_path)
    assert m.run("original", "next", f.policy) == {"saved": True}
    assert json.loads((f.parent / m.CLAIM).read_bytes()) == f.selection
    assert json.loads((f.child / "input.json").read_bytes()) == f.selection
    names = [x[0] for x in f.calls]
    assert (
        names.index("lease")
        < names.index("origin")
        < names.index("start")
        < names.index("execute")
        < names.index("persist")
    )
    assert names.count("execute") == 1


@pytest.mark.parametrize("fault", ["started", "claim", "input"])
def test_unknown_or_conflicting_window_never_reissues(monkeypatch, tmp_path, fault):
    f = setup(monkeypatch, tmp_path)
    f.child.mkdir()
    if fault == "started":
        (f.child / "started.json").write_text("{}")
    elif fault == "claim":
        (f.parent / m.CLAIM).write_text('{"different":true}')
    else:
        (f.child / "input.json").write_text('{"different":true}')
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}
    with pytest.raises(
        ValueError, match="generic_definition_(started|successor_conflict|output_conflict)"
    ):
        m.run("original", "next", f.policy)
    assert not any(x[0] in {"route", "start", "execute", "persist"} for x in f.calls)
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*.json")}


def test_repeated_saved_export_needs_no_provider_or_new_window(monkeypatch, tmp_path):
    f = setup(monkeypatch, tmp_path)
    f.child.mkdir()
    (f.child / "export-reference.json").write_text("{}")
    monkeypatch.setattr(
        m, "load_selection", lambda *a: pytest.fail("already persisted owner export")
    )
    assert m.export_saved("next")["status"] == "already_recorded"
    assert m.export_saved("next")["status"] == "already_recorded"
    assert f.calls == [("export", "next"), ("export", "next")]


@pytest.mark.parametrize("existing_result", [False, True])
def test_committed_failed_phase_gets_separate_local_closure_no_rescore(
    monkeypatch, tmp_path, existing_result
):
    from packages.evaluation_preparation import generic_run

    class Docs:
        run_id = "RUN:REVIEW"

        def __init__(self):
            self.saved = {}

        def read(self, key):
            return self.saved.get(key)

        def write(self, key, value):
            if key in self.saved:
                assert self.saved[key] == value
            self.saved[key] = copy.deepcopy(value)

        @contextmanager
        def session(self):
            yield

    docs = Docs()
    phase = dict(
        run_id=docs.run_id,
        target=None,
        accepted=False,
        acceptance_key=None,
        error="local_html_failed",
    )
    old = dict(phases=[phase], new_requests=2, prior_graph_requests=10, complete=False)
    if existing_result:
        docs.saved["dynamic-case-result"] = (copy.deepcopy(old),)
    output = tmp_path / "evidence"
    review = output / "review"
    review.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(old))
    before = (output / "result.json").read_bytes()
    detail = {"unchanged_score": "original committed owner score"}
    proof = {"acceptance_key": "accepted-key", "report_checksum": digest(detail)}
    campaign = SimpleNamespace(
        output=output,
        sample="synthetic",
        base=SimpleNamespace(binding={"case_code": "CASE:ONE"}),
        max_new_requests=5,
        protocol={"previous": {"reservations": 10}},
        export_suffix="_increment",
        usage=lambda: 2,
        phase_path=lambda _: review,
        accepted_observations=lambda: {},
    )
    exports = []
    campaign.export_result = lambda result: exports.append(copy.deepcopy(result))
    monkeypatch.setattr(m, "directory", lambda _: tmp_path)
    monkeypatch.setattr(m, "load_selection", lambda _: campaign)
    monkeypatch.setattr(m, "committed_state", lambda _: (docs, None, phase, (proof, detail)))
    monkeypatch.setattr(generic_run, "persist_export", lambda key, c, result: dict(result=result))
    result = m.export_saved("new")["result"]
    assert result["phases"][0]["accepted"] and result["new_requests"] == 2
    assert result["local_closure"]["api_requests"] == 0
    assert campaign.report_key == "generic-definition-local-case-report"
    assert (output / "result.json").read_bytes() == before
    if existing_result:
        assert docs.saved["dynamic-case-result"] == (old,)
    assert docs.saved["generic-definition-local-case-result"] == (result,)
    assert json.loads((review / "review-report.json").read_bytes()) == detail
    assert len(exports) == 1
