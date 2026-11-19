"""Bounded accepted-fact method change, not a pending-task or score rewrite."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_method_review as method
from packages.evaluation_preparation.generic_review import run_review
from packages.grounding.models import digest
from packages.llm_provider.config import read_provider_config


def old_config():
    return read_provider_config(
        json.loads(Path("configs/agents/fairvlmed_relay_review_single_v1.json").read_bytes())
    )


def test_only_explicit_text_method_delta_and_no_same_method_loop():
    origin = old_config()
    before = origin.model_dump_json()
    selected = method.configuration(origin, method.PROFILE)
    assert origin.model_dump_json() == before
    a, b = origin.model_dump(mode="json"), selected.model_dump(mode="json")
    assert a["roles"] == b["roles"]
    assert b["transport"]["model"] != a["transport"]["model"]
    assert b["transport"]["max_retries"] == 1
    assert b["max_extra_requests"] == a["max_extra_requests"] == 3
    assert b["transport"]["timeout_seconds"] == 180
    assert b["review_batching"]["batch_size"] == 3
    b["transport"] = a["transport"]
    b["review_batching"] = a["review_batching"]
    assert a == b
    with pytest.raises(ValueError, match="origin_configuration"):
        method.configuration(selected, method.PROFILE)
    with pytest.raises(ValueError, match="profile"):
        method.configuration(origin, "unbounded")


@pytest.mark.parametrize(
    "options",
    [
        dict(method_origin="origin"),
        dict(review_method_profile=method.PROFILE),
        dict(method_origin="origin", review_method_profile=method.PROFILE, review_profile="old"),
        dict(
            method_origin="origin",
            review_method_profile=method.PROFILE,
            acknowledge_truncated_request="accepted-cannot-be-pending",
        ),
    ],
)
def test_existing_entry_rejects_ambiguous_scope_before_owner_or_dispatch(options):
    with pytest.raises(ValueError, match="explicit_scope"):
        run_review("root", "new", **options)


def test_existing_entry_routes_explicit_method_once(monkeypatch):
    calls = []
    monkeypatch.setattr(method, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or {})
    assert (
        run_review("root", "new", method_origin="aggregate", review_method_profile=method.PROFILE)
        == {}
    )
    assert calls == [
        (("root", "new"), dict(method_origin="aggregate", review_method_profile=method.PROFILE))
    ]


def test_prior_rechecks_original_committed_state_and_journal(monkeypatch):
    control_raw = {"phase": "stopped", "pending": None}
    bundle = ({"score": "immutable-original"},)
    records = {
        "aggregate": {
            "multimodal-case-result": ({"complete": True},),
            "accepted-aggregation-manifest": ({"run_id": "aggregate"},),
        },
        "old": {"accepted:key": bundle},
    }

    class Docs:
        def __init__(self, run):
            self.run_id = run

        def read(self, key):
            return records[self.run_id].get(key)

    journals = {"aggregate": (), "old": ({"phase": "confirmed"},)}
    monkeypatch.setattr(
        method, "RuntimeAgentJournal", lambda d: SimpleNamespace(read=lambda: journals[d.run_id])
    )
    campaign = SimpleNamespace(
        documents=Docs, sessions=SimpleNamespace(read=lambda run: control_raw)
    )
    proof = dict(
        control=digest(control_raw),
        previous_requests=15,
        origins=[
            dict(
                run_id="old",
                control_checksum=digest(control_raw),
                journal_checksum=digest(journals["old"]),
                acceptance_key="key",
                acceptance_checksum=digest(bundle),
            )
        ],
    )
    current = object.__new__(method.AcceptedMethodReview)
    current.method_origin = SimpleNamespace(
        campaign=campaign,
        documents=Docs("aggregate"),
        proof=proof,
        result={"complete": True},
        manifest={"run_id": "aggregate"},
    )
    assert current.prior() == dict(reservations=15, accepted_method_origin=proof)
    journals["old"] = (*journals["old"], {"phase": "reserved"})
    with pytest.raises(ValueError, match="ancestor_changed"):
        current.prior()
    journals["old"] = ({"phase": "confirmed"},)
    control_raw["pending"] = "unknown"
    with pytest.raises(ValueError, match="origin_changed"):
        current.prior()


def test_before_review_revalidates_proof_and_does_not_archive_as_memory_success(monkeypatch):
    current = object.__new__(method.AcceptedMethodReview)
    proof = dict(origin="exact", model_ids=["a", "b"])
    current.method_origin = SimpleNamespace(proof=proof)
    calls = []

    def reader(path, profile):
        calls.append((path, profile))
        return SimpleNamespace(proof=dict(proof))

    monkeypatch.setattr(method, "current_origin", reader)
    assert current.before_review() is None
    assert calls == [("exact", method.PROFILE)]
    monkeypatch.setattr(
        method, "current_origin", lambda *a: SimpleNamespace(proof={**proof, "model_ids": ["a"]})
    )
    with pytest.raises(ValueError, match="current_origin_changed"):
        current.before_review()
    result = dict(complete=True)
    assert current.archive_completed(result) is result


def test_counts_determine_finite_budget_without_changing_original_40():
    assert method.request_limit(10, 15) == 6
    assert method.request_limit(1, 37) == 3
    for count, previous in ((0, 15), (16, 0), (1, -1), (1, 41), (10, 35)):
        with pytest.raises(ValueError, match="generic_method_"):
            method.request_limit(count, previous)


def test_original_budget_hook_narrows_dispatch_and_preserves_finalization(monkeypatch):
    from packages.evaluation_preparation.generic_review import GenericReview
    from packages.llm_provider.provider import ProviderError

    calls = []
    budget = SimpleNamespace(before_reserve=lambda sha: calls.append(sha))
    monkeypatch.setattr(GenericReview, "budget_factory", lambda self, **kw: budget)
    current = object.__new__(method.AcceptedMethodReview)
    current.max_new_requests = 6
    used = [4]
    current.usage = lambda: used[0]
    bounded = current.budget_factory()
    bounded.before_reserve("exact-config")
    assert calls == ["exact-config"]
    used[0] = 5
    with pytest.raises(ProviderError, match="budget"):
        bounded.before_reserve("exact-config")
    assert calls == ["exact-config"]


def test_claim_recovery_does_not_dispatch_twice_or_allow_another_key(monkeypatch, tmp_path):
    from packages.evaluation_preparation import generic_control

    parent = tmp_path / "root"
    parent.mkdir()
    (parent / "binding.json").write_text(json.dumps({"case": "original"}))
    origin = SimpleNamespace(proof={"root_key": "root", "facts": "retained"})
    monkeypatch.setattr(method, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(method, "current_origin", lambda *args: origin)
    calls = []
    campaign = SimpleNamespace(execute=lambda: calls.append("execute") or {"complete": True})
    monkeypatch.setattr(method, "AcceptedMethodReview", lambda *args: campaign)
    monkeypatch.setattr(generic_control, "start", lambda *a: calls.append("start"))
    monkeypatch.setattr(generic_control, "guard", lambda *a: None)
    monkeypatch.setattr(method, "finish_method_export", lambda key: {"saved": key})

    def persist(key, current, result):
        assert current is campaign
        (tmp_path / key / "export-reference.json").write_text("{}")
        return dict(result=result)

    monkeypatch.setattr(method, "persist_method_export", persist)
    args = dict(method_origin="exact-aggregate", review_method_profile=method.PROFILE)
    assert method.run("root", "new", **args) == {"result": {"complete": True}}
    claim = (parent / "accepted-method-successor.json").read_bytes()
    assert method.run("root", "new", **args)["status"] == "already_recorded"
    assert calls == ["start", "execute"]
    assert (parent / "accepted-method-successor.json").read_bytes() == claim
    with pytest.raises(ValueError, match="successor_already_claimed"):
        method.run("root", "another", **args)
    assert not (tmp_path / "another").exists()
    assert not (parent / "review-successor.json").exists()


def test_wrong_output_rejected_before_claim(monkeypatch, tmp_path):
    parent, out = tmp_path / "root", tmp_path / "other"
    parent.mkdir()
    out.mkdir()
    (out / "input.json").write_text(json.dumps({"unrelated": True}))
    monkeypatch.setattr(method, "directory", lambda key: tmp_path / key)
    monkeypatch.setattr(
        method, "current_origin", lambda *a: SimpleNamespace(proof={"root_key": "root"})
    )
    with pytest.raises(ValueError, match="output_already_exists"):
        method.run("root", "other", method_origin="aggregate", review_method_profile=method.PROFILE)
    assert not (parent / "accepted-method-successor.json").exists()
    assert json.loads((out / "input.json").read_bytes()) == {"unrelated": True}


def method_media_fixture(tmp_path, monkeypatch):
    from tests.evaluation_preparation.test_review_media_origin import fixture

    parent, child, report, calls = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(method, "directory", lambda key: tmp_path / key)
    selection = dict(
        schema_version=method.VERSION,
        parent_key="parent",
        key="child",
        profile=method.PROFILE,
        origin=dict(root_key="parent"),
    )
    (child / "input.json").write_text(json.dumps(selection))
    (parent / "accepted-method-successor.json").write_text(json.dumps(selection))
    return parent, child, report, calls


def test_method_media_uses_exact_parent_without_relaxing_old_collector(tmp_path, monkeypatch):
    from packages.evaluation_preparation.evidence_media import acquisition_root, checked_file
    from packages.object_store import checksum
    from tests.evaluation_preparation.test_evidence_tree_v2 import PNG

    parent, child, report, calls = method_media_fixture(tmp_path, monkeypatch)
    assert acquisition_root(report, "child")[0] == child  # Unchanged legacy semantics.
    root, binding = method.media_origin(report, "child")
    assert root == parent and binding["case_code"] == "CASE:TEST" and calls == ["parent"]
    mask = parent / "mask.png"
    mask.write_bytes(PNG)
    assert checked_file(root, mask.as_uri(), checksum(PNG)) == PNG
    other = tmp_path / "other.png"
    other.write_bytes(PNG)
    with pytest.raises(ValueError, match="media_file_outside_run"):
        checked_file(root, other.as_uri(), checksum(PNG))


@pytest.mark.parametrize("fault", ["claim", "binding", "version", "parent", "case", "native"])
def test_method_media_rejects_other_original_artifacts(tmp_path, monkeypatch, fault):
    parent, child, report, _ = method_media_fixture(tmp_path, monkeypatch)
    path = parent / "accepted-method-successor.json"
    if fault == "binding":
        path = child / "binding.json"
    elif fault in {"version", "parent"}:
        path = child / "input.json"
    elif fault in {"case", "native"}:
        path = parent / "report.json"
    raw = json.loads(path.read_bytes())
    if fault == "version":
        raw["schema_version"] = "unsupported"
    elif fault == "parent":
        raw["parent_key"] = "child"
    elif fault == "case":
        raw["result"]["case_code"] = "CASE:OTHER"
    elif fault == "native":
        raw["native_tools"] = {"other": True}
    else:
        raw["changed"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="generic_method_media_"):
        method.media_origin(report, "child")


def test_method_export_reference_uses_unchanged_original_owner_and_repeat(tmp_path, monkeypatch):
    from packages.evaluation_preparation import generic_run
    from tests.evaluation_preparation.test_generic_export import fixture

    out, _, artifacts = fixture(tmp_path, monkeypatch)
    existing = (out / "export-reference.json").read_bytes()
    reference = json.loads(existing)
    result = dict(phases=[dict(run_id=reference["run_id"])], complete=True)
    docs = SimpleNamespace(
        run_id=reference["run_id"],
        root=SimpleNamespace(checksum=reference["root_checksum"]),
        key=lambda _: reference["bundle_key"],
    )
    campaign = SimpleNamespace(
        documents=lambda _: docs,
        base=SimpleNamespace(
            binding=dict(case_code=reference["case_code"]),
            runtime=SimpleNamespace(read_observation_bundle=lambda _: artifacts),
        ),
    )
    # Existing fixture's descriptor predates complete/clinical fields. Keep it
    # as history, then test this adapter's newly created descriptor and recovery.
    (out / "export-reference.json").rename(out / "original-reference.json")
    monkeypatch.setattr(method, "finish_method_export", generic_run.export)
    first = method.persist_method_export("export-test", campaign, result)
    assert first == method.persist_method_export("export-test", campaign, result)
    assert first["exports"]["api_requests"] == first["exports"]["model_starts"] == 0
    assert (out / "original-reference.json").read_bytes() == existing


@pytest.mark.parametrize("valid_second", [False, True])
def test_real_provider_one_bounded_correction_not_until_expected_score(tmp_path, valid_second):
    from packages.llm_provider.provider import ProviderError
    from tests.agent_provider.batch_support import batch_reply, fixture, generate
    from tests.evaluation_preparation.test_generic_review_shape import malformed
    from tests.expert_agent.batch_case_support import initial_state

    path = tmp_path / "selected.json"
    path.write_text(method.configuration(old_config(), method.PROFILE).model_dump_json())
    f = fixture(
        3,
        replies=[malformed, batch_reply if valid_second else malformed, batch_reply],
        config_path=str(path),
    )
    f.owner.state = initial_state(f.binding, f.config, 3)[0]
    if valid_second:
        sub = generate(f)
        f.acceptance.submit(sub, key="new-method-synthetic")
        assert f.acceptance.accept(key="new-method-synthetic").status == "accepted"
        assert generate(f) == sub
    else:
        with pytest.raises(ProviderError, match="schema"):
            generate(f)
    assert f.budget.used == 2
    assert len(f.provider.transport.calls) == 2
