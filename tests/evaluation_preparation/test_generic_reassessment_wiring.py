"""Verify actual CLI, cold recovery and ancestral budget accounting."""

import json
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import generic_input
from packages.evaluation_preparation import generic_pending_review as pending
from packages.evaluation_preparation import generic_reassessment as candidate
from packages.evaluation_preparation import generic_review_followup as followup
from packages.grounding.models import digest


def test_real_cli_shape_dispatches_exact_requests(monkeypatch):
    from packages.evaluation_preparation import generic_review as module

    calls = []
    monkeypatch.setattr(candidate, "run", lambda *a, **kw: calls.append((a, kw)) or {})
    assert (
        module.main(
            [
                "--key",
                "root",
                "--review-key",
                "new",
                "--accepted-review-key",
                "accepted",
                "--acknowledge-reassessment-request",
                "request:a",
                "--acknowledge-reassessment-request",
                "request:b",
                "--preflight",
            ]
        )
        == 0
    )
    assert calls == [
        (("root", "new", "accepted", ["request:a", "request:b"]), dict(preflight=True))
    ]


@pytest.mark.parametrize(
    "extra",
    [
        dict(prior_review_key="old"),
        dict(network_route="route"),
        dict(review_profile="old"),
        dict(method_origin="old"),
        dict(review_method_profile="old"),
        dict(acknowledge_unknown_request="request:old"),
        dict(acknowledge_truncated_request="request:old"),
    ],
)
def test_new_path_cannot_mix_old_permissions(monkeypatch, extra):
    from packages.evaluation_preparation import generic_review as module

    with pytest.raises(ValueError, match="generic_reassessment_explicit_scope"):
        module.run_review(
            "root",
            "new",
            accepted_review_key="accepted",
            acknowledge_reassessment_request=["request:a"],
            **extra,
        )


def test_legacy_followup_dispatch_unchanged(monkeypatch):
    from packages.evaluation_preparation import generic_review as module

    calls = []
    monkeypatch.setattr(followup, "run", lambda *a, **kw: calls.append((a, kw)) or {})
    module.run_review("root", "new", prior_review_key="old", network_route="route", preflight=True)
    assert calls == [(("root", "new", "old", "route"), dict(preflight=True))]


@pytest.mark.parametrize("version", ["1.5.0", "1.6.0"])
def test_saved_version_dispatches_exact_restore_owner(tmp_path, monkeypatch, version):
    from packages.evaluation_preparation import generic_pending_review as module

    (tmp_path / "input.json").write_text(
        json.dumps(dict(schema_version="generic-review-selection/" + version))
    )
    monkeypatch.setattr(generic_input, "directory", lambda _: tmp_path)
    calls = []
    monkeypatch.setattr(
        candidate if version == "1.6.0" else followup,
        "restore",
        lambda path: calls.append(path) or "original-owner",
    )
    assert module.restore_campaign(tmp_path) == "original-owner"
    assert calls == [tmp_path]


@pytest.mark.parametrize("kind", ["legacy", "followup", "reassessment"])
def test_aggregation_uses_verified_full_ancestry_not_only_root(tmp_path, monkeypatch, kind):
    from packages.multimodal_research import accepted_aggregation as module

    calls = []
    cls = candidate.ReassessmentReview if kind == "reassessment" else followup.FollowupReview
    c = SimpleNamespace() if kind == "legacy" else object.__new__(cls)
    old = dict(
        phases=[dict(run_id="current", target=None, accepted=True, acceptance_key="k")],
        new_requests=2,
    )
    prior = dict(phases=[dict(run_id="root")], new_requests=7)
    (tmp_path / "root/evidence").mkdir(parents=True)
    (tmp_path / "root/evidence/result.json").write_text(json.dumps(prior))
    c.acquisition = SimpleNamespace(
        output=tmp_path / "root/evidence",
        documents=lambda _: SimpleNamespace(read=lambda _: (prior,)),
    )
    c.documents = lambda run: SimpleNamespace(read=lambda _: ("state",), run_id=run)
    c.phase_class = lambda *a, **kw: SimpleNamespace()
    c.base = None
    c.config = None
    c.accepted_observations = lambda: None
    c.read_model = lambda: None
    monkeypatch.setattr(pending, "restore_campaign", lambda _: (c, old))
    monkeypatch.setattr(module, "checked_path", lambda _: tmp_path / "current")
    monkeypatch.setattr(module, "validate", lambda *a: "state")
    monkeypatch.setattr(module, "RuntimeAgentJournal", lambda _: None)
    monkeypatch.setattr(module, "HistoricalRequests", lambda _: None)
    monkeypatch.setattr(module, "AcceptedOriginReader", lambda **kw: SimpleNamespace())
    previous = [
        dict(directory="root", checksum=digest(prior), requests=7),
        dict(directory="failed", checksum="failed", requests=1),
    ]
    if kind == "reassessment":
        previous.append(dict(directory="accepted", checksum="accepted", requests=3))
    monkeypatch.setattr(
        candidate, "aggregation_history", lambda owner: calls.append(owner) or previous
    )
    reader = module.origin_reader(module.OriginPath(campaign="current"), generic=True)
    expected = 9 if kind == "legacy" else 10 if kind == "followup" else 13
    assert sum(x["requests"] for x in reader.accounting) == expected
    assert calls == ([] if kind == "legacy" else [c])
