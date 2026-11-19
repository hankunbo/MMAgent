"""Original owners join aliases and retained batches; no new medical judgment."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.generic_aliases import alias_review
from packages.evidence_details.accepted_aggregation import AcceptedReviewSet
from packages.grounding.models import digest
from tests.agent_provider import test_accepted_review_reuse as reuse
from tests.expert_agent.test_reused_origin_aggregation import fixture as retained


def prepared(monkeypatch):
    original = reuse.fixture

    def aliased():
        f = original()
        f.owner.state, f.alias_manifest = alias_review(f.owner.current())
        return f

    monkeypatch.setattr(reuse, "fixture", aliased)
    f, current, readers, sub = retained()
    mapping = f.old.alias_manifest
    for reader in readers:
        reader.alias_manifests = lambda: (mapping, mapping)
        reader.retained_identity_aliases = True
    return f, current, readers, sub


def test_original_owner_retained_alias_union_preserves_old_values(monkeypatch):
    f, current, readers, sub = prepared(monkeypatch)
    before = digest((f.old.documents.saved, f.documents.saved, f.budget.journal.read()))
    combined, requests = AcceptedReviewSet(readers, f.old.config).current(current)
    expected = {i.assessment_id: i for i in sub.historical.submission.candidate.items}
    expected.update({i.assessment_id: i for i in sub.candidate.items})
    assert {i.assessment_id: i for i in combined.candidate.items} == expected
    assert readers[1].read(current)[0].item_requests.keys() == {"item:additional"}
    assert sum(len(o.item_requests) for o in combined.origins) == len(expected)
    assert len({r.request_id for r in requests}) == len(requests)
    assert before == digest((f.old.documents.saved, f.documents.saved, f.budget.journal.read()))
    assert combined.schema_version == "research-batched-submission/17.0.0"
    assert all(t.before == t.after for o in combined.origins for t in o.alias_projection.transfers)


@pytest.mark.parametrize(
    "fault",
    ["legacy", "missing_owner", "source_changed", "duplicate", "omitted", "manifest_changed"],
)
def test_retained_alias_still_requires_current_ancestor_and_complete_union(monkeypatch, fault):
    f, current, readers, _ = prepared(monkeypatch)
    if fault == "legacy":
        readers[1].retained_identity_aliases = False
    elif fault == "missing_owner":
        readers[1].confirmed_review_reader = None
    elif fault == "source_changed":
        f.old.owner.state = f.old.owner.state.model_copy(
            update={"config_checksum": digest("changed")}
        )
    elif fault == "duplicate":
        readers.append(readers[1])
    elif fault == "omitted":
        readers = readers[1:]
    else:
        old = f.old.alias_manifest
        changed = deepcopy(old)
        changed["checksum"] = digest("changed")
        readers[1].alias_manifests = lambda: (old, changed)
    before = digest((f.old.documents.saved, f.documents.saved, f.budget.journal.read()))
    with pytest.raises(ValueError):
        AcceptedReviewSet(readers, f.old.config).current(current)
    assert before == digest((f.old.documents.saved, f.documents.saved, f.budget.journal.read()))


def test_new_plan_enables_both_existing_owner_paths_without_changing_old_plans(
    monkeypatch, tmp_path
):
    import json

    from packages.multimodal_research import accepted_aggregation as a

    raw = dict(facts=str(tmp_path / "facts"), origins=[dict(campaign=str(tmp_path / "origin"))])
    plan = a.read_plan(dict(raw, schema_version="accepted-review-aggregation-plan/1.9.0"))
    assert isinstance(plan, a.AliasAggregationPlan)
    assert isinstance(plan, a.RetainedOriginAggregationPlan)
    for version, other in (
        ("1.5.0", a.AliasAggregationPlan),
        ("1.6.0", a.RetainedOriginAggregationPlan),
    ):
        old = a.read_plan(dict(raw, schema_version="accepted-review-aggregation-plan/" + version))
        assert not isinstance(old, other)
        assert a.read_plan(old.model_dump(mode="json")) == old
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.model_dump(mode="json")))
    monkeypatch.setattr(a, "plan_campaign", lambda _: (SimpleNamespace(config={}), {}, None))
    calls = []

    def reader(entry, **kwargs):
        calls.append(kwargs)
        raise ValueError("end_original_owner_routing_probe")

    monkeypatch.setattr(a, "origin_reader", reader)
    with pytest.raises(ValueError, match="^end_original_owner_routing_probe$"):
        a.aggregate_entry(path, tmp_path / "out")
    assert calls[0]["reused_origin_new_items_only"] is True
    assert calls[0]["retained_identity_aliases"] is True
    assert callable(calls[0]["current_alias_reader"])
    assert not (tmp_path / "out").exists()
