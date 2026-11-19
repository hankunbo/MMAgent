"""Selection contracts only; actual source/receipt qualification is a later gate."""

from types import SimpleNamespace

import pytest

from packages.evidence_details import accepted_aggregation as a
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_scoring.review_protocol import ReasonedReviewItem
from packages.grounding.models import digest, validate
from packages.multimodal_research.review_context import supplementary
from tests.agent_provider.batch_support import generate
from tests.evaluation_preparation.test_directional_review import explanation, fixture


def sample(value=0.03):
    f = fixture()
    candidate = generate(f).candidate
    item = next(i for i in candidate.items if i.status == "assessed")
    target = item.assessment_id
    old_raw = item.model_dump(mode="json")
    old_raw.update(status="method_mismatch", direction_assessment=explanation(None))
    old_raw["judgment"].update(
        value=None,
        lower=None,
        upper=None,
        method="not_assessed",
        missing_reason="Old method context unavailable",
    )
    old_item = validate(ReasonedReviewItem, old_raw)
    new_raw = item.model_dump(mode="json")
    new_raw.update(
        status="assessed" if value is not None else "method_mismatch",
        direction_assessment=explanation(value),
    )
    new_raw["judgment"].update(
        value=value,
        lower=-0.1 if value is not None else None,
        upper=0.1 if value is not None else None,
        method="manual_source_bound_review" if value is not None else "not_assessed",
        missing_reason=None if value is not None else "Still not interpretable after new context",
    )
    new_item = validate(ReasonedReviewItem, new_raw)
    old_state = f.owner.current()
    extra = supplementary(
        {
            target: dict(
                unit_basis=None,
                source_limits=["Synthetic new method metadata"],
                transform={"kind": "unchanged"},
                numeric={"value": 87},
                interpretation_limits={"not_clinical": True},
                owner_method={"version": "synthetic/2"},
            )
        }
    )
    raw = old_state.model_dump(mode="json")
    raw["material"]["observations"].extend(o.model_dump(mode="json") for o in extra)
    raw["material"]["scope"]["run_id"] = "RUN:NULL_REVISION"
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    new_state = validate(DomainState, raw)
    old_items = tuple(old_item if i.assessment_id == target else i for i in candidate.items)

    def origin(state, items, prefix):
        return a.AcceptedOrigin(
            run_id=state.material.scope.run_id,
            acceptance_key=prefix,
            acceptance_checksum=digest(prefix + "acceptance"),
            submission_checksum=digest(prefix + "submission"),
            state_checksum=digest(state),
            control_checksum=digest(prefix + "control"),
            journal_checksum=digest(prefix + "journal"),
            permission_checksum=digest(prefix + "grant"),
            item_requests={i.assessment_id: prefix + ":" + str(n) for n, i in enumerate(items)},
            unused_request_ids=(),
        )

    old_origin = origin(old_state, old_items, "old")
    new_origin = origin(new_state, (new_item,), "new")
    revision = a.NullReviewRevision(
        assessment_id=target,
        previous_run_id=old_origin.run_id,
        replacement_run_id=new_origin.run_id,
        previous_submission_checksum=old_origin.submission_checksum,
        replacement_submission_checksum=new_origin.submission_checksum,
        previous_item_checksum=digest(old_item),
        replacement_item_checksum=digest(new_item),
        new_context_checksum=digest(a._target_context(new_state, target)),
    )
    return SimpleNamespace(
        values=[(old_origin, old_items, ()), (new_origin, (new_item,), ())],
        states={old_origin.run_id: old_state, new_origin.run_id: new_state},
        revision=revision,
        target=target,
    )


@pytest.mark.parametrize("value", [0.03, 0.0, None])
def test_new_review_selects_one_version_without_forcing_direction(value):
    f = sample(value)
    old = repr(f.values)
    result = a.select_null_revisions(f.values, f.states, (f.revision,))
    assert repr(f.values) == old
    items = [i for _, rows, _ in result for i in rows]
    assert len(items) == len(f.values[0][1])
    assert len({i.assessment_id for i in items}) == len(items)
    assert next(i for i in items if i.assessment_id == f.target).judgment.value == value
    assert all(i in items for i in f.values[0][1] if i.assessment_id != f.target)
    assert result[0][0].submission_checksum == f.values[0][0].submission_checksum
    assert f.values[0][0].item_requests[f.target] in result[0][0].unused_request_ids


@pytest.mark.parametrize(
    "field",
    [
        "previous_submission_checksum",
        "replacement_submission_checksum",
        "previous_item_checksum",
        "replacement_item_checksum",
        "new_context_checksum",
    ],
)
def test_any_changed_binding_is_rejected(field):
    f = sample()
    change = f.revision.model_copy(update={field: digest("wrong")})
    with pytest.raises(ValueError, match="null_revision"):
        a.select_null_revisions(f.values, f.states, (change,))


@pytest.mark.parametrize("value", [0.0, 0.03, -0.02])
def test_old_numeric_or_neutral_judgment_never_superseded(value):
    f = sample(value)
    new = f.values[1][1][0]
    oldproof, rows, records = f.values[0]
    f.values[0] = (
        oldproof,
        tuple(new if i.assessment_id == f.target else i for i in rows),
        records,
    )
    revision = f.revision.model_copy(update={"previous_item_checksum": digest(new)})
    with pytest.raises(ValueError, match="null_revision_accepted_item_binding"):
        a.select_null_revisions(f.values, f.states, (revision,))


def test_duplicate_or_same_origin_and_state_drift_rejected():
    f = sample()
    with pytest.raises(ValueError, match="null_revision_duplicate_selection"):
        a.select_null_revisions(f.values, f.states, (f.revision, f.revision))
    with pytest.raises(ValueError, match="null_revision_requires_distinct"):
        a.select_null_revisions(
            f.values,
            f.states,
            (f.revision.model_copy(update={"replacement_run_id": f.revision.previous_run_id}),),
        )
    f.states[f.revision.replacement_run_id] = f.states[f.revision.previous_run_id]
    with pytest.raises(ValueError, match="null_revision_original_state_binding"):
        a.select_null_revisions(f.values, f.states, (f.revision,))
