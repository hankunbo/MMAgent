"""Explicit bounded slices of owner-eligible work, not manual medical definitions."""

import pytest

from packages.multimodal_research.active_continuation import select_acquisition_targets


def test_subset_preserves_schedule_order_and_pending_inventory():
    plan = select_acquisition_targets(["a", "b", "c", "d"], ["c", "a"])
    assert plan["selected"] == ["a", "c"]
    assert plan["deferred"] == ["b", "d"]
    assert plan["eligible"] == ["a", "b", "c", "d"]
    modern = select_acquisition_targets(["a", "b", "c", "d"], ["c", "a"], owner_methods=True)
    assert modern["schema_version"] == "bounded-acquisition-selection/1.1.0"
    assert modern["eligibility_owner"] == "MultimodalCampaign.available_targets"
    assert modern["selected"] == plan["selected"]


@pytest.mark.parametrize("selected", [[], ["a", "a"], ["unknown"], ["a", "b", "c", "d"]])
def test_empty_duplicate_ineligible_and_unbounded_scope_rejected(selected):
    with pytest.raises(ValueError, match="acquisition_selection"):
        select_acquisition_targets(["a", "b", "c", "d"], selected)


def test_no_implicit_attempt_or_input_authorization():
    # Eligible list must already exclude missing input, attempted and UNKNOWN targets.
    for selected in (["old_unknown"], ["missing_input"], ["already_confirmed"]):
        with pytest.raises(ValueError, match="acquisition_selection"):
            select_acquisition_targets(["fresh_check"], selected)


def test_current_inventory_does_not_lose_work_after_text_only_or_many_successors():
    from types import SimpleNamespace

    from packages.multimodal_research.campaign_renewal import unattempted_targets

    rows = [
        dict(id=name, component="image", acquisition_method="vision")
        for name in ("done", "unknown", "next", "later")
    ]
    rows += [
        dict(id="missing", component=None, acquisition_method="vision"),
        dict(
            id="blocked", component="image", acquisition_method="vision", acquisition_allowed=False
        ),
        dict(id="numeric", component="image", acquisition_method="source_measurement"),
        dict(id="unprojected", component="other", acquisition_method="vision"),
    ]
    parent = SimpleNamespace(
        initial_targets=(),
        base=SimpleNamespace(
            ledger=rows,
            manifest=SimpleNamespace(
                components=[
                    SimpleNamespace(id="image", kind="image"),
                    SimpleNamespace(id="other", kind="image"),
                ]
            ),
            views=dict(
                views=[
                    dict(component="image", status="obtained"),
                    dict(component="other", status="missing"),
                ]
            ),
        ),
    )
    assert unattempted_targets(
        parent, {"unknown"}, {"done"}, full_inventory=True, owner_methods=True
    ) == [
        "next",
        "later",
    ]
    assert unattempted_targets(parent, {"unknown"}, {"done"}, full_inventory=True) == [
        "next",
        "later",
        "unprojected",
    ]  # Old frozen v1.0 interpretation remains byte-compatible.
    assert unattempted_targets(parent, {"unknown"}, {"done"}) == []  # Legacy dispatch unchanged.


@pytest.mark.parametrize("owner_methods", [False, True])
def test_reconstruction_rejects_changed_history_or_newly_ineligible_selection(
    tmp_path, owner_methods
):
    import json
    from types import SimpleNamespace

    from packages.grounding.models import digest
    from packages.multimodal_research.campaign_renewal import validate_acquisition_selection

    original = dict(phases=[dict(target="unknown")])
    protocol = dict(version="original")
    (tmp_path / "result.json").write_text(json.dumps(original))
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    parent = SimpleNamespace(
        base=SimpleNamespace(
            ledger=[dict(id="next", component="image", acquisition_method="vision")],
            manifest=SimpleNamespace(components=[SimpleNamespace(id="image", kind="image")]),
            views=dict(views=[dict(component="image", status="obtained")]),
        ),
        accepted_observations=lambda: {},
    )
    plan = dict(
        schema_version="multimodal-campaign-renewal/1.26.0",
        mode="acquisition",
        acquisition_targets=["next"],
        acquisition_selection=select_acquisition_targets(
            ["next"], ["next"], owner_methods=owner_methods
        ),
        previous_attempts=[dict(directory=str(tmp_path), checksum=digest(original))],
        tip=dict(path=str(tmp_path), result=digest(original), protocol=digest(protocol)),
    )
    validate_acquisition_selection(parent, plan)
    parent.base.ledger[0]["component"] = None
    with pytest.raises(ValueError, match="acquisition_selection"):
        validate_acquisition_selection(parent, plan)
    parent.base.ledger[0]["component"] = "image"
    (tmp_path / "result.json").write_text(json.dumps(dict(phases=[])))
    with pytest.raises(ValueError, match="renewal_history_changed"):
        validate_acquisition_selection(parent, plan)


def test_selection_on_review_is_rejected_before_loading_owners(monkeypatch):
    from packages.multimodal_research import campaign_renewal as renewal

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid scope reached owner assembly")

    monkeypatch.setattr(renewal, "completed_parent", forbidden)
    with pytest.raises(ValueError, match="renewal_acquisition_selection_mode"):
        renewal.prepare(
            facts="unused",
            previous="unused",
            output="unused",
            mode="review",
            authorization="synthetic",
            acquisition_target=["next"],
        )


@pytest.mark.parametrize(
    "kind",
    [
        "confirmed_source_schema_failure",
        "terminal_source_unknown_deferred",
        "confirmed_source_schema_deferred",
    ],
)
def test_source_failure_history_is_not_a_missing_case_result(tmp_path, kind):
    import json
    from types import SimpleNamespace

    from db.runtime.repository.case_pipeline import identity
    from packages.grounding.models import digest
    from packages.multimodal_research.campaign_renewal import attempted_acquisition_targets

    protocol = dict(targets=["fresh", "later"], input="input", task="task")
    failure = dict(
        campaign_id=tmp_path.name,
        stage="definitions",
        protocol_checksum=digest(protocol),
        input_checksum="input",
        run_id=identity(
            "RUN", "task-research:mm-live:" + digest((tmp_path.name, "definition-expansion"))[7:55]
        ),
        new_requests=1,
        failure_kind="dispatched_unknown"
        if kind == "terminal_source_unknown_deferred"
        else "confirmed_response_or_domain_refusal",
    )
    attempt = dict(
        directory=str(tmp_path),
        requests=1,
        checksum=digest(failure),
        kind=kind,
    )
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "stage-failure.json").write_text(json.dumps(failure))
    reads = []

    def context(run_id):
        reads.append(run_id)
        raise ValueError("research_context_required")

    parent = SimpleNamespace(
        base=SimpleNamespace(
            manifest=SimpleNamespace(checksum="input"), task=SimpleNamespace(checksum="task")
        ),
        documents=lambda run_id: SimpleNamespace(read=lambda key: (failure,)),
        contexts=SimpleNamespace(read_research_run=context),
    )
    assert attempted_acquisition_targets(parent, [attempt]) == set()
    assert len(reads) == 3  # Both acquisition targets and Review must be absent.
    parent.contexts.read_research_run = lambda run_id: object()
    with pytest.raises(ValueError, match="renewal_source_history_acquisition_started"):
        attempted_acquisition_targets(parent, [attempt])
    (tmp_path / "stage-failure.json").write_text(json.dumps(dict(failure, new_requests=2)))
    with pytest.raises(ValueError, match="renewal_acquisition_history_changed"):
        attempted_acquisition_targets(parent, [attempt])
