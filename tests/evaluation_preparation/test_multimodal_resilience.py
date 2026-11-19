import json
from types import SimpleNamespace

import pytest

from packages.grounding.models import digest
from packages.multimodal_research.campaign import MultimodalCampaign
from packages.multimodal_research.continuation import load
from packages.multimodal_research.entry import registration_key


def test_circuit_stops_images_only_when_opted_in(tmp_path):
    c = object.__new__(MultimodalCampaign)
    c.base = SimpleNamespace(resilient_vision=True)
    c.output, c.targets = tmp_path, ("a", "b", "c")
    assert not c.acquisition_exhausted()
    for target in c.targets[:2]:
        path = c.phase_path(target)
        path.mkdir()
        (path / "result.json").write_text(json.dumps(dict(accepted=False, error="timeout")))
    assert c.acquisition_exhausted()
    c.base.resilient_vision = False
    assert not c.acquisition_exhausted()


def test_continuation_requires_matching_incomplete_parent_and_single_generation(tmp_path):
    spec = dict(
        input_checksum=digest("input"), intent=dict(task="same"), independent_propositions=True
    )
    result = dict(complete=False, trees=[], new_requests=3)
    (tmp_path / "entry.json").write_text(json.dumps(spec))
    (tmp_path / "suite-result.json").write_text(json.dumps(result))
    (tmp_path / "suite-checksum.json").write_text(json.dumps(dict(checksum=digest(result))))
    assert load(tmp_path, checksum=spec["input_checksum"], intent=spec["intent"]) == (spec, result)
    new = dict(spec, continuation_of=str(tmp_path))
    assert registration_key(tmp_path / "new", new) == registration_key(tmp_path, spec)
    with pytest.raises(ValueError):
        load(tmp_path, checksum=digest("different"), intent=spec["intent"])
    (tmp_path / "entry.json").write_text(json.dumps(new))
    with pytest.raises(ValueError):
        load(tmp_path, checksum=spec["input_checksum"], intent=spec["intent"])


def test_remaining_review_load_scope_deadline_and_single_generation(tmp_path):
    from packages.multimodal_research.remaining_review import load as read

    spec = dict(
        input_checksum=digest("input"), intent=dict(task="same"), independent_propositions=True
    )
    result = dict(
        complete=False, error="deadline", trees=[dict(complete=True)], shared_acquisition=[]
    )
    for name, value in (
        ("entry.json", spec),
        ("suite-result.json", result),
        ("suite-checksum.json", dict(checksum=digest(result))),
    ):
        (tmp_path / name).write_text(json.dumps(value))
    assert read(tmp_path, input_checksum=spec["input_checksum"], intent=spec["intent"]) == (
        spec,
        result,
    )
    with pytest.raises(ValueError):
        read(tmp_path, input_checksum=digest("other"), intent=spec["intent"])
    for change in (
        dict(complete=True),
        dict(error="timeout"),
        dict(trees=[]),
        dict(shared_acquisition=[dict(accepted=False)]),
    ):
        bad = dict(result, **change)
        (tmp_path / "suite-result.json").write_text(json.dumps(bad))
        (tmp_path / "suite-checksum.json").write_text(json.dumps(dict(checksum=digest(bad))))
        with pytest.raises(ValueError):
            read(tmp_path, input_checksum=spec["input_checksum"], intent=spec["intent"])
    (tmp_path / "suite-result.json").write_text(json.dumps(result))
    (tmp_path / "suite-checksum.json").write_text(json.dumps(dict(checksum=digest(result))))
    (tmp_path / "entry.json").write_text(json.dumps(dict(spec, review_from="prior")))
    with pytest.raises(ValueError):
        read(tmp_path, input_checksum=spec["input_checksum"], intent=spec["intent"])


def test_remaining_registration_keeps_original_case_through_acquisition(tmp_path):
    original, acquired = tmp_path / "original", tmp_path / "acquired"
    original.mkdir()
    acquired.mkdir()
    spec = dict(input_checksum=digest("same"), intent=dict(task="same"))
    (original / "entry.json").write_text(json.dumps(spec))
    (acquired / "entry.json").write_text(json.dumps(dict(spec, continuation_of=str(original))))
    new = dict(spec, review_from=str(acquired))
    assert registration_key(tmp_path / "new", new) == registration_key(original, spec)


def test_definition_failure_successor_keeps_original_patient_registration(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    spec = dict(input_checksum=digest("same"), intent=dict(task="same"))
    (original / "entry.json").write_text(json.dumps(spec))
    reference = dict(parent=str(original), entry=digest(spec))
    successor = dict(spec, definition_failure_of=reference)
    assert registration_key(tmp_path / "successor", successor) == registration_key(original, spec)
    # A later Review must traverse the definition successor as well.
    middle = tmp_path / "successor"
    middle.mkdir()
    (middle / "entry.json").write_text(json.dumps(successor))
    assert registration_key(tmp_path / "review", dict(spec, review_from=str(middle))) == (
        registration_key(original, spec)
    )


@pytest.mark.parametrize("change", ["entry", "input", "intent", "ambiguous"])
def test_definition_successor_registration_rejects_changed_parent(tmp_path, change):
    original = tmp_path / "original"
    original.mkdir()
    spec = dict(input_checksum=digest("same"), intent=dict(task="same"))
    (original / "entry.json").write_text(json.dumps(spec))
    reference = dict(parent=str(original), entry=digest(spec))
    successor = dict(spec, definition_failure_of=reference)
    if change == "entry":
        reference["entry"] = digest("changed")
    elif change == "input":
        successor["input_checksum"] = digest("different")
    elif change == "intent":
        successor["intent"] = dict(task="different")
    else:
        successor["previous"] = str(tmp_path / "unrelated")
    with pytest.raises(ValueError, match="multimodal_previous_"):
        registration_key(tmp_path / "successor", successor)


def test_remaining_report_preserves_exact_json_roundtrip():
    from packages.multimodal_research.remaining_review import RemainingReview

    pool = object.__new__(RemainingReview)
    pool.path, pool.suite, pool.previous = "parent", {"complete": False}, []
    pool.labels = ("non_glaucoma", "early_glaucoma")
    pool.values_checksum = digest({})
    pool.campaign = SimpleNamespace(
        propositions={p: None for p in (*pool.labels, "intermediate_glaucoma", "advanced_glaucoma")}
    )
    report = pool.report()
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize(
    "resilient,completed,expected", [(False, 0, 2700), (True, 0, 5400), (True, 2, 3600)]
)
def test_case_deadline_scales_frozen_propositions_not_requests(
    tmp_path, resilient, completed, expected
):
    from datetime import datetime

    c = object.__new__(MultimodalCampaign)
    c.output, c.campaign_id, c.targets, c.config = tmp_path, "test", (), {}
    task = SimpleNamespace(checksum=digest("task"), model_dump=lambda **_: {})
    c.base = SimpleNamespace(
        resilient_vision=resilient,
        manifest=SimpleNamespace(checksum=digest("input")),
        visible_value=SimpleNamespace(checksum=digest("visible")),
        task=task,
        view_checksum=digest("views"),
        sources=SimpleNamespace(initial=[]),
    )
    c.propositions = {str(i): task for i in range(4)}
    c.previous_attempts = lambda: []
    if completed:
        c.remaining_review = SimpleNamespace(labels=tuple(range(completed)), report=lambda: {})
    frozen = c.freeze()
    assert frozen["seconds"] == expected and frozen["max_new_requests"] == 40
    assert (
        datetime.fromisoformat(frozen["deadline"]) - datetime.fromisoformat(frozen["started_at"])
    ).total_seconds() == expected
    assert c.freeze() == frozen
