"""Bounded drafting without clinical acceptance, network or invented thresholds."""

import copy
from types import SimpleNamespace

import pytest

from packages.evidence_reuse.definitions import prepare_definition
from packages.grounding.models import digest


def fixture():
    source = dict(
        code="SOURCE:A",
        checksum=digest("text"),
        text="A qualitative relationship.",
        purpose="research_material",
    )
    draft = dict(
        schema_version="source-definition-draft/1.0.0",
        target="check:a",
        relation="qualitative",
        examination="Inspect the visible region.",
        interpretation="A qualitative association, no population threshold.",
        applicability="Synthetic source; not a patient diagnosis.",
        input_concepts=["SLO"],
        method=None,
        unit=None,
        excerpts=[
            dict(
                source_code=source["code"],
                source_checksum=source["checksum"],
                start=0,
                end=len(source["text"]),
                text=source["text"],
            )
        ],
    )
    return source, draft


def test_existing_source_first_and_one_draft_no_approval():
    source, draft = fixture()
    owner = SimpleNamespace(read=lambda _: copy.deepcopy(source))
    calls = []
    result = prepare_definition(
        target="check:a",
        existing_codes=[source["code"]],
        source_owner=owner,
        proposer=lambda **_: draft,
        search=lambda **_: pytest.fail("existing source"),
        before_request=lambda: calls.append(1),
    )
    assert result["status"] == "draft_pending_owner_acceptance"
    assert result["queries"] == 0 and result["proposals"] == len(calls) == 1
    assert not result["auto_installed"]


@pytest.mark.parametrize("change", ["text", "checksum", "target", "autoapprove"])
def test_wrong_draft_never_admitted(change):
    source, draft = fixture()
    if change == "text":
        draft["excerpts"][0]["text"] = "invented"
    elif change == "checksum":
        draft["excerpts"][0]["source_checksum"] = digest("other")
    elif change == "target":
        draft["target"] = "check:other"
    else:
        draft["automatically_accepted"] = True
    with pytest.raises(ValueError):
        prepare_definition(
            target="check:a",
            existing_codes=["SOURCE:A"],
            source_owner=SimpleNamespace(read=lambda _: source),
            proposer=lambda **_: draft,
            before_request=lambda: None,
        )


def test_search_no_progress_stops_and_no_proposal_without_sources():
    calls = []
    result = prepare_definition(
        target="check:a",
        existing_codes=[],
        source_owner=None,
        proposer=lambda **_: pytest.fail("no evidence"),
        search=lambda **_: [],
        before_request=lambda: calls.append(1),
    )
    assert result["queries"] == len(calls) == 1 and result["proposals"] == 0


def test_cancellation_before_proposal_and_return_time_source_change():
    source, draft = fixture()

    def cancelled():
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        prepare_definition(
            target="check:a",
            existing_codes=["SOURCE:A"],
            source_owner=SimpleNamespace(read=lambda _: source),
            proposer=lambda **_: pytest.fail("must not dispatch"),
            before_request=cancelled,
        )
    count = []

    def read(_):
        count.append(1)
        return source if len(count) < 3 else dict(source, purpose="withdrawn")

    with pytest.raises(ValueError, match="source_changed"):
        prepare_definition(
            target="check:a",
            existing_codes=["SOURCE:A"],
            source_owner=SimpleNamespace(read=read),
            proposer=lambda **_: draft,
            before_request=lambda: None,
        )
