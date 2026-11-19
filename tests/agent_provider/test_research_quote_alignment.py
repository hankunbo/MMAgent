"""Saved real failure and synthetic negative controls; no network or database."""

import json
from pathlib import Path

import pytest

from packages.knowledge_claims.research_quote import align_research_quote


def test_real_200_response_has_only_layout_difference_and_is_not_rewritten():
    package = Path("docs/acceptance/12-completion-cde-target-live-20261001")
    request = json.loads((package / "actual-requests.json").read_text())[0]
    original_response = request["raw_response"][0]["raw_model_text"]
    candidate = json.loads(original_response)
    documents = json.loads((package / "original-case-documents.json").read_text())
    intent = next(
        v for k, v in documents.items() if k.startswith("review:") and k.endswith(":intent")
    )
    material = intent[0]["state"]["material"]
    source = next(
        s for s in material["sources"] if s["reference"]["code"] == candidate["source_code"]
    )
    assert align_research_quote(candidate["quote"], source["text"]) is None
    proof = align_research_quote(candidate["quote"], source["text"], allow_layout=True)
    assert proof["mode"] == "layout_whitespace"
    assert source["text"][proof["start"] : proof["end"]] == proof["original_text"]
    assert proof["original_text"].replace("\n", "") == candidate["quote"]
    assert request["raw_response"][0]["raw_model_text"] == original_response


@pytest.mark.parametrize(
    "quote",
    [
        "This is glaucoma.",
        "This is not; glaucoma.",
        "This IS not glaucoma.",
        "Threshold >= 0.8.",
        "Threshold > 0.7.",
        "Threshold >= 0.7 mm.",
        "A B D",
        "A\u200bB C",
        "Because",
        "",
        " \n ",
    ],
)
def test_layout_alignment_does_not_change_medical_characters_or_join_words(quote):
    source = "This is not\nglaucoma. Threshold >=\n0.7. A\nB\nC. Becau\nse"
    assert align_research_quote(quote, source, allow_layout=True) is None


def test_ambiguous_layout_match_rejected_but_literal_compatibility_preserved():
    source = "A\nB; A\tB"
    assert align_research_quote("A B", source, allow_layout=True) is None
    literal = align_research_quote("A\nB", source)
    assert literal["mode"] == "literal" and literal["original_text"] == "A\nB"


def test_unique_layout_span_retains_unicode_offsets_and_all_original_whitespace():
    source = "前言：measure\t \n0.7\u00a0mm；结尾"
    proof = align_research_quote("measure 0.7 mm", source, allow_layout=True)
    assert proof["start"] == 3
    assert proof["original_text"] == "measure\t \n0.7\u00a0mm"
    assert source[proof["start"] : proof["end"]] == proof["original_text"]


def test_contiguous_excerpt_need_not_include_terminal_punctuation():
    proof = align_research_quote(
        "This is not glaucoma", "This is not\nglaucoma.", allow_layout=True
    )
    assert proof["original_text"] == "This is not\nglaucoma"
