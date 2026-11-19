"""Presentation/inventory regression: no clinical inference or new API execution."""

import copy
import json

import pytest

from packages.evaluation_preparation.evidence_inventory import inventory
from packages.evaluation_preparation.evidence_view import TYPES, convert, normalize, render


def report():
    rows = []
    for identifier, value in (
        ("risk:age", 0.3),
        ("feature:normal", -0.2),
        ("feature:neutral", 0),
        ("feature:unknown", None),
    ):
        rows.append(
            dict(
                id=identifier,
                title=identifier,
                judgment=dict(value=value, reason="<script>bad()</script>"),
                observation=dict(value=0.8),
                relation=dict(value=0.5),
                applicability=dict(value=0.4),
                support=max(value or 0, 0),
                refutation=max(-(value or 0), 0),
                role="risk" if identifier == "risk:age" else "diagnostic",
            )
        )
    scores = dict(
        schema_version="case-research-scoring/8.0.0",
        root=dict(summary=dict(confidence=dict(value=0.55))),
        reviews=rows,
    )
    return dict(
        schema_version="dynamic-case-report/1.0.0",
        scores=scores,
        original_review=dict(
            scores=copy.deepcopy(scores), records=dict(review_tasks=dict(tasks=[]))
        ),
        requirements=[
            dict(id=r["id"], examination=r["title"], sources=[], modality=["SLO"]) for r in rows
        ]
        + [dict(id="missing", examination="Missing", sources=[])],
        process_evaluations=[dict(id="missing", availability_score=0)],
        result=dict(sample="arbitrary"),
    )


def test_view_preserves_scores_and_six_types_without_extra_votes():
    original = report()
    before = copy.deepcopy(original)
    view = normalize(original, input_checksum="sha256:" + "a" * 64)
    assert original == before and view["scores"] == before["scores"]
    assert view["counts"]["nonzero_contributors"] == 2
    assert view["counts"]["numeric"] == 3 and view["counts"]["unassessed"] == 1
    assert len(TYPES) == 6
    page = render(view)
    assert "55.0000 / 100" in page and "<script>bad()" not in page
    assert page.count("<body>") == 1 and page.endswith("</body></html>")


def test_render_old_snapshot_without_replacing_results():
    old = dict(
        schema="evidence-graph-html-snapshot/2.1.0", current_scoring=report()["scores"], nodes=[]
    )
    view = normalize(old, input_checksum="sha256:" + "b" * 64)
    assert view["counts"]["requirements"] is None
    assert view["scores"] is old["current_scoring"]


def test_score_binding_and_duplicate_ids_rejected():
    data = report()
    data["scores"]["reviews"].append(data["scores"]["reviews"][0])
    with pytest.raises(ValueError, match="duplicate"):
        normalize(data, input_checksum="sha256:a")
    data = report()
    data["scores"]["root"] = {}
    with pytest.raises(ValueError, match="score_binding"):
        normalize(data, input_checksum="sha256:a")


def test_conversion_is_readonly_and_refuses_overwrite(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(report()))
    before = source.read_bytes()
    out = tmp_path / "view.html"
    result = convert(source, out)
    assert result == convert(source, out) and result["new_requests"] == 0
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match="overwrite"):
        convert(source, source)
    out.write_text("original other view")
    with pytest.raises(ValueError, match="exists"):
        convert(source, out)


def test_every_candidate_has_disposition_and_no_pending_denominator():
    catalog = dict(
        tasks={
            k: dict(selected_source_ids=["source"] if k == "a" else [], definition=None)
            for k in ("a", "b", "c")
        }
    )
    value = inventory(catalog, ["a"], mappings={"b": dict(target="a", reason="same question")})
    assert value["candidate_count"] == 3 and value["scheduled_count"] == 1
    assert [r["disposition"] for r in value["rows"]] == [
        "scheduled",
        "merged",
        "source_binding_pending",
    ]
    assert sum(r["counts_in_coverage"] for r in value["rows"]) == 1
    with pytest.raises(ValueError, match="merge"):
        inventory(catalog, ["a"], mappings={"b": dict(target="not-selected", reason="bad")})
