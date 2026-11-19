"""No inference: shared reader shell, exact scores, safe generic conversion."""

import copy
import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from apps.evidence_view import main
from packages.evaluation_preparation.evidence_tree import TEMPLATES, VERSION, project, render
from packages.evaluation_preparation.evidence_view import convert
from tests.evaluation_preparation.test_unified_view import report


def test_shared_original_shell_and_exact_score_projection():
    original = report()
    before = copy.deepcopy(original)
    page = render(original, input_checksum="sha256:source")
    soup = BeautifulSoup(page, "html.parser")
    shell = BeautifulSoup((TEMPLATES / "readable_evidence_graph.html").read_text(), "html.parser")
    assert soup.style.string == shell.style.string
    for identifier in ("tree", "detail", "overview", "criteria", "search", "filter"):
        assert soup.find(id=identifier).name == shell.find(id=identifier).name
    data = json.loads(soup.find(id="evidence-view").string)
    assert original == before
    assert data["scores"] == original["scores"]
    assert data["reader"]["version"] == VERSION
    assert data["counts"]["nonzero_contributors"] == 2
    assert len(soup.find_all("script")) == 2
    assert "__SNAPSHOT__" not in page and "__TITLE__" not in page
    assert "connect-src 'none'" in page
    assert "innerHTML" not in page and "fetch(" not in page
    assert "<script>bad()" not in page


def test_identity_markup_cannot_escape_json_or_title():
    value = report()
    injected = '</script><img src=x onerror="alert(1)">\u2028'
    value["result"]["sample"] = injected
    value["requirements"][0]["examination"] = injected
    page = render(value, input_checksum="sha256:test")
    soup = BeautifulSoup(page, "html.parser")
    assert not soup.find_all("img")
    assert soup.title.string == injected + " · 病例证据图"
    assert json.loads(soup.find(id="evidence-view").string)["sample"] == injected


def test_absent_reference_or_observation_is_not_filled_from_another_case():
    data = project(report(), input_checksum="sha256:test")
    missing = next(n for n in data["nodes"] if n["id"] == "check:missing")
    assert missing["payload"]["definition"] == {}
    assert "missing" not in data["reader"]["tasks"]
    assert not any(n["id"] == "check:missing:review" for n in data["nodes"])
    assert next(n for n in data["nodes"] if n["id"] == "check:missing:quality")["payload"] == {
        "id": "missing",
        "availability_score": 0,
    }


def test_cli_defaults_to_tree_and_cards_remain_reproducible(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "input.json", tmp_path / "tree.html"
    source.write_text(json.dumps(report()))
    monkeypatch.setattr(
        "sys.argv", ["evidence_view", "--input", str(source), "--output", str(output)]
    )
    main()
    assert json.loads(capsys.readouterr().out)["layout"] == "tree-v3"
    assert 'id="evidence-view"' in output.read_text()
    assert convert(source, output, layout="tree-v3")["new_requests"] == 0
    assert convert(source, tmp_path / "v2.html", layout="tree-v2")["new_requests"] == 0
    assert convert(source, tmp_path / "v1.html", layout="tree-v1")["new_requests"] == 0
    with pytest.raises(ValueError, match="exists"):
        convert(source, output, layout="cards-v1")
    with pytest.raises(ValueError, match="unsupported_layout"):
        convert(source, tmp_path / "wrong.html", layout="unversioned")


@pytest.mark.parametrize("version", ["1.19.0", "2.1.0"])
def test_historical_adapter_keeps_original_score_and_structure(version):
    old = dict(
        schema="evidence-graph-html-snapshot/" + version,
        current_scoring=report()["scores"],
        nodes=[dict(key="historic", title="Historical branch")],
    )
    data = project(old, input_checksum="sha256:old")
    assert data["scores"] == old["current_scoring"]
    assert (
        next(n for n in data["nodes"] if n["id"] == "historical-structure")["payload"]
        == old["nodes"]
    )


def test_current_saved_case_remains_a_24_requirement_16_review_report():
    path = Path("docs/exports/FairVLMed_fv-unified-live-r2-20261005.json")
    # This is read-only presentation evidence, not a fresh medical or PG test.
    data = json.loads(path.read_bytes())
    view = project(data, input_checksum="sha256:fixture")
    assert view["counts"]["requirements"] == 24
    assert view["counts"]["reviewed"] == 16
    assert view["confidence"] == data["scores"]["root"]["summary"]["confidence"]["value"]
    assert len(view["reader"]["interpretations"]) == 16
    assert view["reader"]["tasks"]["criterion:AGE"]["method"] == "dataset_public_age"
