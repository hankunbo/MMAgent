import copy
import json

import pytest

from apps.evidence_graph_html import extended_records, render_html, validate_snapshot
from apps.glaucoma_evidence_bindings import dump, revised_records
from apps.glaucoma_evidence_plan import checksum
from tests.database.test_glaucoma_bindings import fixture


def snapshot():
    value = {
        "title": "Synthetic graph",
        "nodes": [{"key": "ROOT", "parent": None}],
        "relationships": [],
    }
    return {**value, "checksum": checksum(value)}


def rehash(value):
    value["checksum"] = checksum({k: v for k, v in value.items() if k != "checksum"})
    return value


def test_html_escapes_remote_script_and_has_no_external_dependencies():
    data = snapshot()
    data["title"] = '</script><script>alert("remote")</script>'
    output = render_html(rehash(data))
    assert '<script>alert("remote")</script>' not in output
    assert "\\u003c/script\\u003e" in output
    assert "connect-src 'none'" in output
    assert 'src="https://' not in output
    assert "innerHTML" not in output
    raw = output.split('<script id="snapshot" type="application/json">')[1].split("</script>")[0]
    assert json.loads(raw) == data


@pytest.mark.parametrize("mutation", ["duplicate", "cycle", "missing", "edge", "checksum"])
def test_invalid_export_topology_or_checksum_rejected(mutation):
    data = copy.deepcopy(snapshot())
    if mutation == "duplicate":
        data["nodes"].append(data["nodes"][0].copy())
    elif mutation == "cycle":
        data["nodes"].extend([{"key": "A", "parent": "B"}, {"key": "B", "parent": "A"}])
    elif mutation == "missing":
        data["nodes"].append({"key": "A", "parent": "missing"})
    elif mutation == "edge":
        data["relationships"].append({"parent": "ROOT", "child": "missing"})
    if mutation != "checksum":
        rehash(data)
    else:
        data["title"] = "tampered"
    with pytest.raises(ValueError):
        validate_snapshot(data)


def test_egs_bindings_append_versions_without_inventing_medical_edges():
    source, plan, binding = fixture()
    previous = [dump(r) for r in revised_records(plan, binding)]
    pages = [
        {
            "binding": {"nodes": ["HEMORRHAGE"], "terms": ["synthetic"]},
            "source_ref": dump(source["source_version"].reference()),
            "chunks": [
                {"text": "Synthetic fixture " * 10, "ref": dump(source["chunk"].reference())}
            ],
        }
    ]
    records = extended_records(previous, pages)
    assert len(records) == 163
    assert all(r.version == "1.4.0" and r.supersedes.version == "1.3.0" for r in records)
    node = next(r for r in records if r.kind.value == "node" and r.code.endswith(":HEMORRHAGE"))
    assert source["chunk"].reference() in node.binding_refs
    assert node.template_ref.version == node.concept_ref.version == "1.4.0"
    assert len([r for r in records if r.kind.value == "edge"]) == 25
    assert all(
        r.acceptance_rule_ref is None for r in records if r.kind.value == "evidence_requirement"
    )
    repaired = extended_records([dump(r) for r in records], pages, version="1.5.0")
    assert all(r.version == "1.5.0" and r.supersedes.version == "1.4.0" for r in repaired)
    repaired_node = next(r for r in repaired if r.kind.value == "node" and r.code == node.code)
    assert repaired_node.template_ref.version == repaired_node.concept_ref.version == "1.5.0"
    assert repaired_node.binding_refs == node.binding_refs
