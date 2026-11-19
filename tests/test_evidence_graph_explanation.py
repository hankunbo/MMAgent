import copy
import json
import re
import shutil
import subprocess

import pytest

from apps.evidence_graph_explanation import explain
from apps.evidence_graph_html import render_html, write_export
from tests.test_evidence_graph_html import rehash, snapshot


def test_relevant_bound_passages_only_and_no_heading_or_intro():
    data = snapshot()
    data["nodes"] = [
        {
            "key": "STRUCTURE",
            "parent": None,
            "title": "视神经结构",
            "knowledge": [{"chunk_ref": {"checksum": str(i)}} for i in range(4)],
        }
    ]
    data["source_chunks"] = [
        {"ref": {"checksum": str(i)}, "text": text, "position": {"kind": kind}}
        for i, (kind, text) in enumerate(
            [
                ("heading", "1.2.1"),
                ("paragraph", "Offer all tests:"),
                ("list", "optic nerve assessment"),
                ("list", "visual field assessment"),
            ]
        )
    ]
    before = copy.deepcopy(data)
    e = explain(data)["nodes"]["STRUCTURE"]
    assert [s["text"] for s in e["sources"]] == ["optic nerve assessment"]
    assert e["sources"][0]["reference"] == data["nodes"][0]["knowledge"][2]
    assert data == before


def test_historical_receipt_selects_exact_paragraph_not_other_text():
    data = snapshot()
    data["nodes"][0]["knowledge"] = [{"page_id": "nei", "paragraph_checksums": ["bound"]}]
    data["historical_pages"] = [
        {
            "id": "nei",
            "url": "https://www.nei.nih.gov/",
            "paragraphs": [
                {"text_checksum": "bound", "text": "Bound fact"},
                {"text_checksum": "other", "text": "Other fact"},
            ],
        }
    ]
    assert [s["text"] for s in explain(data)["nodes"]["ROOT"]["sources"]] == ["Bound fact"]


def test_readable_page_keeps_raw_data_inert_and_preserves_old_exports(tmp_path):
    data = snapshot()
    data["nodes"][0]["title"] = "Example"
    data["nodes"][0]["patient_evidence"] = {"summary": "</script><script>bad()</script>"}
    data = rehash(data)
    before = copy.deepcopy(data)
    output = render_html(data)
    assert "来源事实 → 提炼理由 → 取证需求 → 工具结果 → 判读" in output
    # Saved-record foldouts may format JSON as inert text. Test the actual
    # injection boundary, rather than banning a JavaScript function name.
    scripts = re.findall(r"<script([^>]*)>(.*?)</script>", output, re.S)
    payload = next(body for attrs, body in scripts if 'id="snapshot"' in attrs)
    assert json.loads(payload) == data
    assert all("bad()" not in body for attrs, body in scripts if "application/json" not in attrs)
    assert "<pre" not in output and "下载完整JSON" not in output
    assert "来源位置和版本" not in output
    assert "</script><script>bad()" not in output
    assert "innerHTML" not in output
    assert data == before
    path = tmp_path / "graph.html"
    write_export(data, path)
    old = path.read_bytes()
    write_export(data, path, replace=True)
    assert any(p.read_bytes() == old for p in (tmp_path / "history").glob("*.html"))
    assert json.loads(path.with_suffix(".json").read_text()) == data


@pytest.mark.parametrize("with_criteria", [False, True])
def test_all_node_clicks_with_javascript_dom_harness(with_criteria):
    """Execute UI logic, not just substring checks; no browser dependencies needed."""
    from tests.test_glaucoma_case_view import data, project

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js unavailable for DOM harness")
    source, report = data()
    value = project(source, report)
    if with_criteria:
        from apps.glaucoma_research_criteria import apply
        from tests.test_glaucoma_research_criteria import example

        value = apply(example())
    page = render_html(value)
    scripts = re.findall(r"<script([^>]*)>(.*?)</script>", page, re.S)
    values = {
        re.search(r'id="([^"]+)"', attrs)[1]: content
        for attrs, content in scripts
        if 'type="application/json"' in attrs
    }
    script = next(content for attrs, content in scripts if not attrs)
    harness = r"""
const fs=require('fs'),vm=require('vm'),input=JSON.parse(fs.readFileSync(0,'utf8'));
const created=[];
class Element {
 constructor(tag){this.tag=tag;created.push(this);this.children=[];this.textContent='';this.value='';this.style={}}
 append(...xs){this.children.push(...xs)}
 replaceChildren(...xs){this.children=xs}
 setAttribute(){}
 querySelectorAll(){return []}
}
const elements={};
for(const [key,text] of Object.entries(input.values)){
 elements[key]=new Element();elements[key].textContent=text;
}
const document={getElementById:k=>elements[k]??(elements[k]=new Element()),
 createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag)};
const context=vm.createContext({document,URL,console});
vm.runInContext(input.script,context);
vm.runInContext('for(const n of D.nodes) show(n.key); overview();',context);
if(!elements.detail.children.length) throw Error('empty detail');
if(created.some(e=>e.tag==='pre')) throw Error('raw record displayed in default view');
if(vm.runInContext('JSON.stringify(D)',context)!==JSON.stringify(JSON.parse(input.values.snapshot)))
 throw Error('UI changed the saved snapshot');
console.log('all node clicks passed');
"""
    result = subprocess.run(
        [node, "-e", harness],
        input=json.dumps({"values": values, "script": script}),
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    assert "all node clicks passed" in result.stdout
