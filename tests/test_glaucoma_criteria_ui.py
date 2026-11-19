"""Presentation contract: disease conditions and descriptive measurements stay separate."""

import json
import re
import subprocess

from apps.evidence_graph_html import render_html
from apps.glaucoma_criteria_v1_2 import apply
from tests.test_glaucoma_inspection import fixture


def test_disease_only_cards_descriptions_and_historical_display_labels():
    # First render a fully validated historical object. Mutate only the embedded
    # browser fixture to test the new rendering contract independently of models.
    page = render_html(apply(fixture()))
    scripts = re.findall(r"<script([^>]*)>(.*?)</script>", page, re.S)
    values = {
        re.search(r'id="([^"]+)"', a)[1]: c for a, c in scripts if 'type="application/json"' in a
    }
    snapshot = json.loads(values["snapshot"])
    criteria = snapshot["research_criteria"]
    criteria["schema"] = "case-research-criteria/1.3.0"
    medical = {"VCDR_GE_07"}
    criteria["descriptive_criteria"] = [
        r for r in criteria["policy"]["criteria"] if r["id"] not in medical
    ]
    criteria["policy"]["criteria"] = [
        r for r in criteria["policy"]["criteria"] if r["id"] in medical
    ]
    criteria["descriptive_results"] = [
        r for r in criteria["results"] if r["criterion_id"] not in medical
    ]
    criteria["results"] = [r for r in criteria["results"] if r["criterion_id"] in medical]
    criteria["descriptive_node_summaries"] = {
        k: {"criterion_ids": [i for i in s["criterion_ids"] if i not in medical]}
        for k, s in criteria["node_summaries"].items()
    }
    criteria["node_summaries"] = {
        k: {"criterion_ids": [i for i in s["criterion_ids"] if i in medical]}
        for k, s in criteria["node_summaries"].items()
    }
    for node in snapshot["nodes"]:
        if node["key"] == "RNFL_REFLECTANCE":
            node["patient_evidence"]["status"] = "已有观察·判读不确定"
        elif node["key"] == "DISC_BLEED":
            node["patient_evidence"]["status"] = "已有观察"
    snapshot["observation_summary"] = {"已有观察·判读不确定": 1, "已有观察": 1, "已有结果": 3}
    values["snapshot"] = json.dumps(snapshot, ensure_ascii=False)
    explanations = json.loads(values["explanations"])
    for key in ("CUP_RIM_MEASURE", "DENSITY"):
        explanations["nodes"][key].update(
            interpretation="STALE_INTERPRETATION_MUST_NOT_RENDER",
            next_action="STALE_ACTION_MUST_NOT_RENDER",
        )
    values["explanations"] = json.dumps(explanations, ensure_ascii=False)
    script = next(c for a, c in scripts if not a)
    assert "回看本页测量结果" not in script
    harness = r"""
const vm=require('vm'),input=JSON.parse(require('fs').readFileSync(0,'utf8')),elements={};
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.style={};this.value=''}
 set id(v){elements[v]=this} append(...xs){this.children.push(...xs)}
 replaceChildren(...xs){this.children=xs} setAttribute(){} querySelectorAll(){return []}
}
for(const [k,v] of Object.entries(input.values)){
 elements[k]=new Element();elements[k].textContent=v
}
const document={getElementById:k=>elements[k]??(elements[k]=new Element()),
createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t)};
const c=vm.createContext({document,URL,console});vm.runInContext(input.script,c);
function all(e){return [e,...e.children.flatMap(all)]}
function text(e){return all(e).map(x=>x.textContent).join(' ')}
function section(prefix){
 const children=elements.detail.children;
 const start=children.findIndex(x=>x.tag==='h3'&&x.textContent.startsWith(prefix));
 const rest=children.slice(start+1),end=rest.findIndex(x=>x.tag==='h3');
 return rest.slice(0,end<0?rest.length:end);
}
function sectionText(prefix){return section(prefix).map(text).join(' ')}
vm.runInContext("show('CUP_RIM_MEASURE')",c);
if(!sectionText('4 ·').includes('杯面积')||!sectionText('4 ·').includes('面积杯盘比'))
 throw Error('descriptive measures lost');
const fifth=sectionText('5 ·');
if(!fifth.includes('垂直杯盘比')||fifth.includes('面积杯盘比')||fifth.includes('杯面积'))
 throw Error('non-disease measurements leaked into interpretation');
if(text(elements.detail).includes('STALE_'))throw Error('old fallback returned');
vm.runInContext("show('DENSITY')",c);
if(!sectionText('4 ·').includes('血管面积密度'))throw Error('measurement explanation missing');
if(!sectionText('5 ·').includes('无独立疾病判读条件')||sectionText('5 ·').includes('血管面积密度'))
 throw Error('empty disease conditions have incorrect fallback');
for(const key of ['RNFL_REFLECTANCE','DISC_BLEED']){
 vm.runInContext('show('+JSON.stringify(key)+')',c);
 const badge=elements.detail.children.find(x=>x.className==='badge');
 if(badge.textContent!=='已有结果')throw Error('old status not translated');
}
vm.runInContext('overview()',c);
if(!text(elements.detail).includes('已有结果：5个节点'))throw Error('status counts not merged');
if(text(elements.detail).includes('已有观察'))throw Error('old status in overview');
vm.runInContext('for(const n of D.nodes)show(n.key);',c);
console.log('disease-only rendering passed');
"""
    result = subprocess.run(
        ["/opt/data/test/envs/MMAgent1/bin/node", "-e", harness],
        input=json.dumps({"values": values, "script": script}),
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    assert "disease-only rendering passed" in result.stdout
