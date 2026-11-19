"""The visible medical tree follows criteria, while inputs remain navigable."""

import json
import re
import subprocess

from apps.evidence_graph_html import render_html
from apps.glaucoma_criteria_v1_3 import apply
from apps.glaucoma_evidence_structure import apply as organize
from tests.test_glaucoma_disease_conditions import fixture


def browser(snapshot, assertions):
    page = render_html(snapshot)
    scripts = re.findall(r"<script([^>]*)>(.*?)</script>", page, re.S)
    values = {
        re.search(r'id="([^"]+)"', attrs)[1]: code
        for attrs, code in scripts
        if 'type="application/json"' in attrs
    }
    script = next(code for attrs, code in scripts if not attrs)
    harness = r"""
const vm=require('vm'),input=JSON.parse(require('fs').readFileSync(0,'utf8')),elements={};
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.value='';this.attrs={}}
 set id(v){elements[v]=this} append(...xs){this.children.push(...xs)}
 replaceChildren(...xs){this.children=xs} setAttribute(k,v){this.attrs[k]=v}
 querySelectorAll(){return []}
}
for(const [k,v] of Object.entries(input.values)){
 elements[k]=new Element();elements[k].textContent=v;
}
const document={getElementById:k=>elements[k]??(elements[k]=new Element()),
 createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t)};
const c=vm.createContext({document,URL,console});vm.runInContext(input.script,c);
function all(e){return [e,...e.children.flatMap(all)]}
function text(e){return all(e).map(x=>x.textContent).join(' ')}
function run(js){return vm.runInContext(js,c)}
function check(value,message){if(!value)throw Error(message)}
function treeKeys(){return all(elements.tree).map(x=>x.attrs?.['data-node-key']).filter(Boolean)}
function fifth(){const cs=elements.detail.children;
 const start=cs.findIndex(x=>x.tag==='h3'&&x.textContent.startsWith('5 ·'));
 if(start<0)return '';const rest=cs.slice(start+1),end=rest.findIndex(x=>x.tag==='h3');
 return rest.slice(0,end<0?rest.length:end).map(text).join(' ')}
"""
    result = subprocess.run(
        ["/opt/data/test/envs/MMAgent1/bin/node", "-e", harness + assertions],
        input=json.dumps({"values": values, "script": script}),
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_main_tree_and_all_nonmedical_groups_are_separate(monkeypatch):
    snapshot = organize(apply(fixture(monkeypatch)))
    browser(
        snapshot,
        r"""
const expected=run('S.main_node_keys');
check(treeKeys().length===expected.length,'main count differs');
check(treeKeys().every(k=>expected.includes(k)),'technical node leaked into main tree');
check(!treeKeys().includes('DISC_MASK')&&!treeKeys().includes('DISC_SIZE'),
 'old disc-size branch retained');
check(elements['structure-navigation'].children.length===4,'missing category controls');
const seen=[];
for(const group of ['main','dependency','deferred','auxiliary']){
 run('groupOverview('+JSON.stringify(group)+')');
 const keys=treeKeys();seen.push(...keys);
 check(keys.length===run('S.counts['+JSON.stringify(group)+']'),'group count differs');
 check(keys.every(k=>run('S.node_roles['+JSON.stringify(k)+'].group')===group),'wrong group');
}
check(new Set(seen).size===run('D.nodes.length')&&seen.length===run('D.nodes.length'),
 'results lost or duplicated');
run("groupOverview('dependency')");elements.search.value='视盘掩膜';run('render()');
check(treeKeys().includes('DISC_MASK')&&!treeKeys().includes('DISC_SIZE'),
 'old ancestors returned in dependency search');
elements.search.value='';run("show('CUP_RIM_MEASURE')");
check(treeKeys().every(k=>expected.includes(k)),'navigation back to main failed');
run('overview()');
check(text(elements.summary).includes('医学主树'),'legacy topology count shown');
check(text(elements.detail).includes('计算与解释依赖'),'overview category count missing');
""",
    )


def test_dependencies_explain_their_consumer_without_empty_disease_card(monkeypatch):
    snapshot = organize(apply(fixture(monkeypatch)))
    browser(
        snapshot,
        r"""
run("show('CUP_RIM_MEASURE')");
check(fifth().includes('垂直杯盘比'),'criterion lost');
check(!fifth().includes('杯面积')&&!fifth().includes('血管面积密度'),
 'description became a disease condition');
check(text(elements.detail).includes('6 · 计算依赖与判读背景'),
 'missing calculation dependency section');
check(text(elements.detail).includes('杯盘比=杯垂直径/盘垂直径'),'mask purpose missing');
const buttons=all(elements.detail).filter(x=>x.tag==='button');
check(buttons.some(x=>x.textContent.startsWith('查看依赖：')&&x.textContent.includes('视盘掩膜')),
 'mask navigation missing');
for(const key of ['DISC_MASK','DISC_SIZE','DENSITY','RNFL_THICKNESS','ANGLE']){
 run('show('+JSON.stringify(key)+')');
 check(elements.detail.children.some(x=>x.tag==='h3'&&x.textContent==='5 · 目前如何判读'),
  'unified reading section missing');
 check(!text(elements.detail).includes('无独立疾病判读条件'),'generic defensive fallback returned');
 check(text(elements.detail).includes('2 · 我们的提炼理由'),'nonmedical purpose missing');
 check(!all(elements.detail).some(x=>x.className==='card criterion-card'),
  'supporting input promoted to disease criterion');
}
run("show('DISC_MASK')");
check(text(elements.detail).includes('服务检查：'),'consumer reference missing');
check(!text(elements.detail).includes('取证主题的上下级组织'),'historical parent chain shown');
run("show('DISC_SIZE')");
check(text(elements.detail).includes('物理标尺')&&text(elements.detail).includes('盘大小修饰'),
 'disc-size interpretation lost');
run("show('DENSITY')");
check(text(elements.detail).includes('血管面积密度'),'stored result lost');
run('for(const n of D.nodes)show(n.key)');
check(!text(elements.detail).includes('回看本页'),'self-navigation returned');
""",
    )


def test_historical_export_keeps_original_navigation(monkeypatch):
    browser(
        apply(fixture(monkeypatch)),
        r"""
check(treeKeys().length===run('D.nodes.length'),'historical tree changed');
check(treeKeys().includes('DISC_SIZE')&&treeKeys().includes('DISC_MASK'),'historical nodes lost');
run("show('DISC_MASK')");
check(fifth().includes('无独立疾病判读条件'),'historical display changed');
        """,
    )


def test_new_topology_owns_summary_status_and_condition_counts(monkeypatch):
    browser(
        organize(apply(fixture(monkeypatch))),
        r"""
for(const key of ['STRUCTURE','RNFL','CONTEXT']){
 run('E.nodes['+JSON.stringify(key)+'].result_summary="STALE直接子项";'+
  'byKey.get('+JSON.stringify(key)+').patient_evidence.results='+
  '[{name:"STALE汇总血管",value:1}];'+
  'byKey.get('+JSON.stringify(key)+').patient_evidence.acquisition='+
  '{required_input:"STALE直接子项",label:"已有结果",reason:"STALE聚合"}');
 run('show('+JSON.stringify(key)+')');
 check(!text(elements.detail).includes('STALE'),'historical summary or aggregation leaked');
 const expected=run('S.node_roles['+JSON.stringify(key)+'].result_summary');
 check(text(elements.detail).includes(expected),'new condition summary missing');
}
run("show('CONTEXT')");
check(!text(elements.detail).includes('血管面积密度'),'old auxiliary descendant leaked');
run("byKey.get('DISC_SIZE').patient_evidence.acquisition="+
 '{required_input:"STALE像素尺寸",label:"已有结果",reason:"STALE已取证"}');
run("show('DISC_SIZE')");
const badge=elements.detail.children.find(x=>x.className==='badge');
check(badge.textContent==='当前输入无法取证','physical scale status replaced by pixel result');
check(text(elements.detail).includes('物理')&&!text(elements.detail).includes('STALE'),
 'disc-size physical requirement missing or old acquisition retained');
run('D.observation_summary={"STALE已有结果":999};overview()');
check(!text(elements.detail).includes('STALE'),'old node counts shown');
check(text(elements.detail).includes('共10条判读条件'),'unique condition total missing');
const counts=run('criterionObservationCounts()');
check(Object.values(counts).reduce((a,b)=>a+b,0)===10,'conditions counted by ancestor');
run('detail.replaceChildren();structureSources(detail,'+
 '[{title:"TEST",summary:"SUMMARY_ONLY",url:"https://example.org"}])');
check(text(elements.detail).includes('SUMMARY_ONLY'),'source summary omitted');
""",
    )
