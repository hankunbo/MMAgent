"""Reusable reader layout: synthetic UI fixtures, no new clinical judgments."""

import copy
import json
import re

from apps.evidence_graph_html import render_html
from apps.evidence_graph_scores import attach
from tests.test_evidence_scoring_current import current, graded, interpreted, parent  # noqa: F401
from tests.test_glaucoma_structure_ui import browser


def test_overview_numbering_and_optional_experiments(current):  # noqa: F811
    base, score = current
    browser(
        attach(base, score),
        r"""
const headers=()=>elements.detail.children.filter(x=>x.tag==='h3').map(x=>x.textContent);
check(JSON.stringify(headers())===JSON.stringify([
 '1 · 病例与目标','2 · 结论与关键分数','3 · 证据进度与待补项',
 '4 · 证据移除与关键依赖','5 · 方法、溯源与附录']), 'overview structure');
check(all(elements.detail).filter(x=>x.className==='metric-grid').length===1,
 'duplicate root score dashboard');
const optional=elements.detail.children.find(x=>x.tag==='details'&&
 x.children[0]?.textContent==='可选附加实验 · 已保存的工具替换');
check(optional&&!optional.open,'replacement must be a closed optional appendix');
check(text(optional).includes('不作为本例完成条件'),'replacement scope');
check(text(elements.detail).includes('待补检查与匹配参照'),'missing evidence hidden');
check(text(elements.detail).includes('统一评分口径与阅读说明'),'central definitions');
check(!elements.detail.children.some(x=>x.textContent==='本轮完成与仍缺的资料'),
 'old timeline duplicated');
check(all(elements.detail).filter(x=>x.tag==='a'&&x.href?.startsWith('#case-')).length===5,
 'section navigation');
check(elements.detail.attrs['data-layout-version']==='case-evidence-reader/2.0.0',
 'presentation version absent');
""",
    )


def test_all_nodes_single_owner_reading_and_six_sections(current):  # noqa: F811
    base, score = current
    browser(
        attach(base, score),
        r"""
const titles=['来源事实','我们的提炼理由','本节点的取证需求',
 '本例工具与观察结果','目前如何判读','计算依赖与判读背景'];
const initial=run('JSON.stringify(D)');
for(const n of run('D.nodes')){
 run('show('+JSON.stringify(n.key)+')');
 const headings=elements.detail.children.filter(x=>x.tag==='h3').map(x=>x.textContent);
 check(JSON.stringify(headings)===JSON.stringify(titles.map((t,i)=>(i+1)+' · '+t)),
  'node section numbering '+n.key);
 const cards=all(elements.detail).filter(x=>x.attrs?.['data-assessment-id']);
 const expected=run('ownPairs('+JSON.stringify(n.key)+').map(x=>x.j.assessment_id)');
 check(JSON.stringify(cards.map(x=>x.attrs['data-assessment-id']))===JSON.stringify(expected),
  'record not displayed once at owning node '+n.key);
 check(!text(elements.detail).includes('关键证据集与移除分析'),
  'full overview repeated inside a node');
}
run("show('CUP_RIM_MEASURE')");
check(fifth().includes('检查参照')&&fifth().includes('0.7'),'criterion source threshold lost');
check(fifth().includes('可靠性')&&fifth().includes('关系强度')&&fifth().includes('适用性'),
 'ratings not adjacent to interpretation');
run("show('VASCULAR_SKELETON_DENSITY')");
check(fifth().includes('待补信息'),'reference gap lost');
run('overview()');check(run('JSON.stringify(D)')===initial,'UI mutated evidence or scores');
""",
    )


def test_case_target_and_branch_count_are_data_driven(current):  # noqa: F811
    base, score = current
    browser(
        attach(base, score),
        r"""
// UI-only fixture changes; not new authorized medical content or a schema upgrade.
run("C.sample='synthetic-case-B';byKey.get('ROOT').title='测试疾病乙评估';"+
 "S.main_edges=S.main_edges.filter(e=>e.from!=='ROOT'||['STRUCTURE','CONTEXT'].includes(e.to));overview()");
const h2=elements.detail.children.find(x=>x.tag==='h2');
check(h2.textContent==='测试疾病乙评估','hardcoded disease title');
check(text(elements.detail).includes('synthetic-case-B'),'hardcoded case identifier');
const branches=elements.detail.children.find(x=>x.className==='branch-grid');
check(branches.children.length===2,'hardcoded four glaucoma branches');
run('D.current_scoring.root.summary.confidence.value=null;'+
 'D.current_scoring.root.coverage.slo_image_research.acquisition=null;overview()');
const tiles=elements.detail.children.find(x=>x.className==='metric-grid');
check(text(tiles.children[0]).includes('未评分'),'missing score became zero or 50');
check(!text(elements.detail).includes('NaN')&&!text(elements.detail).includes('undefined'),
 'invalid display fallback');
""",
    )


def test_render_preserves_exact_snapshot_and_offline_contract(current):  # noqa: F811
    base, score = current
    value = attach(base, score)
    before = copy.deepcopy(value)
    page = render_html(value)
    embedded = re.search(
        r'<script id="snapshot" type="application/json">(.*?)</script>', page, re.S
    )[1]
    assert json.loads(embedded) == before == value
    assert "__LAYOUT__" not in page
    assert "innerHTML" not in page
    assert "connect-src 'none'" in page
    assert 'src="https://' not in page
    assert "@media(max-width:850px)" in page
    assert "prefers-reduced-motion" in page
