"""Six-section UI layout; stored observations, thresholds and scope stay intact."""

import copy

from apps.evidence_graph_explanation import explain
from apps.evidence_graph_presentation import PROFILES, build
from apps.glaucoma_tool_evidence import upgrade
from tests.test_glaucoma_structure_ui import browser
from tests.test_glaucoma_tool_evidence import fixture


def test_presentation_is_complete_and_does_not_change_any_fact(monkeypatch):
    _, measured = fixture(monkeypatch)
    value = upgrade(measured)
    before = copy.deepcopy(value)
    presentation = build(value, explain(value))
    assert set(PROFILES) == {n["key"] for n in value["nodes"]}
    assert value == before
    for item in presentation["nodes"].values():
        assert all(item[k] for k in ["rationale", "modality", "data_required", "method"])
        assert "核对以下来源定义的阈值或现象" not in item["data_required"]
    branch = presentation["nodes"]["VENULAR_BRANCH_ANGLE"]
    assert "静脉掩膜与视盘掩膜" in branch["modality"]
    assert "90.00°" in branch["association"]["observed"]
    assert "缺少同模态、同算法" in branch["association"]["assessment"]
    assert "OCT" in presentation["nodes"]["RNFL_THICKNESS"]["modality"]
    assert not presentation["nodes"]["FOVEA_LOCATION"]["saved_tools"]


def test_all_nodes_have_six_sections_and_associations_share_reading_format(monkeypatch):
    _, measured = fixture(monkeypatch)
    value = upgrade(measured)
    browser(
        value,
        r"""
const titles=['1 · 来源事实','2 · 我们的提炼理由','3 · 本节点的取证需求',
 '4 · 本例工具与观察结果','5 · 目前如何判读','6 · 计算依赖与判读背景'];
function section(prefix){const a=elements.detail.children;
 const i=a.findIndex(e=>e.tag==='h3'&&e.textContent===prefix);check(i>=0,'heading missing');
 const rest=a.slice(i+1),j=rest.findIndex(e=>e.tag==='h3');
 return rest.slice(0,j<0?rest.length:j).map(text).join(' ')}
for(const key of run('D.nodes.map(n=>n.key)')){
 run('show('+JSON.stringify(key)+')');
 const hs=elements.detail.children.filter(e=>e.tag==='h3'&&/^\d ·/.test(e.textContent))
  .map(e=>e.textContent);
 check(JSON.stringify(hs)===JSON.stringify(titles),'six-section mismatch '+key);
 const third=section(titles[2]);
 for(const label of ['数据模态：','所需数据：','取证方法：'])
  check(third.includes(label),'input fields '+key);
 check(!third.includes('核对以下来源定义的阈值或现象'),'thresholds in inputs '+key);
 check(!third.includes('≥0.7')&&!third.includes('<0.1'),'disease criteria leaked to third '+key);
 check(section(titles[4]).includes('检查参照：'),'reference missing '+key);
 check(section(titles[4]).includes('本例结果：'),'result missing '+key);
 check(section(titles[4]).includes('判读：'),'assessment missing '+key);
}
run("show('VENULAR_BRANCH_ANGLE')");
check(section(titles[4]).includes('90.00°'),'actual result lost');
check(section(titles[4]).includes('缺少同模态、同算法'),'reference gap hidden');
check(!text(elements.detail).includes('5 · 本项关联如何使用'),'old alternate heading remains');
run("show('CUP_RIM_MEASURE')");
check(section(titles[2]).includes('模型选择'),'selection absent');
check(section(titles[4]).includes('垂直杯盘比'),'condition moved away from judgement');
run("show('RNFL_THICKNESS')");
check(section(titles[2]).includes('OCT'),'modality missing');
check(section(titles[4]).includes('缺少'),'input absence hidden');
run("show('DISC_MASK')");
check(!all(elements.detail).some(e=>e.className==='card criterion-card'),'mask given disease vote');
""",
    )
