import copy
import json
import re
import subprocess

import pytest

from apps import glaucoma_model_selection as selection
from apps.evidence_graph_html import render_html, validate_snapshot
from apps.glaucoma_criteria_v1_1 import apply, build, policy
from apps.glaucoma_evidence_plan import checksum
from apps.glaucoma_research_criteria import apply as legacy_apply
from tests.test_glaucoma_research_criteria import example, leaf


def fixture():
    """Synthetic records of the fixed reviewed shape, not a production approval."""
    v = example()
    c = v["case_evidence"]
    c["input"]["checksum"] = selection.INPUT
    c["artifacts"] = [
        {
            "asset_code": "ASSET:OID_SYNTHETIC_" + str(i),
            "artifact_type_code": kind,
            "checksum": digest,
            "scope": {"case_code": c["case_code"]},
        }
        for i, (kind, digest) in enumerate(selection.MASKS.items())
    ]
    p = leaf(v)
    p["tool"] = {"code": "segman_slo_disc_cup_segmenter", "version": "1.0.0"}
    p["artifact_codes"] = [a["asset_code"] for a in c["artifacts"][1:]]
    p["results"].append({"name": "acdr", "value": 0.25, "unit": "1"})
    s = c["supplement"]
    s.update(
        input_checksum=selection.INPUT,
        artifact_codes=p["artifact_codes"],
        registered_tool=p["tool"],
    )
    s["mask_check"].update(
        pixel_counts={"cup": 25, "disc": 100},
        metrics={"vcdr": {"value": 0.542}, "acdr": {"value": 0.25}},
    )
    c["results"]["avod"]["execution"].update(
        input_checksum=selection.INPUT,
        call={"tool": "sloctolyzer_avod_segmenter", "version": "1.0.0"},
    )
    c["results"]["avod"]["observation"]["model"]["foreground_pixels"]["disc_mask"] = 90
    return v


def test_model_choice_is_case_scoped_and_not_accuracy_vote():
    v = fixture()
    before = copy.deepcopy(v)
    out = selection.apply(v)
    assert v == before
    assert selection.apply(out) == out
    assert out["model_selection"]["accuracy_winner"] is None
    assert out["model_selection"]["primary"]["name"] == "SegMAN-B"
    nodes = {n["key"]: n for n in out["nodes"]}
    assert nodes["DISC_MASK"]["patient_evidence"]["results"][0]["value"] == 100
    assert nodes["DISC_GEOMETRY"]["patient_evidence"]["alternatives"]
    # Task-specific primary choice does not swap the vessel/AV producer.
    assert nodes["TASK_AVOD"] == next(n for n in before["nodes"] if n["key"] == "TASK_AVOD")
    validate_snapshot(apply(out))


@pytest.mark.parametrize("change", ["input", "mask", "scope", "producer", "projection"])
def test_wrong_scope_masks_or_projection_rejected(change):
    v = selection.apply(fixture())
    if change == "input":
        v["case_evidence"]["input"]["checksum"] = "wrong"
    elif change == "mask":
        v["case_evidence"]["artifacts"][0]["checksum"] = "wrong"
    elif change == "scope":
        v["case_evidence"]["artifacts"][0]["scope"]["case_code"] = "CASE:ANON_OTHER"
    elif change == "producer":
        v["case_evidence"]["supplement"]["registered_tool"] = {"code": "wrong"}
    else:
        next(n for n in v["nodes"] if n["key"] == "DISC_MASK")["patient_evidence"]["results"][0][
            "value"
        ] = 90
    with pytest.raises(ValueError, match="model_selection_"):
        selection.validate(v)


def test_geometry_not_paper_cutoff_and_old_replay_unchanged():
    v = fixture()
    old = legacy_apply(v)
    old_bytes = json.dumps(old, sort_keys=True)
    new = apply(v)
    r = new["research_criteria"]["results"][1]
    assert r["equivalent_diameter_ratio"] == 0.5
    assert r["comparison"] == "below_geometric_reference"
    assert r["counted_independently"] is False
    rule = policy()["criteria"][1]
    assert rule["threshold"] == 0.49
    assert rule["derivation"]["published_area_cutoff"] is False
    validate_snapshot(new)
    validate_snapshot(old)
    assert json.dumps(old, sort_keys=True) == old_bytes
    assert old["research_criteria"]["results"][1]["comparison"] == "reference_only"


@pytest.mark.parametrize(
    "x,expected", [(0.49, "above_geometric_reference"), (0.48, "below_geometric_reference")]
)
def test_area_reference_boundary_not_vertical_threshold(x, expected):
    v = fixture()
    leaf(v)["results"][1]["value"] = x
    s = v["case_evidence"]["supplement"]["mask_check"]
    s["metrics"]["acdr"]["value"] = x
    s["pixel_counts"] = {"cup": round(x * 100), "disc": 100}
    out = build(v)
    assert out["results"][1]["comparison"] == expected
    assert out["results"][0]["comparison"] == "not_met"


def test_forged_area_denominator_and_unknown_version_rejected():
    v = fixture()
    v["case_evidence"]["supplement"]["mask_check"]["pixel_counts"]["disc"] = 90
    with pytest.raises(ValueError, match="criterion_area_binding"):
        build(v)
    v = apply(fixture())
    v["research_criteria"]["policy"]["version"] = "caller-policy"
    v["checksum"] = checksum({k: x for k, x in v.items() if k != "checksum"})
    with pytest.raises(ValueError, match="criterion_version_unknown"):
        validate_snapshot(v)


@pytest.mark.parametrize("inspection", [False, True])
def test_concrete_morphology_and_distinct_navigation_actions(inspection):
    if inspection:
        from apps.glaucoma_criteria_v1_2 import apply as new_apply
        from tests.test_glaucoma_inspection import fixture as new_fixture

        page = render_html(new_apply(new_fixture()))
    else:
        page = render_html(apply(selection.apply(fixture())))
    scripts = re.findall(r"<script([^>]*)>(.*?)</script>", page, re.S)
    values = {
        re.search(r'id="([^"]+)"', a)[1]: c for a, c in scripts if 'type="application/json"' in a
    }
    script = next(c for a, c in scripts if not a)
    harness = r"""
const vm=require('vm'),input=JSON.parse(require('fs').readFileSync(0,'utf8')),elements={};
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.textContent='';this.style={};this.value=''}
 set id(v){elements[v]=this} append(...xs){this.children.push(...xs)}
 replaceChildren(...xs){this.children=xs} setAttribute(){} querySelectorAll(){return []}
 scrollIntoView(){this.scrolled=true}
}
for(const [k,v] of Object.entries(input.values)){
 elements[k]=new Element();elements[k].textContent=v
}
const document={getElementById:k=>elements[k]??(elements[k]=new Element()),
createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t)};
const c=vm.createContext({document,URL,console});vm.runInContext(input.script,c);
function all(e){return [e,...e.children.flatMap(all)]}
vm.runInContext("show('CUPDISC')",c);
let button=all(elements.detail).find(x=>x.tag==='button'&&
 (x.textContent.startsWith('前往测量')||x.textContent.startsWith('查看结果')));
button.onclick();if(vm.runInContext('active',c)!=='CUP_RIM_MEASURE')throw Error('wrong target');
if(all(elements.detail).some(x=>x.tag==='button'&&x.textContent.startsWith('回看本页')))
 throw Error('removed self-navigation button returned');
vm.runInContext("show('DISC_BLEED')",c);
const texts=all(elements.detail).map(x=>x.textContent).join(' ');
if(!texts.includes('火焰')||!texts.includes('灰度SLO'))throw Error('missing morphology');
for(const row of all(elements.detail).filter(x=>x.className==='source-row')){
 if(row.children.some(x=>x.tag==='button'))throw Error('source and navigation mixed');
}
vm.runInContext('for(const n of D.nodes)show(n.key);overview();',c);
console.log('navigation and all nodes passed');
"""
    r = subprocess.run(
        ["/opt/data/test/envs/MMAgent1/bin/node", "-e", harness],
        input=json.dumps({"values": values, "script": script}),
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    assert "navigation and all nodes passed" in r.stdout
