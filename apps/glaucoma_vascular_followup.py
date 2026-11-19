"""Append registered A/V measurements to the original case; preserve v4 scoring."""

import argparse
import copy
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from apps import av_vascular_tool as tool
from apps.evidence_graph_details import _read_json, _reference
from apps.glaucoma_evidence_plan import checksum as snapshot_checksum
from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from packages.evidence_details.models import body, digest
from packages.object_store import checksum

VERSION = "case-evidence-graph/1.17.0"
TYPE = "ARTIFACT:SLO_AV_VASCULAR_METRICS"
BASE_VERSION = "case-evidence-graph/1.16.0"
INPUTS = {
    "artery_mask": "ARTIFACT:SLO_ARTERY_MASK",
    "vein_mask": "ARTIFACT:SLO_VEIN_MASK",
    "disc_mask": "ARTIFACT:SLO_SEGMAN_DISC",
}
DIRECTORY = Path(__file__).resolve().parents[1] / "docs/experiments/glaucoma-vascular-followup-v5"
POLICY = {
    "version": "vascular-followup/1.0.0",
    "primary_scope": "slo_image_research",
    "secondary_scope": "comprehensive_research",
    "missing_OCT_VF_IOP": "not_required_for_SLO_research",
    "scoring": "preserve_v4_direction_and_fixed_adequacy; no invented reference thresholds",
    "measurement_scope": "existing_aligned_masks_only_no_new_model",
}


def sources():
    return {
        key: {k: value[k] for k in ["title", "url", "reference", "reading_scope"]}
        for key in ["density", "tortuosity"]
        for value in [json.loads((DIRECTORY / (key + "-archive.json")).read_bytes())]
    }


def comparisons(record):
    a, v = record["classes"]["artery"], record["classes"]["vein"]

    def pair(section, field, unit=""):
        def number(m):
            x = m[section][field]
            return "无有效测量" if x is None else f"{x:.8f}{unit}"

        return "动脉 " + number(a) + "；静脉 " + number(v) + "。"

    return [
        {
            "node": "VASCULAR_SKELETON_DENSITY",
            "title": "动静脉骨架密度",
            "observed": pair("whole_image", "skeleton_density", " px⁻¹"),
            "reference": "Chen等的前瞻性研究报告密度与后续青光眼相关；"
            "方法为骨架长度/图像面积，非血管面积比例。",
            "interpretation": "本例已完成A/V分型骨架测量。论文表2的数值尺度未能"
            "与本工具原生像素单位对齐；不把0.70或0.75作为阈值。",
            "remaining": "同视场、同像素尺度、同分割方法的健康比较分布；文献数值转换说明。",
            "state": "measured_reference_unmatched",
            "source_ids": ["density"],
        },
        {
            "node": "VASCULAR_FRACTAL",
            "title": "动静脉骨架分形维数",
            "observed": pair("skeleton_fractal", "value"),
            "reference": "Chen等使用盒计数；本文实现报告每个网格尺度、占据盒数量、"
            "拟合优度，维数描述所见血管网络复杂度。",
            "interpretation": "本例已取得分型值。论文表2的0.79等数值不能直接套给"
            "本工具盒计数斜率；较低关联不等于任意较小值即可判阳性。",
            "remaining": "文献预处理及转换尺度、同方法同视场参照。",
            "state": "measured_reference_unmatched",
            "source_ids": ["density"],
        },
        {
            "node": "AV_TORTUOSITY",
            "title": "动静脉弧弦迂曲度",
            "observed": pair("tortuosity", "mean")
            + "保留段数：动脉 "
            + str(a["tortuosity"]["count"])
            + "，静脉 "
            + str(v["tortuosity"]["count"])
            + "。",
            "reference": "Amir Hamzah等2026论文区分动静脉，强调视场影响；"
            "表4的HRF健康/青光眼动脉均值接近。",
            "interpretation": "算法分型缺口已关闭。该文BFS像素点数与四舍五入弦长"
            "不等同本工具欧氏弧长；表5出现小于1的比值，因此不能作为本例疾病阈值。",
            "remaining": "同路径定义及同视场参考分布；分段均值不能直接当患者诊断准确率。",
            "state": "measured_reference_unmatched",
            "source_ids": ["tortuosity"],
        },
        {
            "node": "ARTERIOLAR_CALIBER",
            "title": "盘周动静脉相对宽度",
            "observed": pair("annulus", "width_proxy_disc_ratio", " 盘径"),
            "reference": "在距视盘中心1至1.5个等效盘径环带内，分别求血管面积/骨架像素数，"
            "再除以等效盘径。",
            "interpretation": "新增的是盘周相对宽度，不是选取六条独立主干得到的CRAE/CRVE，"
            "也不输出虚构微米值。可用于后续同方法比较。",
            "remaining": "若要CRAE：可靠六主干选择、垂直截面口径、标准汇总与标定；"
            "当前仍为宽度代理量。",
            "state": "proxy_measured",
            "source_ids": [],
        },
    ]


def selected(base):
    refs = {}
    c = base["case_evidence"]
    for port, kind in INPUTS.items():
        matches = [
            Artifact.model_validate(a) for a in c["artifacts"] if a["artifact_type_code"] == kind
        ]
        if len(matches) != 1:
            raise ValueError("vascular_unique_registered_mask_required")
        a = matches[0]
        if (
            a.scope.case_code != c["case_code"]
            or c["input"]["asset_code"] not in a.derived_from
            or list(a.dimensions) != c["input"]["dimensions"]
            or a.laterality != Artifact.model_validate(c["input"]).laterality
        ):
            raise ValueError("vascular_mask_scope_or_alignment")
        refs[port] = a
    return refs


def projection(base, record, artifact, execution):
    tool.validate(record)
    refs = selected(base)
    for port, a in refs.items():
        b = record["input_bindings"][port]
        if b["sha256"] != a.checksum or b["dimensions_hw"] != list(a.dimensions):
            raise ValueError("vascular_measurement_input_mismatch")
    a = Artifact.model_validate(artifact)
    if (
        a.scope.case_code != base["case_evidence"]["case_code"]
        or a.artifact_type_code != TYPE
        or a.checksum != checksum(body(record))
        or set(a.derived_from) != {r.asset_code for r in refs.values()}
        or not a.research_only
    ):
        raise ValueError("vascular_artifact_binding")
    if set(execution) != {
        "tool",
        "version",
        "implementation",
        "installation_checksum",
        "direct_exit_code",
    } or (
        execution["tool"],
        execution["version"],
        execution["implementation"],
        execution["direct_exit_code"],
    ) != (tool.CODE, tool.VERSION, tool.IMPLEMENTATION, 0):
        raise ValueError("vascular_execution_receipt")
    return {
        "schema": "case-vascular-followup/1.0.0",
        "policy": POLICY,
        "policy_checksum": digest(POLICY),
        "record": record,
        "artifact": artifact,
        "execution": execution,
        "sources": sources(),
        "comparisons": comparisons(record),
        "observation_status": "已有结果",
        "score_action": "no_new_disease_direction_assigned",
        "reason": "可计算的四类测量已补齐；来源比较仍不能提供本例可辩护的高低方向，"
        "不用检查缺失推断阳性/阴性。",
    }


def attach(base, record, artifact, execution):
    if "vascular_followup" in base:
        raise ValueError("vascular_recursive_attachment")
    result = copy.deepcopy(base)
    result["vascular_followup"] = projection(base, record, artifact, execution)
    result["checksum"] = snapshot_checksum({k: v for k, v in result.items() if k != "checksum"})
    return result


def base_view(value):
    v = {k: x for k, x in value.items() if k not in {"vascular_followup", "checksum"}}
    return {**v, "checksum": snapshot_checksum(v)}


def validate_view(value):
    base = base_view(value)
    f = value["vascular_followup"]
    if f != projection(base, f["record"], f["artifact"], f["execution"]):
        raise ValueError("vascular_view_reconstruction")
    return base


def read_manifest(document, case_code, current, repo, store, artifact):
    from apps.glaucoma_case_storage import read_view

    if (
        set(document) != {"schema", "case_code", "base", "measurement", "execution"}
        or document["case_code"] != case_code
    ):
        raise ValueError("vascular_manifest_schema_or_scope")
    refs = {}
    for key, kind in [("base", "ARTIFACT:CASE_EVIDENCE_GRAPH"), ("measurement", TYPE)]:
        r = document[key]
        a = current.get(r["code"])
        if (
            a is None
            or a.scope.case_code != case_code
            or a.artifact_type_code != kind
            or a.checksum != r["checksum"]
        ):
            raise ValueError("vascular_manifest_binding")
        refs[key] = a
    if (
        set(artifact.derived_from) != {a.asset_code for a in refs.values()}
        or _read_json(refs["base"], store).get("schema") != BASE_VERSION
    ):
        raise ValueError("vascular_manifest_derivation_or_base")
    base = read_view(case_code, refs["base"].asset_code, repo=repo, store=store)
    return attach(
        base,
        _read_json(refs["measurement"], store),
        refs["measurement"].model_dump(mode="json"),
        document["execution"],
    )


def persist(base_code, record, execution, *, repo, store, case_code):
    """Internal observed-result commit; public run() recomputes registered input bytes."""
    from apps.glaucoma_case_storage import read_view

    current = {a.asset_code: a for a in repo.get_case_bundle(case_code).artifacts}
    original = current[base_code]
    if _read_json(original, store).get("schema") != BASE_VERSION:
        raise ValueError("vascular_base_version")
    base = read_view(case_code, base_code, repo=repo, store=store)
    inputs = selected(base)
    tool.validate(record)
    key = identity(
        "TASK",
        "vascular-followup:" + case_code + base_code + record["checksum"] + digest(execution),
    )
    prior = repo.read_observation_bundle(key)
    if prior:
        if (
            len(prior) not in {1, 2}
            or prior[-1].artifact_type_code != "ARTIFACT:CASE_EVIDENCE_GRAPH"
        ):
            raise ValueError("vascular_request_conflict")
        graph = prior[-1]
        document = _read_json(graph, store)
        measurement = current.get(document["measurement"]["code"])
        if measurement is None or (len(prior) == 2 and prior[0] != measurement):
            raise ValueError("vascular_request_measurement_conflict")
        expected = attach(base, record, measurement.model_dump(mode="json"), execution)
        if read_view(case_code, graph.asset_code, repo=repo, store=store) != expected:
            raise ValueError("vascular_retry_content_conflict")
        return {
            "measurement": measurement.model_dump(mode="json"),
            "graph": graph.model_dump(mode="json"),
            "request": key,
        }
    created = []
    attempted = False

    def artifact(value, kind, parents):
        raw = body(value)
        if len(raw) > 4_194_304:
            raise ValueError("vascular_object_bound")
        sha = checksum(raw)
        code = identity("ASSET", case_code + kind + sha)
        existing = current.get(code)
        if existing:
            if (
                existing.artifact_type_code != kind
                or existing.checksum != sha
                or set(existing.derived_from) != {a.asset_code for a in parents}
                or store.read_bounded(existing.uri, 4_194_304) != raw
            ):
                raise ValueError("vascular_existing_artifact_conflict")
            return existing
        stage = store.stage(raw)
        try:
            uri = store.promote(stage, sha)
        except BaseException:
            store.discard(stage)
            raise
        created.append(uri)
        obj = original.model_dump(mode="json")
        obj.update(
            asset_code=code,
            artifact_type_code=kind,
            checksum=sha,
            uri=uri,
            derived_from=sorted(a.asset_code for a in parents),
            dimensions=[],
            measurements=[],
            research_only=True,
        )
        obj["provenance"].update(
            generator_version="vascular-followup-1.0.0",
            created_at=datetime.now(UTC).isoformat(),
            source_codes=[a.asset_code for a in parents],
            input_checksums=[a.checksum for a in parents],
        )
        return Artifact.model_validate(obj)

    try:
        m = artifact(record, TYPE, list(inputs.values()))
        attach(base, record, m.model_dump(mode="json"), execution)
        document = {
            "schema": VERSION,
            "case_code": case_code,
            "base": _reference(original),
            "measurement": _reference(m),
            "execution": execution,
        }
        g = artifact(document, "ARTIFACT:CASE_EVIDENCE_GRAPH", [original, m])
        attempted = True
        # This port inserts new artifacts. Existing measurements remain exact
        # derivation parents; they must not be reinserted in a later graph bundle.
        repo.register_observation_bundle(
            tuple(a for a in (m, g) if a.asset_code not in current), key
        )
    except ValueError:
        for uri in created:
            store.discard(uri)
        raise
    except BaseException:
        if not attempted:
            for uri in created:
                store.discard(uri)
        raise
    if read_view(case_code, g.asset_code, repo=repo, store=store) != attach(
        base, record, m.model_dump(mode="json"), execution
    ):
        raise ValueError("vascular_readback_mismatch")
    return {
        "measurement": m.model_dump(mode="json"),
        "graph": g.model_dump(mode="json"),
        "request": key,
    }


def run(case_code, base_code, *, repo, store):
    from apps.glaucoma_case_storage import read_view
    from apps.knowledge_source_tools import repository
    from apps.patient_service_assembly import OBJECT_ROOTS
    from apps.vascular_literature_followup import verify
    from packages.object_store import LocalObjectStore

    for name in ["density", "tortuosity"]:
        verify(
            json.loads((DIRECTORY / (name + "-archive.json")).read_bytes()),
            # These are archived research sources, not a Published Release for
            # the clinical Planner. Use the existing curator read capability;
            # verification does not publish content or change database grants.
            repository("editor"),
            LocalObjectStore(OBJECT_ROOTS["knowledge"]),
        )
    base = read_view(case_code, base_code, repo=repo, store=store)
    refs = selected(base)
    tmp = Path(__file__).resolve().parents[1] / ".local"
    with tempfile.TemporaryDirectory(prefix="vascular-owned-", dir=tmp) as d:
        inputs = {}
        for port, a in refs.items():
            raw = store.read_bounded(a.uri, 4_194_304)
            if checksum(raw) != a.checksum:
                raise ValueError("vascular_registered_input_drift")
            inputs[port] = Path(d) / (port + ".png")
            inputs[port].write_bytes(raw)
        output = Path(d) / "metrics.json"
        with tool.native_connection() as c:
            plan = tool.resolve(c, inputs, output)
        record, exit_code = tool.execute(plan, output)
        with tool.native_connection() as c:
            current = tool.resolve(c, inputs, Path(d) / "unused-output.json")
            if {k: v for k, v in plan.items() if k != "argv"} != {
                k: v for k, v in current.items() if k != "argv"
            }:
                raise ValueError("vascular_current_state_changed")
        execution = {
            k: plan[k] for k in ["tool", "version", "implementation", "installation_checksum"]
        }
        execution["direct_exit_code"] = exit_code
    return persist(base_code, record, execution, repo=repo, store=store, case_code=case_code)


def main():
    from apps.evidence_graph_html import write_export
    from apps.glaucoma_case_storage import read_view
    from apps.patient_service_assembly import OBJECT_ROOTS
    from apps.research_single_case import credential_boundary
    from db.runtime.repository.case_evidence import CaseEvidenceRepository
    from packages.object_store import LocalObjectStore

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-code", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix(".json").exists() or a.receipt.exists():
        p.error("new output and receipt required")
    repo = CaseEvidenceRepository(credential_boundary().resolve_target("runtime"))
    store = LocalObjectStore(OBJECT_ROOTS["visible"])
    result = run(a.case_code, a.base, repo=repo, store=store)
    with a.receipt.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    view = read_view(a.case_code, result["graph"]["asset_code"], repo=repo, store=store)
    write_export(view, a.output)
    print(
        json.dumps(
            {
                "graph": result["graph"]["asset_code"],
                "measurement": result["measurement"]["asset_code"],
                "output": str(a.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
