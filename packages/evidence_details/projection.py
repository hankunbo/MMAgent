"""Deterministic binding of saved judgments, not re-inference of patient facts."""

from collections import Counter

from packages.evidence_details.models import (
    FACETS,
    Assessment,
    Facet,
    Ledger,
    binding,
    digest,
    resolve,
)


def build(base, profiles):
    documents = {"base": base, "profiles": profiles}
    nodes = {n["key"]: (i, n) for i, n in enumerate(base["nodes"])}
    case = base["case_evidence"]
    assessments = []
    index = {key: [] for key in nodes}
    rules = base.get("research_criteria", {})
    completion = base.get("case_completion", {})
    structure = base.get("evidence_structure", {})

    def fact(pointer, document="base"):
        return binding(documents, document, pointer)

    def facet(explanation, refs=(), missing=(), *, applicable=True):
        status = (
            "not_applicable"
            if not applicable
            else "partial"
            if refs and missing
            else "missing"
            if missing
            else "bound"
        )
        return Facet(
            status=status, facts=list(refs), explanation=explanation, missing=list(missing)
        )

    def add(identifier, key, title, kind, definition, result=None, path=()):
        ni, node = nodes[key]
        patient = node.get("patient_evidence", {})
        node_ref = fact(f"/nodes/{ni}/patient_evidence")
        definition_ref = fact(definition)
        definition_value = resolve(base, definition)
        result_value = resolve(base, result) if result else {}
        result_refs = [fact(result)] if result else [node_ref]
        tool_codes = set()
        for tool in [patient.get("tool"), *node.get("tools", [])]:
            if isinstance(tool, dict):
                code = (
                    tool.get("code")
                    or tool.get("tool_code")
                    or tool.get("tool_ref", {}).get("code")
                )
                if code:
                    tool_codes.add(code)
        # Multi-step measurements retain their algorithm as well as segmentation.
        for geometry in completion.get("geometry", {}).values():
            if isinstance(geometry, dict) and key in {
                "CUP_RIM_MEASURE",
                "DISC_GEOMETRY",
                "DISC_SHAPE_CONTEXT",
            }:
                code = geometry.get("tool_code")
                if code:
                    tool_codes.add(code)
        if key == "MODEL_ASSESSMENT":
            tool_codes.add("fairclip_slo_glaucoma_classifier")
        if key == base.get("branch_geometry", {}).get("node"):
            tool_codes.add(base["branch_geometry"]["tool"]["code"])
        tool_refs = [
            fact(f"/tools/{i}", "profiles")
            for i, p in enumerate(profiles["tools"])
            if p["code"] in tool_codes
        ]
        evaluation_refs = [
            fact(f"/evaluations/{i}", "profiles")
            for i, p in enumerate(profiles["evaluations"])
            if set(p["tool_codes"]) & tool_codes
        ]
        source_refs = []
        if node.get("knowledge"):
            source_refs.append(fact(f"/nodes/{ni}/knowledge"))
        if kind in {"criterion", "descriptive", "association"}:
            source_refs.append(definition_ref)
        urls = {s.get("url") for s in definition_value.get("sources", []) if isinstance(s, dict)}
        source_profile_refs = [
            fact(f"/sources/{i}", "profiles")
            for i, source in enumerate(profiles.get("sources", []))
            if source["same_url_group"] in urls
        ]
        cited = set()

        def source_ids(value):
            if isinstance(value, dict):
                if value.get("kind") == "source_version":
                    cited.add((value.get("code"), value.get("version"), value.get("checksum")))
                for item in value.values():
                    source_ids(item)
            elif isinstance(value, list):
                for item in value:
                    source_ids(item)

        for ref in source_refs:
            source_ids(resolve(base, ref.pointer))
        source_profile_refs.extend(
            fact(f"/knowledge_sources/{i}", "profiles")
            for i, source in enumerate(profiles.get("knowledge_sources", []))
            if tuple(source["reference"][k] for k in ("code", "version", "checksum")) in cited
        )
        artifacts = sorted(set(patient.get("artifact_codes", [])))
        artifact_refs = [
            fact(f"/case_evidence/artifacts/{i}")
            for i, a in enumerate(case.get("artifacts", []))
            if a["asset_code"] in artifacts
        ]
        extra = []
        if key in {"CUP_RIM_MEASURE", "DISC_GEOMETRY", "DISC_SHAPE_CONTEXT", "DISC_MASK"}:
            if "model_selection" in base:
                extra.append(fact("/model_selection"))
            if completion.get("geometry"):
                extra.append(fact("/case_completion/geometry"))
            if rules.get("radial_rim_measurement"):
                extra.append(fact("/research_criteria/radial_rim_measurement"))
        if key == "MODEL_ASSESSMENT":
            extra.append(fact("/case_evidence/fairclip"))
        if key == base.get("branch_geometry", {}).get("node"):
            extra.append(fact("/branch_geometry/record"))
            extra.append(fact("/source_associations/tool_flows"))
        deps = [
            fact(f"/evidence_structure/dependency_edges/{i}")
            for i, e in enumerate(structure.get("dependency_edges", []))
            if e.get("to") == key or e.get("from") == key
        ]
        visual = kind == "visual"
        missing_input = patient.get("status") == "当前输入无法取证"
        known_missing = ["尚未完成结构化研究设计/偏倚/参考标准/人群适用性审核；已保存原来源说明"]
        input_missing = (
            ["当前没有可核验的检查输入"]
            if missing_input
            else ["格式可用与医学测量等效性分别判断；未记录的设备、眼别、时点不推断"]
        )
        notes = ["本层不生成置信度；保留原始观察、比较和用途，不修改03评分"]
        if identifier.endswith("RIM_RADIAL_LT_01"):
            notes.append(
                "文献方位条件与本图全周最薄径向测量并非自动等效；没有鼻颞定向时，不把图像轴当解剖象限"
            )
        if key == "VENULAR_BRANCH_ANGLE":
            notes.append(
                "算法按盘径归一化且有尺度复核；与论文ROI、血管选择及人群统计量的等效性尚未验证"
            )
        if key == "MODEL_ASSESSMENT":
            notes.append(
                "940例指标属于锁定的离线评价协议；不是本例概率可靠度，不能直接替代当前部署的完整验证"
            )
        facets = {
            "target": facet("精确节点、定义、目标范围；组织关系不是因果或权重。", [definition_ref]),
            "source": facet(
                "区分医学原文、已有来源引用和方法定义。",
                [*source_refs, *source_profile_refs],
                []
                if source_refs
                else ["本项未单独绑定医学来源；技术产物通过服务的判读条件关联来源"],
            ),
            "source_quality": facet(
                "保留研究描述；不将网页/指南名称自动当作高质量评分。",
                [*source_refs, *source_profile_refs],
                known_missing,
                applicable=kind not in {"visual", "node_context"},
            ),
            "reference": facet(
                "逐项绑定原条件、阈值/现象、公式、单位及适用说明；不新增医学阈值。",
                [definition_ref, *result_refs],
            ),
            "tool": facet(
                "工具身份、当前登记快照与历史选择分开；历史记录不授权新执行。",
                [*tool_refs, *extra],
                [] if tool_refs else ["没有本项精确模型/算法身份；观察记录或输入读取见执行来源"],
            ),
            "tool_evaluation": facet(
                "作者/本地评测、数据划分、输出定义和部署对应关系分别保留。",
                evaluation_refs,
                ["未报告的置信区间、亚组、外部验证、重复性不补造"],
                applicable=bool(tool_codes),
            ),
            "input_match": facet(
                "绑定病例、原图摘要、所需模态、原适用性与缺项；同图不等于同眼纵向证据。",
                [fact("/case_evidence/input"), node_ref, *result_refs],
                input_missing,
            ),
            "preprocessing": facet(
                "从执行、测量和锁定资源记录追溯重采样/坐标；缺步骤不推定恒等变换。",
                [*extra, *evaluation_refs],
                ["逐次执行中未保存的裁剪/缩放/方向变换及参数，不能由当前环境倒推"],
            ),
            "execution": facet(
                "助手看图、组件调用和06执行记录按实际来源区分；不补造attempt、模型或提示词。",
                [*result_refs, *extra, *artifact_refs],
                ["历史未记录的逐次执行身份/环境或观察者详细配置保持缺失"],
            ),
            "output_qc": facet(
                "保留输出类型、数值、单位、原QC和来源；技术PASS不是医学正确。",
                [*result_refs, *artifact_refs, *extra],
            ),
            "uncertainty": facet(
                "同图模型差异和已有尺度复核可读；未做的重复测量/分割扰动不写成稳定。",
                extra,
                ["未统一评估边界扰动、重复性与误差向下游的传播"],
            ),
            "interpretation": facet("；".join(notes), [definition_ref, *result_refs]),
            "dependence": facet(
                "同一判读沿父节点引用同一ID；相同图像/产物分组仅表示相关，不表示全都相同或独立。",
                [node_ref, *deps, *extra],
            ),
            "missing_conflict": facet(
                "缺输入、未定位、未达条件和来源冲突沿用原记录；不把缺值填成0。",
                [node_ref, *result_refs, *extra],
            ),
            "assessment_provenance": facet(
                "本轮为助手编写的确定性追溯绑定，不是新临床审核或真实VLM执行。",
                [fact("/assembly", "profiles")],
            ),
        }
        if visual:
            facets["execution"] = facet(
                "原记录明确为助手直接图像观察，未执行部署VLM；原观察时间/模型/提示词未结构化保存。",
                result_refs,
                ["观察者模型版本、完整提示词、裁剪坐标或独立复核记录未保存"],
            )
        record = Assessment(
            id=identifier,
            node=key,
            title=title,
            kind=kind,
            facets=facets,
            same_image_group=case["input"]["checksum"],
            artifact_groups=artifacts,
            input_checks={
                "case": {"status": "bound_to_case", "value": case["case_code"]},
                "image": {"status": "bound_to_input", "checksum": case["input"]["checksum"]},
                "format": {
                    "status": "recorded_not_redecoded",
                    "media_type": case["input"].get("media_type"),
                    "dimensions": case["input"].get("dimensions"),
                },
                "laterality": {
                    "status": "unknown"
                    if case["input"].get("laterality", "UNKNOWN") == "UNKNOWN"
                    else "recorded",
                    "value": case["input"].get("laterality"),
                },
                "modality_and_method": {
                    "status": "use_saved_applicability",
                    "applicability": result_value.get("applicability"),
                    "explanation": result_value.get("applicability_note"),
                },
                "measurement_equivalence": {
                    "status": "not_assumed",
                    "saved_record": patient.get("measurement_equivalence"),
                },
                "required_inputs": {
                    "status": "missing" if missing_input else "see_bound_acquisition",
                    "record": patient.get("acquisition"),
                },
                "time_and_device": {"status": "not_verified", "value": None},
            },
            comparison={
                "metric": definition_value.get("metric"),
                "unit": definition_value.get("unit"),
                "operator": definition_value.get("operator"),
                "threshold": definition_value.get("threshold"),
                "phenomenon": definition_value.get("expected_pattern"),
                "condition": definition_value.get("condition"),
                "observed_value": result_value.get("value"),
                "comparison": result_value.get("comparison"),
                "finding": result_value.get("finding"),
                "assessment": result_value.get("assessment"),
                "medical_effect": result_value.get("medical_effect", "not_assigned"),
                "origin": "exact_saved_result_not_new_computation",
            },
            measurements=patient.get("results", []),
            gap_codes=["UNCERTAINTY_NOT_FULLY_EVALUATED", "NO_CONFIDENCE_RUBRIC_ASSIGNED"]
            + (["STUDY_APPRAISAL_NOT_STRUCTURED"] if kind in {"criterion", "association"} else [])
            + (["REQUIRED_INPUT_UNAVAILABLE"] if missing_input else [])
            + (["HISTORICAL_OBSERVER_CONFIGURATION_INCOMPLETE"] if visual else [])
            + (
                ["NO_GENERAL_MEDICAL_CUTOFF"]
                if kind == "association" and definition_value.get("universal_threshold") is None
                else []
            ),
        )
        assessments.append(record)
        for target in dict.fromkeys([key, *path]):
            if target in index:
                index[target].append(identifier)

    for kind, definitions, results in [
        ("criterion", "/research_criteria/policy/criteria", "results"),
        ("descriptive", "/research_criteria/descriptive_criteria", "descriptive_results"),
    ]:
        definitions_data = (
            rules.get("policy", {}).get("criteria", [])
            if kind == "criterion"
            else rules.get("descriptive_criteria", [])
        )
        for i, definition in enumerate(definitions_data):
            match = next(
                (
                    (j, r)
                    for j, r in enumerate(rules.get(results, []))
                    if r["criterion_id"] == definition["id"]
                ),
                None,
            )
            j, result = match if match else (None, {})
            add(
                kind + ":" + definition["id"],
                definition["leaf"],
                definition["title"],
                kind,
                f"{definitions}/{i}",
                f"/research_criteria/{results}/{j}" if match else None,
                result.get("path", []),
            )
    for i, edge in enumerate(base.get("source_associations", {}).get("association_edges", [])):
        add(
            "association:" + edge["from"],
            edge["from"],
            nodes[edge["from"]][1]["title"],
            "association",
            f"/source_associations/association_edges/{i}",
            path=[edge["to"]],
        )
    for i, visual in enumerate(completion.get("visual_observations", [])):
        add(
            "visual:" + visual["id"],
            visual["nodes"][0],
            visual["title"],
            "visual",
            f"/case_completion/visual_observations/{i}",
            f"/case_completion/visual_observations/{i}",
            visual["nodes"],
        )
    for key, (i, node) in nodes.items():
        add("context:" + key, key, node["title"], "node_context", f"/nodes/{i}")
    relations = []
    for section, names in [
        ("evidence_structure", ["main_edges", "dependency_edges"]),
        ("source_associations", ["association_edges", "dependency_edges", "organization_edges"]),
    ]:
        for name in names:
            relations.extend(
                fact(f"/{section}/{name}/{i}")
                for i, _ in enumerate(base.get(section, {}).get(name, []))
            )
    counts = Counter(f.status for a in assessments for f in a.facets.values())
    coverage = {
        "nodes": {"bound": len(index), "total": len(nodes)},
        "assessments": len(assessments),
        "relations": len(relations),
        "facets": {"total": len(assessments) * len(FACETS), **dict(counts)},
        "meaning": "此分母为追溯清单，不是疾病证据完整率或工具准确率；partial不计为全部信息齐备",
        "clinical_evidence_coverage": None,
        "clinical_evidence_coverage_missing": "未设定经过审核的疾病证据集合和权重",
    }
    raw = dict(
        schema_version="case-judgment-details/1.0.0",
        case_code=case["case_code"],
        input_checksum=case["input"]["checksum"],
        base_checksum=base["checksum"],
        profiles_checksum=digest(profiles),
        builder_version="judgment-binding/1.0.0",
        assessments=[a.model_dump(mode="json") for a in assessments],
        node_index=index,
        relations=[r.model_dump() for r in relations],
        coverage=coverage,
    )
    return Ledger.model_validate({**raw, "checksum": digest(raw)}).model_dump(mode="json")


def validate(base, profiles, ledger):
    from packages.evidence_details.models import validate_bindings

    validate_bindings(ledger, {"base": base, "profiles": profiles})
    if build(base, profiles) != ledger:
        raise ValueError("detail_controlled_projection_mismatch")
