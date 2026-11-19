"""Pure projection of verified supplemental facts into the existing demand nodes."""


def project_supplement(nodes, report):
    s = report.get("supplement")
    if not s:
        return
    if s["case_code"] != report["case_code"] or s["input_checksum"] != report["input"]["checksum"]:
        raise ValueError("supplement_projection_scope")
    known = {a["asset_code"] for a in report["artifacts"]}
    if not s["artifact_codes"] or not set(s["artifact_codes"]) <= known:
        raise ValueError("supplement_projection_source")
    metrics = s["mask_check"]["metrics"]
    rows = [
        {
            "name": name,
            "value": metrics[name]["value"],
            "unit": "1",
            "formula": metrics[name].get("expression"),
        }
        for name in ("vcdr", "acdr", "hcdr", "rdar")
    ]
    rows += [
        {"name": name + "面积", "value": count, "unit": "pixel"}
        for name, count in s["mask_check"]["pixel_counts"].items()
    ]
    rows += [
        {"name": "盘沿包围盒距离·" + k, "value": v, "unit": "pixel"}
        for k, v in metrics["rim_profile"]["values"].items()
    ]
    p = nodes["CUP_RIM_MEASURE"]["patient_evidence"]
    p.update(
        status="已有结果",
        summary="SegMAN-B同图杯/盘沿/盘掩膜及现有几何算法测量",
        results=rows,
        artifact_codes=s["artifact_codes"],
        images=["segman_overlay", "segman_cup", "segman_rim", "segman_disc"],
        tool=s["registered_tool"],
        next_action="查看叠加边界与像素测量；眼别未知，左右方向不命名为鼻侧/颞侧。",
    )
    nodes["CUP_RIM_MEASURE"]["tools"] = [s["registered_tool"]]
    d = nodes["RISK_FIELDS"]["patient_evidence"]
    fields = s["demographics"]["fields"]
    d.update(
        status="已有结果",
        summary="同一源NPZ的6项人口学字段已核对；家族史未提供。",
        results=[
            {
                "name": key,
                "value": v["display_value"],
                "unit": v.get("unit", ""),
                "missing_reason": None
                if v.get("display_value") is not None
                else v.get("reason", "当前输入无法取证"),
            }
            for key, v in fields.items()
        ],
        artifact_codes=s["artifact_codes"],
        next_action="使用已提供字段；没有的家族史本轮不取证。",
    )
    for key, review in s["visual_review"]["observations"].items():
        nodes[key]["patient_evidence"].update(
            status=review["status"],
            summary=review["summary"],
            next_action=review["next_action"],
            results=[{"name": "图像观察", "value": review["summary"], "unit": ""}],
            artifact_codes=s["artifact_codes"],
            images=["input"],
            observation_source={
                "method": s["visual_review"]["method"],
                "reviewer": s["visual_review"]["reviewer"],
                "finding": review["finding"],
                "image_checksum": s["input_checksum"],
            },
        )
