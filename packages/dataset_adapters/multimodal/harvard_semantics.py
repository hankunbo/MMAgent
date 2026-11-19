"""Explicit published-release semantics; not a disease label or a scoring rule.

Opt-in only: old manifests retain their original bytes and interpretation.
Sources are frozen by the implementation snapshot, not fetched during decoding.
"""

PROFILE = "harvard-gdp-v1"
ADAPTER = "multimodal-adapters/1.1.0"
README = (
    "https://raw.githubusercontent.com/Harvard-Ophthalmology-AI-Lab/Harvard-GDP/"
    "3d643726b80f32b36ea59b1b0f8881c8614bb2c2/README.md"
)
PAPER = "https://arxiv.org/abs/2308.13411"
VF_LAYOUT = (
    "https://raw.githubusercontent.com/Harvard-Ophthalmology-AI-Lab/Harvard-GDP/"
    "3d643726b80f32b36ea59b1b0f8881c8614bb2c2/utils/data_handler.py"
)
METHOD_PROFILE = "harvard-gdp-author-methods/1.0.0"
METHOD_SOURCE = "docs/sources/harvard-gdp-author-ICCV2023.pdf"
METHOD_CHECKSUM = "sha256:7b8929b0ca83e2b689af652dfe1956334c0b98dd3026b193610b4ef1af731784"
METHOD_EXCERPT = "sha256:ccff963df469069810d747bb22edbce7af2344d705461d0cbd03a814187f69a6"


def method_context(concept):
    """Author device/area metadata, not eye registration or a normative comparison."""
    from pathlib import Path

    from packages.knowledge_index.pdf_preview import preview
    from packages.object_store import checksum

    if concept != "rnflt":
        raise ValueError("harvard_method_concept")
    path = Path(METHOD_SOURCE)
    readme = Path("docs/sources/harvard-gdp-author-readme-3d643726.md")
    if any(p.is_symlink() for p in (path, readme, *path.parents)):
        raise ValueError("harvard_method_source_binding")
    raw, description = path.read_bytes(), readme.read_bytes()
    if (
        checksum(raw) != METHOD_CHECKSUM
        or checksum(description)
        != "sha256:9a8e1a43be7894989745a2947e61fbc1e4f6de6fdbd641f1899d7e22802d0b10"
    ):
        raise ValueError("harvard_method_source_binding")
    parsed = preview(raw, page_start=3, page_count=1, max_chars=16000)
    if parsed["status"] != "succeeded" or parsed["excerpt_truncated"]:
        raise ValueError("harvard_method_pdf_preview")
    text = parsed["excerpt"]
    start = text.find("The RNFLT (")
    end = text.find("The VF testing", start)
    excerpt = text[start:end].strip()
    if start < 0 or end <= start or checksum(excerpt.encode()) != METHOD_EXCERPT:
        raise ValueError("harvard_method_excerpt_binding")
    return dict(
        schema_version=METHOD_PROFILE,
        concept=concept,
        purpose="dataset_release_measurement_method_not_medical_direction",
        source_url=(
            "https://openaccess.thecvf.com/content/ICCV2023/papers/"
            "Luo_Harvard_Glaucoma_Detection_and_Progression_A_Multimodal_Multitask_"
            "Dataset_and_ICCV_2023_paper.pdf"
        ),
        source_checksum=METHOD_CHECKSUM,
        locator=dict(physical_page=3, section="3.1", span_start=start, span_end=end),
        excerpt=excerpt,
        excerpt_checksum=METHOD_EXCERPT,
        release_readme=dict(url=README, checksum=checksum(description), shape=[225, 225]),
        paper_shape=[224, 224],
        unit_qualification="original_owner_um_unchanged",
        interpretation_limits=[
            "Author acquisition area: peripapillary 6x6 mm around ONH, not a ring mean.",
            "Paper 224x224 versus release 225x225 is unresolved; no resampling or mm/pixel claim.",
            "This identifies dataset measurement region, not this eye's sector registration.",
            "Original numeric statistics/negative codes retained; no per-case segmentation QC.",
            "No normative percentile, threshold, stage relation, laterality or direction inferred.",
        ],
    )


def fields(key):
    """Only documented semantics; unknown eye/visit/scan calibration stay unknown."""
    common = ("published_dataset_QC_not_per_case_reliability_certificate",)
    if key == "age":
        return dict(unit="years", unit_basis=README + "#age", semantic_limits=common)
    if key in {"md", "tds"}:
        return dict(
            unit="dB",
            unit_basis=PAPER + "#section-3.1-visual-field",
            axis_semantics="not_applicable" if key == "md" else "source_index_only",
            semantic_limits=common
            + (
                "Humphrey_24-2_mean_deviation"
                if key == "md"
                else "Humphrey_24-2_total_deviation_52",
                "signed_deviation_not_probability_or_category_code",
                "eye_visit_and_per_test_reliability_not_provided",
                "VF_derived_label_leakage_not_blinded_classification_evidence",
            ),
        )
    if key == "rnflt":
        return dict(
            unit="um",
            unit_basis=PAPER + "#section-3.1-RNFLT",
            axis_semantics="source_index_only",
            semantic_limits=common
            + (
                "manufacturer_derived_RNFL_thickness_map_not_raw_intensity",
                "nonnegative_finite_thickness_only_negative_codes_preserved_separately",
                "source_grid_not_eye_registered_anatomical_sectors",
                "no_device_normative_percentiles_or_per_case_segmentation_QC",
            ),
        )
    return dict(
        axis_semantics="source_index_only",
        semantic_limits=common
        + (
            "source_axis_0_slice_axis_1_row_axis_2_column_no_physical_voxel_calibration",
            "sampled_Bscans_not_circumpapillary_ring_or_macular_GCC_measurement",
        ),
    )


def context(concept):
    """Dataset decoding facts, distinct from the cited medical relationship."""
    shared = (
        "数据语义依据为Harvard-GDP作者README、ICCV2023论文3.1节与公开读取代码；"
        "这些说明不是本病例诊断，也不自动建立命题方向。数据集级质控不代表本例可靠性数值已知。"
    )
    notes = {
        "age": "age为年龄（年），不是损害测量；风险关联不能自动当作晚期关系。",
        "md": "md为Humphrey 24-2视野平均偏差（dB），不是未定义类别编码；保留符号及原值。",
        "tds": (
            "tds为52点总偏差（dB），不是0至5级评分。作者映射为8×9源坐标网格；"
            "行列点数为4/6/8/8/8/8/6/4，两中间行跳过第7列（0起）。"
            "不得凭源坐标编造眼别、解剖定位、概率或分期阈值。"
        ),
        "rnflt": (
            "rnflt为设备由层边界导出的RNFL厚度图（微米），不是普通灰度强度。"
            "负值不作为组织厚度参与统计，也不填零；保留其编码计数与覆盖。"
            "无正常库不报正常百分位，但不因此禁止解释有依据的局部形态或分布。"
        ),
        "bscans": (
            "展示的是原OCT体积源索引切片；无物理层厚标定，不声称GCC分割或完整环扫。"
            "先评价实际可见结构，再明确剩余障碍，不因不能单项分期自动放弃已有信息。"
        ),
    }
    return shared + notes.get(concept, "")


def qualify_rows(rows, manifest):
    if manifest.adapter_version != ADAPTER:
        return rows
    from packages.grounding.models import digest, validate

    components = {c.id: c for c in manifest.components}
    result = []
    for row in rows:
        c = components.get(row.get("component"))
        if c is None:
            result.append(row)
            continue
        definition = row["definition"].model_dump(mode="json")
        # TaskDefinition has a deliberate 240-character per-field wire bound.
        # Keep the proposition safeguards; concise dataset facts are not a new
        # medical reference. The full provenance stays in the manifest/unit_basis.
        short = {
            "age": "作者字段：年龄，年；风险背景非损害或分期。",
            "md": "作者字段：Humphrey24-2平均偏差（dB），保留符号，非类别码。",
            "tds": "作者字段：52点总偏差（dB），保留符号，非等级码；眼别未核。",
            "rnflt": "作者字段：设备RNFL厚度图（微米）；负码非组织；无正常百分位。",
            "bscans": "源索引B扫描，非环扫或GCC分割；判读实际可见部分。",
        }
        definition["applicability_conditions"] += " " + short.get(c.concept, "")
        definition["reference"]["checksum"] = digest(
            {k: v for k, v in definition.items() if k != "reference"}
        )
        result.append(dict(row, definition=validate(type(row["definition"]), definition)))
    return result


def numeric(c, payloads):
    """Source statistics only; no direction, clinical thresholds or label access."""
    import math

    from .formats import npy

    body = dict(concept=c.concept, unit=c.unit, semantic_profile=PROFILE)
    if c.kind == "scalar":
        return dict(body, value=c.value)
    if c.kind not in {"vector", "map"}:
        return None
    values = list(npy(payloads[c.payload_ids[0]]).numbers())
    finite = [float(n) for n in values if math.isfinite(n)]
    if c.kind == "vector":
        return dict(
            body,
            values=[float(n) if math.isfinite(n) else None for n in values],
            finite_count=len(finite),
            count=len(values),
            layout="author_8x9_source_grid_not_eye_registered",
            meaning="total_deviation_not_ordinal_code",
        )
    valid = [n for n in finite if n >= 0]
    negative = sorted(set(n for n in finite if n < 0))
    body.update(
        shape=list(c.shape),
        valid_count=len(valid),
        total_count=len(values),
        negative_count=sum(n < 0 for n in finite),
        negative_values=negative[:8],
        nonfinite_count=len(values) - len(finite),
        mean=sum(valid) / len(valid) if valid else None,
        minimum=min(valid) if valid else None,
        maximum=max(valid) if valid else None,
        meaning="nonnegative_device_RNFL_thickness_not_normative_comparison",
    )
    return body
