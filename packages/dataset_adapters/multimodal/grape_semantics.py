"""Opt-in GRAPE author-defined measurement semantics, never a disease label.

Source: Scientific Data 10:520 (2023), Methods and Table 1. No normative
reference, anatomical point map, or unstated measurement unit is invented.
"""

PROFILE = "grape-v1"
ADAPTER = "multimodal-adapters/1.2.0"
PAPER = "https://www.nature.com/articles/s41597-023-02424-4"
SOURCE_CHECKSUM = "sha256:02600be38903d16ed076b7b9fca4288ce99ef4cc8486938d7795fb47e12e7926"
VF_MEANING = "Octopus_G1_light_sensitivity_61_not_Humphrey_TD52"
METHOD_PROFILE = "grape-author-methods/1.0.0"
RNFL_UNIT_PROFILE = "grape-rnfl-device-unit/1.0.0"
TONOPACHY_UNIT_PROFILE = "grape-tonopachy-device-unit/1.0.0"


def method_context(concept, *, profile=METHOD_PROFILE):
    """Versioned release metadata, not a new clinical relationship or unit grant."""
    from pathlib import Path
    from xml.etree import ElementTree

    from packages.object_store import checksum

    if profile not in {METHOD_PROFILE, RNFL_UNIT_PROFILE, TONOPACHY_UNIT_PROFILE}:
        raise ValueError("grape_method_profile")
    if profile == RNFL_UNIT_PROFILE and concept not in {
        "rnfl_mean",
        "rnfl_superior",
        "rnfl_nasal",
        "rnfl_inferior",
        "rnfl_temporal",
    }:
        raise ValueError("grape_rnfl_unit_concept")

    if profile == TONOPACHY_UNIT_PROFILE and concept not in {"iop", "cct"}:
        raise ValueError("grape_tonopachy_unit_concept")

    methods = {
        "rnfl": (
            "Retinal nerve fiber layer (RNFL) thickness",
            "sha256:e6ae8d8a4fbceae54dd17740301798456be26f41469c30ca18f329065628b31f",
        ),
        "iop": ("IOP", "sha256:f6e1d0cc77ef604864240c07d6bd2c09667238671547ecd363b96319ea3c09fc"),
        "cct": (
            "Central corneal thickness (CCT)",
            "sha256:6578ddbe36153f4f315d4adc66e4eeb9bdc025cbe658e835574a130b9d9fdf41",
        ),
    }
    rnfl = {"rnfl_mean", "rnfl_superior", "rnfl_nasal", "rnfl_inferior", "rnfl_temporal"}
    key = "rnfl" if concept in rnfl else concept
    if key not in methods:
        raise ValueError("grape_method_concept")
    path = Path("docs/sources/grape-author-2023-PMC10404253.xml")
    if path.is_symlink():
        raise ValueError("grape_method_source_binding")
    raw = path.read_bytes()
    if checksum(raw) != SOURCE_CHECKSUM:
        raise ValueError("grape_method_source_binding")
    title, expected = methods[key]
    paragraphs = []
    for section in ElementTree.fromstring(raw).findall(".//body//sec"):
        heading = section.find("title")
        if heading is not None and "".join(heading.itertext()) == title:
            paragraphs.extend("".join(p.itertext()) for p in section.findall("p")[:1])
    if len(paragraphs) != 1 or checksum(paragraphs[0].encode()) != expected:
        raise ValueError("grape_method_excerpt_binding")
    context = dict(
        schema_version=METHOD_PROFILE,
        concept=concept,
        purpose="dataset_release_measurement_method_not_medical_direction",
        source_url=PAPER,
        source_checksum=SOURCE_CHECKSUM,
        locator=dict(section=title, paragraph_index=0),
        excerpt=paragraphs[0],
        excerpt_checksum=expected,
        unit_qualification="release_unit_remains_unverified",
        interpretation_limits=[
            "Dataset protocol, not proof of individual measurement quality or treatment.",
            "Device name does not provide normative percentiles, thresholds or a stage relation.",
            "Do not turn supplied quadrant measurements into raw OCT or a new segmentation.",
            "Keep the original value, unit status and observation; this context is not a vote.",
        ],
    )
    if profile == RNFL_UNIT_PROFILE:
        context = dict(context, schema_version=RNFL_UNIT_PROFILE)
        context["unit_bridge"] = rnfl_device_unit_bridge(concept)
        context["interpretation_limits"].append(
            "Explicit device/protocol unit inference supplements, but does not rewrite, "
            "the original release observation unit; no normative or stage result follows."
        )
    if profile == TONOPACHY_UNIT_PROFILE:
        context = dict(context, schema_version=TONOPACHY_UNIT_PROFILE)
        context["unit_bridge"] = tonopachy_device_unit_bridge(concept)
        context["interpretation_limits"].append(
            "Device-scale unit inference only; the original release field remains unverified. "
            "IOP raw/compensated state is unknown; do not apply a CCT correction."
        )
    return context


def rnfl_device_unit_bridge(concept):
    """Exact public method bridge, not a changed measurement or medical grant."""
    from pathlib import Path

    from packages.object_store import checksum

    if concept not in {
        "rnfl_mean",
        "rnfl_superior",
        "rnfl_nasal",
        "rnfl_inferior",
        "rnfl_temporal",
    }:
        raise ValueError("grape_rnfl_unit_concept")
    path = Path("docs/sources/cirrus_rnfl_report_guide_31_011_0002ii.pdf")
    expected = "sha256:1c1e8556e322336a8c9ed06d3f75e3588b345286634e627fb1c27f8d13818995"
    if path.is_symlink() or checksum(path.read_bytes()) != expected:
        raise ValueError("grape_rnfl_manufacturer_source_binding")
    return dict(
        schema_version="dataset-device-unit-bridge/1.0.0",
        concept=concept,
        source_url=(
            "https://www.zeiss.co.uk/content/dam/med-hcp/united-kingdom/downloads/"
            "dbb/oct-imaging-downloads/user-guides/"
            "cirrus_how_to_read_en_ous_en_31_011_0002ii_final3.pdf"
        ),
        source_checksum=expected,
        locator=dict(page_1_based=12, report="RNFL and ONH: Optic Disc Cube 200x200"),
        exact_unit_text="µm",
        proposed_unit="um",
        evidence_location="Average RNFL Thickness table and RNFL Thickness graph axis",
        relation_kind="documented_device_protocol_inference_not_release_field_assertion",
        inference=(
            "GRAPE reports CIRRUS HD-OCT 5000 Optic Disc 200x200 ONH/RNFL analysis "
            "for total and four sectoral mean peripapillary RNFL thickness. "
            "The corresponding manufacturer report labels RNFL thickness in micrometres; "
            "this supports an explicit method-to-field unit inference."
        ),
        source_measurement_changed=False,
        original_release_unit_qualification="release_unit_remains_unverified",
        clinical_qualification=False,
        interpretation_limits=[
            "No exact patient software version, scan quality or segmentation proof.",
            "No individual normative percentile, threshold, thinning or stage inferred.",
            "No IOP/CCT unit qualification or conversion from this RNFL report.",
            "Not an independent disease observation or additional score.",
        ],
    )


def fields(concept):
    limits = (
        "semantic_profile:" + PROFILE,
        "source_document:" + PAPER,
        "source_document_checksum:" + SOURCE_CHECKSUM,
        "dataset_QC_not_individual_reliability_certificate",
    )
    if concept == "age":
        return dict(unit="years", unit_basis=PAPER + "#Table1-age", semantic_limits=limits)
    if concept == "vf":
        return dict(
            axis_semantics="source_index_only",
            semantic_limits=limits
            + (
                VF_MEANING,
                "minus_one_is_unseen_or_blind_point_not_numeric_sensitivity",
                "author_eye_specific_order_anatomical_map_not_bound",
                "no_age_normative_reference_no_TD_or_MD_derivation",
                "physical_unit_not_explicitly_bound_from_release_table",
            ),
        )
    if concept.startswith("rnfl_"):
        limits += (
            "author_peripapillary_RNFL_mean_or_named_quadrant_not_raw_OCT",
            "physical_unit_not_explicitly_bound_from_release_table",
            "device_normative_reference_and_individual_segmentation_QC_not_provided",
        )
    else:
        limits += ("physical_unit_not_explicitly_bound_from_release_table",)
    return dict(semantic_limits=limits)


def context(concept):
    if concept == "vf":
        return (
            "作者定义：Octopus G1 61点光敏感度，非Humphrey总偏差。-1可为盲点或最低刺激仍未见，"
            "不作负测量、不填零；无正常库不推导MD/TD。源点序未绑定解剖图。"
        )
    if concept.startswith("rnfl_"):
        return "作者字段：盘周RNFL均值或命名象限测量，非原始OCT；单位与正常库未核，不造阈值。"
    if concept == "age":
        return "作者字段：基线年龄（年），不是每次随访重新测得的年龄或疾病标签。"
    return "作者已命名临床测量；保留原数值，单位未核，不据此套用数值阈值。"


def qualify_rows(rows, manifest):
    if manifest.adapter_version != ADAPTER:
        return rows
    from packages.grounding.models import digest, validate

    components = {c.id: c for c in manifest.components}
    result = []
    for row in rows:
        c = components.get(row.get("component"))
        if c is None or c.kind not in {"scalar", "vector"}:
            result.append(row)
            continue
        definition = row["definition"].model_dump(mode="json")
        definition["applicability_conditions"] += " " + context(c.concept)
        definition["reference"]["checksum"] = digest(
            {k: v for k, v in definition.items() if k != "reference"}
        )
        result.append(dict(row, definition=validate(type(row["definition"]), definition)))
    return result


def numeric(c, payloads):
    import json
    import math

    body = dict(
        concept=c.concept, unit=c.unit or "source_units_unqualified", semantic_profile=PROFILE
    )
    if c.kind == "scalar":
        return dict(body, value=c.value, meaning=context(c.concept))
    if c.kind != "vector" or c.concept != "vf":
        return None
    values = json.loads(payloads[c.payload_ids[0]])["values"]
    if len(values) != 61 or c.shape != (61,):
        raise ValueError("grape_vf_source_shape")
    valid = [float(v) for v in values if type(v) in {int, float} and math.isfinite(v) and v >= 0]
    return dict(
        body,
        values=values,
        count=len(values),
        valid_count=len(valid),
        coded_unseen_or_blind_count=sum(v == -1 for v in values),
        invalid_negative_count=sum(
            type(v) in {int, float} and math.isfinite(v) and v < 0 and v != -1 for v in values
        ),
        missing_count=sum(type(v) not in {int, float} or not math.isfinite(v) for v in values),
        mean=sum(valid) / len(valid) if valid else None,
        minimum=min(valid) if valid else None,
        maximum=max(valid) if valid else None,
        meaning=VF_MEANING,
        statistics="nonnegative_values_only_not_MD_or_normative_comparison",
        coded_meaning="minus_one_unseen_or_blind_point_not_negative_measurement",
        spatial_layout="source_index_only_no_anatomical_map",
    )


def paired_vector(first, last, left, right):
    """Same-source-index differences, never normative or progression scores."""
    import math

    from packages.grounding.models import digest

    a, b = left["values"], right["values"]
    if len(a) != 61 or len(b) != 61:
        raise ValueError("grape_paired_source_shape")
    comparable = [
        i
        for i, (x, y) in enumerate(zip(a, b, strict=True))
        if all(type(n) in {int, float} and math.isfinite(n) and n >= 0 for n in (x, y))
    ]
    differences = [b[i] - a[i] for i in comparable]
    return dict(
        semantic_profile=PROFILE,
        baseline_component=first.id,
        followup_component=last.id,
        unit=first.unit or "unqualified",
        source_value_checksums=[digest(left), digest(right)],
        aligned_indices=comparable,
        aligned_count=len(comparable),
        excluded_pair_count=61 - len(comparable),
        delta_range=[min(differences), max(differences)] if differences else None,
        delta_mean=sum(differences) / len(differences) if differences else None,
        identical_values=a == b,
        independent_acquisitions_proven=False,
        limits=[
            "source_index_only_no_anatomical_map",
            "minus_one_unseen_or_blind_and_invalid_values_excluded_not_filled",
            "raw_difference_not_TD_MD_or_clinical_progression",
        ],
    )


def tonopachy_device_unit_bridge(concept):
    """Original author/device method bridge, never a converted patient fact."""
    from pathlib import Path

    from packages.object_store import checksum

    if concept not in {"iop", "cct"}:
        raise ValueError("grape_tonopachy_unit_concept")
    path = Path("docs/sources/nidek_tonopachy_nt530p.pdf")
    expected = "sha256:2d8ee2b0280ecbd080ee7e7fa5bb3139a48c1d291005c4e876c4571e5d0674db"
    if path.is_symlink() or checksum(path.read_bytes()) != expected:
        raise ValueError("grape_tonopachy_manufacturer_source_binding")
    return dict(
        schema_version="dataset-device-unit-bridge/1.0.0",
        concept=concept,
        source_url="https://www.nidek.co.jp/wp-content/uploads/2024/05/TONOPACHY-NT-530P.pdf",
        source_checksum=expected,
        locator=dict(page_1_based=4, section="Specifications", device="TONOPACHY NT-530P"),
        exact_unit_text="mmHg" if concept == "iop" else "µm",
        proposed_unit="mmHg" if concept == "iop" else "um",
        relation_kind="documented_device_protocol_inference_not_release_field_assertion",
        inference=(
            "GRAPE author methods name NT-530P for the supplied IOP and CCT measurements. "
            "Manufacturer specifications identify IOP in mmHg "
            "and corneal thickness in micrometres. "
            "Assume the dataset retains untransformed device-scale values; the release itself "
            "does not explicitly qualify the field unit."
        ),
        source_measurement_changed=False,
        original_release_unit_qualification="release_unit_remains_unverified",
        clinical_qualification=False,
        iop_compensation_status="unknown_in_dataset_release",
        interpretation_limits=[
            "Device units do not establish individual measurement quality or disease stage.",
            "Manufacturer page 3 supports CCT-compensated IOP; GRAPE does not state "
            "which IOP output was retained. Do not select raw/compensated status or correct twice.",
            "GRAPE reports IOP medication use in follow-up, not an untreated individual baseline.",
            "No CCT normative threshold, individual treatment effect "
            "or disease direction inferred.",
            "Not an independent observation or an additional score; retain original numeric value.",
        ],
    )
