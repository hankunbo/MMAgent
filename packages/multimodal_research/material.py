"""Original Knowledge and input owners feed task-specific, dynamic review tasks."""

import json
import math
from types import SimpleNamespace

from apps.vectorengine_case_materials import ref
from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV2, TaskPolicyV5
from packages.evidence_scoring.continuous_policy import POLICY as LEGACY_POLICY
from packages.evidence_scoring.task_policy import POLICY, SCOPES
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import Material, ObservationInput, TaskDefinition

RULES = {
    "age": (
        "年龄背景",
        ("NEI-GLAUCOMA-RISK",),
        "年龄是风险背景，不是视神经损害或本次变化的独立证明。",
        "risk",
    ),
    "md": (
        "视野总体偏差",
        ("aao_context",),
        "按提供的MD定义评估功能损害线索；非特异性损害需保留，不凭MD单独确诊。",
        "diagnostic",
    ),
    "tds": (
        "视野偏差序列",
        ("aao_context",),
        "按原52点偏差序列评估功能线索；未绑定空间布局，不能发明缺损位置。",
        "diagnostic",
    ),
    "vf": (
        "视野敏感度序列",
        ("aao_context",),
        "这是原始敏感度，不是TD；无年龄正常库不制造概率图。已取得数值仍可作有局限的功能评价。",
        "diagnostic",
    ),
    "iop": (
        "眼压背景",
        ("NICE-NG81-REFERRAL",),
        "眼压需结合结构和功能评估，不能单凭眼压确诊或排除。未确认单位和方法时不套阈值。",
        "risk",
    ),
    "cct": ("角膜厚度背景", ("NICE-NG81-REFERRAL",), None, "measurement"),
    "rnfl": (
        "RNFL结构观察",
        ("aao_signs", "EGS-OCT-2020"),
        "RNFL结构为损害线索；缺设备匹配正常库、单位或分割质控时降低适用性，不制造正常百分位或确诊。",
        "diagnostic",
    ),
    "fundus": (
        "眼底视盘与盘沿观察",
        ("aao_signs",),
        "局灶盘沿变薄/切迹、杯盘形态为结构线索；结合盘大小和可见范围，不用固定杯盘比确诊。",
        "visual",
    ),
    "oct": (
        "OCT显示视野内结构观察",
        ("aao_signs", "EGS-OCT-2020"),
        "仅评价实际展示切片可见的结构；OCT不能单独确诊。无物理尺度/层分割时不输出微米厚度。",
        "visual",
    ),
}


def category(c):
    if c.concept.startswith("rnfl"):
        return "rnfl"
    if c.modality in {"color_fundus", "SLO"}:
        return "fundus"
    if c.modality == "OCT":
        return "oct"
    return c.concept if c.concept in RULES else None


class MedicalSources:
    """Exact existing archived passages with current ancestor/withdrawal checks."""

    def __init__(self):
        from apps.knowledge_source_tools import repository
        from apps.patient_service_assembly import OBJECT_ROOTS
        from apps.vectorengine_round_materials import RoundCaseEvidence
        from packages.evaluation_preparation.unified_requirements import catalog
        from packages.llm_provider.models import Source
        from packages.object_store import LocalObjectStore

        self.catalog = catalog()
        selected = {k for _, names, _, _ in RULES.values() for k in names}
        subset = dict(
            self.catalog, sources=[s for s in self.catalog["sources"] if s["id"] in selected]
        )
        placeholder = Source(
            reference=ref("source", "POLICY:NOT_MEDICAL", digest("unused")),
            locator="unused",
            purpose="research_material",
            state="research_only",
            text="Not a medical reference.",
        )
        self.reader = RoundCaseEvidence(
            SimpleNamespace(
                # Original research-reader composition uses editor visibility for
                # unpublished archived passages, never Published Planner authority.
                knowledge=repository("editor"),
                knowledge_objects=LocalObjectStore(OBJECT_ROOTS["knowledge"]),
                fixed_source=placeholder,
            ),
            subset,
        )
        self.reader.chunks = [c for c in self.reader.chunks if not c.get("project_definition")]
        self.initial = self.read()

    def read(self):
        from packages.evidence_details.accepted_read_scope import historical_source

        return historical_source(self)

    def _history_binding(self):
        # Keep exact owner/method identity; never fingerprint credentials or DSNs.
        from packages.evidence_details.accepted_read_scope import _callable_identity

        reader = self.reader
        return (
            id(reader),
            id(reader.knowledge),
            id(reader.knowledge_objects),
            id(getattr(reader.knowledge, "_target", None)),
            getattr(reader.knowledge, "_role", None),
            digest(self.catalog),
            digest(reader.chunks),
            _callable_identity(self._read_current),
            _callable_identity(reader.read_sources),
            _callable_identity(reader.knowledge.get),
            _callable_identity(getattr(reader.knowledge, "get_many", None)),
            _callable_identity(reader.knowledge_objects.read_bounded),
        )

    def _read_current(self):
        from packages.evaluation_preparation.unified_requirements import catalog

        if catalog() != self.catalog:
            raise ValueError("multimodal_medical_catalog_changed")
        return {
            c["id"]: s for c, s in zip(self.reader.chunks, self.reader.read_sources(), strict=True)
        }


def definition(identifier, name, interpretation, sources, *, change=False):
    if change and interpretation:
        interpretation = (
            "本项只有单时点观察，不能单独证明变化；对当前变化目标仅作背景，e=0。必须有同眼配对结果才评价变化；不得用年龄或已有病变推断发生进展。"
            + interpretation
        )
    body = dict(
        examination=name[:240],
        applicability_conditions="研究用；单位、方法、采样范围和缺项以本例投影为准；不能用标签或未观察数据。",
        interpretation=interpretation,
        interpretation_status="installed" if interpretation else "missing",
        source_refs=tuple(s.reference for s in sources),
    )
    return TaskDefinition(
        reference=ref("definition", "POLICY:MM:" + digest(identifier)[7:23], digest(body)), **body
    )


def numeric(c, payloads):
    """Only decoded approved payload values; never a label or patient-note container."""
    if c.status not in {"obtained", "partial"}:
        return None
    basis = getattr(c, "unit_basis", None)
    if "semantic_profile:grape-v1" in c.semantic_limits:
        from packages.dataset_adapters.multimodal.grape_semantics import numeric as qualified

        return qualified(c, payloads)
    if basis and (
        "2308.13411#section-3.1" in basis
        or "3d643726b80f32b36ea59b1b0f8881c8614bb2c2/README.md#age" in basis
    ):
        from packages.dataset_adapters.multimodal.harvard_semantics import numeric as qualified

        return qualified(c, payloads)
    value = dict(
        concept=c.concept,
        unit=c.unit or "source_units_unqualified",
        visit=c.visit,
        limits=list(c.semantic_limits),
        source_locator=c.source_locator,
    )
    if c.kind == "scalar":
        value["value"] = c.value
    elif c.kind in {"vector", "map"}:
        raw = payloads[c.payload_ids[0]]
        if raw.startswith(b"\x93NUMPY"):
            from packages.dataset_adapters.multimodal.formats import npy

            numbers = list(npy(raw).numbers())
        else:
            decoded = json.loads(raw)
            numbers = decoded["values"] if isinstance(decoded, dict) else decoded
        valid = [
            float(n)
            for n in numbers
            if n is not None and type(n) in {int, float} and math.isfinite(n)
        ]
        if not valid:
            return None
        value.update(
            count=len(numbers),
            finite_count=len(valid),
            minimum=min(valid),
            maximum=max(valid),
            mean=sum(valid) / len(valid),
            axes=c.axis_semantics,
        )
        if c.kind == "vector":
            value["values"] = [round(float(n), 3) if n is not None else None for n in numbers]
        else:
            value["meaning"] = "raw_map_summary_not_normative_percentile_or_segmented_measurement"
    else:
        return None
    if len(canonical(value).decode()) > 1000:
        value.pop("values", None)
        value["full_values"] = "original_visible_payload_by_checksum"
    return value


def observation(
    identifier,
    value,
    *,
    method="source_measurement_projection",
    unit="source_units_unqualified",
    image=None,
):
    body = dict(
        status="obtained", method=method, unit=unit, value=canonical(value).decode(), image=image
    )
    return ObservationInput(
        reference=ref("observation", "OBS:MM:" + digest(identifier)[7:19], digest(body)), **body
    )


def requirements(manifest, medical, task, *, directional=False):
    result = []
    for c in manifest.components:
        cat = category(c)
        if cat is None or c.modality not in task.allowed_modalities or c.status == "withheld":
            continue
        name, source_ids, interpretation, role = RULES[cat]
        sources = tuple(medical[k] for k in source_ids)
        identifier = "check:" + c.id
        result.append(
            dict(
                id=identifier,
                component=c.id,
                category=cat,
                role=role,
                definition=definition(
                    identifier,
                    name + " · " + c.id,
                    interpretation,
                    sources,
                    change=task.output_kind == "observed_change",
                ),
                input_modalities=(c.modality,),
                basis_kind="qualitative",
                missing_reason="not_yet_observed"
                if c.kind in {"image", "volume", "image_series"}
                else c.reason,
            )
        )
    from .hypotheses import expand_requirements
    from .single_proposition import bind_requirements

    return bind_requirements(
        expand_requirements(result, medical, task), task, directional=directional
    )


def paired_numeric(manifest, payloads, medical):
    """Compare only declared same-concept/same-eye arrays in the visible cutoff.

    Raw change is not a clinical progression label; sentinel interpretation and
    unqualified units remain unresolved, and identical payloads are not new tests.
    """
    ledger, observations = [], {}
    groups = {}
    for c in manifest.components:
        if (
            category(c) in {"iop", "vf", "rnfl", "md", "tds"}
            and c.kind in {"scalar", "vector"}
            and c.status in {"obtained", "partial"}
        ):
            groups.setdefault((category(c), c.concept), []).append(c)
    for (cat, concept), components in groups.items():
        if len(components) < 2:
            continue
        first, last = components[0], components[-1]
        if (
            first.visit == last.visit
            or first.laterality != last.laterality
            or first.unit != last.unit
        ):
            continue
        left, right = numeric(first, payloads), numeric(last, payloads)
        if left is None or right is None:
            continue
        a, b = left.get("values", [left.get("value")]), right.get("values", [right.get("value")])
        comparable = len(a) == len(b) and all(
            type(x) in {float, int} and math.isfinite(x) for x in a + b
        )
        values = dict(
            baseline=left,
            followup=right,
            same_eye=True,
            matching_source_order=comparable,
            delta=[y - x for x, y in zip(a, b, strict=True)] if comparable else None,
            independent_acquisitions_proven=False,
            limits=[
                "raw_difference_not_clinical_progression",
                "negative_values_may_be_sentinel_no_normative_binding",
            ],
        )
        if len(canonical(values).decode()) > 1000:
            differences = values["delta"]
            values = dict(
                baseline_component=first.id,
                followup_component=last.id,
                unit=first.unit or "unqualified",
                source_value_checksums=[digest(left), digest(right)],
                aligned_count=len(a) if comparable else 0,
                delta_range=[min(differences), max(differences)] if differences else None,
                delta_mean=sum(differences) / len(differences) if differences else None,
                identical_values=a == b,
                independent_acquisitions_proven=False,
                limits=values["limits"],
            )
        if (
            concept == "vf"
            and left.get("semantic_profile") == "grape-v1"
            and right.get("semantic_profile") == "grape-v1"
        ):
            from packages.dataset_adapters.multimodal.grape_semantics import paired_vector

            values = paired_vector(first, last, left, right)
        identifier = "comparison:" + concept
        names = RULES[cat][1]
        d = definition(
            identifier,
            "同眼配对变化 · " + concept,
            "只评价已给出的配对差异；相同数值/重复影像不支持发生变化，e可为0。眼压变化不是结构或功能进展。没有单位/正常库/测试重复性，不声称临床进展或未来风险。",
            tuple(medical[k] for k in names),
        )
        ledger.append(
            dict(
                id=identifier,
                component=first.id + ":" + last.id,
                category=cat,
                role="diagnostic",
                definition=d,
                input_modalities=(first.modality,),
                basis_kind="qualitative",
                missing_reason=None,
            )
        )
        observations[identifier] = observation(
            identifier, values, method="same_eye_source_order_numeric_comparison"
        )
    return ledger, observations


def requirement_source_refs(ledger):
    """Exact source closure of definitions retained in policy requirements/items.

    Missing-input requirements remain explicit Review context and denominator;
    their definitions still require sources. A retrieved paragraph not referenced
    by any retained definition is history, not an implicit Review dependency.
    """
    references = {}
    for row in ledger:
        definition_ = row["definition"]
        if not isinstance(definition_, TaskDefinition):
            definition_ = validate(TaskDefinition, definition_)
        for reference in definition_.source_refs:
            if reference.code in references and references[reference.code] != reference:
                raise ValueError("multimodal_requirement_source_version_conflict")
            references[reference.code] = reference
    return references


def requirement_sources(medical, ledger):
    references = requirement_source_refs(ledger)
    available = {}
    for source in medical.values():
        code = source.reference.code
        if code in available and available[code] != source:
            raise ValueError("multimodal_requirement_source_content_conflict")
        available[code] = source
    if any(
        code not in available or available[code].reference != ref
        for code, ref in references.items()
    ):
        raise ValueError("multimodal_requirement_source_missing_or_changed")
    return {code: available[code] for code in references}


def state(
    *,
    context,
    visible,
    root,
    config,
    medical,
    ledger,
    observations,
    resources,
    acquisition=None,
    supplementary_observations=(),
):
    """Acquisition requests use original v2. Only obtained evidence enters v5 scores."""
    sources = tuple(dict((s.reference.code, s) for s in medical.values()).values())
    target = context.request.task.target_code
    summary = {
        "binary_evidence": "本例青光眼相关证据的支持与反驳（非患病概率）",
        "ordinal_evidence": "本例青光眼结构功能损害线索（本方向分不等于GAMMA三分类概率或分级）",
        "observed_change": "截至已绑定访视的同眼不利变化证据（非未来进展预测）",
        "evidence_review": "本例来源限定的证据评阅",
    }[context.request.task.output_kind]
    from .single_proposition import meaning

    directional = bool(getattr(config, "direction_policy", None))
    summary = meaning(context.request.task, directional=directional) or summary
    requirements_ = [
        {
            k: v
            for k, v in r.items()
            if k in {"id", "definition", "input_modalities", "basis_kind", "missing_reason"}
        }
        for r in ledger
    ]
    items = []
    selected = []
    for r in ledger:
        o = observations.get(r["id"])
        if acquisition is not None and r["id"] != acquisition:
            continue
        if o is None or (acquisition is None and o.status != "obtained"):
            continue
        if acquisition is None and not r.get("review_allowed", True):
            # This is an owner-reconstructed row. Observation persists separately;
            # missing interpretation is also enforced by the original acceptor.
            continue
        selected.append(o)
        paths = r.get("dependency_paths", ((resources[0].code,),))
        q = r.get("quality", 1.0)
        if acquisition is None and o.value and "partial-visual-observation/1.0.0" in o.value:
            from packages.evaluation_preparation.partial_observation import evaluate

            q = evaluate(o)["effective_quality"]
        item = dict(
            id=r["id"],
            node="node:" + r["component"],
            kind="criterion",
            role=r["role"],
            observation_code=o.reference.code,
            method=o.method,
            unit=o.unit,
            laterality=visible.case.laterality,
            source_codes=tuple(s.code for s in r["definition"].source_refs),
            applicable=True,
            correlation_group=r["category"],
            dependence_block="modality:" + r["input_modalities"][0],
            dependency_paths=paths,
            definition=r["definition"],
        )
        if acquisition is None:
            item["quality"] = dict(
                value=q,
                method="partial_visibility_or_source_measurement",
                reason="Q仅用于已观察内容；原单位/比较参照不足另由A与e体现。",
                source=r.get("quality_source", resources[0]),
            )
        items.append(item)
    if acquisition is not None:
        if len(items) != 1:
            raise ValueError("multimodal_acquisition_one_task")
        policy = TaskPolicyV2(
            version="multimodal-acquisition-1.0.0",
            scale=ref("scale", "POLICY:REVIEW_SCALE", digest(LEGACY_POLICY)),
            aggregation_checksum=digest(LEGACY_POLICY),
            items=tuple(items),
            groups=(
                dict(
                    id="acquire",
                    slots=((acquisition,),),
                    importance=dict(slo_image_research=0.0, comprehensive_research=1.0),
                ),
            ),
        )
    else:
        for r in requirements_:
            if r["id"] in {i["id"] for i in items}:
                r["missing_reason"] = None
            elif r["id"] in observations:
                r["missing_reason"] = "observation_" + observations[r["id"]].status
        from packages.evidence_details.acceptance_models import TaskPolicyV7

        policy = (TaskPolicyV7 if directional else TaskPolicyV5)(
            version="task-relative-quality-1.0.0",
            target_code=target,
            target_summary=summary,
            scale=ref("scale", "POLICY:REVIEW_SCALE", digest(LEGACY_POLICY)),
            aggregation_checksum=digest(POLICY),
            items=tuple(items),
            requirements=tuple(requirements_),
            groups=tuple(
                dict(id=r["id"], slots=((r["id"],),), importance=dict.fromkeys(SCOPES, 1.0))
                for r in requirements_
            ),
        )
        from .hypotheses import enabled, expand_policy

        if enabled(context.request.task):
            policy = expand_policy(policy)
            summary = "无、早、中、晚四个研究命题分别评阅；共享证据，不是四分类概率或临床分级。"
            policy = policy.model_copy(update=dict(target_summary=summary))
    scope = dict(
        case_code=visible.case.case_code,
        run_id=context.run_id,
        graph=dict(
            role="graph",
            code=root.asset_code,
            version="multimodal-input-object/1.0.0",
            checksum=root.checksum,
            kind="research_artifact",
            parent=None,
        ),
        visible_checksum=visible.checksum,
        input_checksum=visible.components.checksum,
        policy_checksum=digest(policy),
    )
    material = dict(
        scope=scope,
        goal_code=target,
        goal_summary=summary,
        visibility="approved_visible_projection/1.0.0",
        sources=sources,
        observations=(*selected, *supplementary_observations),
        policy_ref=ref("policy", "POLICY:MM_TASK", digest(policy), policy.schema_version),
        missing=(),
        memory=(),
    )
    scope["material_checksum"] = digest(material)
    return DomainState(
        material=validate(Material, material),
        policy=policy,
        laterality=visible.case.laterality,
        resources=resources,
        prompt_checksum=next(
            r.prompt_checksum
            for r in config.roles
            if r.role == ("vision" if acquisition else "review")
        ),
        config_checksum=digest(config),
        nodes=tuple(dict.fromkeys([target] + [i.node for i in policy.items])),
    )
