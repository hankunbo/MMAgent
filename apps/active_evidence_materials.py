"""First bounded active-acquisition pilot: exact sources -> targeted image -> review.

The packet is a derived research reference, not a new publication. Its two
original Knowledge documents, exact excerpts and task/member mapping remain
explicit. All old sources/tasks/coverage denominators remain in the material.
"""

import json
from pathlib import Path

from apps.active_evidence_sources import DESTINATION, verify
from apps.evidence_acquisition import requirements_for
from apps.vectorengine_case_materials import FirstCaseEvidence, ref
from apps.vectorengine_round_materials import RoundCaseEvidence, load_catalog
from packages.evidence_acquisition.materials import (
    TaskTargetedReader,
    _rebuild,
    prepare_definition,
    prepare_visual_task,
)
from packages.evidence_acquisition.planning import plan_acquisition
from packages.grounding.models import digest, validate
from packages.llm_provider.models import Source, TaskDefinition

MANIFEST = DESTINATION / "manifest.json"
PILOTS = ("criterion:VCDR_GE_07", "criterion:RIM_RADIAL_LT_01", "criterion:RIM_NOTCH")
RULES = {
    PILOTS[0]: dict(
        member="phene",
        examination="垂直杯盘比：读取同图工具测量值，与0.7参考线比较。",
        interpretation="Phene等对可转诊青光眼性视神经病变的研究：垂直杯盘比≥0.7是关联特征。依据本例数值给连续支持/反驳评分；低于0.7不等于排除青光眼。",
        applicability_conditions="原研究为彩色眼底照片和可转诊GON；本例SLO，迁移适用性需降权。结合盘大小/盘沿。不得把研究特征解释为个体确诊概率。",
    ),
    PILOTS[1]: dict(
        member="rim",
        examination="最薄径向盘沿宽度÷同轴视盘直径；读取工具测量方法与数值。",
        interpretation="EGS示例：最薄处径向盘沿宽度/同轴盘径<0.1提示青光眼性损害，尤其颞下盘沿；结合盘大小及对应视野。只对定义相符的归一化测量判读，不用杯像素面积代替。",
        applicability_conditions="EGS教育示例，不是普适确诊阈值；本例SLO及掩膜算法，需核测量同轴性。眼别/扇区及视野未提供应降低适用性，不凭全局比值虚构扇区或视野结果。",
    ),
    PILOTS[2]: dict(
        member="phene",
        examination="仅观察视盘盘沿是否存在局灶切迹/局灶变薄；描述部位、可见程度及图像限制。不要诊断，不要用图像质量结论代替形态观察。",
        interpretation="Phene研究中局灶盘沿切迹是与可转诊GON相关的特征。存在明确局灶缺损可支持；清晰可见而未见切迹可给弱反驳；不可见/无法评估保留缺失，不作阴性。",
        applicability_conditions="本例单张SLO，非立体彩照，眼别未知。VLM只能给图像观察；局灶切迹并非确诊条件，不以无切迹排除其他类型青光眼。",
    ),
}


def load_manifest(path=MANIFEST):
    raw = json.loads(Path(path).read_text())
    if raw["schema_version"] != "active-acquisition-manifest/1.0.0" or raw["rules"] != RULES:
        raise ValueError("active_manifest_rules")
    if raw["checksum"] != digest({k: v for k, v in raw.items() if k != "checksum"}):
        raise ValueError("active_manifest_checksum")
    for member in raw["members"]:
        if digest(json.loads(Path(member["archive"]).read_text())) != member["archive_checksum"]:
            raise ValueError("active_archive_changed")
    if {m["id"] for m in raw["members"]} != {"phene", "rim"}:
        raise ValueError("active_member_inventory")
    return raw


class ActiveCaseEvidence:
    targeted_acquisition = True

    def __init__(
        self, original, catalog, *, catalog_path, manifest_path=MANIFEST, short_sources=False
    ):
        self.base = RoundCaseEvidence(original, catalog, catalog_path=catalog_path)
        self.manifest_path = Path(manifest_path)
        self.manifest = load_manifest(manifest_path)
        self.policies = {}
        self.short_sources = short_sources
        self.alias_manifest = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def packet(self):
        from packages.evidence_details.accepted_read_scope import historical_source

        return historical_source(self)

    def _history_binding(self):
        from packages.evidence_details.accepted_read_scope import _callable_identity

        return (
            id(self.base),
            id(self.knowledge),
            id(self.knowledge_objects),
            id(getattr(self.knowledge, "_target", None)),
            getattr(self.knowledge, "_role", None),
            _callable_identity(self.knowledge.get),
            _callable_identity(getattr(self.knowledge, "get_many", None)),
            _callable_identity(self.knowledge_objects.read_bounded),
            _callable_identity(self._read_current),
            _callable_identity(verify),
            str(self.manifest_path),
            digest(self.manifest),
        )

    def _read_current(self):
        # First and final reads still use the original archive owner. This is
        # never a retained source qualification between public operations.
        if load_manifest(self.manifest_path) != self.manifest:
            raise ValueError("active_manifest_changed")
        pieces = []
        for member in self.manifest["members"]:
            entry = json.loads(Path(member["archive"]).read_text())
            verify(self.knowledge, self.knowledge_objects, entry)
            exact = {c["reference"]["code"]: c for c in entry["chunks"]}
            for excerpt in member["excerpts"]:
                chunk = exact[excerpt["reference"]["code"]]
                a, b = excerpt["span"]
                if (
                    chunk["reference"] != excerpt["reference"]
                    or chunk["text"][a:b] != excerpt["text"]
                ):
                    raise ValueError("active_packet_exact_passage")
                pieces.append(
                    "MEMBER " + member["id"] + " | " + entry["url"] + "\n" + excerpt["text"]
                )
        text = "\n\n".join(pieces)
        return Source(
            reference=ref("source", "SOURCE:ACTIVE_RESEARCH_PACKET:ONH", digest(self.manifest)),
            locator=(
                "Derived packet: PMID31561879 (vCDR/notch) + EGS 2011-06-30 (radial rim). "
                "Exact original archive refs in manifest."
            ),
            purpose="research_material",
            state="research_only",
            text=text,
        )

    def read_source_ref(self, reference):
        packet = self.packet()
        if packet.reference != reference:
            raise ValueError("active_packet_reference")
        return packet

    def definition(self, target):
        rule = RULES[target]
        body = dict(
            examination=rule["examination"],
            applicability_conditions=rule["applicability_conditions"],
            interpretation=rule["interpretation"],
            interpretation_status="installed",
            source_refs=[self.packet().reference.model_dump(mode="json")],
        )
        return validate(
            TaskDefinition,
            dict(
                **body,
                reference=ref("definition", "POLICY:ACTIVE:" + target, digest(body)).model_dump(
                    mode="json"
                ),
            ),
        )

    def read_definition(self, target, reference):
        result = self.definition(target)
        if result.reference != reference:
            raise ValueError("active_definition_binding")
        return result

    def read_sources(self):
        sources = self.base.read_sources() + (self.packet(),)
        if self.alias_manifest is not None:
            from packages.evidence_acquisition.source_aliases import alias_sources

            sources, manifest = alias_sources(sources)
            if manifest != self.alias_manifest:
                raise ValueError("active_source_alias_binding_changed")
        return sources

    def project(self, *, run_id, config):
        self.alias_manifest = None
        state, manifest = self.base.project(run_id=run_id, config=config)
        before = plan_acquisition(state, requirements_for(state, self.base.catalog))
        for target in PILOTS:
            state = prepare_definition(
                state,
                target,
                definition_reference=self.definition(target).reference,
                definition_owner=self,
                source_owner=self,
            )
        # Read the actual original numeric-method record, never its old verdict.
        radial = self.base.view["research_criteria"]["radial_rim_measurement"]
        if (
            radial["input_checksum"] != image_checksum(state)
            or radial["definition"]
            != "(disc_ray_radius-cup_ray_radius)/(disc_ray_radius+opposite_disc_ray_radius)"
        ):
            raise ValueError("active_radial_method_binding")
        method_context = {
            PILOTS[0]: " 方法：同图视杯垂直包围盒高度/视盘垂直包围盒高度，无量纲。",
            PILOTS[1]: (
                " 方法：视盘质心射线；(盘半径-杯半径)/(盘半径+反向盘半径)，"
                "360方向、1°步长、径向0.25px；最小值仅图像坐标，不赋解剖象限。"
            ),
        }
        items = []
        for item in state.policy.items:
            if item.id in method_context:
                raw = item.definition.model_dump(mode="json", exclude={"reference"})
                raw["examination"] += method_context[item.id]
                raw["reference"] = ref(
                    "definition", "POLICY:ACTIVE_METHOD:" + item.id, digest(raw)
                ).model_dump(mode="json")
                item = item.model_copy(update={"definition": validate(TaskDefinition, raw)})
            items.append(item)
        state = _rebuild(state, items=tuple(items))
        image = next(o.image for o in state.material.observations if o.image is not None)
        state = prepare_visual_task(
            state, PILOTS[2], image=image, instruction=RULES[PILOTS[2]]["examination"]
        )
        state = _rebuild(state, sources=self.read_sources() + (self.fixed_source,))
        if self.short_sources:
            from packages.evidence_acquisition.source_aliases import prepare_source_aliases

            state, self.alias_manifest = prepare_source_aliases(state)
            manifest["source_aliases"] = self.alias_manifest
        self.policies[run_id] = digest(state.policy)
        manifest.update(
            active_acquisition=self.manifest,
            initial_gap_plan=before.model_dump(mode="json"),
            acquired_definition_tasks=PILOTS,
            vision_acquisition_slots=[PILOTS[2]],
            old_task_count=len(state.policy.items),
            base_source_slots=manifest["sources"],
            sources=len(state.material.sources),
            source_packet_is_not_a_publication=True,
            historical_tool_measurements_reused_not_rerun=True,
            radial_method_provenance={
                k: radial[k]
                for k in (
                    "schema",
                    "definition",
                    "method",
                    "input_checksum",
                    "evidence_refs",
                    "direction",
                    "angle_step_degrees",
                    "radial_step_px",
                    "precision",
                )
            },
        )
        return state, manifest

    def verify(self, state):
        if load_catalog(self.base.catalog_path) != self.base.catalog:
            raise ValueError("active_base_catalog_changed")
        if self.policies.get(state.material.scope.run_id) != digest(state.policy):
            raise ValueError("active_installed_policy_changed")
        FirstCaseEvidence.verify(self, state)

    def requirements(self, state):
        return tuple(
            r.model_copy(update={"missing_source_ids": ()}) if r.assessment_id in PILOTS else r
            for r in requirements_for(state, self.base.catalog)
        )


class ActiveMaterialReader(TaskTargetedReader):
    def __init__(self, owner, evidence, documents):
        super().__init__(owner)
        self.evidence, self.documents = evidence, documents

    @property
    def role(self):
        return self.owner.role

    @role.setter
    def role(self, value):
        self.owner.role = value

    def acquisition_checkpoint(self, stage):
        saved = self.documents.read("active-acquisition:" + stage)
        if saved is not None:
            # Exact historical checkpoint only. Original owner still authorizes
            # every subsequent operation; an old plan never grants a new call.
            return saved[0]
        state = self.owner.current()
        self.documents.write("active-acquisition:source-binding", (self.evidence.manifest,))
        aliases = getattr(self.evidence, "alias_manifest", None)
        if aliases is not None:
            self.documents.write("active-acquisition:source-aliases", (aliases,))
        plan = plan_acquisition(state, self.evidence.requirements(state))
        record = dict(
            schema_version="active-acquisition-checkpoint/1.0.0",
            stage=stage,
            manifest_checksum=digest(self.evidence.manifest),
            material_checksum=digest(state.material),
            plan=plan.model_dump(mode="json"),
            pilot_tasks=PILOTS,
            continuation="remaining_missing_inputs_not_invented; bounded_pilot_only",
        )
        self.documents.write("active-acquisition:" + stage, (record,))
        return record


def image_checksum(state):
    return next(o.image.reference.checksum for o in state.material.observations if o.image)
