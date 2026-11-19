"""Bounded, strict trace records. A complete inventory is not complete evidence."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

FACETS = {
    "target": "判读目标与作用范围",
    "source": "医学来源与原文定位",
    "source_quality": "研究设计与证据质量",
    "reference": "检查参照、公式与单位",
    "tool": "工具身份与选择依据",
    "tool_evaluation": "工具评测与适用域",
    "input_match": "本例输入匹配",
    "preprocessing": "预处理与坐标变换",
    "execution": "执行或观察来源",
    "output_qc": "输出与质量检查",
    "uncertainty": "重复性、误差与敏感性",
    "interpretation": "比较与解释链",
    "dependence": "依赖、同源与重复引用",
    "missing_conflict": "缺项与冲突",
    "assessment_provenance": "整理者与规则版本",
}


def body(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return "sha256:" + hashlib.sha256(body(value)).hexdigest()


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Binding(Strict):
    document: Literal["base", "profiles"]
    pointer: str = Field(max_length=512, pattern=r"^/")
    checksum: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")


class Facet(Strict):
    status: Literal["bound", "partial", "missing", "not_applicable"]
    facts: list[Binding] = Field(max_length=64)
    explanation: str = Field(min_length=1, max_length=2500)
    missing: list[str] = Field(max_length=32)

    @model_validator(mode="after")
    def consistency(self):
        if self.status in {"missing", "partial"} and not self.missing:
            raise ValueError("detail_missing_reason_required")
        if self.status == "bound" and (not self.facts or self.missing):
            raise ValueError("detail_bound_requires_facts")
        return self


class Assessment(Strict):
    id: str = Field(min_length=1, max_length=128)
    node: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=512)
    kind: Literal["criterion", "descriptive", "association", "visual", "node_context"]
    facets: dict[str, Facet]
    same_image_group: str
    artifact_groups: list[str] = Field(max_length=64)
    input_checks: dict[str, JsonValue]
    comparison: dict[str, JsonValue]
    measurements: list[JsonValue] = Field(max_length=128)
    gap_codes: list[str] = Field(max_length=32)
    confidence_score: None = None

    @model_validator(mode="after")
    def complete_inventory(self):
        if set(self.facets) != set(FACETS):
            raise ValueError("detail_facet_inventory_mismatch")
        return self


class Ledger(Strict):
    schema_version: Literal["case-judgment-details/1.0.0"]
    case_code: str
    input_checksum: str
    base_checksum: str
    profiles_checksum: str
    builder_version: Literal["judgment-binding/1.0.0"]
    assessments: list[Assessment] = Field(min_length=1, max_length=512)
    node_index: dict[str, list[str]]
    relations: list[Binding] = Field(max_length=1024)
    coverage: dict[str, JsonValue]
    checksum: str

    @model_validator(mode="after")
    def seal(self):
        raw = self.model_dump(mode="json")
        if self.checksum != digest({k: v for k, v in raw.items() if k != "checksum"}):
            raise ValueError("detail_checksum_mismatch")
        ids = [a.id for a in self.assessments]
        if len(set(ids)) != len(ids) or any(
            len(set(items)) != len(items) or not set(items) <= set(ids)
            for items in self.node_index.values()
        ):
            raise ValueError("detail_duplicate_or_unknown_assessment")
        return self


def resolve(document, pointer):
    value = document
    for part in pointer.removeprefix("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def binding(documents, document, pointer):
    return Binding(
        document=document, pointer=pointer, checksum=digest(resolve(documents[document], pointer))
    )


def validate_bindings(ledger, documents):
    parsed = Ledger.model_validate(ledger)
    refs = [r for a in parsed.assessments for f in a.facets.values() for r in f.facts]
    for ref in [*refs, *parsed.relations]:
        if digest(resolve(documents[ref.document], ref.pointer)) != ref.checksum:
            raise ValueError("detail_fact_binding_mismatch")
    return parsed
