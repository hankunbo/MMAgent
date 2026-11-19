"""Opt-in v2 projection of existing definitions and exact archived passages.

This reader never retrieves remote literature, writes Knowledge, or runs a tool.
Observation instructions and missing interpretation references remain distinct.
"""

import json
from pathlib import Path

from apps.vectorengine_case_materials import BASELINE, FirstCaseEvidence, ref
from db.knowledge.models import KnowledgeReference
from packages.grounding.models import digest
from packages.llm_provider.models import Source
from packages.object_store import checksum

CATALOG = Path("configs/research/vectorengine_round2_materials.json")
ASSOCIATIONS = Path("docs/experiments/glaucoma-associations-v1/sources.json")


def load_catalog(path=CATALOG):
    value = json.loads(Path(path).read_text())
    if value["schema_version"] != "first-case-round-material-catalog/1.0.0":
        raise ValueError("round_material_version")
    for file, key in (
        (BASELINE, "baseline_checksum"),
        (Path("apps/glaucoma_associations.py"), "definition_owner_checksum"),
        (ASSOCIATIONS, "association_archive_checksum"),
    ):
        if checksum(file.read_bytes()) != value[key]:
            raise ValueError("round_definition_owner_changed")
    if len({s["reference"]["code"] for s in value["sources"]}) != len(value["sources"]):
        raise ValueError("round_duplicate_source")
    for source in value["sources"]:
        if checksum(source["text"].encode()) != source["text_checksum"]:
            raise ValueError("round_excerpt_checksum")
    return value


def extract_passage(raw, selector):
    text = raw.decode("utf-8")
    if selector["kind"] == "text":
        a, b = selector["start"], selector["end"]
        if not 0 <= a < b <= len(text):
            raise ValueError("round_excerpt_span")
        return text[a:b]
    if selector["kind"] == "json_blocks":
        blocks = json.loads(text)["blocks"]
        return "\n".join(blocks[i]["text"] for i in selector["indices"])
    raise ValueError("round_excerpt_selector")


def definitions(catalog):
    result = {}
    for identifier, row in catalog["tasks"].items():
        original = row["definition"] or {}
        instruction = original.get("condition", original.get("demand")) or row.get(
            "observation_instruction"
        )
        interpretation = original.get(
            "interpretation_rule", original.get("direction", original.get("condition"))
        )
        limits = original.get("gap", original.get("operationalization", ""))
        if original.get("gap"):
            limits = "原定义版本的限制/当时缺项（本轮观察状态以observations为准）：" + limits
        result[identifier] = dict(
            sources=[dict(id=s) for s in row["selected_source_ids"]],
            examination=instruction,
            interpretation_rule=interpretation,
            interpretation_installed=row["interpretation_installed"],
            applicability_conditions=(
                "单张SLO研究；眼别UNKNOWN；无临床结论。"
                + (
                    "人群关联不能直接套用个体阈值。"
                    if identifier.startswith("association:")
                    else ""
                )
                + limits
                + (
                    "仅观察映射，未安装独立疾病判读参照。"
                    if identifier.startswith("visual:")
                    else ""
                )
            ),
        )
    return result


class RoundCaseEvidence:
    """Reuse original visible/image/Runtime owner, replacing only the material projection."""

    def __init__(self, original, catalog, *, catalog_path=CATALOG):
        self.original, self.catalog = original, catalog
        self.catalog_path = Path(catalog_path)
        self.definition_source = Source(
            reference=ref(
                "source", "POLICY:ROUND_DEFINITIONS", digest(catalog), catalog["version"]
            ),
            locator="Project task definitions; exact owners/digests in round material catalog",
            purpose="research_material",
            state="research_only",
            text=(
                "Project observation/measurement/quality instructions only. This reference is not "
                "medical literature or a disease cutoff. Tasks without an installed interpretation "
                "must preserve missing_reference; no unrelated default literature is substituted."
            ),
        )
        self.chunks = [dict(s, source_id=s["id"]) for s in catalog["sources"]] + [
            dict(source_id="PROJECT_DEFINITION_ONLY", project_definition=True)
        ]
        # Construction may be followed by original-owner catalog specialization.
        # Do not memoize this preliminary inventory as the final reader identity.
        self.sources = self._read_current() + (self.fixed_source,)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def read_source(self, passage):
        return self._read_source(passage, self.knowledge.get)

    def read_sources(self):
        from packages.evidence_details.accepted_read_scope import historical_source

        return historical_source(self)

    def _history_binding(self):
        from packages.evidence_details.accepted_read_scope import _callable_identity

        return (
            id(self.knowledge),
            id(self.knowledge_objects),
            id(getattr(self.knowledge, "_target", None)),
            getattr(self.knowledge, "_role", None),
            _callable_identity(self.knowledge.get),
            _callable_identity(getattr(self.knowledge, "get_many", None)),
            _callable_identity(self.knowledge_objects.read_bounded),
            _callable_identity(self._read_source),
            _callable_identity(self._read_current),
            digest(self.chunks),
            digest(self.definition_source),
        )

    def _read_current(self):
        # Historical reuse is bounded by the original fact scope. Outside it,
        # and at its return boundary, these remain fresh lifecycle/object reads.
        if not hasattr(self.knowledge, "get_many"):
            return tuple(self.read_source(c) for c in self.chunks)
        # Scope is this call only. Every verification reads all current lifecycle
        # rows again, and reads/checks every original object byte below.
        snapshots = {}

        def fetch(references):
            unique = {digest(r): r for r in references}
            pending = {key: r for key, r in unique.items() if key not in snapshots}
            if pending:
                values = self.knowledge.get_many(tuple(pending.values()))
                snapshots.update(zip(pending, values, strict=True))

        refs = tuple(
            KnowledgeReference.model_validate_json(json.dumps(c["reference"]))
            for c in self.chunks
            if not c.get("project_definition")
        )
        fetch(refs)
        children = [snapshots[digest(r)].record for r in refs]
        fetch(
            [
                parent
                for child in children
                for parent in (child.source_version_ref,)
                + ((child.document_ref,) if hasattr(child, "document_ref") else ())
            ]
        )
        fetch([snapshots[digest(c.source_version_ref)].record.source_ref for c in children])
        return tuple(self._read_source(c, lambda r: snapshots[digest(r)]) for c in self.chunks)

    def _read_source(self, passage, get):
        if passage.get("project_definition"):
            return self.definition_source
        r = KnowledgeReference.model_validate_json(json.dumps(passage["reference"]))
        current = get(r)
        record = current.record
        parents = [current, get(record.source_version_ref)]
        if hasattr(record, "document_ref"):
            parents.append(get(record.document_ref))
        parents.append(get(parents[1].record.source_ref))
        if any(p.withdrawn or p.status.value == "deprecated" for p in parents):
            raise ValueError("source_withdrawn")
        raw = self.knowledge_objects.read_bounded(record.object_uri, 4194304)
        if checksum(raw) != record.content_checksum or len(raw) != record.size_bytes:
            raise ValueError("source_object_changed")
        text = extract_passage(raw, passage["selector"])
        if text != passage["text"] or checksum(text.encode()) != passage["text_checksum"]:
            raise ValueError("round_exact_passage_changed")
        return Source(
            reference=ref("source", r.code, r.checksum, r.version),
            locator=passage["locator"],
            purpose="research_material",
            state="research_only",
            text=text,
        )

    def project(self, *, run_id, config):
        state, manifest = FirstCaseEvidence.project(
            self, run_id=run_id, config=config, definitions=definitions(self.catalog)
        )
        manifest.update(
            projection_version=self.catalog["version"],
            definition_catalog_checksum=digest(self.catalog),
            exact_passages=[
                {k: s[k] for k in ("id", "reference", "selector", "text_checksum", "locator")}
                for s in self.catalog["sources"]
            ],
            vision_acquisition_slots=["visual:quality"],
            other_visual_slots="requires_task_targeted_interface_no_quality_fanout",
            current_source_qualification="original_Knowledge_get_and_object_bytes",
        )
        return state, manifest

    def verify(self, state):
        if load_catalog(self.catalog_path) != self.catalog:
            raise ValueError("round_definition_catalog_changed")
        FirstCaseEvidence.verify(self, state)
