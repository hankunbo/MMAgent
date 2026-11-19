"""First-case visible projection through the existing Runtime and Knowledge owners.

No dataset iterator, NPZ, evaluation store, original note, or old visual interpretation
is a dependency. Historical numeric tools are reused; no executable is installed here.
"""

import json
from pathlib import Path

from db.knowledge.models import KnowledgeReference
from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV2
from packages.evidence_scoring.continuous_policy import POLICY
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import ImageBinding, Material, Source
from packages.object_store import checksum
from packages.planner_control.models import Graph, Reference

CASE = "CASE:ANON_85A135B787D34A23861AB252ADC26150"
GRAPH = "ASSET:OID_88EE6F9E0126FE7059566DC8FD14EBC3E0FC0C19295F2070"
IMAGE = "sha256:2ba00b68ec05340ae68a7b475786bff05c2153344cd9ab67ae88e58c6b42c9cd"
BASELINE = Path("docs/exports/青光眼证据图.json")
BASELINE_SHA = "sha256:42bca80c33fedfab0f57a9c6c9b3b0e9e42f2e23adc76d4a5126f5c32d5b83e2"


def ref(role, code, sha, version="1.0.0"):
    return Reference(role=role, code=code, checksum=sha, version=version)


class FirstCaseEvidence:
    def __init__(self, *, runtime, visible, objects, knowledge, knowledge_objects):
        self.runtime, self.visible, self.objects = runtime, visible, objects
        self.knowledge, self.knowledge_objects = knowledge, knowledge_objects
        raw = BASELINE.read_bytes()
        if checksum(raw) != BASELINE_SHA:
            raise ValueError("first_case_baseline_changed")
        self.view = json.loads(raw)
        self.visible_value = visible.read(CASE)
        images = self.visible_value.case.artifacts
        if len(images) != 1 or images[0].checksum != IMAGE:
            raise ValueError("fixed_first_image_required")
        self.image_artifact = images[0]
        self.root = next(
            a for a in runtime.get_case_bundle(CASE).artifacts if a.asset_code == GRAPH
        )
        self.root_bytes = self.objects.read_bounded(self.root.uri, 4194304)
        if checksum(self.root_bytes) != self.root.checksum:
            raise ValueError("first_case_graph_bytes")
        # Exact registered graph is the authority for the saved export, not a fixture reader.
        from apps.glaucoma_case_storage import read_view

        if read_view(CASE, GRAPH, repo=runtime, store=objects) != self.view:
            raise ValueError("first_case_runtime_export_mismatch")
        self.chunks = [
            dict(c, source_id="NICE-NG81")
            for i, c in enumerate(self.view["source_chunks"])
            if i in (3, 8, 11)
        ]
        for page in self.view["egs_pages"]:
            # Fixed source/document order; selection uses no patient labels or prior score.
            content = [c for c in page["chunks"] if 120 <= len(c["text"]) <= 4000]
            self.chunks.extend(dict(c, source_id=page["id"]) for c in content[:2])
        self.fixed_source = Source(
            reference=ref(
                "source", self.root.asset_code, self.root.checksum, "case-evidence-graph/1.19.0"
            ),
            locator=(
                "/case_evidence/fairclip/outputs; original domain fixed mapping, historical only"
            ),
            purpose="research_material",
            state="research_only",
            text=(
                "Historical fixed FairCLIP observation and deterministic mapping are held by the "
                "domain owner. No new model execution or medical confirmation is implied."
            ),
        )
        self.sources = self.read_sources() + (self.fixed_source,)
        if not self.sources or len(self.sources) > 15:
            raise ValueError("first_case_source_bound")

    def read_source(self, chunk):
        r = KnowledgeReference.model_validate_json(json.dumps(chunk["ref"]))
        current = self.knowledge.get(r)
        record = current.record
        for snapshot in (
            current,
            self.knowledge.get(record.source_version_ref),
            self.knowledge.get(record.document_ref),
        ):
            if snapshot.withdrawn or snapshot.status.value == "deprecated":
                raise ValueError("source_withdrawn")
        raw = self.knowledge_objects.read_bounded(record.object_uri, 4194304)
        if checksum(raw) != record.content_checksum:
            raise ValueError("source_object_changed")
        # Indexed text is the literal chunk payload, not an old medical summary.
        text = raw.decode("utf-8")
        if text != chunk["text"] or checksum(raw) != chunk["text_checksum"]:
            raise ValueError("source_text_changed")
        return Source(
            reference=ref("source", r.code, r.checksum, r.version),
            locator=(
                chunk["source_id"]
                + ": "
                + record.location.section
                + "; paragraph="
                + str(record.location.paragraph)
            )[:240],
            purpose="research_material",
            state="research_only",
            text=text,
        )

    def read_sources(self):
        return tuple(self.read_source(c) for c in self.chunks)

    def image(self, reference):
        a = self.image_artifact
        if reference.code != a.asset_code or reference.checksum != a.checksum:
            raise ValueError("first_case_image_scope")
        raw = self.objects.read_bounded(a.uri, 4194304)
        if checksum(raw) != IMAGE:
            raise ValueError("first_case_image_changed")
        return raw

    def project(self, *, run_id, config, definitions=None):
        import struct

        a = self.image_artifact
        image_ref = ref("image", a.asset_code, a.checksum)
        raw = self.image(image_ref)
        if raw[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", raw[16:24]) != (512, 664):
            raise ValueError("first_case_image_metadata_requires_review")
        offset = 8
        while offset < len(raw):
            length = struct.unpack(">I", raw[offset : offset + 4])[0]
            if raw[offset + 4 : offset + 8] not in {b"IHDR", b"IDAT", b"IEND"}:
                raise ValueError("first_case_image_metadata_requires_review")
            offset += length + 12
        if offset != len(raw):
            raise ValueError("first_case_image_encoding")
        image_binding = ImageBinding(
            reference=image_ref,
            media_type="image/png",
            width=512,
            height=664,
            coordinate_system="pixel_xy_top_left",
            preprocessing=ref(
                "preprocessing",
                "POLICY:VISIBLE_PNG_UNCHANGED",
                digest(dict(input=IMAGE, transform="none")),
            ),
            location=(0, 0, 512, 664),
        )
        observations, items, missing_resources = [], [], []
        rows = {r["id"]: r for r in self.view["current_scoring"]["reviews"]}
        criteria = {
            "criterion:" + c["id"]: c for c in self.view["research_criteria"]["policy"]["criteria"]
        }
        by_source = {}
        for c, s in zip(self.chunks, self.sources[:-1], strict=True):
            by_source.setdefault(c["source_id"], []).append(s)
        missing_definitions = []
        for old in self.view["evidence_records"]["observations"]:
            identifier = old["assessment_id"]
            row = rows[identifier]
            kind = old["result_kind"]
            values = (
                old["values"]
                if kind in {"measurement", "proxy_measurement", "provided_information"}
                else []
            )
            if kind == "provided_information":
                values = [v for v in values if v["name"] == "age"]
            # Textual legacy/VLM/assistant observations and model answers never enter the prompt.
            values = [
                {"name": v["name"], "value": v["value"], "unit": v["unit"]}
                for v in values
                if type(v["value"]) in (int, float)
            ]
            fixed = row["role"] == "model"
            value = (
                canonical(values).decode()
                if values
                else (
                    "Historical fixed model output retained by domain; not supplied as an answer."
                    if fixed
                    else None
                )
            )
            if value and len(value) > 1000:
                raise ValueError("first_case_observation_projection_too_large")
            units = {v["unit"] for v in values}
            unit = next(iter(units)) if len(units) == 1 and next(iter(units)) else "mixed_units"
            unit = {"px^-1": "per_pixel", "°": "degree"}.get(unit, unit)
            method = (
                "reused_numeric_tool"
                if values
                else "fixed_historical_model"
                if fixed
                else "unobserved"
            )
            # Only one explicit VLM observation slot. It cannot mark other missing tests obtained.
            image = image_binding if identifier == "visual:quality" else None
            if image:
                method, unit = "live_image_observation", "qualitative"
            observation = dict(
                status="obtained" if value else "unknown",
                method=method,
                unit=unit,
                value=value,
                image=image.model_dump(mode="json") if image else None,
            )
            observation["reference"] = ref(
                "observation", old["id"], digest(observation)
            ).model_dump(mode="json")
            observations.append(observation)
            definition = (criteria if definitions is None else definitions).get(identifier)
            required = [s["id"] for s in definition.get("sources", [])] if definition else []
            exact = bool(required) and all(s in by_source for s in required)
            sources = (
                tuple(s for name in required for s in by_source.get(name, ()))
                if exact
                else (
                    self.sources[:1]
                    if definitions is None
                    else tuple(
                        s for s in self.sources if s.reference.code == "POLICY:ROUND_DEFINITIONS"
                    )
                )
            )
            if fixed:
                sources = (self.fixed_source,)
            interpretation = None
            if exact and (definitions is None or definition.get("interpretation_installed")):
                interpretation = definition.get("interpretation_rule", definition.get("condition"))
            if interpretation is None:
                missing_definitions.append(identifier)
            definition_value = dict(
                examination=(definition or {}).get("examination", old["title"][:240]),
                applicability_conditions=(
                    (definition or {}).get("applicability_conditions")
                    or "Single SLO research; laterality UNKNOWN; no clinical conclusion."
                ),
                interpretation=interpretation,
                interpretation_status="installed" if interpretation else "missing",
                source_refs=[s.reference.model_dump(mode="json") for s in sources],
            )
            definition_value["reference"] = ref(
                "definition", "POLICY:LIVE:" + identifier, digest(definition_value), "2.0.0"
            ).model_dump(mode="json")
            # Original numeric fixed model row is copied solely into the domain policy.
            policy_kind = (
                "model" if fixed else "measurement" if row["kind"] == "descriptive" else row["kind"]
            )
            paths = row["dependency_paths"]
            if not paths:
                missing_ref = ref("missing_resource", "missing:" + identifier, digest(observation))
                missing_resources.append(missing_ref)
                paths = [[missing_ref.code]]
            fixed_row = (
                dict(row, kind=policy_kind, source_binding_checksum=digest(list(sources)))
                if fixed
                else None
            )
            items.append(
                dict(
                    id=identifier,
                    node=row["node"],
                    kind=policy_kind,
                    role=row["role"],
                    observation_code=old["id"],
                    method=method,
                    unit=unit,
                    laterality="UNKNOWN",
                    source_codes=[s.reference.code for s in sources],
                    applicable=True,
                    correlation_group=row["correlation_group"],
                    dependence_block=row["dependence_block"],
                    dependency_paths=paths,
                    definition=definition_value,
                    fixed_model_row=fixed_row,
                    fixed_observation_ref=observation["reference"] if fixed else None,
                )
            )
        protocol = json.loads(Path("configs/research/evidence_review_protocol_v1.json").read_text())
        groups = [
            dict(
                id=g["id"],
                importance=g["importance"],
                slots=[s["assessment_ids"] for s in g["slots"]],
            )
            for g in self.view["current_scoring"]["groups"]
        ]
        policy = validate(
            TaskPolicyV2,
            dict(
                version="first-case-live-1.0.0" if definitions is None else "first-case-live-2.0.0",
                scale=ref("scale", "POLICY:CASE_REVIEW", digest(protocol)),
                aggregation_checksum=digest(POLICY),
                items=items,
                groups=groups,
            ),
        )
        scope = dict(
            case_code=CASE,
            run_id=run_id,
            graph=Graph(
                role="graph",
                code=GRAPH,
                version="case-evidence-graph/1.19.0",
                checksum=self.root.checksum,
                kind="research_artifact",
            ).model_dump(mode="json"),
            visible_checksum=digest(self.visible_value),
            input_checksum=self.visible_value.checksum,
            policy_checksum=digest(policy),
        )
        from apps.vectorengine_case_preflight import graph_target_binding

        target_binding = graph_target_binding(self.view, policy, scope["graph"])
        material = dict(
            scope=scope,
            goal_code=target_binding["root_code"],
            goal_summary=(
                "Single-image research evidence planning; "
                "preserve missing observations and references."
            ),
            visibility="approved_visible_projection/1.0.0",
            sources=[s.model_dump(mode="json") for s in self.sources],
            observations=observations,
            policy_ref=ref("policy", "POLICY:FIRST_CASE_LIVE", digest(policy)).model_dump(
                mode="json"
            ),
            missing=["IOP", "OCT", "visual_field", "laterality"],
            memory=[],
        )
        scope["material_checksum"] = digest(material)
        resources = tuple(
            ref("resource", key, digest(value))
            for key, value in self.view["current_scoring"]["resources"].items()
        ) + tuple(missing_resources)
        state = DomainState(
            material=validate(Material, material),
            policy=policy,
            laterality="UNKNOWN",
            resources=resources,
            prompt_checksum=next(r.prompt_checksum for r in config.roles if r.role == "review"),
            config_checksum=digest(config),
            nodes=tuple(sorted({i.node for i in policy.items} | {target_binding["root_code"]})),
        )
        return state, dict(
            schema_version="bounded-live-visible-manifest/1.0.0",
            graph_target_binding=target_binding,
            case_code=CASE,
            selection="existing_07_first_case_no_label_access",
            images=1,
            image_checksum=IMAGE,
            image_bytes=len(raw),
            width=512,
            height=664,
            image_metadata={},
            visible_identifiers_observed=False,
            external_transfer_restriction="none_explicit_in_registered_visible_metadata_and_project_scope",
            restriction_check_scope="registered_visible_metadata_and_shared_07_spec_only",
            allowed_demographics=["age"],
            old_visual_interpretations_sent=False,
            old_model_answers_sent=False,
            old_ratings_sent=False,
            original_note_or_label_read=False,
            sources=len(self.sources),
            source_research_only=True,
            reused_tool_execution=True,
            new_tool_execution=False,
            missing_reference_definitions=missing_definitions,
            material_checksum=digest(state.material),
            policy_checksum=digest(policy),
            relay_processes_inputs=True,
            store_false_proves_zero_retention=False,
        )

    def verify(self, state):
        from apps.vectorengine_case_preflight import graph_target_binding

        target = graph_target_binding(self.view, state.policy, state.material.scope.graph)
        if state.material.goal_code != target["root_code"] or set(state.nodes) != (
            {i.node for i in state.policy.items} | {target["root_code"]}
        ):
            raise ValueError("case_graph_target_changed")
        if self.visible.read(CASE) != self.visible_value:
            raise ValueError("visible_input_changed")
        current = (
            self.runtime.read_case_artifact(CASE, GRAPH)
            if hasattr(self.runtime, "read_case_artifact")
            else next(
                a for a in self.runtime.get_case_bundle(CASE).artifacts if a.asset_code == GRAPH
            )
        )
        if (
            current != self.root
            or self.objects.read_bounded(current.uri, 4194304) != self.root_bytes
        ):
            raise ValueError("research_parent_changed")
        sources = self.read_sources() + (self.fixed_source,)
        if sources != state.material.sources:
            raise ValueError("source_withdrawn")
        self.image(ref("image", self.image_artifact.asset_code, IMAGE))
