"""04C trusted assembly over 04B7 admission and stage-03 public APIs, no dataset reader."""

import json
from dataclasses import dataclass, replace

from apps.knowledge_retrieval import PublishedKnowledgeService, SourceCandidateService
from db.knowledge.models import Release, Template
from db.knowledge.repository import KnowledgeConflictError, KnowledgeNotFoundError
from packages.contracts import ActorType, ImmutablePolicyReference, VersionedReference
from packages.disease_packs import DiseasePack, ResearchDiseasePack, validate_conclusion
from packages.disease_packs.catalog import validate_prepared_case_contract
from packages.graph_engine import canonical_checksum, instantiate_template
from packages.rule_engine import (
    RULE_PACKAGE_PAYLOAD_VERSION,
    GateDefinition,
    KnowledgeGap,
    RulePackage,
    compute_rule_package_checksum,
    propagate,
)


@dataclass(frozen=True)
class PackAdmission:
    pack: DiseasePack | ResearchDiseasePack
    template: Template | None
    knowledge_gap: KnowledgeGap | None
    manifest_checksum: str | None


class DiseasePackService:
    """Inject only the reader identity. A descriptor's Published string is not authorization."""

    def __init__(self, catalog, repository, object_store, *, synthetic_test_mode=False):
        self._published = PublishedKnowledgeService(repository, object_store)
        self._repository = repository
        self._catalog = catalog
        if type(synthetic_test_mode) is not bool:
            raise TypeError("explicit synthetic test mode required")
        self._synthetic = synthetic_test_mode

    def admit(self, pack_ref, query, context, config, *, runtime=None, run_id=None):
        pack = self._catalog.pack(pack_ref)
        if isinstance(pack, ResearchDiseasePack):
            from packages.knowledge_review.research import validate_use

            validate_use(
                pack.research_scope, pack.research_policy_ref.to_versioned_reference(), context
            )
        elif getattr(context, "use", None) is not None:
            raise KnowledgeConflictError("KNO_B03_LEGACY_USE_DENIED")
        if pack.definition.synthetic and not self._synthetic:
            raise ValueError("synthetic pack cannot enter a formal clinical run")
        if (runtime is None) != (run_id is None):
            raise ValueError("Runtime and run identity must be supplied together")

        def gap(reason):
            return PackAdmission(pack, None, KnowledgeGap("KNO04C_NOT_ADMITTED", reason, ()), None)

        if pack.status != "Published":
            return gap("; ".join(pack.knowledge_gaps))
        if not set(pack.definition.applicable_conditions) <= set(context.conditions) or (
            set(pack.definition.excluded_conditions) & set(context.conditions)
        ):
            return gap("pack applicability or exclusion condition did not match")
        binding = pack.bindings
        if runtime is not None:
            ref = pack.reference()
            expected_policy = ImmutablePolicyReference(
                policy_code=ref.code, version=ref.version, checksum=ref.checksum
            )
            if expected_policy not in runtime.get_run_policy_references(run_id):
                raise ValueError("exact DiseasePack version/checksum is not pinned by this run")
            lock = runtime.get_knowledge_lock(run_id)
            if isinstance(pack, ResearchDiseasePack) and (
                lock.get("use") != context.use.model_dump(mode="json")
                or lock.get("pack_ref") != pack.reference().model_dump(mode="json")
            ):
                raise KnowledgeConflictError("KNO_B03_RUN_USE_LOCK")
            if lock["release_ref"] != binding.release_ref.model_dump(mode="json"):
                raise ValueError("pack release differs from immutable Runtime lock")
            result = self._published.retrieve_for_run(runtime, run_id, query, context, config)
        else:
            result = self._published.retrieve_published_knowledge(
                binding.release_ref, query, context, config
            )
        records = {h["artifact"].record.reference(): h["artifact"].record for h in result["hits"]}
        release = Release.model_validate_json(json.dumps(result["current"]["release"]["record"]))
        if isinstance(pack, ResearchDiseasePack):
            if (
                release.research_scope != pack.research_scope
                or release.policy_ref != pack.research_policy_ref
            ):
                raise KnowledgeConflictError("KNO_B03_RELEASE_BINDING")
        elif release.research_scope is not None:
            raise KnowledgeConflictError("KNO_B03_LEGACY_USE_DENIED")
        if release.synthetic != pack.definition.synthetic:
            raise ValueError("release synthetic scope differs from pack")
        if release.reference() != binding.release_ref or not set(
            (*binding.artifacts(), *binding.evidence_requirement_refs)
        ) <= set(release.member_refs):
            raise ValueError("all pack bindings must be exact members of the locked release")
        # No top-k widening or Draft fallback: an absent required exact artifact is a gap.
        if not set(binding.artifacts()) <= set(records):
            return gap(
                "required Published Claim/Template/Rule/StopRule absent from exact retrieval"
            )
        template = records[binding.template_ref]
        public = template.graph_template
        if public is None or public.extensions.get("disease_pack_definition") != {
            "code": pack.definition.pack_code,
            "version": pack.definition.version,
            "checksum": pack.definition.checksum,
        }:
            raise ValueError("reviewed template does not bind this exact pack definition")
        if public.goal_node_code != pack.definition.target_node_code:
            raise ValueError("pack goal differs from reviewed template")
        if set(template.rule_refs) != set(binding.rule_refs):
            raise ValueError("pack rules differ from reviewed template rules")
        if any(record.synthetic != pack.definition.synthetic for record in records.values()):
            raise ValueError("synthetic Knowledge cannot authorize a real disease pack")
        node_codes = {n.node_code for n in public.nodes}
        if not {s.node_code for s in pack.definition.evidence_slots} <= node_codes:
            raise ValueError("pack evidence slots absent from reviewed template")
        required_nodes = set()
        for ref in binding.evidence_requirement_refs:
            # Exact public reads preserve full live review/withdrawal dependencies.
            try:
                requirement = self._repository.get_published_evidence(ref, context=context).record
                node = self._repository.get_published_evidence(
                    requirement.node_ref, context=context
                ).record
            except (KnowledgeConflictError, KnowledgeNotFoundError):
                return gap("EvidenceRequirement or its node is not currently Published/applicable")
            if node.template_ref != binding.template_ref or node.code not in node_codes:
                raise ValueError("EvidenceRequirement belongs to another template")
            # Reference equality proves the reviewed binding only. Each exact read
            # above independently checked the actual complete request against its scope.
            for field in ("source_refs", "applicability_refs", "binding_refs"):
                if set(getattr(requirement, field) or ()) != set(getattr(template, field) or ()):
                    raise ValueError("EvidenceRequirement must share the filtered template scope")
            if requirement.synthetic != pack.definition.synthetic:
                raise ValueError("EvidenceRequirement synthetic scope mismatch")
            if requirement.acceptance_rule_ref not in binding.rule_refs:
                raise ValueError("EvidenceRequirement lacks the bound reviewed acceptance rule")
            slot = next(
                (s for s in pack.definition.evidence_slots if s.node_code == node.code), None
            )
            if slot is None or slot.output_artifact_code != requirement.artifact_type_code:
                raise ValueError("EvidenceRequirement artifact differs from declaration")
            required_nodes.add(node.code)
        if required_nodes != {s.node_code for s in pack.definition.evidence_slots}:
            raise ValueError("incomplete EvidenceRequirement coverage")
        # Check live authorization again after resolving requirements, including cache hits.
        self._published.retrieve_published_knowledge(
            binding.release_ref, query, context, config, run_id=run_id
        )
        return PackAdmission(pack, template, None, result["manifest_checksum"])

    def admit_research(self, pack_ref, query, context, config, **kwargs):
        if not isinstance(self._catalog.pack(pack_ref), ResearchDiseasePack):
            raise KnowledgeConflictError("KNO_B03_RESEARCH_DECLARATION_REQUIRED")
        return self.admit(pack_ref, query, context, config, **kwargs)

    def admit_clinical(self, pack_ref, query, context, config, **kwargs):
        # B03 introduces no clinical evaluation authority. A legacy L4 descriptor
        # remains readable via its old port, but cannot authorize this new port.
        raise KnowledgeConflictError("KNO_B03_CLINICAL_DENIED")

    def instantiate(self, pack_ref, case, instantiation_context, query, context, config):
        admission = self.admit(pack_ref, query, context, config)
        if admission.knowledge_gap:
            return admission, None
        case = validate_prepared_case_contract(admission.pack, case)
        public = admission.template.graph_template
        if instantiation_context.rule_versions != tuple(
            r.to_versioned_reference() for r in admission.pack.bindings.rule_refs
        ):
            raise ValueError("instantiation must pin the exact pack rule versions")
        return admission, instantiate_template(public, case, instantiation_context)

    def propagate_for_run(
        self, pack_ref, runtime, run_id, graph, package, context, query, retrieval_context, config
    ):
        admission = self.admit(
            pack_ref, query, retrieval_context, config, runtime=runtime, run_id=run_id
        )
        if admission.knowledge_gap:
            return admission, propagate(graph, None, context)
        if graph.scope.run_id != run_id:
            raise ValueError("graph run scope differs from locked run")
        # Reject caller-injected AND/OR gates or stop semantics even if they were re-hashed.
        expected = build_rule_package(admission, graph)
        if package != expected:
            raise ValueError("rule package differs from the reviewed exact pack/template")
        self._published.validate_rule_package_for_run(
            runtime, run_id, graph, package, query, retrieval_context, config
        )
        return admission, propagate(graph, package, context)

    def validate_conclusion(self, pack_ref, conclusion):
        return validate_conclusion(self._catalog.pack(pack_ref), conclusion)


def build_rule_package(admission, graph):
    if admission.knowledge_gap is not None or admission.template is None:
        raise ValueError("KnowledgeGap cannot be compiled into a Published RulePackage")
    pack, template = admission.pack, admission.template
    if (
        graph.template_ref.code != template.code
        or graph.template_ref.checksum != canonical_checksum(template.graph_template)
    ):
        raise ValueError("graph does not originate from the exact reviewed template")
    grouped = {}
    for edge in template.edges:
        if edge.edge_type.value in {"AND_CHILD", "OR_CHILD"}:
            grouped.setdefault(
                (edge.target_node_code, edge.edge_type.value, edge.rule_ref), []
            ).append(edge.source_node_code)
    gates = tuple(
        GateDefinition(
            gate_code=f"RULE:PACK_GATE_{index}",
            parent_node_code=parent,
            operator=kind.split("_")[0],
            child_node_codes=tuple(sorted(children)),
            rule_ref=VersionedReference.model_validate(ref.model_dump(mode="python")),
        )
        for index, ((parent, kind, ref), children) in enumerate(
            sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].code))
        )
    )
    refs = tuple(r.to_versioned_reference() for r in pack.bindings.rule_refs)
    provenance = graph.provenance.model_copy(
        update={
            "actor": ActorType.RULE_ENGINE,
            "generator_version": "04C.1.0.0",
            "input_checksums": (pack.checksum, graph.provenance.input_checksums[0]),
            "knowledge_versions": (pack.bindings.release_ref.to_versioned_reference(),),
            "rule_versions": refs,
        }
    )
    package = RulePackage(
        RULE_PACKAGE_PAYLOAD_VERSION,
        "RULE:PACK:" + pack.definition.pack_code.upper(),
        pack.definition.version,
        "sha256:" + "0" * 64,
        "PUBLISHED",
        str(graph.graph_code),
        graph.template_ref,
        refs,
        gates,
        provenance,
    )
    return replace(package, package_checksum=compute_rule_package_checksum(package))


class DiseasePackReplayService:
    """Offline replay has a separate identity and never returns a PackAdmission."""

    def __init__(self, catalog, repository, object_store):
        self._catalog = catalog
        self._history = SourceCandidateService(repository, object_store)

    def replay(self, pack_ref):
        pack = self._catalog.pack(pack_ref)
        history = self._history.replay_release(pack.bindings.release_ref) if pack.bindings else None
        return {"pack": pack, "frozen_history": history, "grants_current_admission": False}


def read_engineering_stop_basis(reader, checksum, *, trigger, reason_code, runtime_config_checksum):
    """04C explicit v2 sidecar consumer; it never returns a PackAdmission/RulePackage.

    Legacy PackBindings.stop_rule_refs still require Published medical StopRules.
    Engineering provenance alone cannot acquire medical publication authority.
    """
    return reader.require_trigger(
        checksum,
        trigger=trigger,
        reason_code=reason_code,
        runtime_config_checksum=runtime_config_checksum,
    )
