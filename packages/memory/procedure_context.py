"""Read-side method prerequisites from original accepted inputs, never patient values.

This does not upgrade a Memory candidate to published or executable experience.
Old archive bytes remain unchanged. Missing provenance is an explicit mismatch,
not permission to infer a component from a dataset or a modality name.
"""

import json

from db.runtime.repository.research import ResearchRunRepository
from packages.grounding.models import digest, validate
from packages.research_runtime.task_context import check_current_inputs

VERSION = "memory-method-context/1.0.0"


def original_definition(definition, sources, aliases):
    """Reverse only the original owner's exact alias map; no string heuristics."""
    from packages.evidence_acquisition.source_aliases import alias_sources
    from packages.llm_provider.models import Source

    if definition["reference"]["version"] != "research-source-alias/1.0.0":
        return definition
    if not aliases or len(aliases) != 1:
        raise ValueError("memory_method_alias_provenance_missing")
    manifest = aliases[0]
    bindings = {b["alias"]["code"]: b for b in manifest["bindings"]}
    if len(bindings) != len(manifest["bindings"]) or set(bindings) != {
        s["reference"]["code"] for s in sources
    }:
        raise ValueError("memory_method_alias_inventory")
    originals = []
    for source in sources:
        binding = bindings[source["reference"]["code"]]
        if source["reference"] != binding["alias"]:
            raise ValueError("memory_method_alias_binding")
        originals.append(validate(Source, dict(source, reference=binding["original"])))
    rebuilt, proof = alias_sources(tuple(originals))
    if proof != manifest or [s.model_dump(mode="json") for s in rebuilt] != sources:
        raise ValueError("memory_method_alias_binding")
    refs = []
    for reference in definition["source_refs"]:
        binding = bindings.get(reference["code"])
        if binding is None or reference != binding["alias"]:
            raise ValueError("memory_method_alias_definition_binding")
        refs.append(binding["original"])
    return dict(definition, source_refs=refs)


def component_contract(component):
    return dict(
        representation=component.kind,
        modality=component.modality,
        concept=component.concept,
        unit=component.unit,
        unit_basis=component.unit_basis,
        unit_qualified=bool(component.unit and component.unit_basis),
        axis_semantics=component.axis_semantics,
        physical_scale_authorized=False,
    )


def contract(task, definition, component):
    """Comparable prerequisites only; no dataset, patient, payload or score."""
    value = definition.model_dump(mode="json") if hasattr(definition, "model_dump") else definition
    return dict(
        schema_version=VERSION,
        target=task.target_code,
        task_checksum=task.checksum,
        definition_checksum=digest({k: v for k, v in value.items() if k != "reference"}),
        component=component_contract(component),
        pair_binding="single_component_only",
        qualification="method_hint_requires_current_scope_and_tool_owners",
    )


def read(owner, record, *, inputs, contexts):
    """Finish one historical input proof before returning a method prerequisite."""
    from packages.dataset_adapters.current_read_scope import _independent_projection_visible_reads

    with _independent_projection_visible_reads():
        return _read(owner, record, inputs=inputs, contexts=contexts)


def _read(owner, record, *, inputs, contexts):
    """Reconstruct a single-component procedure from original Memory/domain owners.

    The old accepted item's node is the original producer's exact component binding.
    We deliberately do not split compound node IDs or guess a paired measurement.
    """
    from packages.memory.aggregation_archive import Documents
    from packages.memory.archive import ArchiveSources

    if type(owner) is not ArchiveSources or not isinstance(contexts, ResearchRunRepository):
        raise ValueError("memory_method_original_owners_required")
    owner.verify_record(record)
    if record.kind != "tool_experience":
        raise ValueError("memory_method_branch_required")
    context = contexts.read_research_run(record.run_id)
    visible = check_current_inputs(context, inputs)
    if context.request.scope.case_code != record.case_code:
        raise ValueError("memory_method_case_binding")
    _, raw = owner.artifact(record.case_code, record.detail.origin_code)
    envelope = json.loads(raw)
    docs = Documents(
        owner, record.case_code, record.run_id, envelope["root_checksum"], {}, None, None
    )
    parent = docs.read("dynamic-phase-input")
    if not parent:
        raise ValueError("memory_method_context_missing")
    state = parent[0]
    items = [i for i in state["policy"]["items"] if i["id"] == record.detail.target]
    requirements = [r for r in state["policy"]["requirements"] if r["id"] == record.detail.target]
    if len(items) != 1 or len(requirements) != 1:
        raise ValueError("memory_method_context_missing")
    item, requirement = items[0], requirements[0]
    matches = [
        c
        for c in visible.components.components
        if item["node"] == "node:" + c.id and c.id in context.request.selected_components
    ]
    if len(matches) != 1:
        raise ValueError("memory_method_component_binding_missing")
    component = matches[0]
    if component.status not in {"obtained", "partial"}:
        raise ValueError("memory_method_input_unavailable")
    if (
        state["material"]["goal_code"] != context.request.task.target_code
        or item["definition"] != requirement["definition"]
        or any(step.source_checksum != digest(item) for step in record.detail.steps)
    ):
        raise ValueError("memory_method_definition_binding")
    aliases = docs.read("generic-review-source-aliases")
    definition = original_definition(
        requirement["definition"], state["material"]["sources"], aliases
    )
    return dict(
        contract=contract(context.request.task, definition, component),
        provenance=dict(
            context_checksum=context.checksum,
            parent_checksum=digest(state),
            input_checksum=visible.components.checksum,
            component_checksum=digest(component),
            alias_checksum=digest(aliases),
        ),
        patient_group=visible.components.patient_group,
    )


def query(base, row):
    """Revalidate this case using its original input reader before matching."""
    base.read_dependency_input()
    current = [r for r in base.ledger if r["id"] == row["id"]]
    if len(current) != 1:
        raise ValueError("memory_method_current_requirement_missing")
    # Definition expansion may replace a ledger row after initial selection.
    # A previously captured Python row must not authorize that new definition.
    row = current[0]
    components = [c for c in base.manifest.components if c.id == row.get("component")]
    if len(components) != 1:
        raise ValueError("memory_method_component_binding_missing")
    if components[0].status not in {"obtained", "partial"}:
        raise ValueError("memory_method_input_unavailable")
    return dict(
        contract=contract(base.task, row["definition"], components[0]),
        patient_group=base.manifest.patient_group,
    )


def mismatch(historical, current):
    # Unknown patient identifiers never prove equality or independence.
    group = historical.get("patient_group")
    if group is not None and group == current.get("patient_group"):
        return "same_patient_not_external_experience"
    old, new = historical["contract"], current["contract"]
    for field in (
        "schema_version",
        "target",
        "task_checksum",
        "definition_checksum",
        "component",
        "pair_binding",
    ):
        if old[field] != new[field]:
            return "memory_method_" + field + "_mismatch"
    return None
