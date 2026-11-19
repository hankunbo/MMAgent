"""Explicit synthetic disc producer, distinct from its ratio consumer."""

from packages.grounding import CatalogSnapshot, Mapping
from packages.grounding.synthetic import SyntheticCatalogReader, ref
from tests.grounding.query_support import ready, request_for
from tests.grounding.test_models import reseal

DISC_CAPABILITY = "CAPABILITY:SYNTHETIC_DISC_SEGMENTATION"


def add_producer(mapping, catalog, *, mapped=True, validation="auto_validated_research"):
    _, template, _, source, _, _ = ready(chain=True)
    tool = source.tools[-1].model_dump(mode="json")
    candidate = template.candidates[-1].model_dump(mode="json")
    output = mapping.candidates[0].inputs[0].constraint.model_dump(mode="json")
    image = mapping.candidates[0].inputs[2].constraint.model_dump(mode="json")
    capability = ref("capability", DISC_CAPABILITY).model_dump(mode="json")
    tool.update(capabilities=[capability], outputs=[output], validation_status=validation)
    tool["inputs"][0]["constraint"] = image
    candidate.update(capability_ref=capability, output=output, inputs=tool["inputs"])
    catalog = reseal(
        CatalogSnapshot,
        {
            **catalog.model_dump(mode="json"),
            "version": "1.3.0",
            "tools": [*(t.model_dump(mode="json") for t in catalog.tools), tool],
        },
    )
    mapping = reseal(
        Mapping,
        {
            **mapping.model_dump(mode="json"),
            "version": "1.3.0",
            "catalog_ref": catalog.reference(),
            "accepted_artifacts": [*mapping.accepted_artifacts, output],
            "candidates": [*mapping.candidates, *([candidate] if mapped else [])],
        },
    )
    return mapping, catalog


def producer_ready(*, mapped=True, validation="auto_validated_research"):
    q, m, p, c, k, _ = ready()
    m, c = add_producer(m, c, mapped=mapped, validation=validation)
    q = request_for(m, p, artifacts=q.artifacts[:1], patient_data_types=[])
    return q, m, p, c, k, SyntheticCatalogReader([c])


def alternative_graph():
    """Separate locked requirement for disc production with two explicit OR tools."""
    from packages.grounding import validate
    from packages.grounding.locks import GroundingLock
    from packages.grounding.ports import KnowledgeProjection
    from packages.grounding.synthetic import SyntheticKnowledgeReader
    from tests.grounding.query_support import metadata, paths, updated

    q, mapping, policy, catalog, knowledge, _ = producer_ready()
    old_tool = catalog.tools[-1].tool_ref
    old_impl = catalog.tools[-1].implementation_ref
    new_tool = ref("tool", "synthetic_disc_alternative")
    new_impl = ref("implementation", "synthetic_disc_alternative_impl")

    def replace(value):
        if isinstance(value, dict):
            if value == old_tool.model_dump(mode="json"):
                return new_tool.model_dump(mode="json")
            if value == old_impl.model_dump(mode="json"):
                return new_impl.model_dump(mode="json")
            return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list):
            return [replace(v) for v in value]
        return value

    tool = replace(catalog.tools[-1].model_dump(mode="json"))
    alternative = replace(mapping.candidates[-1].model_dump(mode="json"))
    alternative["code"] = "CAPABILITY:SYNTHETIC:ALTERNATIVE"
    catalog = reseal(
        CatalogSnapshot, {**catalog.model_dump(mode="json"), "tools": [*catalog.tools, tool]}
    )
    root = reseal(
        Mapping,
        {
            **mapping.model_dump(mode="json"),
            "catalog_ref": catalog.reference(),
            "candidates": mapping.candidates[:2],
            "accepted_artifacts": mapping.accepted_artifacts[:2],
        },
    )
    requirement = ref("evidence_requirement", "EVIDENCE:SYNTHETIC:DISC_REQUIREMENT")
    producer = reseal(
        Mapping,
        {
            **mapping.model_dump(mode="json"),
            "code": "BINDING:OID_PRODUCERS",
            "requirement_ref": requirement,
            "catalog_ref": catalog.reference(),
            "candidates": [mapping.candidates[-1], alternative],
            "accepted_artifacts": [mapping.candidates[-1].output],
        },
    )
    producer_knowledge = SyntheticKnowledgeReader(
        validate(
            KnowledgeProjection,
            {
                **knowledge.projection.model_dump(mode="json"),
                "requirement_ref": requirement,
                "artifact_type_code": mapping.candidates[-1].output.artifact_ref.code,
            },
        )
    )
    reader = SyntheticCatalogReader([catalog])
    request = request_for(
        root, policy, artifacts=q.artifacts, patient_data_types=[], environment=q.environment[:2]
    )
    lock = reseal(
        GroundingLock,
        {
            **request.lock.model_dump(mode="json"),
            "mapping_refs": [root.reference(), producer.reference()],
            "knowledge_refs": [*request.lock.knowledge_refs, requirement],
        },
    )
    environment = [
        *request.environment,
        {
            "implementation_ref": new_impl,
            "availability": "available",
            "preconditions": "satisfied",
            "source_ref": ref("source_version", "SOURCE:SYNTHETIC:ENV"),
        },
    ]
    request = updated(
        request,
        lock=lock,
        environment=environment,
        artifacts=[
            *request.artifacts,
            metadata(request, mapping.candidates[-1].inputs[0].constraint, "IMAGE001"),
        ],
    )
    views = (
        *paths(root, policy, knowledge, reader),
        *paths(producer, policy, producer_knowledge, reader),
    )
    return request, views


def attribution_examples():
    """Rebuild shareable synthetic output examples without database or tool I/O."""
    from packages.grounding import evaluate_query
    from tests.grounding.query_support import metadata, paths, updated

    results = {}
    q, m, p, _, k, reader = ready()
    results["unknown_masks"] = evaluate_query(
        updated(q, artifacts=[], patient_data_types=[]), paths(m, p, k, reader)
    )
    q, m, p, _, k, reader = producer_ready(mapped=False)
    results["unmapped_disc_producer"] = evaluate_query(q, paths(m, p, k, reader))
    q, m, p, _, k, reader = producer_ready()
    q = updated(
        q,
        environment=q.environment[:2],
        artifacts=[*q.artifacts, metadata(q, m.candidates[-1].inputs[0].constraint, "IMAGE001")],
    )
    results["disc_producer_environment_unknown"] = evaluate_query(q, paths(m, p, k, reader))
    return results
