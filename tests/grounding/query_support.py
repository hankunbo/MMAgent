"""Small explicit synthetic query graph builders; no real tool or medical facts."""

from datetime import UTC, datetime

from packages.grounding import Mapping, digest, seal, validate
from packages.grounding.locks import GroundingLock
from packages.grounding.query_models import QueryConfig, QueryRequest
from packages.grounding.synthetic import SyntheticCatalogReader, fixture, ref
from packages.grounding.validation import evaluate_paths
from packages.retrieval.knowledge.models import RetrievalContext
from tests.grounding.test_models import reseal

NOW = datetime(2026, 1, 2, tzinfo=UTC)


def config(**changes):
    return seal(
        QueryConfig,
        version="1.0.0",
        max_depth=6,
        max_evaluations=128,
        top_k=1,
        information_gain_enabled=True,
        order=["information_gain", "directness", "cost", "latency", "failure_risk"],
        **changes,
    )


def request_for(mapping, policy, **changes):
    lock = seal(
        GroundingLock,
        run_id="RUN:OID_QUERY0001",
        graph_code="GRAPH:OID_QUERY0001",
        graph_version=1,
        graph_checksum=digest("synthetic graph"),
        catalog_ref=mapping.catalog_ref,
        mapping_refs=[mapping.reference()],
        release_ref=mapping.release_ref,
        template_ref=mapping.template_ref,
        graph_template_ref=mapping.template_ref,
        knowledge_refs=sorted(
            set(
                (
                    mapping.requirement_ref,
                    *mapping.claim_refs,
                    *mapping.source_refs,
                    *mapping.applicability_refs,
                )
            ),
            key=lambda r: r.code,
        ),
        policy=policy,
    )
    fields = dict(
        lock=lock,
        requirement_ref=mapping.requirement_ref,
        target_node_code="EVIDENCE:SYNTHETIC:NODE",
        scope=dict(
            case_code="CASE:OID_QUERY0001",
            patient_hash=digest("synthetic patient"),
            encounter_code="ENCOUNTER:OID_QUERY0001",
            laterality="NOT_APPLICABLE",
        ),
        artifacts=[],
        availability_source_ref=ref("source_version", "SOURCE:SYNTHETIC:AVAILABILITY"),
        snapshot_at=NOW,
        window_start=datetime(2026, 1, 1, tzinfo=UTC),
        window_end=NOW,
        patient_data_types=list(
            {i.constraint.artifact_ref for c in mapping.candidates for i in c.inputs}
        ),
        satisfied_terms=["CONCEPT:SYNTHETIC"],
        environment=[
            dict(
                implementation_ref=c.implementation_ref,
                availability="available",
                preconditions="satisfied",
                source_ref=ref("source_version", "SOURCE:SYNTHETIC:ENV"),
            )
            for c in mapping.candidates
        ],
        context=RetrievalContext(),
        config=config(),
    )
    fields.update(changes)
    return validate(QueryRequest, fields)


def metadata(request, constraint, suffix):
    return dict(
        asset_code="ASSET:OID_QUERY" + suffix,
        checksum=digest("synthetic " + suffix),
        case_code=request.scope.case_code,
        patient_hash=request.scope.patient_hash,
        encounter_code=request.scope.encounter_code,
        run_id=None,
        artifact_ref=constraint.artifact_ref,
        modality=constraint.modality,
        laterality=constraint.laterality,
        coordinate_space=constraint.coordinate_space,
        unit=constraint.unit,
        scope_terms=constraint.scope_terms,
        qc_status="PASS",
        acquired_at=NOW,
    )


def ready(*, chain=False, cycle=False):
    m, p, c, k, _ = fixture()
    if chain:
        # The third synthetic capability produces a required intermediate from an image.
        tool = c.tools[0].model_dump(mode="json")
        old_ref = tool["tool_ref"]
        new_ref = ref("tool", "synthetic_intermediate").model_dump(mode="json")

        def replace(v):
            if isinstance(v, dict):
                return (
                    new_ref if v == old_ref else {key: replace(child) for key, child in v.items()}
                )
            if isinstance(v, list):
                return [replace(child) for child in v]
            return v

        tool = replace(tool)
        tool["implementation_ref"] = ref(
            "implementation", "synthetic_intermediate_impl"
        ).model_dump(mode="json")
        tool["outputs"] = [m.candidates[0].inputs[0].constraint.model_dump(mode="json")]
        tool["inputs"] = [tool["inputs"][2]]
        tool["inputs"][0].update(required=True, cardinality_min=1)
        if cycle:
            tool["inputs"][0]["constraint"] = m.candidates[0].output.model_dump(mode="json")
        new_c = reseal(
            type(c),
            {
                **c.model_dump(mode="json"),
                "tools": [*(t.model_dump(mode="json") for t in c.tools), tool],
            },
        )
        candidate = replace(m.candidates[0].model_dump(mode="json"))
        candidate.update(
            code="CAPABILITY:SYNTHETIC:INTERMEDIATE",
            implementation_ref=tool["implementation_ref"],
            output=tool["outputs"][0],
            inputs=tool["inputs"],
        )
        m = reseal(
            Mapping,
            {
                **m.model_dump(mode="json"),
                "catalog_ref": new_c.reference(),
                "accepted_artifacts": [
                    *(a.model_dump(mode="json") for a in m.accepted_artifacts),
                    tool["outputs"][0],
                ],
                "candidates": [*(x.model_dump(mode="json") for x in m.candidates), candidate],
            },
        )
        c = new_c
    request = request_for(m, p)
    inputs = [m.candidates[0].inputs[1].constraint]
    inputs.append(
        m.candidates[-1].inputs[0].constraint
        if chain and not cycle
        else m.candidates[0].inputs[0].constraint
    )
    if cycle:
        inputs = inputs[:1]
    request = validate(
        QueryRequest,
        {
            **request.model_dump(mode="json"),
            "artifacts": [metadata(request, x, str(n).zfill(4)) for n, x in enumerate(inputs)],
        },
    )
    reader = SyntheticCatalogReader([c])
    return request, m, p, c, k, reader


def paths(mapping, policy, knowledge, reader, context=None):
    return (evaluate_paths(mapping, policy, knowledge, reader, context or RetrievalContext()),)


def updated(request, **changes):
    return validate(QueryRequest, {**request.model_dump(mode="json"), **changes})
