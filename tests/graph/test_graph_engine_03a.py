"""Stage 03A deterministic graph validation, instantiation, and persistence tests."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from db import manage
from db.runtime.repository import PostgresRuntimeRepository
from packages.contracts import (
    CapabilityAvailability,
    CaseBundle,
    DataAvailability,
    EdgeType,
    GECEGInstance,
    GECEGTemplate,
    GraphEdge,
    GraphNodeStatus,
    NodeType,
    TemplateNode,
    VersionedReference,
)
from packages.graph_engine import (
    CLINICAL_EDGE_TYPES,
    DATA_FLOW_EDGE_TYPES,
    INSTANTIATION_INPUT_PAYLOAD_VERSION,
    Applicability,
    ArtifactNodeBinding,
    GraphEngineError,
    GraphEngineErrorCode,
    InstantiationContext,
    NodeResolution,
    ResolutionKind,
    canonical_checksum,
    instantiate_template,
    validate_template,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "packages" / "contracts" / "examples" / "v1"
SHA = "sha256:" + "a" * 64


def _example(model_type: type, filename: str):
    return model_type.model_validate(json.loads((EXAMPLES / filename).read_text()))


def _ref(code: str = "RULE:INSTANTIATION:SYNTHETIC") -> VersionedReference:
    return VersionedReference(code=code, version="1.0.0", checksum=SHA)


def _template() -> GECEGTemplate:
    base = _example(GECEGTemplate, "GECEGTemplate.valid.json")
    data = base.model_dump(mode="python")
    data["nodes"] = (
        TemplateNode(
            node_code="GOAL:SYNTHETIC_SCREEN",
            node_type=NodeType.GOAL,
            concept_code="CONCEPT:SYNTHETIC_GOAL",
            required=True,
        ),
        TemplateNode(
            node_code="EVIDENCE:IMAGE_USABLE",
            node_type=NodeType.EVIDENCE,
            concept_code="CONCEPT:IMAGE_USABLE",
            required=True,
        ),
        TemplateNode(
            node_code="ARTIFACT:SYNTHETIC_INPUT",
            node_type=NodeType.DATA,
            concept_code="CONCEPT:IMAGE_AVAILABLE",
            required=True,
        ),
        TemplateNode(
            node_code="TOOL:SYNTHETIC_ANALYZER",
            node_type=NodeType.TOOL,
            concept_code="CONCEPT:SYNTHETIC_TOOL",
            required=False,
        ),
        TemplateNode(
            node_code="ARTIFACT:SYNTHETIC_OUTPUT",
            node_type=NodeType.DATA,
            concept_code="CONCEPT:SYNTHETIC_OUTPUT",
            required=False,
        ),
        TemplateNode(
            node_code="CLAIM:SYNTHETIC_QC",
            node_type=NodeType.QC,
            concept_code="CONCEPT:SYNTHETIC_QC",
            required=False,
        ),
        TemplateNode(
            node_code="EVIDENCE:OPTIONAL_PATH",
            node_type=NodeType.EVIDENCE,
            concept_code="CONCEPT:OPTIONAL_PATH",
            required=False,
        ),
    )
    data["edges"] = (
        GraphEdge(
            source_node_code="EVIDENCE:IMAGE_USABLE",
            target_node_code="GOAL:SYNTHETIC_SCREEN",
            edge_type=EdgeType.SUPPORTS,
            rule_ref=_ref(),
        ),
        GraphEdge(
            source_node_code="ARTIFACT:SYNTHETIC_INPUT",
            target_node_code="TOOL:SYNTHETIC_ANALYZER",
            edge_type=EdgeType.REQUIRES_INPUT,
        ),
        GraphEdge(
            source_node_code="TOOL:SYNTHETIC_ANALYZER",
            target_node_code="ARTIFACT:SYNTHETIC_OUTPUT",
            edge_type=EdgeType.PRODUCES,
        ),
        GraphEdge(
            source_node_code="ARTIFACT:SYNTHETIC_OUTPUT",
            target_node_code="CLAIM:SYNTHETIC_QC",
            edge_type=EdgeType.VALIDATED_BY,
        ),
        GraphEdge(
            source_node_code="CLAIM:SYNTHETIC_QC",
            target_node_code="EVIDENCE:IMAGE_USABLE",
            edge_type=EdgeType.SUPPORTS,
        ),
        GraphEdge(
            source_node_code="EVIDENCE:OPTIONAL_PATH",
            target_node_code="GOAL:SYNTHETIC_SCREEN",
            edge_type=EdgeType.OR_CHILD,
        ),
    )
    return GECEGTemplate.model_validate(data)


def _case() -> CaseBundle:
    return _example(CaseBundle, "CaseBundle.valid.json")


def _context(template: GECEGTemplate) -> InstantiationContext:
    return InstantiationContext(
        graph_code="GRAPH:OID_GRAPH03A1",
        graph_version=1,
        template_checksum=canonical_checksum(template),
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        generator_version="03A.1.0",
        source_codes=("SOURCE:SYNTHETIC_03A",),
        rule_versions=(_ref(),),
        resolutions=(
            NodeResolution(
                node_code="ARTIFACT:SYNTHETIC_INPUT",
                applicability=Applicability.APPLICABLE,
                availability=DataAvailability.AVAILABLE,
                artifact_bindings=(
                    ArtifactNodeBinding(
                        asset_code="ASSET:OID_ASSET001",
                        binding_code="BINDING:OID_BIND0301",
                    ),
                ),
                observation_concept_codes=("CONCEPT:IMAGE_AVAILABLE",),
                reason_code="RULE:INSTANTIATION:EXISTING_DATA",
                rule_ref=_ref("RULE:INSTANTIATION:EXISTING_DATA"),
            ),
            NodeResolution(
                node_code="TOOL:SYNTHETIC_ANALYZER",
                applicability=Applicability.APPLICABLE,
                availability=DataAvailability.UNAVAILABLE,
                capability=CapabilityAvailability.UNAVAILABLE_CAPABILITY,
                reason_code="RULE:INSTANTIATION:CAPABILITY_UNAVAILABLE",
                rule_ref=_ref("RULE:INSTANTIATION:CAPABILITY_UNAVAILABLE"),
            ),
            NodeResolution(
                node_code="EVIDENCE:OPTIONAL_PATH",
                applicability=Applicability.NOT_APPLICABLE,
                availability=DataAvailability.NOT_APPLICABLE,
                reason_code="RULE:INSTANTIATION:NOT_APPLICABLE",
                rule_ref=_ref("RULE:INSTANTIATION:NOT_APPLICABLE"),
            ),
        ),
    )


def _error_code(error: pytest.ExceptionInfo[GraphEngineError]) -> GraphEngineErrorCode:
    return error.value.code


def test_minimal_legal_template_instantiates_and_reparses() -> None:
    template = _example(GECEGTemplate, "GECEGTemplate.valid.json")
    result = instantiate_template(template, _case(), _context_for_minimal(template))
    assert validate_template(template) is template
    assert GECEGInstance.model_validate(result.instance.model_dump(mode="json")) == result.instance
    assert result.instance.template_ref.checksum == canonical_checksum(template)
    assert result.instance.provenance.input_checksums == (result.trace.input_checksum,)
    assert result.trace.input_payload_version == INSTANTIATION_INPUT_PAYLOAD_VERSION


def _context_for_minimal(template: GECEGTemplate) -> InstantiationContext:
    return InstantiationContext(
        graph_code="GRAPH:OID_GRAPH03M1",
        graph_version=1,
        template_checksum=canonical_checksum(template),
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
        generator_version="03A.1.0",
        source_codes=("SOURCE:SYNTHETIC_03A",),
        resolutions=(
            NodeResolution(
                node_code="EVIDENCE:IMAGE_USABLE",
                applicability=Applicability.APPLICABLE,
                availability=DataAvailability.AVAILABLE,
                artifact_bindings=(
                    ArtifactNodeBinding(
                        asset_code="ASSET:OID_ASSET001",
                        binding_code="BINDING:OID_BIND0302",
                    ),
                ),
                reason_code="RULE:INSTANTIATION:EXISTING_DATA",
            ),
        ),
    )


def test_repeat_is_deterministic_resolution_order_independent_and_template_immutable() -> None:
    template = _template()
    before = template.model_dump_json()
    context = _context(template)
    reversed_context = InstantiationContext(
        graph_code=context.graph_code,
        graph_version=context.graph_version,
        template_checksum=context.template_checksum,
        created_at=context.created_at,
        generator_version=context.generator_version,
        source_codes=context.source_codes,
        resolutions=tuple(reversed(context.resolutions)),
        rule_versions=context.rule_versions,
    )
    first = instantiate_template(template, _case(), context)
    second = instantiate_template(template, _case(), context)
    reordered = instantiate_template(template, _case(), reversed_context)
    assert first == second == reordered
    assert template.model_dump_json() == before


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("graph_code", "GRAPH:OID_GRAPH03A2"),
        ("graph_version", 2),
        ("run_id", "RUN:OID_RUN03A01"),
        ("created_at", datetime(2026, 8, 31, tzinfo=UTC) + timedelta(seconds=1)),
        ("generator_version", "03A.1.1"),
        ("source_codes", ("SOURCE:SYNTHETIC_03A", "SOURCE:SYNTHETIC_03A_ALT")),
        (
            "rule_versions",
            (_ref(), _ref("RULE:INSTANTIATION:SYNTHETIC_ALT")),
        ),
    ),
)
def test_input_checksum_changes_for_every_output_affecting_context_field(
    field: str, value: object
) -> None:
    template = _template()
    context = _context(template)
    original = instantiate_template(template, _case(), context)
    changed = instantiate_template(template, _case(), replace(context, **{field: value}))
    assert changed.trace.input_checksum != original.trace.input_checksum


def test_set_semantics_normalize_source_rule_and_resolution_order_in_output() -> None:
    template = _template()
    context = replace(
        _context(template),
        source_codes=("SOURCE:SYNTHETIC_03A_ALT", "SOURCE:SYNTHETIC_03A"),
        rule_versions=(_ref("RULE:INSTANTIATION:SYNTHETIC_ALT"), _ref()),
    )
    reordered = replace(
        context,
        source_codes=tuple(reversed(context.source_codes)),
        rule_versions=tuple(reversed(context.rule_versions)),
        resolutions=tuple(reversed(context.resolutions)),
    )
    first = instantiate_template(template, _case(), context)
    second = instantiate_template(template, _case(), reordered)
    assert first.trace.input_checksum == second.trace.input_checksum
    assert first == second


def test_canonical_checksum_rejects_unknown_non_json_objects() -> None:
    with pytest.raises(TypeError, match="canonical JSON"):
        canonical_checksum(object())


def test_clinical_and_data_flow_edges_are_distinct_and_validated() -> None:
    assert CLINICAL_EDGE_TYPES.isdisjoint(DATA_FLOW_EDGE_TYPES)
    assert CLINICAL_EDGE_TYPES | DATA_FLOW_EDGE_TYPES == frozenset(EdgeType)
    validate_template(_template())

    base = _example(GECEGTemplate, "GECEGTemplate.valid.json")
    invalid_edge = GraphEdge(
        source_node_code="EVIDENCE:IMAGE_USABLE",
        target_node_code="GOAL:SYNTHETIC_SCREEN",
        edge_type=EdgeType.PRODUCES,
    )
    invalid = base.model_copy(update={"edges": (invalid_edge,)})
    with pytest.raises(GraphEngineError) as error:
        validate_template(invalid)
    assert _error_code(error) is GraphEngineErrorCode.EDGE_NODE_TYPE_MISMATCH


def test_goal_missing_and_duplicate_nodes_have_stable_errors() -> None:
    base = _example(GECEGTemplate, "GECEGTemplate.valid.json")
    without_goal = base.model_copy(
        update={
            "nodes": tuple(node for node in base.nodes if node.node_type is not NodeType.GOAL),
            "edges": (),
        }
    )
    with pytest.raises(GraphEngineError) as missing:
        validate_template(without_goal)
    assert _error_code(missing) is GraphEngineErrorCode.GOAL_COUNT

    duplicate = base.model_copy(update={"nodes": (*base.nodes, base.nodes[0])})
    with pytest.raises(GraphEngineError) as repeated:
        validate_template(duplicate)
    assert _error_code(repeated) is GraphEngineErrorCode.DUPLICATE_NODE


@pytest.mark.parametrize(
    ("edge", "code"),
    (
        (
            GraphEdge(
                source_node_code="EVIDENCE:UNKNOWN",
                target_node_code="GOAL:SYNTHETIC_SCREEN",
                edge_type=EdgeType.SUPPORTS,
            ),
            GraphEngineErrorCode.UNKNOWN_EDGE_ENDPOINT,
        ),
        (
            GraphEdge(
                source_node_code="EVIDENCE:IMAGE_USABLE",
                target_node_code="EVIDENCE:IMAGE_USABLE",
                edge_type=EdgeType.SUPPORTS,
            ),
            GraphEngineErrorCode.SELF_REFERENCE,
        ),
    ),
)
def test_unknown_endpoint_and_self_reference_have_stable_errors(
    edge: GraphEdge, code: GraphEngineErrorCode
) -> None:
    base = _example(GECEGTemplate, "GECEGTemplate.valid.json")
    invalid = base.model_copy(update={"edges": (edge,)})
    with pytest.raises(GraphEngineError) as error:
        validate_template(invalid)
    assert _error_code(error) is code


def test_duplicate_edges_and_unexplained_disconnected_structure_are_rejected() -> None:
    base = _example(GECEGTemplate, "GECEGTemplate.valid.json")
    duplicate = base.model_copy(update={"edges": (base.edges[0], base.edges[0])})
    with pytest.raises(GraphEngineError) as repeated:
        validate_template(duplicate)
    assert _error_code(repeated) is GraphEngineErrorCode.DUPLICATE_EDGE

    orphan = TemplateNode(
        node_code="EVIDENCE:ORPHAN",
        node_type=NodeType.EVIDENCE,
        concept_code="CONCEPT:ORPHAN",
        required=False,
    )
    disconnected = base.model_copy(update={"nodes": (*base.nodes, orphan)})
    with pytest.raises(GraphEngineError) as unreachable:
        validate_template(disconnected)
    assert _error_code(unreachable) is GraphEngineErrorCode.UNREACHABLE_NODE


def test_four_availability_cases_are_distinct_and_missing_is_not_medical_negative() -> None:
    result = instantiate_template(_template(), _case(), _context(_template()))
    nodes = {str(node.node_code): node for node in result.instance.nodes}
    traces = {trace.node_code: trace for trace in result.trace.node_results}

    existing = nodes["ARTIFACT:SYNTHETIC_INPUT"]
    assert existing.availability is DataAvailability.AVAILABLE
    assert existing.evidence_bindings == ("BINDING:OID_BIND0301",)
    assert traces["ARTIFACT:SYNTHETIC_INPUT"].mapped_artifact_codes == ("ASSET:OID_ASSET001",)
    assert traces["ARTIFACT:SYNTHETIC_INPUT"].observation_concept_codes == (
        "CONCEPT:IMAGE_AVAILABLE",
    )
    missing = nodes["ARTIFACT:SYNTHETIC_OUTPUT"]
    assert missing.availability is DataAvailability.UNAVAILABLE
    assert (
        traces["ARTIFACT:SYNTHETIC_OUTPUT"].resolution_kind is ResolutionKind.MAPPING_NOT_PROVIDED
    )
    capability_gap = nodes["TOOL:SYNTHETIC_ANALYZER"]
    assert capability_gap.availability is DataAvailability.UNAVAILABLE
    assert (
        traces["TOOL:SYNTHETIC_ANALYZER"].capability
        is CapabilityAvailability.UNAVAILABLE_CAPABILITY
    )
    assert "EVIDENCE:OPTIONAL_PATH" not in nodes
    assert result.trace.pruned_nodes[0].node_code == "EVIDENCE:OPTIONAL_PATH"
    assert result.trace.pruned_nodes[0].rule_ref is not None
    assert result.trace.pruned_nodes[0].input_checksum == result.trace.input_checksum

    assert missing.status is GraphNodeStatus.PENDING
    assert capability_gap.status is GraphNodeStatus.PENDING
    assert all(
        node.status not in {GraphNodeStatus.REFUTED, GraphNodeStatus.EXCLUDED}
        for node in result.instance.nodes
    )


def test_available_observation_mapping_is_retained_in_trace_without_artifact_guessing() -> None:
    template = _example(GECEGTemplate, "GECEGTemplate.valid.json")
    context = replace(
        _context_for_minimal(template),
        resolutions=(
            NodeResolution(
                node_code="EVIDENCE:IMAGE_USABLE",
                applicability=Applicability.APPLICABLE,
                availability=DataAvailability.AVAILABLE,
                observation_concept_codes=("CONCEPT:IMAGE_AVAILABLE",),
                reason_code="RULE:INSTANTIATION:EXISTING_OBSERVATION",
            ),
        ),
    )
    result = instantiate_template(template, _case(), context)
    trace = next(
        item
        for item in result.trace.node_results
        if item.node_code == context.resolutions[0].node_code
    )
    assert trace.mapped_artifact_codes == ()
    assert trace.observation_concept_codes == ("CONCEPT:IMAGE_AVAILABLE",)
    assert (
        next(node for node in result.instance.nodes if node.node_type is NodeType.EVIDENCE).status
        is GraphNodeStatus.PENDING
    )


def test_available_without_artifact_or_observation_source_is_rejected() -> None:
    template = _example(GECEGTemplate, "GECEGTemplate.valid.json")
    context = replace(
        _context_for_minimal(template),
        resolutions=(
            NodeResolution(
                node_code="EVIDENCE:IMAGE_USABLE",
                applicability=Applicability.APPLICABLE,
                availability=DataAvailability.AVAILABLE,
            ),
        ),
    )
    with pytest.raises(GraphEngineError) as error:
        instantiate_template(template, _case(), context)
    assert _error_code(error) is GraphEngineErrorCode.INVALID_RESOLUTION


@pytest.mark.parametrize(
    "bridge_node_code",
    (
        "EVIDENCE:IMAGE_USABLE",
        "CLAIM:SYNTHETIC_QC",
        "ARTIFACT:SYNTHETIC_OUTPUT",
        "TOOL:SYNTHETIC_ANALYZER",
    ),
)
def test_pruning_a_bridge_cannot_leave_goal_disconnected_subgraphs(
    bridge_node_code: str,
) -> None:
    template = _template()
    context = replace(
        _context(template),
        resolutions=(
            NodeResolution(
                node_code=bridge_node_code,
                applicability=Applicability.NOT_APPLICABLE,
                availability=DataAvailability.NOT_APPLICABLE,
                reason_code="RULE:INSTANTIATION:BRIDGE_NOT_APPLICABLE",
                rule_ref=_ref("RULE:INSTANTIATION:BRIDGE_NOT_APPLICABLE"),
            ),
        ),
    )
    with pytest.raises(GraphEngineError) as error:
        instantiate_template(template, _case(), context)
    assert _error_code(error) is GraphEngineErrorCode.PRUNING_DISCONNECTS_GRAPH


def test_artifacts_are_never_guessed_and_bad_explicit_mapping_is_rejected() -> None:
    template = _example(GECEGTemplate, "GECEGTemplate.valid.json")
    no_mapping = _context_for_minimal(template)
    no_mapping = InstantiationContext(
        graph_code=no_mapping.graph_code,
        graph_version=1,
        template_checksum=no_mapping.template_checksum,
        created_at=no_mapping.created_at,
        generator_version=no_mapping.generator_version,
        source_codes=no_mapping.source_codes,
        resolutions=(),
    )
    result = instantiate_template(template, _case(), no_mapping)
    evidence = next(node for node in result.instance.nodes if node.node_type is NodeType.EVIDENCE)
    assert evidence.evidence_bindings == ()
    assert evidence.availability is DataAvailability.UNAVAILABLE

    bad = _context_for_minimal(template)
    bad_resolution = bad.resolutions[0]
    bad = InstantiationContext(
        graph_code=bad.graph_code,
        graph_version=1,
        template_checksum=bad.template_checksum,
        created_at=bad.created_at,
        generator_version=bad.generator_version,
        source_codes=bad.source_codes,
        resolutions=(
            NodeResolution(
                node_code=bad_resolution.node_code,
                applicability=bad_resolution.applicability,
                availability=bad_resolution.availability,
                artifact_bindings=(
                    ArtifactNodeBinding(
                        asset_code="ASSET:OID_UNKNOWN01",
                        binding_code="BINDING:OID_BIND0303",
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(GraphEngineError) as error:
        instantiate_template(template, _case(), bad)
    assert _error_code(error) is GraphEngineErrorCode.UNKNOWN_ARTIFACT


def test_checksum_mismatch_is_rejected() -> None:
    template = _template()
    context = _context(template)
    context = InstantiationContext(
        graph_code=context.graph_code,
        graph_version=1,
        template_checksum="sha256:" + "0" * 64,
        created_at=context.created_at,
        generator_version=context.generator_version,
        source_codes=context.source_codes,
        resolutions=context.resolutions,
    )
    with pytest.raises(GraphEngineError) as error:
        instantiate_template(template, _case(), context)
    assert _error_code(error) is GraphEngineErrorCode.TEMPLATE_CHECKSUM_MISMATCH


def test_template_code_and_version_are_revalidated_by_frozen_contract() -> None:
    template = _example(GECEGTemplate, "GECEGTemplate.valid.json")
    invalid = template.model_copy(update={"template_version": "bad/version"})
    with pytest.raises(GraphEngineError) as error:
        validate_template(invalid)
    assert _error_code(error) is GraphEngineErrorCode.TEMPLATE_CONTRACT_INVALID


def test_public_import_does_not_load_database_agents_openai_or_tool_runtime() -> None:
    code = """
import sys
import packages.graph_engine
forbidden = ('db', 'agents', 'openai', 'apps', 'Algorithm_Tool', 'Seg_Model')
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden)
)
assert not loaded, loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def repository(database_lease) -> Iterator[PostgresRuntimeRepository]:
    database_lease.start(_free_port())
    try:
        admin = manage.test_target("bootstrap")
        migration = manage.test_target("migration")
        runtime = manage.test_target("runtime")
        manage.bootstrap(admin, migration.user)
        manage.bootstrap_runtime_principal(admin, runtime.user)
        manage.migrate(migration)
        yield PostgresRuntimeRepository(runtime)
    finally:
        database_lease.cleanup()


def test_instance_round_trips_through_runtime_repository_public_api(
    repository: PostgresRuntimeRepository,
) -> None:
    case = _case()
    template = _template()
    result = instantiate_template(template, case, _context(template))
    repository.create_case_bundle(case, "TASK:OID_CASE03A1")
    stored = repository.create_graph_snapshot(result.instance, "TASK:OID_GRAPH03A1")
    loaded = repository.get_graph_snapshot(stored.graph_code, stored.graph_version)
    assert loaded == result.instance
    assert GECEGInstance.model_validate(loaded.model_dump(mode="json")) == loaded
