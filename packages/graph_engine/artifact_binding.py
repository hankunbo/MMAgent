"""Controlled append-only Artifact binding; never assigns a medical node status.

This is a 03 application port beside frozen v1 GraphDelta. Binding events preserve
all existing facts and replay before subsequent accepted status deltas. Authorization
and exact reviewed node/type mapping belong to the application/Knowledge owner.
"""

import json

from packages.contracts import Artifact, DataAvailability, GECEGInstance

from .engine import canonical_checksum, validate_instance_structure

BINDING_VERSION = "03_ARTIFACT_BINDING_V1"


def bind_artifacts(base, bindings, *, expected_checksum, rule_ref):
    base = validate_instance_structure(base)
    if (
        canonical_checksum(base) != expected_checksum
        or rule_ref not in base.provenance.rule_versions
    ):
        raise ValueError("artifact_binding_base_or_rule_mismatch")
    nodes = {str(n.node_code): n for n in base.nodes}
    material = []
    for node_code, artifact in bindings:
        artifact = Artifact.model_validate_json(artifact.model_dump_json())
        node = nodes.get(node_code)
        if (
            node is None
            or node.node_type.value not in {"EVIDENCE", "DATA", "QC"}
            or artifact.scope.case_code != base.case_code
            or artifact.scope.run_id not in (None, base.scope.run_id)
            or artifact.qc_status.value != "PASS"
            or artifact.laterality != base.laterality
        ):
            raise ValueError("artifact_binding_scope_or_qc_mismatch")
        if artifact.asset_code in node.evidence_bindings:
            raise ValueError("artifact_binding_already_present")
        material.append((node_code, artifact))
    if not material or len({(n, a.asset_code) for n, a in material}) != len(material):
        raise ValueError("artifact_binding_empty_or_duplicate")
    material.sort(key=lambda v: (v[0], v[1].asset_code))
    for node_code, artifact in material:
        node = nodes[node_code]
        nodes[node_code] = node.model_copy(
            update={
                "availability": DataAvailability.AVAILABLE,
                "evidence_bindings": tuple(sorted((*node.evidence_bindings, artifact.asset_code))),
            }
        )
    # Binding changes availability only. In particular QC PASS never sets SUPPORTED.
    document = base.model_dump(mode="json")
    document.update(
        graph_version=base.graph_version + 1,
        nodes=[nodes[str(n.node_code)].model_dump(mode="json") for n in base.nodes],
    )
    result = GECEGInstance.model_validate_json(json.dumps(document))
    validate_instance_structure(result)
    event = dict(
        payload_version=BINDING_VERSION,
        base_graph_version=base.graph_version,
        new_graph_version=result.graph_version,
        base_checksum=expected_checksum,
        rule_ref=rule_ref.model_dump(mode="json"),
        bindings=[dict(node_code=n, artifact=a.model_dump(mode="json")) for n, a in material],
        result_checksum=canonical_checksum(result),
    )
    event["event_checksum"] = canonical_checksum(event)
    return result, event


def replay_artifact_binding(base, event):
    from packages.contracts import VersionedReference

    result, expected = bind_artifacts(
        base,
        tuple((b["node_code"], Artifact.model_validate(b["artifact"])) for b in event["bindings"]),
        expected_checksum=event["base_checksum"],
        rule_ref=VersionedReference.model_validate(event["rule_ref"]),
    )
    if canonical_checksum(event) != canonical_checksum(expected):
        raise ValueError("artifact_binding_event_mismatch")
    return result
