"""Pure 03 binding checks: no medical state assignment and type-sensitive event replay."""

from copy import deepcopy

import pytest

from packages.contracts import Artifact, GECEGInstance, VersionedReference
from packages.graph_engine import bind_artifacts, canonical_checksum, replay_artifact_binding
from tests.graph.test_graph_engine_03b import _base, _example, _rule_ref


def fixture():
    base = _base()
    base = GECEGInstance.model_validate_json(
        base.model_copy(
            update={
                "provenance": base.provenance.model_copy(update={"rule_versions": (_rule_ref(),)})
            }
        ).model_dump_json()
    )
    a = _example(Artifact, "Artifact.valid.json")
    a = Artifact.model_validate_json(
        a.model_copy(
            update={
                "asset_code": "ASSET:OID_SYNTHETIC_BINDING",
                "laterality": base.laterality,
                "scope": a.scope.model_copy(update={"case_code": base.case_code}),
                "qc_status": "PASS",
            }
        ).model_dump_json()
    )
    return base, a


def test_binding_preserves_status_and_replays():
    base, a = fixture()
    original = canonical_checksum(base)
    new, event = bind_artifacts(
        base, (("EVIDENCE:IMAGE_USABLE", a),), expected_checksum=original, rule_ref=_rule_ref()
    )
    assert new.graph_version == base.graph_version + 1
    assert [n.status for n in new.nodes] == [n.status for n in base.nodes]
    assert replay_artifact_binding(base, event) == new
    assert canonical_checksum(base) == original
    with pytest.raises(ValueError):
        bind_artifacts(
            new,
            (("EVIDENCE:IMAGE_USABLE", a),),
            expected_checksum=canonical_checksum(new),
            rule_ref=_rule_ref(),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_graph_version", True),
        ("new_graph_version", 7),
        ("result_checksum", "sha256:" + "b" * 64),
        ("extra", "x"),
    ],
)
def test_binding_event_tampering_rejected(field, value):
    base, a = fixture()
    _, event = bind_artifacts(
        base,
        (("EVIDENCE:IMAGE_USABLE", a),),
        expected_checksum=canonical_checksum(base),
        rule_ref=_rule_ref(),
    )
    event = deepcopy(event)
    event[field] = value
    with pytest.raises(ValueError):
        replay_artifact_binding(base, event)


@pytest.mark.parametrize("mode", ["case", "run", "qc", "eye", "rule", "checksum", "goal"])
def test_binding_scope_and_authority_guards(mode):
    base, a = fixture()
    node = "EVIDENCE:IMAGE_USABLE"
    rule = _rule_ref()
    expected = canonical_checksum(base)
    if mode == "case":
        a = a.model_copy(
            update={"scope": a.scope.model_copy(update={"case_code": "CASE:OID_OTHER0001"})}
        )
    if mode == "run":
        a = a.model_copy(
            update={
                "scope": {
                    "scope_type": "RUN",
                    "case_code": base.case_code,
                    "run_id": "RUN:OID_OTHER0001",
                }
            }
        )
    if mode == "qc":
        a = a.model_copy(update={"qc_status": "FAIL"})
    if mode == "eye":
        a = a.model_copy(update={"laterality": "LEFT"})
    if mode == "rule":
        rule = VersionedReference(code="RULE:OTHER", version="1.0.0", checksum="sha256:" + "b" * 64)
    if mode == "checksum":
        expected = "sha256:" + "b" * 64
    if mode == "goal":
        node = base.goal_node_code
    with pytest.raises(ValueError):
        bind_artifacts(base, ((node, a),), expected_checksum=expected, rule_ref=rule)
