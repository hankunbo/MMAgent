"""05A strict model, manifest scope, OR/AND and immutable external port contracts."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from packages.grounding import GroundingError, Mapping, Policy, digest, validate, verify
from packages.grounding.synthetic import SyntheticCatalogReader, fixture
from packages.retrieval.knowledge.models import RetrievalContext


def reseal(model, data):
    data = copy.deepcopy(data)
    data.pop("checksum", None)
    data["checksum"] = digest(data)
    return validate(model, data)


def test_multi_artifact_or_candidates_have_complete_and_inputs():
    m, p, c, k, _ = fixture()
    assert len(m.accepted_artifacts) == len(m.candidates) == 2
    assert all(
        [i.required for i in candidate.inputs] == [True, True, False] for candidate in m.candidates
    )
    assert verify(m, p, k, SyntheticCatalogReader([c]), RetrievalContext(), admission=True) == m
    assert validate(Mapping, json.loads(m.model_dump_json())) == m


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "null",
        "empty",
        "patient",
        "unknown_enum",
        "scope",
        "port",
        "checksum",
        "duplicate",
        "type",
        "free_term",
    ],
)
def test_illegal_mapping(mutation):
    m, p, c, k, _ = fixture()
    data = m.model_dump(mode="json")
    if mutation == "missing":
        del data["candidates"][0]["inputs"][0]["required"]
    elif mutation == "null":
        data["candidates"][0]["inputs"] = None
    elif mutation == "empty":
        data["accepted_artifacts"][0]["unit"]["code"] = ""
    elif mutation == "patient":
        data["patient_id"] = "CASE:OID_PATIENT1"
    elif mutation == "unknown_enum":
        data["candidates"][0]["validation_required"] = "validated"
    elif mutation == "scope":
        data["candidates"][0]["inputs"][0]["port"]["tool_ref"] = data["candidates"][1]["tool_ref"]
    elif mutation == "port":
        data["candidates"][0]["output_port"]["tool_ref"]["version"] = "0.9.0"
    elif mutation == "checksum":
        data["checksum"] = "sha256:" + "0" * 64
    elif mutation == "duplicate":
        data["candidates"][1]["code"] = data["candidates"][0]["code"]
    elif mutation == "type":
        data["synthetic"] = "true"
    elif mutation == "free_term":
        data["candidates"][0]["output"]["scope_terms"] = ["CASE:OID_PATIENT1"]
    with pytest.raises((ValidationError, ValueError)):
        if mutation == "checksum":
            validate(Mapping, data)
        else:
            reseal(Mapping, data)


@pytest.mark.parametrize(
    "mutation",
    [
        "catalog_checksum",
        "catalog_version",
        "tool_checksum",
        "capability",
        "implementation",
        "missing_input",
        "wrong_input_type",
        "workflow",
        "unregistered_scope",
    ],
)
def test_exact_external_references_and_complete_input_sets(mutation):
    m, p, c, k, _ = fixture()
    data = m.model_dump(mode="json")
    candidate = data["candidates"][0]
    if mutation.startswith("catalog_"):
        data["catalog_ref"][mutation.split("_")[1]] = (
            "0.9.0" if mutation.endswith("version") else "sha256:" + "b" * 64
        )
    elif mutation == "tool_checksum":
        for r in [
            candidate["tool_ref"],
            candidate["output_port"]["tool_ref"],
            *(i["port"]["tool_ref"] for i in candidate["inputs"]),
        ]:
            r["checksum"] = "sha256:" + "c" * 64
    elif mutation in ("capability", "implementation", "workflow"):
        candidate[mutation + "_ref"]["checksum"] = "sha256:" + "c" * 64
    elif mutation == "missing_input":
        candidate["inputs"].pop(0)
    elif mutation == "wrong_input_type":
        candidate["inputs"][0]["constraint"]["artifact_ref"] = data["accepted_artifacts"][0][
            "artifact_ref"
        ]
    elif mutation == "unregistered_scope":
        candidate["inputs"][0]["constraint"]["scope_terms"] = ["CONCEPT:ARBITRARY"]
    bad = reseal(Mapping, data)
    with pytest.raises(GroundingError):
        verify(bad, p, k, SyntheticCatalogReader([c]), RetrievalContext(), admission=True)


def test_catalog_is_immutable_and_safety_is_separate():
    m, p, c, k, raw = fixture()
    reader = SyntheticCatalogReader([c])
    assert c.manifest_checksum == digest(raw)
    assert c.tools[0].tool_ref.checksum == digest(raw[0])
    for tool, manifest in zip(c.tools, raw, strict=True):
        assert all(
            set(port)
            == {
                "direction",
                "port_name",
                "artifact_code",
                "artifact_name_zh",
                "required",
                "cardinality_min",
                "cardinality_max",
                "constraints",
            }
            for port in manifest["ports"]
        )
        assert tool.output_ports[0].port_name == manifest["ports"][-1]["port_name"]
    later = reseal(type(c), {**c.model_dump(mode="json"), "synthetic": False})
    with pytest.raises(GroundingError):
        reader.add(later)
    reader.disable(c.tools[0].implementation_ref)
    assert reader.read_snapshot(c.reference()) == c
    with pytest.raises(GroundingError, match="tool_safety_revoked"):
        verify(m, p, k, reader, RetrievalContext(), admission=True)
    verify(m, p, k, reader, RetrievalContext(), admission=False)


@pytest.mark.parametrize(
    "field,value", [("max_candidates", 1), ("max_inputs", 1), ("max_bytes", 1024)]
)
def test_configuration_limits(field, value):
    m, p, c, k, _ = fixture()
    new_policy = reseal(Policy, {**p.model_dump(mode="json"), field: value})
    m = reseal(
        Mapping,
        {**m.model_dump(mode="json"), "policy_ref": new_policy.reference().model_dump(mode="json")},
    )
    with pytest.raises(GroundingError):
        verify(m, new_policy, k, SyntheticCatalogReader([c]), RetrievalContext(), admission=True)


def test_construct_copy_and_draft_knowledge_cannot_bypass():
    m, p, c, k, _ = fixture()
    bad = m.model_copy(update={"synthetic": "false"})
    with pytest.raises(ValidationError):
        verify(bad, p, k, SyntheticCatalogReader([c]), RetrievalContext(), admission=True)
    k.published = False
    with pytest.raises(GroundingError):
        verify(m, p, k, SyntheticCatalogReader([c]), RetrievalContext(), admission=True)
    verify(m, p, k, SyntheticCatalogReader([c]), RetrievalContext(), admission=False)


def test_schemas_config_examples_are_current():
    from db.runtime.repository.grounding import GroundingLock
    from packages.grounding import CatalogSnapshot

    root = Path(__file__).resolve().parents[2]
    for model in (Mapping, Policy, CatalogSnapshot, GroundingLock):
        assert (
            json.loads(
                (root / "packages/grounding/schemas" / f"{model.__name__}.schema.json").read_text()
            )
            == model.model_json_schema()
        )
    assert (
        validate(Policy, json.loads((root / "configs/grounding_policy.json").read_text()))
        == fixture()[1]
    )

    mapping, _, catalog, _, raw = fixture()
    folder = root / "packages/grounding/examples"
    assert validate(Mapping, json.loads((folder / "Mapping.synthetic.json").read_text())) == mapping
    assert (
        validate(
            CatalogSnapshot, json.loads((folder / "CatalogSnapshot.synthetic.json").read_text())
        )
        == catalog
    )
    assert json.loads((folder / "native-manifest.synthetic.json").read_text()) == raw
