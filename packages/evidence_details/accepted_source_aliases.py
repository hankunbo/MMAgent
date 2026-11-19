"""Reversible alias-only projection for a NEW original-owner aggregation.

Never repair model responses or alter saved acceptances. Both manifests must be
rebuilt by their installed material owners; original contents and all medical
policy checks remain mandatory after projecting identities.
"""

from typing import Literal

from pydantic import Field

from packages.evidence_acquisition.source_aliases import alias_sources
from packages.grounding.models import Checksum, Value, digest, validate
from packages.planner_control.models import Reference

from .acceptance_models import DomainState


class AliasTransfer(Value):
    original: Reference
    before: Reference
    after: Reference
    source_checksum: Checksum


class AliasProjection(Value):
    schema_version: Literal["accepted-source-alias-projection/1.0.0"] = (
        "accepted-source-alias-projection/1.0.0"
    )
    original_manifest_checksum: Checksum
    current_manifest_checksum: Checksum
    transfers: tuple[AliasTransfer, ...] = Field(min_length=1, max_length=256)
    projected_state_checksum: Checksum
    projected_candidate_checksum: Checksum
    medical_values_and_explanation_unchanged: Literal[True] = True


def _originals(state, manifest):
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "bindings",
        "checksum",
    }:
        raise ValueError("aggregation_alias_manifest_required")
    bindings = manifest["bindings"]
    if not isinstance(bindings, list) or len(bindings) != len(state.material.sources):
        raise ValueError("aggregation_alias_manifest_scope")
    by_alias = {b["alias"]["code"]: b for b in bindings}
    if len(by_alias) != len(bindings):
        raise ValueError("aggregation_alias_duplicate")
    originals = []
    for source in state.material.sources:
        binding = by_alias.get(source.reference.code)
        if binding is None or binding["alias"] != source.reference.model_dump(mode="json"):
            raise ValueError("aggregation_alias_source_binding")
        raw = source.model_dump(mode="json")
        raw["reference"] = binding["original"]
        original = validate(type(source), raw)
        if digest(original) != binding["source_checksum"]:
            raise ValueError("aggregation_alias_source_content_changed")
        originals.append(original)
    aliases, rebuilt = alias_sources(tuple(originals))
    if aliases != state.material.sources or rebuilt != manifest:
        raise ValueError("aggregation_alias_manifest_changed")
    return {s.reference.code: s for s in originals}, by_alias


def project_aliases(old, current, candidate, original_manifest, current_manifest):
    """Exact reversible names only; caller still runs compatible_local_ids."""
    originals, bindings = _originals(old, original_manifest)
    current_originals, new_bindings = _originals(current, current_manifest)
    by_original = {b["original"]["code"]: b for b in new_bindings.values()}
    mapping, transfers = {}, []
    for code, binding in bindings.items():
        original_code = binding["original"]["code"]
        if current_originals.get(original_code) != originals[original_code]:
            raise ValueError("aggregation_alias_original_source_changed")
        following = by_original[original_code]
        mapping[code] = following["alias"]
        transfers.append(
            validate(
                AliasTransfer,
                dict(
                    original=binding["original"],
                    before=binding["alias"],
                    after=following["alias"],
                    source_checksum=binding["source_checksum"],
                ),
            )
        )
    raw = old.model_dump(mode="json")
    for part in (raw["policy"]["items"], raw["policy"]["requirements"]):
        for item in part:
            if "source_codes" in item:
                item["source_codes"] = [mapping[c]["code"] for c in item["source_codes"]]
            definition = item["definition"]
            refs = []
            for ref in definition["source_refs"]:
                if ref != bindings[ref["code"]]["alias"]:
                    raise ValueError("aggregation_alias_definition_binding")
                refs.append(mapping[ref["code"]])
            definition["source_refs"] = refs
            definition["reference"].update(
                version="research-source-alias/1.0.0",
                checksum=digest({k: v for k, v in definition.items() if k != "reference"}),
            )
    material = raw["material"]
    for source in material["sources"]:
        source["reference"] = mapping[source["reference"]["code"]]
    material["policy_ref"]["checksum"] = material["scope"]["policy_checksum"] = digest(
        raw["policy"]
    )
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    projected = validate(DomainState, raw)
    rows = candidate.model_dump(mode="json")
    for item in rows["items"]:
        if any(c not in mapping for c in item["reference_citations"]):
            raise ValueError("aggregation_alias_unknown_citation")
        item["reference_citations"] = [mapping[c]["code"] for c in item["reference_citations"]]
    rows["scope_checksum"] = digest(projected.material.scope)
    projected_candidate = validate(type(candidate), rows)
    proof = AliasProjection(
        original_manifest_checksum=digest(original_manifest),
        current_manifest_checksum=digest(current_manifest),
        transfers=tuple(transfers),
        projected_state_checksum=digest(projected),
        projected_candidate_checksum=digest(projected_candidate),
    )
    return projected, projected_candidate, proof
