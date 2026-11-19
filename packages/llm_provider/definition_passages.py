"""Versioned literal passage selection, resolved before original definition admission.

Selection replaces transcription only. No approximate matching, repaired response,
medical inference, or current-source qualification is supplied here.
"""

from copy import deepcopy
from typing import Literal

from pydantic import Field

from packages.evidence_reuse.definitions import (
    DefinitionProposal,
    OwnerBoundDefinitionProposal,
    QuotedOwnerBoundConfirmedInterpretationDraft,
    QuotedOwnerBoundResearchDefinitionDraft,
)
from packages.grounding.models import Value, digest, validate

from .definition_identity import material_map

PROMPT = "definition-draft-6.8.0"


class PassageSelection(Value):
    passage_id: str = Field(min_length=1, max_length=80)


class SelectedResearchDraft(QuotedOwnerBoundResearchDefinitionDraft):
    schema_version: Literal["source-definition-draft/6.2.0"] = "source-definition-draft/6.2.0"
    excerpts: tuple[PassageSelection, ...] = Field(min_length=1, max_length=8)


class SelectedConfirmedDraft(QuotedOwnerBoundConfirmedInterpretationDraft):
    schema_version: Literal["source-definition-draft/7.2.0"] = "source-definition-draft/7.2.0"
    excerpts: tuple[PassageSelection, ...] = Field(min_length=1, max_length=8)


class SelectedDefinitionProposal(DefinitionProposal):
    schema_version: Literal["research-definition-proposal/6.0.0"] = (
        "research-definition-proposal/6.0.0"
    )
    definitions: tuple[SelectedResearchDraft | SelectedConfirmedDraft, ...] = Field(max_length=3)


def choices(payload):
    """Lossless contiguous source coverage; no normalization or semantic filtering."""
    result = {}
    for code, source in material_map(payload).items():
        text, start = source["text"], 0
        while start < len(text):
            end = min(start + 1800, len(text))
            if end < len(text):
                boundary = max(
                    text.rfind("\n", start + 600, end), text.rfind(". ", start + 600, end)
                )
                if boundary >= start + 600:
                    end = boundary + 1
            raw = text[start:end]
            identifier = "passage:" + digest((code, source["checksum"], start, end, raw))[7:]
            result[identifier] = dict(
                source_code=code,
                source_checksum=source["checksum"],
                start=start,
                end=end,
                text=raw,
                prefix=text[max(0, start - 240) : start] or None,
                suffix=text[end : end + 240] or None,
            )
            start = end
    return result


def project(payload):
    result = deepcopy(payload)
    selected = choices(payload)
    result["source_materials"] = [
        dict(
            code=s["code"],
            checksum=s["checksum"],
            passage_ids=[key for key, row in selected.items() if row["source_code"] == s["code"]],
        )
        for s in payload["source_materials"]
    ]
    for group in result.get("candidate_contract_choices", {}).values():
        if isinstance(group, list):
            for item in group:
                fields = item.get("copy_fields", {})
                if fields.get("schema_version") in {
                    "source-definition-draft/6.1.0",
                    "source-definition-draft/7.1.0",
                }:
                    fields["schema_version"] = fields["schema_version"].replace(".1.0", ".2.0")
    result.pop("citation_contract", None)
    result.pop("definition_identity_contract", None)
    result["definition_passage_contract"] = dict(
        schema_version="definition-passage-selection/1.0.0",
        target=payload["target"],
        passages=[dict(passage_id=k, **v) for k, v in selected.items()],
        meaning="Choose exact passage_id only. "
        "Full source text is preserved across ordered passages. "
        "Selection is citation identity, not proof of medical entailment or current acceptance.",
    )
    return result


def decode(raw, payload):
    candidate = validate(SelectedDefinitionProposal, raw)
    value = candidate.model_dump(mode="json")
    selected = choices(payload)
    used = []
    for draft in value["definitions"]:
        if draft["target"] != payload["target"]:
            raise ValueError("definition_passage_target")
        ids = [e["passage_id"] for e in draft["excerpts"]]
        if len(set(ids)) != len(ids) or not set(ids) <= selected.keys():
            raise ValueError("definition_passage_selection")
        draft["excerpts"] = [
            {k: v for k, v in selected[key].items() if k not in {"start", "end"}} for key in ids
        ]
        draft["schema_version"] = {
            "source-definition-draft/6.2.0": "source-definition-draft/6.1.0",
            "source-definition-draft/7.2.0": "source-definition-draft/7.1.0",
        }[draft["schema_version"]]
        used.extend(ids)
    value["schema_version"] = "research-definition-proposal/5.0.0"
    proposal = validate(OwnerBoundDefinitionProposal, value)
    return proposal, dict(
        schema_version="definition-passage-resolution/1.0.0",
        selection_checksum=digest(candidate),
        input_checksum=digest(payload),
        proposal_checksum=digest(proposal),
        passages={key: selected[key] for key in dict.fromkeys(used)},
        medically_accepted=False,
    )
