"""Typed Source contract feedback; never fill or erase a proposed dependency."""

from copy import deepcopy

from pydantic import ValidationError

from packages.evidence_reuse.definitions import (
    QuotedOwnerBoundConfirmedInterpretationDraft,
    QuotedOwnerBoundResearchDefinitionDraft,
)
from packages.grounding.models import validate

PROMPT = "definition-draft-6.7.0"
MODELS = {
    "source-definition-draft/6.1.0": QuotedOwnerBoundResearchDefinitionDraft,
    "source-definition-draft/7.1.0": QuotedOwnerBoundConfirmedInterpretationDraft,
}
RULES = {
    "definition_dependency_input_selector": (
        "kind=patient_input requires nonempty modality and concept, including unresolved. "
        "State the genuinely required input type; unresolved has null owner selectors."
    ),
    "definition_dependency_non_input_selector": (
        "kind=method or quality requires modality=concept=null."
    ),
    "definition_dependency_unresolved_selector": (
        "unresolved requires input_alias, requirement_alias, method, unit, quality_check=null; "
        "it does not remove the modality/concept requirement of kind=patient_input."
    ),
    "definition_dependency_input_alias": (
        "Available-input predicates require kind=patient_input and a matching input_alias; "
        "requirement_alias and quality_check are null."
    ),
    "definition_dependency_numeric_selector": (
        "source_numeric_available requires method=source_numeric_read and the exact supplied unit."
    ),
    "definition_dependency_presence_selector": "input_available requires method=unit=null.",
    "definition_dependency_confirmed_selector": (
        "Confirmed predicates require the matching requirement_alias/method/unit; "
        "input_alias=null. Technical QC additionally requires the exact supported quality_check."
    ),
    "definition_dependency_technical_scope": (
        "Use the exact owner-provided technical condition text only for that narrow condition; "
        "preserve separate required medical limitations as unresolved."
    ),
    "definition_dependency_coverage": "Indices uniquely cover all dependency strings from zero.",
    "definition_dependency_binding": (
        "Contract text equals its dependency string; excerpt indices exist."
    ),
    "definition_dependency_duplicate": "Blocks and excerpt indices contain no duplicates.",
    "definition_dependency_confirmed_draft_required": "Confirmed selectors require a 7.1.0 draft.",
    "definition_dependency_confirmed_scope": (
        "Confirmed selectors match the draft method, unit and alias."
    ),
    "definition_vision_method_unit": "Vision uses live_image_observation and qualitative.",
    "definition_quantitative_method_unit": "Quantitative definitions require a method and unit.",
    "definition_input_aliases": "Raw input aliases must be valid, unique input:N values.",
    "definition_proposal_outcome": (
        "An empty proposal needs a reason; a nonempty proposal has no empty reason."
    ),
}


def project(payload):
    result = deepcopy(payload)
    schema = QuotedOwnerBoundResearchDefinitionDraft.model_json_schema()
    excerpt = schema["$defs"]["QuotedExcerpt"]["properties"]
    result["definition_format_contract"] = dict(
        schema_version="definition-format-contract/1.0.0",
        constraint_rules=RULES,
        quote_context_limits={k: excerpt[k] for k in ("prefix", "suffix")},
        unresolved_is_not_waiver=(
            "Do not remove, relabel or mark a required condition satisfied to obtain a score. "
            "Available input choices permit only original-owner verification, not medical approval."
        ),
    )
    return result


def fields(error, raw):
    """Validate the declared known variant to avoid irrelevant union-branch errors."""
    from .review_correction import fields as legacy_fields

    if not isinstance(error, ValidationError) or not isinstance(raw, dict):
        return legacy_fields(error)
    definitions = raw.get("definitions")
    if not isinstance(definitions, list) or len(definitions) > 3:
        return legacy_fields(error)
    rows = []
    for index, candidate in enumerate(definitions):
        model = MODELS.get(candidate.get("schema_version")) if isinstance(candidate, dict) else None
        if model is None:
            return legacy_fields(error)
        try:
            validate(model, candidate)
        except ValidationError as selected_error:
            for item in selected_error.errors(include_url=False, include_input=False):
                row = dict(path=["definitions", index, *item["loc"]], type=item["type"])
                code = item["msg"].removeprefix("Value error, ")
                if code in RULES:
                    row.update(constraint=code, contract=RULES[code])
                for bound in ("max_length", "min_length"):
                    limit = item.get("ctx", {}).get(bound)
                    if type(limit) is int and 0 <= limit <= 100000:
                        row[bound] = limit
                rows.append(row)
    return rows[:16] if rows else legacy_fields(error)
