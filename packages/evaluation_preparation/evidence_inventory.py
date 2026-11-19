"""Account for every discovery candidate without creating compulsory result slots."""

from packages.grounding.models import digest


def inventory(catalog, requirements, *, mappings=None):
    """No source = an explicit research backlog, not an asserted clinical need.

    Mappings must be versioned owner data. They never supply patient values or
    silently merge independently measured observations into additional votes.
    """
    selected = set(requirements)
    mappings = mappings or {}
    unknown = selected - set(catalog["tasks"])
    if unknown:
        raise ValueError("inventory_unknown_requirement")
    rows = []
    for identifier, task in sorted(catalog["tasks"].items()):
        definition = task.get("definition") or {}
        mapping = mappings.get(identifier)
        if mapping:
            target = mapping["target"]
            if target not in selected or identifier in selected or not mapping.get("reason"):
                raise ValueError("inventory_invalid_merge")
            status, reason = "merged", mapping["reason"]
        elif identifier in selected:
            target = identifier
            status, reason = "scheduled", "current_source_bound_requirement"
        else:
            target = None
            status = "source_binding_pending"
            reason = "exact_current_source_or_definition_not_bound"
        rows.append(
            dict(
                id=identifier,
                title=definition.get("title", identifier),
                disposition=status,
                reason=reason,
                target=target,
                selected_sources=task.get("selected_source_ids", []),
                required_sources=task.get("required_source_ids", []),
                missing_sources=task.get("missing_exact_source_ids", []),
                definition_installed=task.get("interpretation_installed", False),
                counts_as_disease_evidence=False,
                counts_in_coverage=identifier in selected,
                followup="none" if target else "retrieve_verify_archive_and_bind_source",
            )
        )
    return dict(
        schema_version="evidence-requirement-inventory/1.0.0",
        catalog_checksum=digest(catalog),
        discovery_scope="versioned_catalog_not_exhaustive_medical_search",
        candidate_count=len(rows),
        scheduled_count=len(selected),
        rows=rows,
        no_silent_omissions=len(rows) == len(catalog["tasks"]),
    )
