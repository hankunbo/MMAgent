"""Conservative phase projection over current owner input, not clinical entailment.

Presence is not method/quality validation. This initial contract never resolves a
blocking dependency from metadata alone; qualified methods/quality need their own
future original-owner evidence. Old receipts are never reinterpreted here.
"""

from packages.grounding.models import digest


def assess(draft, manifest, input_checksum):
    if manifest.checksum != input_checksum:
        raise ValueError("definition_dependency_input_binding")
    items = []
    for contract in draft.dependency_contracts:
        components = [
            c
            for c in manifest.components
            if contract.kind == "patient_input"
            and c.status != "withheld"
            and c.modality == contract.modality
            and c.concept == contract.concept
        ]
        present = any(c.status in {"obtained", "partial"} for c in components)
        status = (
            ("patient_input_present_unverified" if present else "patient_input_missing")
            if contract.kind == "patient_input"
            else (
                "method_owner_evidence_missing"
                if contract.kind == "method"
                else "quality_observation_pending"
            )
        )
        items.append(
            dict(
                contract=contract.model_dump(mode="json"),
                text=draft.dependencies[contract.index],
                text_checksum=digest(draft.dependencies[contract.index]),
                status=status,
                satisfied=False,
                components=[dict(id=c.id, checksum=digest(c)) for c in components],
            )
        )
    return dict(
        schema_version="research-definition-dependencies/1.0.0",
        input_checksum=input_checksum,
        # Bind the actual visible component set as well as its enclosing identity.
        component_set_checksum=digest(manifest.components),
        items=items,
        acquisition_allowed=not any("acquisition" in c.blocks for c in draft.dependency_contracts),
        review_allowed=draft.relation_use == "directional" and not draft.dependency_contracts,
        semantics="source_proposed_stage_scope_owner_checked_availability_not_medical_proof",
    )


def assess_evidence(draft, manifest, input_checksum, fact_reader):
    """A new immutable receipt, rebuilt by original input owners at every current read."""
    if fact_reader is None:
        raise ValueError("definition_dependency_fact_owner_missing")
    assessment = assess(draft, manifest, input_checksum)
    assessment["schema_version"] = "research-definition-dependencies/2.0.0"
    for contract, item in zip(draft.dependency_contracts, assessment["items"], strict=True):
        if contract.predicate == "unresolved":
            continue
        fact = fact_reader(contract)
        if fact["input_checksum"] != input_checksum or fact["selector"] != dict(
            modality=contract.modality, concept=contract.concept
        ):
            raise ValueError("definition_dependency_fact_binding")
        item["evidence"] = fact
        item["status"] = fact["status"]
        item["satisfied"] = fact["status"] == "input_available"
        if item["satisfied"] and contract.predicate == "source_numeric_available":
            numeric = fact.get("numeric")
            item["satisfied"] = numeric is not None and numeric["unit"] == contract.unit
            item["status"] = (
                "source_numeric_available" if item["satisfied"] else "numeric_method_unit_mismatch"
            )
    for stage, field in (
        ("acquisition", "acquisition_allowed"),
        ("interpretation", "review_allowed"),
    ):
        assessment[field] = not any(
            stage in c.blocks and not item["satisfied"]
            for c, item in zip(draft.dependency_contracts, assessment["items"], strict=True)
        )
    assessment["review_allowed"] &= draft.relation_use == "directional"
    return assessment


def input_aliases(manifest):
    """Public, per-manifest aliases; values/identifiers never leave this owner."""
    components = [c for c in manifest.components if c.status != "withheld"]
    if len(components) > 1000 or len({c.id for c in components}) != len(components):
        raise ValueError("definition_input_alias_capacity_or_duplicate")
    return {"input:" + str(i): c for i, c in enumerate(components)}


def alias_component(manifest, alias, modality, concept):
    component = input_aliases(manifest).get(alias)
    if component is None or (component.modality, component.concept) != (modality, concept):
        raise ValueError("definition_input_alias_binding")
    return component


def selected_inputs(draft, manifest):
    """New acquisition drafts cannot substitute a derived concept for raw input."""
    available = input_aliases(manifest)
    selected = []
    for alias in draft.input_aliases:
        component = available.get(alias)
        if (
            component is None
            or component.modality not in draft.modalities
            or component.concept not in draft.input_concepts
        ):
            raise ValueError("definition_input_alias_binding")
        selected.append(component)
    if not selected and any(
        c.modality in draft.modalities and c.concept in draft.input_concepts
        for c in available.values()
    ):
        raise ValueError("definition_input_alias_omitted")
    return selected


def assess_owner_evidence(draft, manifest, input_checksum, fact_reader, confirmed):
    """Only original-owner facts resolve typed conditions. Unknown stays unknown."""
    if fact_reader is None:
        raise ValueError("definition_dependency_fact_owner_missing")
    assessment = assess(draft, manifest, input_checksum)
    assessment["schema_version"] = "research-definition-dependencies/3.0.0"
    assessment["input_alias_map_checksum"] = digest(input_aliases(manifest))
    for contract, item in zip(draft.dependency_contracts, assessment["items"], strict=True):
        if contract.predicate == "unresolved":
            continue
        if contract.predicate.startswith("confirmed_"):
            if (
                confirmed is None
                or contract.requirement_alias != draft.requirement_alias
                or (contract.method, contract.unit) != (confirmed["method"], confirmed["unit"])
            ):
                raise ValueError("definition_dependency_confirmed_binding")
            satisfied = (
                contract.predicate == "confirmed_method"
                or confirmed["technical_checks"].get(contract.quality_check) is True
            )
            item.update(
                satisfied=satisfied,
                status="confirmed_method_identity"
                if contract.predicate == "confirmed_method"
                else "technical_qc_confirmed"
                if satisfied
                else "technical_qc_unresolved",
                evidence=dict(
                    basis_checksum=digest(confirmed),
                    observation=confirmed["observation"],
                    method=confirmed["method"],
                    unit=confirmed["unit"],
                    quality_check=contract.quality_check,
                    qualification="technical_condition_only_not_medical_accuracy",
                ),
            )
            continue
        component = alias_component(
            manifest, contract.input_alias, contract.modality, contract.concept
        )
        fact = fact_reader(contract)
        if (
            fact["input_checksum"] != input_checksum
            or fact["selector"] != dict(modality=contract.modality, concept=contract.concept)
            or fact["components"] != [digest(component)]
        ):
            raise ValueError("definition_dependency_fact_binding")
        satisfied = fact["status"] == "input_available"
        if satisfied and (fact.get("component_id"), fact.get("component_checksum")) != (
            component.id,
            digest(component),
        ):
            raise ValueError("definition_dependency_fact_binding")
        status = fact["status"]
        if satisfied and contract.predicate == "source_numeric_available":
            numeric = fact.get("numeric")
            satisfied = numeric is not None and numeric["unit"] == contract.unit
            status = "source_numeric_available" if satisfied else "numeric_method_unit_mismatch"
        item.update(
            evidence=fact,
            status=status,
            satisfied=satisfied,
            components=[dict(id=component.id, checksum=digest(component))],
        )
    for stage, field in (
        ("acquisition", "acquisition_allowed"),
        ("interpretation", "review_allowed"),
    ):
        assessment[field] = not any(
            stage in c.blocks and not item["satisfied"]
            for c, item in zip(draft.dependency_contracts, assessment["items"], strict=True)
        )
    assessment["review_allowed"] &= draft.relation_use == "directional"
    return assessment
