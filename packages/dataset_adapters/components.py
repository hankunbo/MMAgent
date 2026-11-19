"""Common projection from existing visible-input owners; no label/notes fallback.

PAPILA is preparation only until its original runtime admission is implemented.
The common envelope is not a caller-fillable runtime grant.
"""

from packages.evidence_reuse.models import Component, InputComponents
from packages.grounding.models import digest


def visible_components(visible, context=None):
    """Caller must use the original VisibleInputReader (and context reader)."""
    case = visible.case
    components = []
    for asset in case.artifacts:
        modality = {
            "ARTIFACT:SLO_IMAGE": "SLO",
            "ARTIFACT:FUNDUS_IMAGE": "color_fundus",
        }.get(asset.artifact_type_code)
        if modality is None:
            continue
        components.append(
            Component(
                id=asset.asset_code,
                kind="image",
                concept=modality,
                status="obtained",
                source_checksum=asset.checksum,
                source_locator=asset.asset_code,
                shape=tuple(asset.dimensions),
                laterality=asset.laterality.value
                if hasattr(asset.laterality, "value")
                else asset.laterality,
                transforms=("owner_bound_visible_projection",),
            )
        )
    if context is not None:
        if (
            context["case_code"] != case.case_code
            or context["visible_checksum"] != visible.checksum
            or context["image_checksum"] not in {c.source_checksum for c in components}
            or context.get("hidden_fields_used") is not False
        ):
            raise ValueError("components_context_scope")
        age = context["age"]
        components.append(
            Component(
                id="context:age",
                kind="scalar",
                concept="age",
                status=age["status"],
                value=age["value"],
                unit=age["unit"],
                unit_basis="owner_projection",
                reason=age["reason"],
                source_checksum=digest(context),
                source_locator="dataset-public-context/1.0.0",
                source_field=context["source_field"],
            )
        )
    return InputComponents(
        dataset=visible.dataset_ref.code,
        adapter_version=visible.adapter_version,
        scope=case.case_code,
        input_checksum=visible.checksum,
        split="training",
        patient_group=None,
        components=tuple(components),
        runtime_ready=True,
    )


def papila_components(owner, sample_id, *, profile="image_demographics_clinical"):
    prepared = owner.prepare(sample_id, profile=profile)
    value = owner.recheck(prepared)
    common = dict(laterality=value.laterality, timepoint="unknown")
    components = [
        Component(
            id="image:" + sample_id,
            kind="image",
            concept=value.modality,
            status="obtained",
            source_checksum=value.image.checksum,
            source_locator=value.image.relative_path,
            shape=value.dimensions,
            transforms=("identity",),
            **common,
        ),
        Component(
            id="context:age",
            kind="scalar",
            concept="age",
            status="missing" if value.age_years is None else "obtained",
            value=value.age_years,
            unit="year",
            unit_basis="owner_projection",
            reason="not_recorded" if value.age_years is None else None,
            source_checksum=value.clinical_source.checksum,
            source_locator=value.clinical_source.relative_path,
            source_field="Age",
            **common,
        ),
    ]
    for key, field in value.clinical.items():
        components.append(
            Component(
                id="clinical:" + key,
                kind="scalar",
                concept=key,
                status="obtained" if field.acquired else "missing",
                value=field.value,
                unit=field.unit,
                unit_basis=field.unit_basis,
                method=field.method,
                reason=field.missing_reason,
                source_checksum=field.source_checksum,
                source_locator=value.clinical_source.relative_path + "#" + field.source_cell,
                source_field=field.source_field,
                **common,
            )
        )
    for modality in value.unavailable_modalities:
        components.append(
            Component(
                id="missing:" + modality,
                kind="missing_modality",
                concept=modality,
                status="missing",
                reason="not_provided_by_dataset",
                source_checksum=digest(value),
                source_locator="papila-clinical-preparation/1.0.0",
                **common,
            )
        )
    return InputComponents(
        dataset=value.dataset,
        adapter_version=value.schema_version,
        scope=value.patient_group + ":" + value.sample_id,
        input_checksum=digest(value),
        split="development_preparation",
        patient_group=value.patient_group,
        components=tuple(components),
        runtime_ready=False,
    )


def available_concepts(components):
    return tuple(sorted({c.concept for c in components.components if c.status == "obtained"}))
