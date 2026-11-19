"""Owner-derived display/numeric context; never an additional disease vote.

Only approved component payloads and the exact view behind an acquired observation
are consumed. Historical observations, definitions, quality and judgments stay intact.
"""

import math

from apps.vectorengine_case_materials import ref
from packages.dataset_adapters.multimodal.formats import npy
from packages.grounding.models import canonical, digest
from packages.llm_provider.models import ObservationInput
from packages.object_store import checksum

VERSION = "review-display-context/1.0.0"


def array_summary(raw, component, view):
    """Source/display facts, not normative thresholds or anatomical measurements."""
    array = npy(raw)
    if list(array.shape) != list(component.shape) or len(array.shape) != 2:
        raise ValueError("review_context_array_shape")
    values = [float(n) for n in array.numbers()]
    if len(values) > 1048576:
        raise ValueError("review_context_array_bound")
    finite = [n for n in values if math.isfinite(n)]
    transform = view["transform"]
    qualified = (
        component.concept == "rnflt"
        and component.unit == "um"
        and transform.get("invalid_display") == "magenta"
        and transform.get("scalar_scale") == "linear_valid_thickness_not_normative_color_code"
    )
    retained = [n for n in finite if n >= 0] if qualified else finite
    if not retained or transform.get("kind") != "global_finite_minmax_display":
        raise ValueError("review_context_transform_unsupported")
    low, high = min(retained), max(retained)
    if (low, high) != (transform["minimum"], transform["maximum"]):
        raise ValueError("review_context_display_scale_changed")
    positive = [n for n in finite if n > 0]
    return dict(
        shape=list(array.shape),
        total=len(values),
        finite=len(finite),
        negative=sum(n < 0 for n in finite),
        zero=sum(n == 0 for n in finite),
        positive=len(positive),
        nonfinite=len(values) - len(finite),
        retained_min=low,
        retained_max=high,
        retained_mean=sum(retained) / len(retained),
        positive_mean=sum(positive) / len(positive) if positive else None,
        quantized_black_retained=sum(
            int(min(1, max(0, (n - low) / max(high - low, 1e-9))) * 255) == 0 for n in retained
        ),
        unit=component.unit or "unqualified",
        subset_semantics="numeric_subsets_not_tissue_QC_or_normative_comparison",
        negative_excluded_by_renderer=qualified,
    )


def collect(base, observations, selected, *, metadata_profile=None):
    """Rebuild on preparation and each Phase projection; never trust caller context."""
    if metadata_profile is not None:
        return collect_methods(base, selected, metadata_profile)
    base.read_dependency_input()
    rows = {r["id"]: r for r in base.ledger}
    payloads = dict(base.prepared.payload_bytes)
    declared = {p.id: p for p in base.manifest.payloads}
    contexts = {}
    for target in selected:
        row, observed = rows.get(target), observations.get(target)
        if row is None or observed is None or observed.status != "obtained" or not observed.image:
            raise ValueError("review_context_requires_confirmed_image_observation")
        component = next(c for c in base.manifest.components if c.id == row["component"])
        if component.kind != "map" or component.status not in {"obtained", "partial"}:
            raise ValueError("review_context_component_not_map")
        image, _, view = base.view(target)
        if image != observed.image or len(component.payload_ids) != 1:
            raise ValueError("review_context_observation_image_binding")
        key = component.payload_ids[0]
        record, raw = declared[key], payloads[key]
        if (checksum(raw), len(raw)) != (record.checksum, record.byte_size):
            raise ValueError("review_context_payload_changed")
        contexts[target] = dict(
            schema_version=VERSION,
            observation=observed.reference.model_dump(mode="json"),
            component=component.id,
            input_checksum=base.manifest.checksum,
            payload_checksum=record.checksum,
            image_checksum=image.reference.checksum,
            preprocessing_checksum=image.preprocessing.checksum,
            unit_basis=component.unit_basis,
            source_limits=list(component.semantic_limits),
            transform=view["transform"],
            numeric=array_summary(raw, component, view),
            interpretation_limits=dict(
                black="Display minimum or quantization; not proof of missing tissue or disease.",
                zero="Raw zero retained; biological validity is not established by rendering.",
                magenta="Negative/nonfinite exclusion if declared by this transform; not tissue.",
                coverage="Original VLM estimate, not the numerical nonnegative fraction.",
                coordinates="Source indices, not verified eye, anatomical sector or percentile.",
                conflict="Color-based clinical claims are hypotheses, not verified input facts.",
                quality="Original observation/Q retained; no automatic quality replacement.",
                direction="Use source-supported interpretable features; no forced sign or epsilon.",
            ),
        )
    return contexts


def collect_methods(base, selected, profile):
    """Original decoded scalar owner plus exact author metadata; no remeasurement."""
    from packages.dataset_adapters.multimodal import harvard_semantics
    from packages.dataset_adapters.multimodal.grape_semantics import (
        ADAPTER,
        METHOD_PROFILE,
        RNFL_UNIT_PROFILE,
        SOURCE_CHECKSUM,
        TONOPACHY_UNIT_PROFILE,
        method_context,
    )

    from .material import numeric, observation

    if profile == harvard_semantics.METHOD_PROFILE:
        return collect_harvard_method(base, selected)
    if (
        profile not in {METHOD_PROFILE, RNFL_UNIT_PROFILE, TONOPACHY_UNIT_PROFILE}
        or base.manifest.dataset != "grape"
        or base.manifest.adapter_version != ADAPTER
        or not selected
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("review_method_context_profile")
    base.read_dependency_input()
    rows = {r["id"]: r for r in base.ledger}
    components = {c.id: c for c in base.manifest.components}
    payloads = dict(base.prepared.payload_bytes)
    contexts = {}
    for target in selected:
        row = rows.get(target)
        component = components.get(row.get("component")) if row else None
        original = base.observations.get(target)
        if (
            component is None
            or component.kind != "scalar"
            or component.status != "obtained"
            or original is None
            or "semantic_profile:grape-v1" not in component.semantic_limits
            or "source_document_checksum:" + SOURCE_CHECKSUM not in component.semantic_limits
        ):
            raise ValueError("review_method_context_original_scalar")
        measured = numeric(component, payloads)
        rebuilt = observation(target, measured, unit=component.unit or "source_units_unqualified")
        if rebuilt != original:
            raise ValueError("review_method_context_observation_changed")
        method = (
            method_context(component.concept, profile=profile)
            if profile in {RNFL_UNIT_PROFILE, TONOPACHY_UNIT_PROFILE}
            else method_context(component.concept)
        )
        contexts[target] = dict(
            schema_version="review-method-context/1.0.0",
            observation=original.reference.model_dump(mode="json"),
            component=component.id,
            input_checksum=base.manifest.checksum,
            source_locator=component.source_locator,
            unit_basis=component.unit_basis,
            source_limits=list(component.semantic_limits),
            transform=dict(kind="unchanged_source_measurement_no_conversion"),
            numeric=measured,
            author_method=method,
            interpretation_limits=dict(
                metadata="Author acquisition method, not a new medical source qualification.",
                units="Original release unit status unchanged; no guessed physical conversion.",
                quality="No per-case reliability, normative comparison or stage relation inferred.",
                direction="Review source-supported interpretable information; no forced sign.",
            ),
        )
    return contexts


def collect_harvard_method(base, selected):
    """Rebuild original numeric RNFLT fact; do not substitute a visual observation."""
    from packages.dataset_adapters.multimodal import harvard_semantics as author

    from .material import numeric, observation

    if (
        base.manifest.dataset != "harvard_gdp"
        or base.manifest.adapter_version != author.ADAPTER
        or len(selected) != 1
    ):
        raise ValueError("review_method_context_harvard_profile")
    base.read_dependency_input()
    target = selected[0]
    row = next((r for r in base.ledger if r["id"] == target), None)
    component = next(
        (c for c in base.manifest.components if row and c.id == row["component"]), None
    )
    original = base.observations.get(target)
    if (
        component is None
        or component.concept != "rnflt"
        or component.kind != "map"
        or component.status != "obtained"
        or component.unit != "um"
        or component.unit_basis != author.PAPER + "#section-3.1-RNFLT"
        or "manufacturer_derived_RNFL_thickness_map_not_raw_intensity"
        not in component.semantic_limits
        or original is None
        or len(component.payload_ids) != 1
    ):
        raise ValueError("review_method_context_harvard_measurement")
    key = component.payload_ids[0]
    record = next(p for p in base.manifest.payloads if p.id == key)
    payloads = dict(base.prepared.payload_bytes)
    raw = payloads[key]
    if (checksum(raw), len(raw)) != (record.checksum, record.byte_size):
        raise ValueError("review_method_context_harvard_payload")
    measured = numeric(component, payloads)
    if (
        measured["shape"] != list(component.shape)
        or observation(target, measured, unit="um") != original
    ):
        raise ValueError("review_method_context_harvard_observation_changed")
    method = author.method_context(component.concept)
    return {
        target: dict(
            schema_version="review-method-context/1.0.0",
            observation=original.reference.model_dump(mode="json"),
            component=component.id,
            input_checksum=base.manifest.checksum,
            payload_checksum=record.checksum,
            source_locator=component.source_locator,
            unit_basis=component.unit_basis,
            source_limits=list(component.semantic_limits),
            transform=dict(kind="unchanged_source_measurement_no_conversion"),
            numeric=measured,
            author_method=method,
            interpretation_limits=dict(
                metadata="Author measurement area; not a new normative or stage relationship.",
                units="Original um and measurement summary unchanged, not a ring average.",
                coordinates="Do not infer pixel mm scale, eye sectors or anatomical registration.",
                direction="Evaluate sourced features; no forced sign or null-to-zero conversion.",
            ),
        )
    }


def collect_campaign(base, observations):
    """Current approved views of any modality; no fabricated numerical measurements."""
    base.read_dependency_input()
    components = {c.id: c for c in base.manifest.components}
    result = {}
    for row in base.ledger:
        target = row["id"]
        observed = observations.get(target)
        component = components.get(row["component"])
        if observed is None or observed.status != "obtained" or observed.image is None:
            continue
        if component is None or component.status not in {"obtained", "partial"}:
            raise ValueError("review_context_component_missing")
        image, _, view = base.view(target)
        if image != observed.image:
            raise ValueError("review_context_observation_image_binding")
        if (
            component.kind == "map"
            and view["transform"].get("kind") == "global_finite_minmax_display"
        ):
            result.update(collect(base, observations, [target]))
            continue
        result[target] = dict(
            schema_version="review-display-context/1.1.0",
            observation=observed.reference.model_dump(mode="json"),
            component=component.id,
            input_checksum=base.manifest.checksum,
            image_checksum=image.reference.checksum,
            preprocessing_checksum=image.preprocessing.checksum,
            unit_basis=component.unit_basis,
            source_limits=list(component.semantic_limits),
            transform={
                k: view[k]
                for k in (
                    "transform",
                    "indices",
                    "available_members",
                    "inspected_fraction",
                    "limits",
                )
                if k in view
            },
            numeric=None,
            interpretation_limits=dict(
                display="Sampling/resize is not a new measurement or complete tissue inspection.",
                scope="Use visible features; do not infer hidden slices or anatomy registration.",
                quality="Retain original Q; no automatic replacement by sampled fraction.",
                direction="Source-supported interpretable features; no forced sign or epsilon.",
            ),
        )
    return result


def context_values(target, part, value):
    """Lossless bounded wire records; concatenate JSON fragments, never truncate facts."""
    original = canonical(
        dict(assessment_id=target, part=part, context=VERSION, data=value)
    ).decode()
    if len(original) <= 1000:
        return [(VERSION, None, original)]
    # Binary-search chunks against the encoded envelope: escaped quotes, backslashes
    # and Unicode count toward the existing ObservationInput field, too.
    chunks, offset = [], 0
    while offset < len(original):
        lo, hi = 1, min(1000, len(original) - offset)
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = canonical(
                dict(
                    encoding="concatenate_fragments_then_parse_json",
                    assessment_id=target,
                    part=part,
                    checksum=digest(original),
                    index=len(chunks),
                    count=999,
                    fragment=original[offset : offset + mid],
                )
            ).decode()
            if len(candidate) <= 1000:
                lo = mid + 1
            else:
                hi = mid - 1
        if hi < 1 or len(chunks) >= 128:
            raise ValueError("review_context_fragment_bound")
        chunks.append(original[offset : offset + hi])
        offset += hi
    return [
        (
            "review-display-context-fragment/1.0.0",
            i,
            canonical(
                dict(
                    encoding="concatenate_fragments_then_parse_json",
                    assessment_id=target,
                    part=part,
                    checksum=digest(original),
                    index=i,
                    count=len(chunks),
                    fragment=chunk,
                )
            ).decode(),
        )
        for i, chunk in enumerate(chunks)
    ]


def supplementary(contexts):
    """Unscored context observations bound to each task; original primary untouched."""
    result = []
    for target, context in sorted(contexts.items()):
        identity = {
            k: v
            for k, v in context.items()
            if k
            not in {"unit_basis", "source_limits", "transform", "numeric", "interpretation_limits"}
        }
        parts = dict(
            identity=identity,
            provenance=dict(unit_basis=context["unit_basis"], limits=context["source_limits"]),
            display=context["transform"],
            numeric=context["numeric"],
            limits=context["interpretation_limits"],
        )
        records = [
            (part, version, index, text)
            for part, value in parts.items()
            for version, index, text in context_values(target, part, value)
        ]
        for part, version, index, text in records:
            body = dict(
                status="obtained",
                method="owner_display_context_not_disease_evidence",
                unit="context_only",
                value=text,
                image=None,
            )
            result.append(
                ObservationInput(
                    reference=ref(
                        "observation",
                        "OBS:DISPLAY_CONTEXT:"
                        + digest(
                            (target, part) if index is None else (target, part, version, index)
                        )[7:27],
                        digest(body),
                        version,
                    ),
                    **body,
                )
            )
    return tuple(result)


def retain_observations(base, current, selected, supplementary):
    scope = getattr(base, "review_context_scope", None)
    if scope is None:
        return supplementary
    reader = getattr(base, "confirmed_review_reader", None)
    if scope != "current-owner-prior-visible/1.0.0" or reader is None:
        raise ValueError("review_retained_context_scope")
    candidates = {}
    for observation in (*current.values(), *supplementary):
        code = observation.reference.code
        if code in candidates and candidates[code] != observation:
            raise ValueError("review_retained_context_ambiguous")
        candidates[code] = observation
    included = {o.reference.code for o in (*selected.values(), *supplementary)}
    previous = reader.state_reader().material.observations
    missing = [o for o in previous if o.status == "obtained" and o.reference.code not in included]
    if any(o.reference.code not in candidates for o in missing) and (
        getattr(base, "execution_profile", None) or getattr(base, "review_display_context", None)
    ):
        from packages.multimodal_research.review_context import collect_campaign
        from packages.multimodal_research.review_context import supplementary as project_context

        # Rebuild metadata through its original producer, never copy historical
        # bytes as a substitute for a currently readable observation/view.
        for observation in project_context(collect_campaign(base, current)):
            code = observation.reference.code
            if code in candidates and candidates[code] != observation:
                raise ValueError("review_retained_context_ambiguous")
            candidates[code] = observation
    retained = []
    for old in previous:
        if old.status != "obtained" or old.reference.code in included:
            continue
        value = candidates.get(old.reference.code)
        if value != old:
            raise ValueError("review_retained_context_current_fact_changed")
        retained.append(value)
        included.add(value.reference.code)
    return (*supplementary, *retained)
