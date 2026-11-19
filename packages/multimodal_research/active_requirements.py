"""Opt-in source-bound requirement discovery; no patient values or invented cutoffs.

The registry is an auditable collection of operationalized research questions,
not a list of result slots. Every row must match a currently readable Knowledge
passage. Missing inputs and unsupported definitions have explicit dispositions.
"""

import json
from pathlib import Path

from packages.grounding.models import digest

PROFILE = Path("configs/research/multimodal_active_requirements_v1.json")
NUMERIC_INPUT_DISPOSITION = "component-status/1.0.0"


def numeric_input_disposition(profile):
    """Explicit projection revision; absence preserves historical plan bytes."""
    if "numeric_input_disposition" not in profile:
        return False
    if profile["numeric_input_disposition"] != NUMERIC_INPUT_DISPOSITION:
        raise ValueError("active_numeric_input_disposition")
    return True


def load(path=PROFILE):
    value = json.loads(Path(path).read_bytes())
    if "requirement_catalog" in value:
        from .requirement_catalog import expand_profile

        value = expand_profile(value)
    elif "capability_matching" in value:
        raise ValueError("active_catalog_reference_required")
    numeric_input_disposition(value)
    if "procedure_reuse_policy" in value:
        from .reuse import CAPABILITY_POLICY

        if (
            value["procedure_reuse_policy"] != CAPABILITY_POLICY
            or "requirement_catalog" not in value
        ):
            raise ValueError("active_procedure_reuse_policy")
    if "input_scope_policy" in value:
        from .input_scope import read_policy

        if "requirement_catalog" not in value:
            raise ValueError("input_scope_requires_catalog")
        read_policy(value["input_scope_policy"])
    if "scope_selection_policy" in value:
        from .scope_selection import read_policy as selection_policy

        if "input_scope_policy" not in value:
            raise ValueError("scope_selection_requires_input_survey")
        selection_policy(value)
    if "native_unavailable_policy" in value:
        from .entry_preflight import NATIVE_ALTERNATIVE

        if (
            value.get("schema_version") != "source-bound-acquisition-profile/5.0.0"
            or value["native_unavailable_policy"] != NATIVE_ALTERNATIVE
        ):
            raise ValueError("active_native_unavailable_policy")
    if "native_component_policy" in value:
        policy = value["native_component_policy"]
        if (
            value.get("schema_version") != "source-bound-acquisition-profile/5.0.0"
            or set(policy) != {"schema_version", "max_components"}
            or policy["schema_version"] != "bounded-native-components/1.0.0"
            or type(policy["max_components"]) is not int
            or not 1 <= policy["max_components"] <= 4
        ):
            raise ValueError("active_native_component_policy")
    if "review_reserve_policy" in value:
        from .active_continuation import review_reserve_policy

        review_reserve_policy(value["review_reserve_policy"])
        if value.get("schema_version") != "source-bound-acquisition-profile/5.0.0":
            raise ValueError("active_review_reserve_version")
    if "acquisition_scope_policy" in value:
        from .acquisition_scope import POLICY

        if (
            value.get("schema_version") != "source-bound-acquisition-profile/5.0.0"
            or value["acquisition_scope_policy"] != POLICY
        ):
            raise ValueError("active_acquisition_scope_policy")
    if value.get("schema_version") not in {
        "source-bound-acquisition-profile/1.0.0",
        "source-bound-acquisition-profile/2.0.0",
        "source-bound-acquisition-profile/3.0.0",
        "source-bound-acquisition-profile/4.0.0",
        "source-bound-acquisition-profile/5.0.0",
    }:
        raise ValueError("active_requirement_profile_version")
    if value["schema_version"] in {
        "source-bound-acquisition-profile/2.0.0",
        "source-bound-acquisition-profile/3.0.0",
        "source-bound-acquisition-profile/4.0.0",
        "source-bound-acquisition-profile/5.0.0",
    }:
        from .definition_expansion import read_expansion_policy

        policy = read_expansion_policy(value["definition_expansion"])
        modern = value["schema_version"] in {
            "source-bound-acquisition-profile/3.0.0",
            "source-bound-acquisition-profile/4.0.0",
            "source-bound-acquisition-profile/5.0.0",
        }
        expected = {
            "source-bound-acquisition-profile/2.0.0": "campaign-definition-expansion/1.0.0",
            "source-bound-acquisition-profile/3.0.0": "campaign-definition-expansion/1.1.0",
            "source-bound-acquisition-profile/4.0.0": "campaign-definition-expansion/1.2.0",
            "source-bound-acquisition-profile/5.0.0": "campaign-definition-expansion/1.23.0",
        }
        permitted = {expected[value["schema_version"]]}
        if value["schema_version"] == "source-bound-acquisition-profile/5.0.0":
            # Same profile structure; explicit opt-in policy/prompt identity is
            # frozen in the entry. Existing 1.23 plans retain their old prompt.
            permitted.add("campaign-definition-expansion/1.24.0")
            permitted.add("campaign-definition-expansion/1.26.0")
            permitted.add("campaign-definition-expansion/1.27.0")
            permitted.add("campaign-definition-expansion/1.28.0")
            permitted.add("campaign-definition-expansion/1.29.0")
            permitted.add("campaign-definition-expansion/1.44.0")
        if policy.schema_version not in permitted:
            raise ValueError("active_target_policy_version")
        if modern and value.get("observation_failure_policy") != "confirmed-only-closure/1.0.0":
            raise ValueError("active_observation_closure_policy")
    if (
        value["schema_version"]
        not in {
            "source-bound-acquisition-profile/3.0.0",
            "source-bound-acquisition-profile/4.0.0",
            "source-bound-acquisition-profile/5.0.0",
        }
        and "observation_failure_policy" in value
    ):
        raise ValueError("active_observation_closure_version")
    nodes = {n["id"]: n for n in value["nodes"]}
    if len(nodes) != len(value["nodes"]) or "root" in nodes:
        raise ValueError("active_requirement_duplicate_group")
    for key in nodes:
        seen, current = set(), key
        while current != "root":
            if current in seen or current not in nodes:
                raise ValueError("active_requirement_hierarchy")
            seen.add(current)
            current = nodes[current]["parent"]
    checks = value["checks"]
    if "measurement_fulfillment" in value:
        from .requirement_catalog import validate_fulfillment

        validate_fulfillment(value)
    limit = 128 if "requirement_catalog" in value else 48
    if not checks or len(checks) > limit or len({c["id"] for c in checks}) != len(checks):
        raise ValueError("active_requirement_inventory")
    if any(c["parent"] not in nodes or len(c["quote"]) < 12 for c in checks):
        raise ValueError("active_requirement_binding")
    from packages.knowledge_search.broad import TOPICS

    if not set(value["search_topics"]) <= set(TOPICS) or not (
        len(value["search_topics"]) <= value["max_queries"] <= 3
        and 1 <= value["max_hits_per_query"] <= 5
    ):
        raise ValueError("active_requirement_search_bound")
    return value


def project(manifest, medical, task, profile, *, capabilities=None):
    from .material import definition
    from .single_proposition import bind_requirements

    capability_matching = "requirement_catalog" in profile
    if capability_matching:
        from .requirement_catalog import disposition, match, verify_capabilities, verify_projection

        verify_projection(profile)
        capabilities = verify_capabilities(manifest, capabilities)
    qualified_numeric = numeric_input_disposition(profile)
    if profile.get("input_semantics"):
        expected = {
            "harvard-gdp-v1": ("harvard_gdp", "multimodal-adapters/1.1.0"),
            "grape-v1": ("grape", "multimodal-adapters/1.2.0"),
        }.get(profile["input_semantics"])
        if expected != (manifest.dataset, manifest.adapter_version):
            raise ValueError("active_input_semantics_binding")

    if task.target_code not in profile["goals"]:
        raise ValueError("active_requirement_goal_not_supported")
    rows, dispositions = [], []
    components = {c.modality: [] for c in manifest.components}
    for c in manifest.components:
        if c.status != "withheld":
            components[c.modality].append(c)
    for check in profile["checks"]:
        if capability_matching and task.target_code not in check["goals"]:
            dispositions.append(dict(id=check["id"], state="target_not_applicable"))
            continue
        if capability_matching and check["equivalent_to"] is not None:
            dispositions.append(
                dict(
                    id=check["id"], state="equivalent_merged", equivalent_to=check["equivalent_to"]
                )
            )
            continue
        source = medical.get(check["source"])
        normalized = " ".join(source.text.split()) if source else ""
        quote = " ".join(check["quote"].split())
        if not source or quote not in normalized:
            dispositions.append(
                dict(id=check["id"], state="definition_backlog", reason="exact_source_not_bound")
            )
            continue
        candidates = components.get(check["modality"], [])
        matched, excluded = {}, []
        if capability_matching:
            matches, excluded = match(check, capabilities)
            matched = {c.component_id: c for c in matches}
            candidates = [c for c in manifest.components if c.id in matched]
        if len(candidates) > 1:
            # Each actual input owns its own requirement. Never silently choose an eye/visit.
            suffixes = [(c, ":" + c.id) for c in candidates]
        else:
            suffixes = [(candidates[0] if candidates else None, "")]
        for component, suffix in suffixes:
            identifier = "active:" + check["id"] + suffix
            current_check, variant = check, None
            if capability_matching:
                from .requirement_catalog import select_input_variant

                current_check, variant = select_input_variant(check, component)
            modality = current_check["modality"]
            present = bool(component and component.status in {"obtained", "partial"})
            reason = None if present else "patient_input_missing:" + modality
            selection = None
            if capability_matching:
                selection = disposition(
                    check, matched.get(component.id) if component else None, excluded
                )
                if selection["state"] == "missing_input":
                    present, reason = False, selection["reason"]
            rows.append(
                dict(
                    id=identifier,
                    component=component.id if component else None,
                    category=check["evidence_unit"],
                    role="diagnostic",
                    definition=definition(
                        identifier, check["title"], current_check["instruction"], (source,)
                    ),
                    input_modalities=(modality,),
                    basis_kind="quantitative"
                    if check["method"] == "disc_cup_measurement"
                    else "qualitative",
                    missing_reason=reason or "not_yet_observed",
                    acquisition_method=check["method"],
                    evidence_unit=check["evidence_unit"],
                    parent=check["parent"],
                    **({"input_variant": variant} if variant is not None else {}),
                    **(
                        {"capability_selection": selection, "acquisition_allowed": present}
                        if capability_matching
                        else {}
                    ),
                )
            )
            dispositions.append(
                dict(
                    id=identifier,
                    state="scheduled" if present else "input_gap",
                    reason=reason,
                    source=source.reference.model_dump(mode="json"),
                    quote=quote,
                    component=component.id if component else None,
                    method=check["method"],
                    parent=check["parent"],
                    operationalization="source_supported_research_question_not_validated_stage_rule",
                    **({"input_variant": variant} if variant is not None else {}),
                    **({"capability_selection": selection} if capability_matching else {}),
                )
            )
    from types import SimpleNamespace

    from .material import RULES, category, requirements

    bound = bind_requirements(rows, task, directional="definition_expansion" in profile)
    numeric_components = {
        c.id for c in manifest.components if c.kind in {"scalar", "vector", "map"}
    }
    selected = [
        c
        for c in manifest.components
        if c.id in numeric_components
        and category(c) in RULES
        and all(k in medical for k in RULES[category(c)][1])
    ]
    numeric_by_id = {c.id: c for c in selected}
    for existing in (
        requirements(
            SimpleNamespace(components=selected),
            medical,
            task,
            directional="definition_expansion" in profile,
        )
        if selected
        else ()
    ):
        if existing["component"] not in numeric_components:
            continue
        row = dict(
            existing,
            acquisition_method="source_measurement",
            evidence_unit=existing["category"],
            parent="CONTEXT",
        )
        bound.append(row)
        component = numeric_by_id[row["component"]]
        present = not qualified_numeric or component.status in {"obtained", "partial"}
        dispositions.append(
            dict(
                id=row["id"],
                state="scheduled" if present else "input_gap",
                reason=None if present else component.reason,
                component=row["component"],
                method="source_measurement",
                parent="CONTEXT",
                source_refs=[s.model_dump(mode="json") for s in row["definition"].source_refs],
                **({"component_status": component.status} if qualified_numeric else {}),
            )
        )
    from packages.dataset_adapters.multimodal.harvard_semantics import qualify_rows

    bound = qualify_rows(bound, manifest)
    if manifest.adapter_version == "multimodal-adapters/1.2.0":
        from packages.dataset_adapters.multimodal.grape_semantics import qualify_rows as grape_rows

        bound = grape_rows(bound, manifest)
    fulfillment = None
    if "measurement_fulfillment" in profile:
        from .requirement_catalog import fulfill_measurements

        bound, dispositions, fulfillment = fulfill_measurements(
            bound, dispositions, manifest, profile
        )
    trace = dict(
        schema_version="source-bound-acquisition-plan/1.3.0"
        if fulfillment is not None
        else "source-bound-acquisition-plan/1.2.0"
        if capability_matching
        else "source-bound-acquisition-plan/1.1.0"
        if qualified_numeric
        else "source-bound-acquisition-plan/1.0.0",
        profile_checksum=digest(profile),
        input_checksum=manifest.checksum,
        target=task.target_code,
        rows=dispositions,
        nodes=profile["nodes"],
        requirement_checksum=digest(bound),
        independent_evidence_count_claimed=False,
        source_discovery="current_owner_passage_matching",
        scope="catalog_matching_input_representation_visual_scope_pending"
        if capability_matching
        else "registered_source_catalog_not_exhaustive_literature",
        **({"numeric_input_disposition": NUMERIC_INPUT_DISPOSITION} if qualified_numeric else {}),
        **({"measurement_fulfillment": fulfillment} if fulfillment is not None else {}),
        **(
            dict(
                capability_matching=profile["capability_matching"],
                capability_checksum=capabilities.checksum,
                requirement_catalog=profile["requirement_catalog"],
            )
            if capability_matching
            else {}
        ),
    )
    return bound, trace


def discover(directory, profile, *, searcher=None):
    """Original persistent Knowledge search. Public concepts only, at most 3 queries.

    This is discovery, not automatic acceptance of arbitrary search snippets as
    interpretation rules. Every candidate remains a backlog until source-bound.
    """
    from apps.vectorengine_single_case import write_once

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    if searcher is None:
        from apps.knowledge_literature_tools import search
        from apps.knowledge_source_tools import repository

        repo = repository("editor")

        def searcher(topic, key):
            return search(repo, ("europepmc",), topic, limit=profile["max_hits_per_query"], key=key)

    results = []
    for topic in profile["search_topics"]:
        path = out / (topic + ".json")
        key = "active-search:" + digest((profile, str(out.absolute()), topic))[7:55]
        if path.exists():
            value = json.loads(path.read_bytes())
            if value["request_key"] != key or value["topic"] != topic:
                raise ValueError("active_search_binding")
        else:
            value = searcher(topic, key)
            if value["request_key"] != key or value["topic"] != topic:
                raise ValueError("active_search_receipt")
            write_once(path, value)
        results.append(dict(topic=topic, checksum=digest(value), result=value))
    return dict(
        schema_version="active-knowledge-discovery/1.0.0",
        queries=results,
        automatic_medical_acceptance=False,
        patient_data_sent=False,
    )
