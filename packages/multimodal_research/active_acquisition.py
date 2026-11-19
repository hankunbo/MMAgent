"""Install checked demand, bounded native acquisition and precise provenance.

This composes the existing input, Knowledge, native Tool and artifact owners.
Old entry versions do not implicitly acquire tools or change their questions.
"""

import json

from apps.vectorengine_case_materials import ref
from apps.vectorengine_single_case import write_once
from packages.grounding.models import canonical, digest

from .active_requirements import discover, project


def input_capabilities(base, profile):
    """Rebuild from the original prepared payload owner, not caller eligibility."""
    if "requirement_catalog" not in profile:
        return None
    from packages.dataset_adapters.multimodal.analysis import build
    from packages.dataset_adapters.multimodal.capabilities import project as capability_projection

    if base.prepared.manifest != base.manifest:
        raise ValueError("active_capability_prepared_binding")
    return capability_projection(
        base.manifest, build(base.manifest, dict(base.prepared.payload_bytes))
    )


def validate_parent_transition(base, revision, rows):
    parent = getattr(base, "capability_parent", None)
    transition = getattr(base, "capability_transition", None)
    if parent is None or not isinstance(transition, dict) or parent is base:
        raise ValueError("capability_transition_definition_ledger_changed")
    seen, current = {id(base)}, parent
    for _ in range(16):
        if id(current) in seen:
            raise ValueError("capability_transition_parent_cycle")
        seen.add(id(current))
        current = getattr(current, "capability_parent", None)
        if current is None:
            break
    else:
        raise ValueError("capability_transition_parent_depth")
    if (
        digest(transition.get("requirements")) != digest(rows)
        or transition.get("definition_revision_checksum") != digest(revision)
        or transition.get("input_checksum") != base.manifest.checksum
        or transition.get("task_checksum") != base.task.checksum
    ):
        raise ValueError("capability_transition_original_revision_binding")
    expected = successor_projection(
        parent,
        base.active_owner.profile,
        explicit_measurement_transition=transition["schema_version"]
        == "capability-requirement-transition/1.2.0",
    )
    # Compare the original-owner reconstruction in its durable representation.
    # Native references are typed in memory and JSON objects after storage.
    if canonical(expected) != canonical(transition):
        raise ValueError("capability_transition_parent_owner_changed")


def successor_projection(base, profile, *, explicit_measurement_transition=False):
    """Reconcile automatic selection with current, owner-admitted historical demand.

    No old ledger is installed or overwritten. A subsequent bounded campaign must
    still qualify scope/methods and bind its Review to these exact definitions.
    """
    from .reuse import root_entry_contract

    root_entry_contract(profile, base.task, base.manifest.checksum)
    base.read_dependency_input()
    if getattr(base, "active_owner", None) is not None:
        base.active_owner.check()
    old = json.loads(canonical(base.ledger))
    reader = getattr(base, "definition_current", None)
    revision = reader() if reader is not None else None
    if revision is not None and digest(revision["rows"]) != digest(old):
        # A selected ledger can differ from its original admitted-definition
        # revision. Reconstruct the exact owner transition, never bless a caller
        # supplied ledger merely because its saved checksum matches.
        validate_parent_transition(base, revision, old)
    capabilities = input_capabilities(base, profile)
    rows, plan = project(
        base.manifest, base.sources.read(), base.task, profile, capabilities=capabilities
    )
    retained_native = retain_native_observations(base, rows)
    new = json.loads(canonical(rows))
    selected = {r["id"]: r for r in new}
    if len(selected) != len(new) or len({r["id"] for r in old}) != len(old):
        raise ValueError("capability_transition_duplicate_requirement")
    changes = []
    # Only presentation/capability metadata may differ without invalidating a
    # prior interpretation; method, dependencies and medical content stay exact.
    presentation = {
        "capability_selection",
        "input_variant",
        "parent",
        "title",
        "acquisition_allowed",
    }
    for prior in old:
        key = prior["id"]
        candidate = selected.get(key)
        selected_key = key
        receipt_key = prior.get("definition_receipt_key")
        if receipt_key is not None:
            receipt = (revision or {}).get("accepted", {}).get(receipt_key)
            # The original revision reader recursively verifies inherited
            # receipts. They need not be duplicated in this revision's accepted
            # map; matching its complete rows above is mandatory in either case.
            inherited = (revision or {}).get("inherited_definition_origin")
            if (
                revision is None
                or (receipt is None and not inherited)
                or (
                    receipt is not None
                    and digest(receipt) != prior.get("definition_receipt_checksum")
                )
            ):
                raise ValueError("capability_transition_definition_receipt_binding")
            # A current accepted revision may supersede the static catalog.
            # Retain its full object and owner, not only the displayed name.
            selected[key] = prior
            if candidate is None:
                new.append(prior)
            else:
                new[new.index(candidate)] = prior
            action = "retained_current_definition"
            reason = "original_definition_owner_revalidated"
            interpretation_same = True
        elif candidate is None:
            merged = (
                fulfilled_target(prior, selected, capabilities)
                if explicit_measurement_transition
                else None
            )
            if merged is None:
                raise ValueError("capability_transition_unresolved_historical_requirement:" + key)
            selected_key = merged
            action = "merged_for_joint_reassessment"
            reason = "original_component_and_current_question_fulfillment"
            interpretation_same = False
        else:
            interpretation_same = digest(
                {k: v for k, v in prior.items() if k not in presentation}
            ) == digest({k: v for k, v in candidate.items() if k not in presentation})
            action = "retained_equivalent" if interpretation_same else "requires_reassessment"
            reason = "same_interpretation_binding" if interpretation_same else "binding_changed"
            if (
                not interpretation_same
                and prior.get("component") is None
                and candidate.get("component") is None
                and digest(
                    {k: v for k, v in prior.items() if k not in presentation | {"missing_reason"}}
                )
                == digest(
                    {
                        k: v
                        for k, v in candidate.items()
                        if k not in presentation | {"missing_reason"}
                    }
                )
            ):
                action, reason = "retained_input_gap", "missing_input_reason_reclassified"
                interpretation_same = True
        changes.append(
            dict(
                requirement_id=key,
                action=action,
                reason=reason,
                previous=digest(prior),
                selected=digest(selected[selected_key]),
                **(
                    {"selected_requirement_id": selected_key}
                    if explicit_measurement_transition
                    else {}
                ),
                interpretation_unchanged=interpretation_same,
                acquisition_condition_changed=(
                    prior.get("acquisition_allowed")
                    != selected[selected_key].get("acquisition_allowed")
                ),
                historical_review_is_current_authorization=False,
            )
        )
    previous_ids = {r["id"] for r in old}
    changes.extend(
        dict(requirement_id=r["id"], action="added_from_catalog", selected=digest(r))
        for r in new
        if r["id"] not in previous_ids
    )
    # Detect a source/input/revision change during projection before saving any
    # successor record. Original owner errors remain technical/revocation errors.
    base.read_dependency_input()
    if reader is not None and digest(reader()) != digest(revision):
        raise ValueError("capability_transition_definition_changed_during_projection")
    if digest(base.ledger) != digest(old):
        raise ValueError("capability_transition_parent_mutated")
    if (
        project(base.manifest, base.sources.read(), base.task, profile, capabilities=capabilities)[
            1
        ]
        != plan
    ):
        raise ValueError("capability_transition_sources_changed")
    if getattr(base, "active_owner", None) is not None:
        base.active_owner.check()
    return dict(
        schema_version=(
            "capability-requirement-transition/1.2.0"
            if explicit_measurement_transition
            else "capability-requirement-transition/1.1.0"
        ),
        input_checksum=base.manifest.checksum,
        task_checksum=base.task.checksum,
        profile_checksum=digest(profile),
        prior_requirements_checksum=digest(old),
        definition_revision_checksum=digest(revision) if revision is not None else None,
        capabilities=capabilities.model_dump(mode="json"),
        catalog_plan=plan,
        requirements=new,
        changes=changes,
        retained_native_observations=retained_native,
        coverage=dict(previous=len(old), catalog=len(rows), selected=len(new)),
        execution_authorized=False,
        qualification="current_definitions_only_scope_method_and_review_revalidation_required",
    )


def bind_native_observation(base, target, current_native):
    """Original Tool result projection; caller must first run its current owner gate."""
    from .material import observation

    record = current_native["measurement"]
    artifact = current_native["artifact"]
    resource = ref("resource", artifact["asset_code"], artifact["checksum"])
    native_resources = tuple(
        ref("resource", a["asset_code"], a["checksum"]) for a in current_native["artifacts"]
    )
    base.resources = tuple(dict.fromkeys((*base.resources, *native_resources)))
    target["quality"] = record["quality"]
    target["dependency_paths"] = ((base.resources[0].code, *(r.code for r in native_resources)),)
    target["quality_source"] = resource
    compact = dict(
        values=record["values"],
        quality=record["quality"],
        analysis_grid=record["coordinate_system"],
        physical_scale=None,
        radial_available=record["radial_rim"] is not None,
        radial_missing_reason=record["radial_missing_reason"],
        validation="research_smoke_only_not_validated_stage_classifier",
        training_overlap=current_native["prediction"]["training_overlap"],
        producer=resource.model_dump(mode="json"),
    )
    base.observations[target["id"]] = observation(
        target["id"], compact, method="registered_fundus_disc_cup_geometry", unit="ratio"
    )


def retain_native_observations(base, rows):
    """Reconstruct exact prior Tool facts, never copy a quality flag as authority."""
    from copy import copy

    from .native_fundus import records

    owner = getattr(base, "active_owner", None)
    if owner is None or getattr(owner, "native", None) is None:
        return []
    owner.check()
    current = copy(base)
    current.observations = {}
    original = {r["id"]: r for r in base.ledger}
    selected = {r["id"]: r for r in rows}
    retained = []
    for native in records(owner.native):
        if native["status"] != "obtained":
            continue
        key = native["plan"]["target"]
        row, prior = selected.get(key), original.get(key)
        if row is None or prior is None:
            raise ValueError("capability_transition_native_requirement_missing")
        if (
            row["component"] != native["plan"]["component"]
            or prior["component"] != row["component"]
            or digest(prior["definition"]) != digest(row["definition"])
            or prior["acquisition_method"] != row["acquisition_method"]
        ):
            raise ValueError("capability_transition_native_binding_changed")
        bind_native_observation(current, row, native)
        if (
            key not in base.observations
            or digest(current.observations[key]) != digest(base.observations[key])
            or any(
                digest(row.get(field)) != digest(prior.get(field))
                for field in ("quality", "quality_source", "dependency_paths")
            )
        ):
            raise ValueError("capability_transition_native_fact_changed")
        retained.append(
            dict(
                requirement_id=key,
                observation_checksum=digest(current.observations[key]),
                native_record_checksum=digest(native),
                quality_source=row["quality_source"],
                current_owner_revalidated=True,
                new_tool_execution=False,
            )
        )
    return retained


class ActiveOwner:
    def __init__(self, base, profile, plan, discovery, native, methods=None):
        self.base, self.profile, self.plan = base, profile, plan
        self.discovery, self.native = discovery, native
        self.methods = methods

    def check(self):
        capabilities = input_capabilities(self.base, self.profile)
        _, current = project(
            self.base.manifest,
            self.base.sources.read(),
            self.base.task,
            self.profile,
            capabilities=capabilities,
        )
        if current != self.plan:
            raise ValueError("active_source_or_requirement_changed")
        if "requirement_catalog" in self.profile:
            saved = json.loads((self.base.output / "input-capabilities.json").read_bytes())
            if saved != capabilities.model_dump(mode="json"):
                raise ValueError("active_capability_snapshot_changed")
        if self.methods is not None and digest(
            json.loads((self.base.output / "active-method-selection.json").read_bytes())
        ) != digest(self.methods):
            raise ValueError("active_method_selection_changed")
        for query in self.discovery["queries"]:
            path = self.base.output / "knowledge-discovery" / (query["topic"] + ".json")
            if digest(json.loads(path.read_bytes())) != query["checksum"]:
                raise ValueError("active_discovery_receipt_changed")
        if self.native:
            from .native_fundus import recheck, records

            current_records = records(self.native)
            if self.native.get("schema_version") == "native-fundus-collection/1.0.0":
                policy = self.profile.get("native_component_policy")
                expected = {
                    (row["id"], row["component"])
                    for row in self.plan["rows"]
                    if row.get("method") == "disc_cup_measurement" and row["state"] == "scheduled"
                }
                eligible = getattr(self.base, "scope_native_targets", None)
                if eligible is not None:
                    expected = {(t, c) for t, c in expected if t in eligible}
                actual = {(r["plan"]["target"], r["plan"]["component"]) for r in current_records}
                if (
                    policy is None
                    or len(current_records) > policy["max_components"]
                    or actual != expected
                ):
                    raise ValueError("active_native_collection_inventory")
            for native in current_records:
                if not any(
                    row["id"] == native["plan"]["target"]
                    and row["component"] == native["plan"]["component"]
                    for row in self.plan["rows"]
                ):
                    raise ValueError("active_native_requirement_binding")
                recheck(self.base, native)

    def report(self):
        value = dict(
            schema_version="active-acquisition-evidence/1.1.0"
            if self.native and self.native.get("schema_version") == "native-fundus-collection/1.0.0"
            else "active-acquisition-evidence/1.0.0",
            plan=self.plan,
            knowledge_discovery=self.discovery,
            native=self.native,
            clinical_acceptance=False,
        )
        if self.methods is not None:
            value.update(
                schema_version="active-acquisition-evidence/1.2.0", method_selection=self.methods
            )
        return value


def install(
    base, profile, *, discoverer=discover, native_runner=None, defer_native=False, read_only=False
):
    from .native_fundus import preparation_guard

    if read_only:
        # Cold trial assembly cannot restart the clock, discovery or native work.
        lease = json.loads((base.output / "active-start.json").read_bytes())
        if lease["input_checksum"] != base.manifest.checksum:
            raise ValueError("active_preparation_input_changed")
        base.active_start = lease

        def guard():
            raise ValueError("active_cold_preparation_missing")
    else:
        guard = preparation_guard(base)
    capabilities = input_capabilities(base, profile)
    rows, plan = project(
        base.manifest, base.sources.initial, base.task, profile, capabilities=capabilities
    )
    directory = base.output
    if capabilities is not None:
        cap_path = directory / "input-capabilities.json"
        if read_only:
            if json.loads(cap_path.read_bytes()) != capabilities.model_dump(mode="json"):
                raise ValueError("active_capability_snapshot_changed")
        else:
            write_once(cap_path, capabilities.model_dump(mode="json"))
    if read_only:
        if json.loads((directory / "active-plan.json").read_bytes()) != plan:
            raise ValueError("active_source_or_requirement_changed")
    else:
        write_once(directory / "active-plan.json", plan)
    if not (directory / "active-discovery.json").exists():
        guard()
    if "definition_expansion" in profile:
        # New definition search is deferred until the campaign identity/deadline are frozen.
        discovery = dict(
            schema_version="active-knowledge-discovery/2.0.0",
            queries=[],
            automatic_medical_acceptance=False,
            patient_data_sent=False,
            status="deferred_to_campaign_revision",
        )
        base.definition_expansion_policy = profile["definition_expansion"]
        if "acquisition_scope_policy" in profile:
            base.acquisition_scope_policy = profile["acquisition_scope_policy"]
        if profile.get("observation_failure_policy"):
            base.observation_failure_policy = profile["observation_failure_policy"]
    else:
        discovery = discoverer(directory / "knowledge-discovery", profile)
    if read_only:
        if json.loads((directory / "active-discovery.json").read_bytes()) != discovery:
            raise ValueError("active_discovery_receipt_changed")
    else:
        write_once(directory / "active-discovery.json", discovery)
    base.ledger = rows
    base.observations = {}
    from .material import numeric, observation

    payloads = dict(base.prepared.payload_bytes)
    for row in rows:
        if row["acquisition_method"] == "source_measurement" and row["component"]:
            component = next(c for c in base.manifest.components if c.id == row["component"])
            value = numeric(component, payloads)
            if value is not None:
                base.observations[row["id"]] = observation(
                    row["id"],
                    value,
                    unit=(component.unit or "source_units_unqualified")
                    if base.manifest.adapter_version
                    in {
                        "multimodal-adapters/1.1.0",
                        "multimodal-adapters/1.2.0",
                        "multimodal-adapters/1.3.0",
                    }
                    else "source_units_unqualified",
                )
    # Only the new policy defers tools until accepted scope evidence is consumed.
    # Legacy profiles keep the exact original preparation behavior.
    scoped = "scope_selection_policy" in profile
    native, methods = install_native(
        base,
        profile,
        rows,
        plan,
        guard=guard,
        native_runner=native_runner,
        defer_native=defer_native or scoped,
        read_only=read_only,
    )
    base.active_owner = ActiveOwner(base, profile, plan, discovery, native, methods)
    if scoped:
        base.initial_active_acquisition = json.loads(canonical(base.active_owner.report()))
    base.active_owner.check()


def install_native(
    base,
    profile,
    rows,
    plan,
    *,
    guard,
    native_runner=None,
    defer_native=False,
    read_only=False,
    eligible_targets=None,
):
    # Native runner owns registration, process claims, output verification and
    # Runtime materialization. A raw caller-supplied measurement is never accepted.
    directory = base.output
    native = None
    native_inputs = {
        c.id
        for c in base.manifest.components
        if c.kind == "image" and c.status in {"obtained", "partial"} and c.payload_ids
    }
    targets = [
        r
        for r in rows
        if r["acquisition_method"] == "disc_cup_measurement"
        and r["component"] in native_inputs
        and r.get("acquisition_allowed", True)
    ]
    if eligible_targets is not None:
        targets = [r for r in targets if r["id"] in eligible_targets]
    component_policy = profile.get("native_component_policy")
    if len(targets) > 1 and component_policy is None:
        raise ValueError("active_native_multiple_images_requires_explicit_selection")
    if component_policy and len(targets) > component_policy["max_components"]:
        raise ValueError("active_native_component_budget")
    methods = None
    if targets and not defer_native and profile.get("native_unavailable_policy"):
        from .entry_preflight import NATIVE_ALTERNATIVE, native_readiness

        if profile["native_unavailable_policy"] != NATIVE_ALTERNATIVE:
            raise ValueError("active_native_unavailable_policy")
        route_path = directory / "active-method-selection.json"
        binding = dict(
            schema_version="bounded-method-selection/1.0.0",
            plan_checksum=digest(plan),
            input_checksum=base.manifest.checksum,
            targets=[dict(id=r["id"], component=r["component"]) for r in targets],
            policy=NATIVE_ALTERNATIVE,
        )
        if read_only:
            methods = json.loads(route_path.read_bytes())
            if {k: methods.get(k) for k in binding} != binding:
                raise ValueError("active_method_selection_binding")
        else:
            guard()
            methods = dict(binding, readiness=native_readiness(allow_alternative=True))
            # Never reinterpret an existing/unknown native process as a fresh alternative.
            if (
                methods["readiness"]["method"] == "vision"
                and (directory / "native-fundus").exists()
            ):
                raise ValueError("active_native_prior_execution_requires_confirmation")
            write_once(route_path, methods)
        if methods["readiness"]["method"] == "vision":
            if methods["readiness"]["native_qualified"] is not False:
                raise ValueError("active_alternative_qualification")
            # Existing source definition explicitly permits qualitative visible cup shape.
            # The VLM path remains without a measurement; no invented mask or ratio is installed.
            for target in targets:
                target["missing_reason"] = "native_unavailable_visible_description_pending"
            targets = []
        elif methods["readiness"]["method"] != "native":
            raise ValueError("active_method_selection_method")
    native_records = []
    if targets and not defer_native:
        if native_runner is None:
            from .native_fundus import acquire

            native_runner = acquire
    for target in targets if not defer_native else ():
        current_native = native_runner(
            base,
            target,
            before_dispatch=guard,
            **({"component_scoped": True} if component_policy else {}),
        )
        native_records.append(current_native)
        if current_native["status"] == "obtained":
            bind_native_observation(base, target, current_native)
    if native_records:
        native = (
            dict(schema_version="native-fundus-collection/1.0.0", records=native_records)
            if component_policy
            else native_records[0]
        )
    return native, methods


def structure(report, base):
    """The source-bound medical grouping is distinct from scoring/AND-OR rules."""
    owner = base.active_owner
    owner.check()
    report["active_acquisition"] = owner.report()
    revision = getattr(base, "definition_revision", None)
    if revision:
        report["definition_revision"] = revision
    nodes = [
        dict(
            id="medical:" + n["id"],
            type="goal",
            title=n["title"],
            semantics="source_bound_medical_question_group_not_extra_vote",
        )
        for n in owner.plan["nodes"]
    ]
    edges = [
        dict(
            source="root" if n["parent"] == "root" else "medical:" + n["parent"],
            target="medical:" + n["id"],
            relation="contains",
        )
        for n in owner.plan["nodes"]
    ]
    rows = {r["id"]: r for r in owner.plan["rows"] if r["state"] != "definition_backlog"}
    if revision:
        nodes.append(
            dict(
                id="medical:DISCOVERED",
                type="goal",
                title="本轮新来源支持的检查",
                semantics="research_definition_revision_not_extra_vote",
            )
        )
        edges.append(dict(source="root", target="medical:DISCOVERED", relation="contains"))
        # The current revision is the full owner-validated ledger. A successor
        # may add nothing while retaining an earlier admitted check and its
        # newly obtained observation. Its local delta is not the display set.
        rows.update({r["id"]: r for r in revision["rows"]})
    for check in report["requirements"]:
        row = rows[check["id"]]
        edges.append(
            dict(
                source="medical:" + row["parent"],
                target="check:" + check["id"],
                relation="contains",
            )
        )
        check["source_requirement"] = row
        if row.get("fulfilled_requirements"):
            sections = check.get("presentation_sections")
            if sections is not None:
                sections["acquisition"]["shared_measurement_questions"] = row[
                    "fulfilled_requirements"
                ]
                sections["rationale"]["review_counting"] = (
                    "同一组件的登记检查与原值读取由一次联合评阅回应；不声称资料齐全，不增加独立票。"
                )
        if getattr(owner, "methods", None) is not None and any(
            t["id"] == check["id"] for t in owner.methods["targets"]
        ):
            check["execution_binding"]["method_selection"] = dict(
                pointer="/active_acquisition/method_selection",
                checksum=digest(owner.methods),
                selected_method=owner.methods["readiness"]["method"],
                native_qualification=owner.methods["readiness"]["native_qualified"],
                reason=owner.methods["readiness"]["reason"],
            )
        from .native_fundus import records

        matched = [
            (i, r)
            for i, r in enumerate(records(owner.native))
            if check["id"] == r["plan"]["target"]
        ]
        if matched:
            index, native = matched[0]
            binding = check["execution_binding"]
            binding.update(
                native_task=native["plan"]["tool"],
                native_result_pointer="/active_acquisition/native"
                + (
                    f"/records/{index}"
                    if owner.native.get("schema_version") == "native-fundus-collection/1.0.0"
                    else ""
                ),
                native_result_checksum=digest(native),
                native_record=native,
                artifacts=native["artifacts"],
            )
            check["presentation_sections"]["dependencies"] = binding
    report["presentation_structure"] = dict(
        schema_version="case-presentation-structure/1.0.0", nodes=nodes, edges=edges
    )
    report["report_production"]["organization"] = (
        "source_bound_question_hierarchy_execution_dependencies_separate"
    )
    return report


def fulfilled_target(prior, selected, capabilities):
    """Only the original current projection may replace a source-measurement question."""
    matches = []
    component = next(
        (c for c in capabilities.components if c.component_id == prior.get("component")), None
    )
    if component is None or prior.get("acquisition_method") != "source_measurement":
        return None
    for key, candidate in selected.items():
        if (
            candidate.get("component") != prior["component"]
            or candidate.get("acquisition_method") != "source_measurement"
        ):
            continue
        for bound in candidate.get("fulfilled_requirements", []):
            if (
                bound.get("schema_version") == "same-component-measurement-fulfillment/1.0.0"
                and bound.get("requirement") == prior["id"]
                and bound.get("component") == component.component_id
                and bound.get("component_checksum") == component.component_checksum
                and bound.get("original_question") == prior["definition"]
                and bound.get("original_parent") == prior["parent"]
                and bound.get("medical_completeness_claimed") is False
                and bound.get("fulfillment") == "shared_observation_and_joint_review_planned"
                and candidate.get("input_modalities") == prior.get("input_modalities")
            ):
                matches.append(key)
    if len(matches) > 1:
        raise ValueError("capability_transition_ambiguous_measurement_fulfillment")
    return matches[0] if matches else None
