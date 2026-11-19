"""Current original Generic measurements for a new research interpretation.

This is a read adapter, not a second input registry or case executor. No tool,
request, observation, historical definition or score is created by this owner.
"""

import json
from dataclasses import dataclass

from packages.grounding.models import canonical, digest


@dataclass(frozen=True)
class ConfirmedOnlyInputView:
    """A restricted dependency view, not a claim that the case lacks raw inputs.

    Only the selected already-confirmed methods are eligible through this adapter.
    Additional patient input conditions require another original input owner; an
    empty component set here must never be reported as a new patient measurement.
    """

    checksum: str
    components: tuple = ()
    schema_version: str = "generic-confirmed-only-dependency-view/1.0.0"


def measurement_basis(state, base, target):
    """Bind the exact measured projection to its original tool and image assets."""
    from .comprehensive_materials import measured_projection

    measurements, origins = measured_projection(base.tools, include_origins=True)
    requirements = {r.id: r for r in state.policy.requirements}
    items = {r.id: r for r in state.policy.items}
    item = items.get(target)
    if item is not None and getattr(item, "method", None) == "registered_segman_geometry":
        return _segmentation_basis(state, base, target, requirements, item)
    if (
        target not in measurements
        or target not in requirements
        or target not in items
        or base.extra_artifact is None
    ):
        raise ValueError("generic_definition_confirmed_measurement_missing")
    item = items[target]
    observed = next(
        (o for o in state.material.observations if o.reference.code == item.observation_code),
        None,
    )
    measured = measurements[target]
    expected = dict(measured, full_record=base.extra_artifact.asset_code)
    if (
        observed is None
        or observed.status != "obtained"
        or (observed.method, observed.unit) != (measured["method"], measured["unit"])
        or observed.image is not None
        or observed.value != canonical(expected).decode()
    ):
        raise ValueError("generic_definition_measurement_identity")
    row = requirements[target].model_dump(mode="json")
    row.update(
        component=base.image_artifact.asset_code,
        acquisition_method="confirmed_tool_observation",
        expected_method=observed.method,
        expected_unit=observed.unit,
    )
    origin = dict(
        schema_version="confirmed-generic-tool-origin/1.0.0",
        input_checksum=state.material.scope.input_checksum,
        target=state.material.goal_code,
        tool_artifact=base.extra_artifact.model_dump(mode="json"),
        image_artifact=base.image_artifact.model_dump(mode="json"),
        tool_result_checksum=digest(base.tools),
        projection_component=origins[target],
        projection_checksum=digest(measured),
        method_semantics={
            k: measured[k]
            for k in ("method", "unit", "definition", "reference_match", "is_CRAE", "use")
            if k in measured
        },
        qualification="original_tool_measurement_not_method_equivalence_or_clinical_accuracy",
    )
    return json.loads(
        canonical(
            dict(
                schema_version="confirmed-generic-interpretation-basis/1.0.0",
                requirement=row,
                observation=observed.reference,
                method=observed.method,
                unit=observed.unit,
                input_projection_checksum=digest((row, origin)),
                observation_binding=digest((observed, origin)),
                origin=origin,
            )
        )
    )


def _segmentation_basis(state, base, target, requirements, item):
    """Expose original geometry facts; do not equate measurement and interpretation.

    The field map is the existing DynamicCaseEvidence measurement contract. It
    is intentionally not a name-based match to a different catalog requirement.
    """
    from apps.active_evidence_materials import PILOTS

    from .generic_ports import read
    from .quality import mask_quality
    from .quality_materials import quality_observation_value

    fields = dict(zip(PILOTS[:2], ("vertical_cdr", "radial_rim_minimum_ratio"), strict=True))
    if target not in fields or target not in requirements:
        raise ValueError("generic_definition_segmentation_target")
    ports = read(base.directory, base.binding, base.runtime, base.objects)
    if ports is None:
        raise ValueError("generic_definition_confirmed_measurement_missing")
    artifacts, raw, measurement = ports
    quality = mask_quality(raw["disc_mask"], raw["cup_mask"])
    if measurement != base.measurement or not quality["measurement_available"]:
        raise ValueError("generic_definition_segmentation_current_measurement")
    value = measurement["measurements"].get(fields[target])
    observed = next(
        (o for o in state.material.observations if o.reference.code == item.observation_code),
        None,
    )
    expected = dict(value=value, input_checksum=base.image_artifact.checksum)
    expected_value = quality_observation_value(canonical(expected).decode(), quality)
    if (
        value is None
        or observed is None
        or observed.status != "obtained"
        or (observed.method, observed.unit) != ("registered_segman_geometry", "ratio")
        or observed.image is not None
        or observed.value != expected_value
    ):
        raise ValueError("generic_definition_measurement_identity")
    row = requirements[target].model_dump(mode="json")
    row.update(
        component=base.image_artifact.asset_code,
        acquisition_method="confirmed_tool_observation",
        expected_method=observed.method,
        expected_unit=observed.unit,
    )
    origin = dict(
        schema_version="confirmed-generic-tool-origin/1.1.0",
        input_checksum=state.material.scope.input_checksum,
        target=state.material.goal_code,
        tool_artifact=artifacts["measurements"].model_dump(mode="json"),
        tool_dependencies={k: a.model_dump(mode="json") for k, a in artifacts.items()},
        image_artifact=base.image_artifact.model_dump(mode="json"),
        tool_result_checksum=digest(measurement),
        projection_component=fields[target],
        projection_checksum=digest(expected),
        method_semantics=dict(
            method=observed.method,
            unit=observed.unit,
            original_projection="DynamicCaseEvidence.project+QualityCaseEvidence.project",
        ),
        quality=quality,
        qualification="original_tool_measurement_not_method_equivalence_or_clinical_accuracy",
    )
    return json.loads(
        canonical(
            dict(
                schema_version="confirmed-generic-interpretation-basis/1.1.0",
                requirement=row,
                observation=observed.reference,
                method=observed.method,
                unit=observed.unit,
                input_projection_checksum=digest((row, origin)),
                observation_binding=digest((observed, origin)),
                origin=origin,
            )
        )
    )


class ConfirmedGenericMeasurements:
    """Read exact completed-root facts through current original Generic owners."""

    def __init__(self, campaign, selected, *, registered_measurement=False):
        from .generic_campaign import GenericCampaign

        if type(campaign) is not GenericCampaign:
            raise ValueError("generic_definition_original_campaign_owner_required")
        selected = tuple(selected)
        if not 1 <= len(selected) <= 8 or len(set(selected)) != len(selected):
            raise ValueError("generic_definition_selected_scope")
        if type(registered_measurement) is not bool:
            raise ValueError("generic_definition_measurement_scope")
        self.registered_measurement = registered_measurement
        self.campaign, self.selected = campaign, selected
        self.snapshot = self._read()
        self.snapshot_checksum = digest(self.snapshot)

    def _read(self):
        from types import SimpleNamespace

        from agents.expert_agent.reporting import read_research_report
        from packages.evidence_details.acceptance import ResearchAcceptance
        from packages.research_runtime.agent_records import RuntimeAgentJournal

        from .generic_input import directory, load
        from .generic_pending_review import original_phase

        campaign = self.campaign
        if (
            load(campaign.spec["key"]) != campaign.spec
            or (directory(campaign.spec["key"]) / "cancel-request.json").exists()
        ):
            raise ValueError("generic_definition_original_input_or_cancel")
        docs, state, phase, final = original_phase(campaign)
        control = campaign.sessions.read(docs.run_id)
        if control is None or control.phase != "stopped" or control.pending or control.cancelled:
            raise ValueError("generic_definition_original_not_closed")
        accepted = read_research_report(
            ResearchAcceptance(
                owner=SimpleNamespace(current=lambda: state), documents=docs, requests=None
            ),
            key=final["acceptance_key"],
            parent=state,
        )
        saved = docs.read("dynamic-case-report")
        if not saved or saved[0].get("scores") != accepted["scores"]:
            raise ValueError("generic_definition_original_score_binding")
        registered = getattr(self, "registered_measurement", False)
        if registered:
            from .generic_registered_measurement import (
                TARGET,
                current_record,
                interpretation_basis,
                project_registered,
            )

            if self.selected != (TARGET,):
                raise ValueError("generic_definition_registered_scope")
            record = current_record(campaign.spec["key"])
            projected, transition = project_registered(state, record)
            facts = {TARGET: interpretation_basis(projected, record, transition)}
        else:
            facts = {
                target: measurement_basis(state, campaign.base, target) for target in self.selected
            }
        phase.verify(state)
        result = dict(
            schema_version="generic-confirmed-definition-origin/1.0.0",
            root_key=campaign.spec["key"],
            input_checksum=state.material.scope.input_checksum,
            target=state.material.goal_code,
            state_checksum=digest(state),
            acceptance_key=final["acceptance_key"],
            acceptance_checksum=digest(accepted),
            run_id=docs.run_id,
            control_checksum=digest(control),
            journal_checksum=digest(RuntimeAgentJournal(docs).read()),
            facts=facts,
        )
        if registered:
            result.update(
                schema_version="generic-confirmed-definition-origin/1.1.0",
                registered_measurement=record,
            )
        return result

    def current(self):
        if digest(self.snapshot) != self.snapshot_checksum:
            raise ValueError("generic_definition_snapshot_changed")
        current = self._read()
        if digest(current) != self.snapshot_checksum:
            raise ValueError("generic_definition_original_changed")
        return current

    def scope(self):
        value = self.current()
        return dict(input_checksum=value["input_checksum"], target=value["target"])

    def interpretation_basis(self, target):
        if target not in self.selected:
            raise ValueError("generic_definition_requirement_scope")
        return self.current()["facts"][target]

    def dependency_basis(self, target):
        return dict(
            self.interpretation_basis(target),
            technical_checks={},
            technical_scope="confirmed_tool_projection_only_no_new_qc_or_medical_validation",
        )

    def input_view(self):
        # This restricted view grants no raw-input predicates. Re-read only the
        # original visible-input owner here; confirmed_basis independently brackets
        # full source/method/acceptance reconstruction before and after every build.
        if digest(self.snapshot) != self.snapshot_checksum:
            raise ValueError("generic_definition_snapshot_changed")
        base = self.campaign.base
        current = base.visible.read(base.binding["case_code"])
        if current != base.visible_value or current.checksum != self.snapshot["input_checksum"]:
            raise ValueError("generic_definition_visible_input_changed")
        return ConfirmedOnlyInputView(current.checksum)

    def dependency_fact(self, contract):
        raise ValueError("generic_definition_additional_input_owner_required")

    def draft_context(self, *, materials, public_queries):
        """Public drafting context only; never export measured values or identities."""
        from packages.multimodal_research.definition_context import (
            candidate_contract_choices,
            public_examination,
            public_gap_context,
            requirement_context,
        )

        current = self.current()
        facts = [current["facts"][target] for target in self.selected]
        rows = [f["requirement"] for f in facts]
        queries = list(public_queries)
        from packages.knowledge_search.privacy import safe_text

        if not 1 <= len(queries) <= 3 or any(
            not isinstance(q, str) or len(q) > 120 or not safe_text(q) for q in queries
        ):
            raise ValueError("generic_definition_public_query_scope")
        plan = [dict(query=q, target_relation_gap_ids=list(self.selected)) for q in queries]
        context = requirement_context(rows, plan, {})
        capabilities = []
        for fact, entry in zip(facts, context["requirements"], strict=True):
            entry["input_alias"] = None
            capabilities.append(
                dict(
                    requirement_alias=entry["requirement_id"],
                    input_alias=None,
                    method=fact["method"],
                    unit=fact["unit"],
                    resolvable_conditions=["confirmed_method"],
                    semantics="current_original_generic_tool_projection_reconstruction_required",
                    method_metadata=fact["origin"]["method_semantics"],
                    exclusions=["clinical_accuracy", "method_equivalence", "new_patient_input"],
                )
            )
        return candidate_contract_choices(
            dict(
                target=current["target"],
                max_definitions=min(3, len(facts)),
                existing_examinations=[public_examination(row) for row in rows],
                existing_requirement_context=context,
                definition_gaps=public_gap_context(rows, plan),
                available_modalities=sorted({m for r in rows for m in r["input_modalities"]}),
                source_materials=materials,
                public_queries=queries,
                input_capabilities=[],
                confirmed_observation_capabilities=capabilities,
                dependency_capability_version="definition-owner-capabilities/1.3.0",
                interpretation_scope=(
                    "Revise only a selected existing confirmed method. No new acquisition or "
                    "patient input qualification. Preserve the original method and unit; "
                    "method metadata is not medical evidence of validity or equivalence."
                ),
            )
        )
