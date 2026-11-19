"""Installed original-06 facts into existing requirements, never JSON authorization.

The consumer must install this before campaign freeze. No tool dispatch, new
requirement, medical direction or replacement of an existing observation occurs.
"""

import json
from copy import deepcopy

from apps.vectorengine_case_materials import ref
from packages.grounding.models import canonical, digest
from packages.llm_provider.models import ObservationInput
from packages.object_store import checksum
from packages.research_runtime.component_steps import ComponentPlannerSteps
from packages.tool_execution.fundus_bundle import MAX_BYTES, decode_bundle

from .fundus_mask_adapter import measure_masks
from .material import observation


def project_fundus(raw, *, input_checksum, artifact):
    """Explicit adapter for this reviewed output protocol, not arbitrary models."""
    if checksum(raw) != artifact.checksum:
        raise ValueError("component_observation_artifact_checksum")
    bundle, masks = decode_bundle(raw)
    if bundle.input_checksum != input_checksum:
        raise ValueError("component_observation_input_checksum")
    record = measure_masks(masks)
    value = dict(
        values=record["values"],
        quality=record["quality"],
        analysis_grid=record["coordinate_system"],
        physical_scale=None,
        radial_available=record["radial_rim"] is not None,
        radial_missing_reason=record["radial_missing_reason"],
        measurement_status=record["status"],
        validation="research_smoke_only_not_validated_stage_classifier",
        training_overlap=bundle.training_overlap,
        producer=ref("resource", artifact.asset_code, artifact.checksum).model_dump(mode="json"),
    )
    return value, record


class ComponentObservations:
    """An installation port with original owners, not caller-provided measurements."""

    def __init__(self, *, base, owner, execution_key, target, guard):
        if type(owner) is not ComponentPlannerSteps or not callable(guard):
            raise ValueError("component_observation_original_owner_required")
        self.base, self.owner, self.execution_key = base, owner, execution_key
        self.target, self.guard = target, guard
        self.input_row = None
        self.value, self.record, self.origin = self.read()
        self.binding = digest((self.value, self.record, self.origin))

    def read(self):
        base, owner = self.base, self.owner
        self.guard()
        manifest = base.read_dependency_input()
        request = owner.requests.parent.request
        if (
            request.scope.case_code != base.binding["case_code"]
            or request.input_checksum != manifest.checksum
            or request.visible_checksum != base.visible_value.checksum
            or request.task != base.task
        ):
            raise ValueError("component_observation_consumer_binding")
        rows = [r for r in base.ledger if r["id"] == self.target]
        if (
            len(rows) != 1
            or rows[0]["component"] != owner.requests.component_id
            or rows[0]["acquisition_method"] != "disc_cup_measurement"
        ):
            raise ValueError("component_observation_requirement_binding")
        if self.input_row is None:
            self.input_row = deepcopy(rows[0])
            self.input_resources = tuple(base.resources)
            self.input_binding = digest((self.input_row, self.input_resources))
        self._check_projection_input()
        state = owner.sessions.read(owner.requests.parent.run_id)

        def decode(workflow):
            if len(workflow.steps) != 1 or len(workflow.steps[0].result.artifacts) != 1:
                raise ValueError("component_observation_single_output_required")
            artifact = workflow.steps[0].result.artifacts[0]
            objects = owner.research.compiler.admission.objects
            raw = objects.read_bounded(artifact.uri, MAX_BYTES)
            registered = owner.requests.for_key(owner.requests.request_key)
            if len(registered.artifacts) != 1:
                raise ValueError("component_observation_single_input_required")
            value, record = project_fundus(
                raw, input_checksum=registered.artifacts[0].checksum, artifact=artifact
            )
            self.guard()
            if base.read_dependency_input() != manifest:
                raise ValueError("component_observation_consumer_changed")
            self._check_projection_input()
            return value, record, registered, artifact

        def final_check():
            self.guard()
            self._check_projection_input()

        # Original owner brackets decoding, consumer checks and source reads with
        # current admission/QC; no external callback follows its final check.
        workflow, (value, record, registered, artifact) = owner.consume_confirmed_result(
            state, self.execution_key, decode, final_check=final_check
        )
        self._check_projection_input()
        origin = dict(
            schema_version="component-observation-origin/1.1.0",
            consumption_policy="confirmed-result-consumption/1.0.0",
            target=self.target,
            component=owner.requests.component_id,
            producer_run=owner.requests.parent.run_id,
            child_run=registered.lock.run_id,
            execution_key=self.execution_key,
            workflow_checksum=digest(workflow),
            plan_checksum=workflow.plan_checksum,
            input_checksum=manifest.checksum,
            visible_checksum=base.visible_value.checksum,
            task_checksum=base.task.checksum,
            artifact=artifact.model_dump(mode="json"),
            adapter="confirmed-fundus-geometry/1.0.0",
            input_projection_checksum=self.input_binding,
        )
        return value, record, origin

    def check(self):
        self.check_fact()
        adoption = getattr(self, "interpretation_adoption", None)
        if adoption is not None:
            receipt = self.interpretation_owner.read_for_report(adoption["key"])
            if digest(receipt) != adoption["receipt_checksum"]:
                raise ValueError("component_interpretation_source_changed")

    def check_fact(self):
        """Original fact qualification, independent of any later interpretation."""
        from packages.evidence_details.accepted_read_scope import current_fact

        return current_fact(self, self._check_fact)

    def _check_fact(self):
        self.check_projection()
        expected = self.binding
        if digest(self.read()) != expected or self.binding != expected:
            raise ValueError("component_observation_changed")
        self.check_projection()

    def check_projection(self):
        """Frozen method/unit and local projection only, NOT current permission.

        Public definition planning uses this metadata without consuming tool
        results. Real observation consumption must use check() and its owner.
        """
        self._check_projection_input()
        if digest((self.value, self.record, self.origin)) != self.binding:
            raise ValueError("component_observation_changed")
        if getattr(self.base, "component_observations", None) is self:
            rows = [r for r in self.base.ledger if r["id"] == self.target]
            resource = self.resource()
            if (
                self.base.observations.get(self.target) != self.observation()
                or resource not in self.base.resources
                or len(rows) != 1
                or rows[0].get("quality") != self.record["quality"]
                or rows[0].get("quality_source") != resource
                or rows[0].get("dependency_paths")
                != ((self.base.resources[0].code, resource.code),)
            ):
                raise ValueError("component_observation_projection_changed")

    def resource(self):
        artifact = self.origin["artifact"]
        return ref("resource", artifact["asset_code"], artifact["checksum"])

    def interpretation_basis(self, requirement_id):
        """Current original-06 fact for a new interpretation, never a new vote.

        Preserve the frozen input row and original observation identity. This
        port neither changes the projection nor dispatches a tool.
        """
        if requirement_id != self.target:
            raise ValueError("component_interpretation_requirement")
        self.check_fact()
        observed = self.observation()
        if observed.status != "obtained":
            raise ValueError("component_interpretation_observation_unavailable")
        return json.loads(
            canonical(
                dict(
                    schema_version="confirmed-interpretation-basis/1.0.0",
                    requirement=self.input_row,
                    observation=observed.reference,
                    method=observed.method,
                    unit=observed.unit,
                    input_projection_checksum=self.input_binding,
                    observation_binding=self.binding,
                    origin=self.origin,
                )
            )
        )

    def dependency_basis(self, requirement_id):
        """Current exact method and narrowly named technical checks, not accuracy.

        interpretation_basis performs the original current admission and artifact
        read. These booleans describe only that same verified output protocol.
        """
        basis = self.interpretation_basis(requirement_id)
        if self.origin.get("adapter") != "confirmed-fundus-geometry/1.0.0":
            raise ValueError("component_dependency_adapter_not_supported")
        return dict(
            basis,
            schema_version="confirmed-interpretation-basis/2.0.0",
            technical_checks=dict(
                artifact_integrity=True,
                mask_measurement_available=self.record["qc"]["measurement_available"] is True,
            ),
            technical_policy=self.record["qc"]["policy_checksum"],
            quality_record_checksum=digest(self.record["qc"]),
            technical_scope="verified_bundle_bytes_and_computable_masks_not_segmentation_accuracy",
        )

    def prepare_interpretation(self, owner, key):
        from packages.evidence_reuse.definition_acceptance import ResearchDefinitionOwner

        if type(owner) is not ResearchDefinitionOwner or owner.confirmed_observations is not self:
            raise ValueError("component_interpretation_original_owner_required")
        # Qualification reconstruction is read-only, including cold export after
        # expiry/cancel. Dispatch and new acceptance retain their separate guards.
        receipt, projection = owner.read_confirmed_projection(key)
        if (
            projection["id"] != self.target
            or receipt["confirmed_basis"]["input_projection_checksum"] != self.input_binding
        ):
            raise ValueError("component_interpretation_projection_binding")
        changes = {
            k: projection[k]
            for k in (
                "definition",
                "basis_kind",
                "definition_receipt_checksum",
                "dependency_assessment",
                "acquisition_allowed",
                "review_allowed",
                "expected_method",
                "expected_unit",
            )
        }
        changes["definition_receipt_key"] = key
        return json.loads(
            canonical(dict(key=key, receipt_checksum=digest(receipt), changes=changes))
        )

    def install_interpretation(self, owner, adoption, rows):
        """Install only a freshly reconstructed same-ID projection, with rollback."""
        rebuilt = self.prepare_interpretation(owner, adoption["key"])
        if canonical(rebuilt) != canonical(adoption):
            raise ValueError("component_interpretation_projection_binding")
        previous = getattr(self, "interpretation_adoption", None)
        if previous is not None and canonical(previous) != canonical(rebuilt):
            raise ValueError("component_interpretation_already_bound")
        old_rows = self.base.ledger
        old_owner = getattr(self, "interpretation_owner", None)
        old_binding = getattr(self, "interpretation_adoption_binding", None)
        self.interpretation_adoption = deepcopy(rebuilt)
        self.interpretation_adoption_binding = digest(rebuilt)
        self.interpretation_owner = owner
        self.base.ledger = deepcopy(rows)
        try:
            self.check_projection()
        except Exception:
            self.base.ledger = old_rows
            self.interpretation_adoption = previous
            self.interpretation_owner = old_owner
            self.interpretation_adoption_binding = old_binding
            raise

    def _check_projection_input(self):
        if digest((self.input_row, self.input_resources)) != self.input_binding:
            raise ValueError("component_observation_input_projection_changed")
        rows = [r for r in self.base.ledger if r["id"] == self.target]
        expected, resources = deepcopy(self.input_row), self.input_resources
        adoption = getattr(self, "interpretation_adoption", None)
        if adoption is not None:
            if digest(adoption) != self.interpretation_adoption_binding:
                raise ValueError("component_interpretation_projection_changed")
            expected.update(deepcopy(adoption["changes"]))
        if getattr(self.base, "component_observations", None) is self:
            resource = self.resource()
            expected.update(
                quality=self.record["quality"],
                quality_source=resource,
                dependency_paths=((self.input_resources[0].code, resource.code),),
            )
            if self.record["status"] == "missing":
                expected["missing_reason"] = "confirmed_geometry_unavailable"
            if resource not in resources:
                resources = (*resources, resource)
        row_matches = (
            canonical(rows) == canonical([expected]) if adoption is not None else rows == [expected]
        )
        if not row_matches or tuple(self.base.resources) != resources:
            raise ValueError("component_observation_projection_changed")

    def observation(self):
        result = observation(
            self.target, self.value, method="confirmed_fundus_disc_cup_geometry", unit="ratio"
        )
        if self.record["status"] != "missing":
            return result
        body = dict(
            status="missing", method=result.method, unit=result.unit, value=result.value, image=None
        )
        return ObservationInput(
            reference=ref("observation", result.reference.code, digest(body)), **body
        )

    def report(self):
        return json.loads(
            canonical(
                dict(origin=self.origin, geometry=self.record, observation_binding=self.binding)
            )
        )

    def install(self):
        if (self.base.output / "protocol.json").exists():
            raise ValueError("component_observation_before_freeze_required")
        self._install()

    def restore(self):
        """Rebuild a cold base only when its existing protocol names this fact.

        Local protocol bytes are not authorization: constructor/check still
        traverse original owners. Campaign.freeze must verify the whole protocol
        before use; its time, budget and prior phases are never renewed here.
        """
        path = self.base.output / "protocol.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("component_observation_frozen_protocol_required")
        protocol = json.loads(path.read_bytes())
        if canonical(protocol.get("component_observations")) != canonical(self.report()):
            raise ValueError("component_observation_frozen_binding_changed")
        self._install()

    def _install(self):
        base = self.base
        if getattr(base, "component_observations", None) is not None:
            raise ValueError("component_observation_already_installed")
        self.check()
        observed = self.observation()
        if self.target in base.observations:
            raise ValueError("component_observation_existing_fact")
        resource = self.resource()
        rows = [dict(r) for r in base.ledger]
        target = next(r for r in rows if r["id"] == self.target)
        target.update(
            quality=self.record["quality"],
            quality_source=resource,
            dependency_paths=((base.resources[0].code, resource.code),),
        )
        if self.record["status"] == "missing":
            target["missing_reason"] = "confirmed_geometry_unavailable"
        base.ledger = rows
        if resource not in base.resources:
            base.resources = (*base.resources, resource)
        base.observations[self.target] = observed
        base.component_observations = self
