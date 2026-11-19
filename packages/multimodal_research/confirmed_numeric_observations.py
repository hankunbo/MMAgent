"""Exact original input measurements for versioned research interpretation reuse.

No dispatch, replacement measurement or medical validity is granted here.
"""

import json
from copy import deepcopy

from packages.grounding.models import canonical, digest
from packages.multimodal_research.campaign import InputEvidence
from packages.multimodal_research.material import numeric, observation
from packages.object_store import checksum


class ConfirmedNumericObservations:
    """Reconstruct source measurements; never rename them to another method."""

    def __init__(self, base, *, method_context_profile=None, method_context_targets=()):
        if type(base) is not InputEvidence:
            raise ValueError("numeric_interpretation_original_input_owner_required")
        self.base = base
        self.input_checksum = base.manifest.checksum
        self.task_checksum = base.task.checksum
        self.input_rows = {
            row["id"]: deepcopy(row)
            for row in base.ledger
            if row.get("acquisition_method") == "source_measurement"
            and row["id"] in base.observations
            and base.observations[row["id"]].method == "source_measurement_projection"
            and base.observations[row["id"]].status == "obtained"
        }
        self.original_observations = {
            key: deepcopy(base.observations[key]) for key in self.input_rows
        }
        self.snapshot = digest((self.input_rows, self.original_observations))
        self.adoptions = {}
        self.adoption_bindings = {}

        self.method_context_profile = method_context_profile
        self.method_context_targets = tuple(method_context_targets)
        if (method_context_profile is None) != (not self.method_context_targets):
            raise ValueError("numeric_interpretation_metadata_profile")
        if len(set(self.method_context_targets)) != len(self.method_context_targets) or not set(
            self.method_context_targets
        ) <= set(self.input_rows):
            raise ValueError("numeric_interpretation_metadata_targets")
        self.method_contexts = self._method_contexts()
        self.method_context_checksum = digest(self.method_contexts)

    def _method_contexts(self):
        if self.method_context_profile is None:
            return {}
        from .review_context import collect_methods

        return collect_methods(self.base, self.method_context_targets, self.method_context_profile)

    def method_metadata(self, target):
        """Current original-author method meaning, not a value or disease relation."""
        if target not in self.method_context_targets:
            return None
        self.check_projection(target)
        current = self._method_contexts()
        if digest(current) != self.method_context_checksum:
            raise ValueError("numeric_interpretation_method_metadata_changed")
        return deepcopy(current[target]["author_method"])

    def check_projection(self, target):
        if digest((self.input_rows, self.original_observations)) != self.snapshot:
            raise ValueError("numeric_interpretation_snapshot_changed")
        if target not in self.input_rows:
            raise ValueError("numeric_interpretation_requirement")
        expected = deepcopy(self.input_rows[target])
        adoption = self.adoptions.get(target)
        if adoption is not None:
            if digest(adoption) != self.adoption_bindings.get(target):
                raise ValueError("numeric_interpretation_adoption_changed")
            expected.update(deepcopy(adoption["changes"]))
        rows = [r for r in self.base.ledger if r["id"] == target]
        if canonical(rows) != canonical([expected]) or (
            self.base.observations.get(target) != self.original_observations[target]
        ):
            raise ValueError("numeric_interpretation_projection_changed")

    def interpretation_basis(self, target):
        self.check_projection(target)
        base = self.base
        manifest = base.read_dependency_input()
        if manifest.checksum != self.input_checksum or base.task.checksum != self.task_checksum:
            raise ValueError("numeric_interpretation_input_scope")
        row = self.input_rows[target]
        selected = [c for c in manifest.components if c.id == row["component"]]
        if len(selected) != 1:
            raise ValueError("numeric_interpretation_component_binding")
        component = selected[0]
        if component.kind not in {"scalar", "vector", "map"} or component.status not in {
            "obtained",
            "partial",
        }:
            raise ValueError("numeric_interpretation_input_unavailable")
        payloads = dict(base.prepared.payload_bytes)
        declared = {p.id: p for p in manifest.payloads}
        proof = []
        for key in component.payload_ids:
            record, data = declared.get(key), payloads.get(key)
            if (
                record is None
                or data is None
                or (checksum(data), len(data)) != (record.checksum, record.byte_size)
            ):
                raise ValueError("numeric_interpretation_payload_binding")
            proof.append(dict(id=key, checksum=record.checksum, byte_size=record.byte_size))
        measured = numeric(component, payloads)
        unit = (
            component.unit or "source_units_unqualified"
            if manifest.adapter_version
            in {"multimodal-adapters/1.1.0", "multimodal-adapters/1.2.0"}
            else "source_units_unqualified"
        )
        if measured is None:
            raise ValueError("numeric_interpretation_observation_unavailable")
        rebuilt = observation(target, measured, unit=unit)
        original = self.original_observations[target]
        if rebuilt != original or original.status != "obtained":
            raise ValueError("numeric_interpretation_method_binding")
        origin = dict(
            schema_version="confirmed-input-numeric-origin/1.0.0",
            input_checksum=manifest.checksum,
            visible_checksum=base.visible_value.checksum,
            task_checksum=base.task.checksum,
            component_id=component.id,
            component_checksum=digest(component),
            payloads=proof,
            adapter_version=manifest.adapter_version,
            numeric_checksum=digest(measured),
            qualification="original_source_projection_not_clinical_accuracy",
        )
        metadata = self.method_metadata(target)
        if metadata is not None:
            origin["schema_version"] = "confirmed-input-numeric-origin/1.1.0"
            origin["author_method"] = metadata
            origin["author_method_scope"] = (
                "release_field_meaning_not_individual_quality_or_direction"
            )
        if base.read_dependency_input() != manifest:
            raise ValueError("numeric_interpretation_input_changed")
        self.check_projection(target)
        return json.loads(
            canonical(
                dict(
                    schema_version="confirmed-numeric-interpretation-basis/1.0.0",
                    requirement=row,
                    observation=original.reference,
                    method=original.method,
                    unit=original.unit,
                    input_projection_checksum=digest((row, component)),
                    observation_binding=digest((original, origin)),
                    origin=origin,
                )
            )
        )

    def dependency_basis(self, target):
        return dict(
            self.interpretation_basis(target),
            technical_checks={},
            technical_scope="confirmed_source_projection_only_no_mask_or_clinical_qc",
        )

    def prepare_interpretation(self, owner, key):
        from packages.evidence_reuse.definition_acceptance import ResearchDefinitionOwner

        if type(owner) is not ResearchDefinitionOwner or owner.confirmed_observations is not self:
            raise ValueError("numeric_interpretation_original_definition_owner_required")
        receipt, projection = owner.read_confirmed_projection(key)
        target = projection["id"]
        basis = self.interpretation_basis(target)
        if (
            receipt["confirmed_basis"]["input_projection_checksum"]
            != basis["input_projection_checksum"]
        ):
            raise ValueError("numeric_interpretation_adoption_binding")
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
            canonical(
                dict(key=key, target=target, receipt_checksum=digest(receipt), changes=changes)
            )
        )

    def install_interpretation(self, owner, adoption, rows):
        rebuilt = self.prepare_interpretation(owner, adoption["key"])
        if canonical(rebuilt) != canonical(adoption):
            raise ValueError("numeric_interpretation_adoption_binding")
        target = rebuilt["target"]
        previous = self.adoptions.get(target)
        if previous is not None and canonical(previous) != canonical(rebuilt):
            raise ValueError("numeric_interpretation_already_bound")
        old_rows = self.base.ledger
        old_binding = self.adoption_bindings.get(target)
        self.adoptions[target] = deepcopy(rebuilt)
        self.adoption_bindings[target] = digest(rebuilt)
        self.base.ledger = deepcopy(rows)
        try:
            self.check_projection(target)
        except Exception:
            self.base.ledger = old_rows
            if previous is None:
                self.adoptions.pop(target, None)
                self.adoption_bindings.pop(target, None)
            else:
                self.adoptions[target] = previous
                self.adoption_bindings[target] = old_binding
            raise
