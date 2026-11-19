"""Original Generic facts viewed by the existing DefinitionExpansion owner.

This view exposes no raw-input acquisition and mutates no historical Generic
catalog, observation or score. A new definition remains campaign research only.
"""

import json
from copy import deepcopy
from types import SimpleNamespace

from packages.evidence_reuse.definition_acceptance import ResearchDefinitionOwner
from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import TaskDefinition
from packages.object_store import checksum

from .generic_definition_facts import ConfirmedGenericMeasurements


def public_proposition(base, target):
    """Project the current original source-driven root, not the old patient tree."""
    from pathlib import Path

    from db.runtime.repository.case_pipeline import identity

    from . import dynamic_materials

    original_structure = dynamic_materials.DynamicCaseEvidence.build_structure(base)
    original = [n for n in original_structure["nodes"] if n["parent"] is None]
    structural = [n for n in base.structure["nodes"] if n["parent"] is None]
    current = [n for n in base.view["nodes"] if n["parent"] is None]
    if len(original) != 1 or len(structural) != 1 or len(current) != 1:
        raise ValueError("generic_definition_public_root_ambiguous")
    fields = ("key", "node_type", "parent")
    old = original[0]
    if (
        base.root_version
        not in {"generic-source-case-root/1.0.0", "generic-source-case-root/2.0.0"}
        or base.structure.get("definition_catalog_checksum")
        != original_structure["definition_catalog_checksum"]
        or {k: old[k] for k in fields} != {k: structural[0][k] for k in fields}
        or {k: old[k] for k in fields} != {k: current[0][k] for k in fields}
        or old["node_type"] != "GOAL"
        or current[0]["node_code"] != target
        or target != identity("GOAL", base.binding["case_code"] + old["key"])
        or not isinstance(old["key"], str)
        or not 1 <= len(old["key"]) <= 240
    ):
        raise ValueError("generic_definition_public_root_binding")
    return dict(
        schema_version="generic-original-public-proposition/1.0.0",
        title=old["key"],
        title_basis="original_source_driven_root_key",
        original_root_version=base.root_version,
        original_catalog_checksum=original_structure["definition_catalog_checksum"],
        root_builder_checksum=checksum(Path(dynamic_materials.__file__).read_bytes()),
        original_root_key=old["key"],
        target=target,
        scope="declared_research_question_not_diagnosis_or_dataset_label",
        previous_patient_results_reused=False,
    )


class GenericDefinitionInputView:
    """Narrow current input/confirmed-method ports for original definition owner."""

    def __init__(self, facts):
        from .generic_pending_review import original_phase

        if type(facts) is not ConfirmedGenericMeasurements:
            raise ValueError("generic_definition_view_original_facts_required")
        current = facts.current()
        _, state, phase, _ = original_phase(facts.campaign)
        phase.verify(state)
        if digest(state) != current["state_checksum"]:
            raise ValueError("generic_definition_view_current_state_changed")
        self.facts = facts
        self.public_goal = public_proposition(facts.campaign.base, current["target"])
        self.public_goal_checksum = digest(self.public_goal)
        self.manifest = facts.input_view()
        self.task = SimpleNamespace(target_code=current["target"])
        self.ledger = [
            dict(
                deepcopy(current["facts"][key]["requirement"]),
                definition=validate(
                    TaskDefinition, current["facts"][key]["requirement"]["definition"]
                ),
            )
            for key in facts.selected
        ]
        by_code = {o.reference.code: o for o in state.material.observations}
        if "registered_measurement" in current:
            from packages.llm_provider.models import ObservationInput

            self.observations = {
                key: validate(ObservationInput, current["facts"][key]["observation_value"])
                for key in facts.selected
            }
        else:
            self.observations = {
                key: by_code[current["facts"][key]["observation"]["code"]] for key in facts.selected
            }
        self.original_rows = deepcopy(self.ledger)
        self.original_rows_checksum = digest(self.original_rows)
        self.original_observations_checksum = digest(self.observations)
        self.adoptions = {}
        self.adoption_bindings = {}
        self.views = {"views": []}
        self.definition_materials = {}

    def read_dependency_input(self):
        return self.facts.input_view()

    def read_dependency_fact(self, contract):
        return self.facts.dependency_fact(contract)

    def check(self):
        current = self.facts.current()
        if (
            current["input_checksum"] != self.manifest.checksum
            or current["target"] != self.task.target_code
            or digest(self.original_rows) != self.original_rows_checksum
            or digest(self.observations) != self.original_observations_checksum
            or digest(self.public_goal) != self.public_goal_checksum
            or public_proposition(self.facts.campaign.base, current["target"]) != self.public_goal
        ):
            raise ValueError("generic_definition_view_changed")
        expected = deepcopy(self.original_rows)
        for row in expected:
            adoption = self.adoptions.get(row["id"])
            if adoption is not None:
                if digest(adoption) != self.adoption_bindings.get(row["id"]):
                    raise ValueError("generic_definition_view_adoption_changed")
                row.update(deepcopy(adoption["changes"]))
        if canonical(expected) != canonical(self.ledger):
            raise ValueError("generic_definition_view_projection_changed")

    def draft_context(self, payload):
        self.check()
        context = self.facts.draft_context(
            materials=payload["source_materials"], public_queries=payload["public_queries"]
        )
        return dict(payload, **context, public_proposition=deepcopy(self.public_goal))

    def prepare_interpretation(self, owner, key):
        if (
            type(owner) is not ResearchDefinitionOwner
            or owner.confirmed_observations is not self.facts
        ):
            raise ValueError("generic_definition_view_original_definition_owner_required")
        self.check()
        receipt, projection = owner.read_confirmed_projection(key)
        target = projection["id"]
        if target not in self.facts.selected:
            raise ValueError("generic_definition_view_unselected")
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
        target = rebuilt["target"]
        if canonical(rebuilt) != canonical(adoption):
            raise ValueError("generic_definition_view_adoption_binding")
        previous = self.adoptions.get(target)
        if previous is not None and canonical(previous) != canonical(rebuilt):
            raise ValueError("generic_definition_view_already_bound")
        expected = deepcopy(self.ledger)
        for row in expected:
            if row["id"] == target:
                row.update(deepcopy(rebuilt["changes"]))
        if canonical(rows) != canonical(expected):
            raise ValueError("generic_definition_view_projection_changed")
        self.adoptions[target] = deepcopy(rebuilt)
        self.adoption_bindings[target] = digest(rebuilt)
        self.ledger = deepcopy(rows)
