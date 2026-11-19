"""Versioned demand projection into original Generic phase owners.

Existing registered questions can cover catalog intents without adopting another
interpretation or copying a score. Additional questions use original acquisition
and review projections. This module creates neither a run nor a dispatch grant.
"""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path

from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import Material

from .comprehensive_materials import ComprehensivePhase
from .dynamic_stages import rebuild
from .generic_aliases import alias_review

VERSION = "generic-capability-demand-projection/1.0.0"
CORRESPONDENCE = "configs/research/generic_requirement_correspondence_v1.json"
_selection_reads = ContextVar("generic_capability_selection_reads", default=None)


def _reader_identity(reader):
    return id(getattr(reader, "__self__", None)), id(getattr(reader, "__func__", reader))


def _selection_binding(owner):
    return (
        id(owner.campaign),
        id(owner.campaign.base),
        id(owner.medical),
        _reader_identity(owner.medical.read),
        _reader_identity(owner._read_current),
        digest(owner.own_claim),
        digest(owner.snapshot),
        digest(owner.policy),
        digest(owner.previous_sources),
    )


@contextmanager
def _current_selection_reads():
    """Share one read-only projection, then repeat its original owner proof.

    This scope must not enclose dispatch or persistence. Nested projections use
    the first current result; mutations are rejected before returning anything.
    No value survives an operation and mutable results are copied to callers.
    """
    if _selection_reads.get() is not None:
        yield
        return
    entries = {}
    token = _selection_reads.set(entries)
    try:
        yield
        _selection_reads.reset(token)
        token = None
        for owner, binding, expected in entries.values():
            if _selection_binding(owner) != binding:
                raise ValueError("generic_capability_selection_owner_changed")
            current = owner._read_current()
            if _selection_binding(owner) != binding or current != expected:
                raise ValueError("generic_capability_selection_return_changed")
    finally:
        if token is not None:
            _selection_reads.reset(token)


def correspondence(snapshot, aliased_seed, policy):
    """Exact registered scope links, never fuzzy names or per-case overrides."""
    if (
        policy.get("schema_version") != "generic-requirement-correspondence/1.0.0"
        or policy.get("scope") != "legacy_generic_source_root_v2_glaucoma_detection"
        or policy.get("canonical_selection_target") != snapshot["task_alias"]["selection_target"]
        or not policy.get("observation_coverage_not_inferred")
        or not policy.get("new_source_interpretation_not_adopted")
        or not policy.get("clinical_equivalence_not_claimed")
    ):
        raise ValueError("generic_capability_correspondence_contract")
    old = {r.id: r for r in aliased_seed.policy.requirements}
    selected = {r["id"]: r for r in snapshot["selected_requirements"]}
    mapped, seen = [], set()
    for rule in policy["rules"]:
        key, target = rule["selected"], rule["existing"]
        if key in seen or rule["relation"] != (
            "reuse_existing_question_scope_not_adopt_new_interpretation"
        ):
            raise ValueError("generic_capability_correspondence_duplicate_or_relation")
        seen.add(key)
        # A missing catalog intent does not invalidate unrelated original facts.
        if key not in selected:
            continue
        source, existing = selected[key], old.get(target)
        if (
            existing is None
            or digest(source["definition"]) != rule["selected_definition_checksum"]
            or digest(existing.definition) != rule["existing_definition_checksum"]
            or list(existing.input_modalities) != rule["existing_modalities"]
            or list(source["input_modalities"]) != rule["selected_modalities"]
        ):
            raise ValueError("generic_capability_correspondence_definition_changed")
        mapped.append(dict(rule, existing_missing_reason=existing.missing_reason))
    return mapped


def project_demands(seed, snapshot, medical, previous_sources, policy):
    """Pure projection; only GenericCapabilityPhase supplies current owner inputs."""
    if (
        seed.material.scope.case_code != snapshot["original_case"]
        or seed.material.scope.input_checksum != snapshot["original_input_checksum"]
        or seed.material.goal_code != snapshot["task_alias"]["canonical_target"]
    ):
        raise ValueError("generic_capability_projection_identity")
    aliased, _ = alias_review(seed, previous=previous_sources)
    links = correspondence(snapshot, aliased, policy)
    covered = {r["selected"] for r in links}
    raw = seed.policy.model_dump(mode="json")
    originals = {r["id"] for r in raw["requirements"]}
    if originals != set(snapshot["old_requirement_ids_preserved"]):
        raise ValueError("generic_capability_original_requirement_set")
    sources = {s.reference.code: s for s in seed.material.sources}
    current = {s.reference.code: s for s in medical.values()}
    additions, dispatch = [], []
    for row in snapshot["selected_requirements"]:
        if row["id"] in covered:
            continue
        if row["id"] in originals or row["id"] in additions:
            raise ValueError("generic_capability_requirement_collision")
        for ref in row["definition"]["source_refs"]:
            source = current.get(ref["code"])
            if source is None or source.reference.model_dump(mode="json") != ref:
                raise ValueError("generic_capability_current_source_binding")
            if source.reference.code in sources and sources[source.reference.code] != source:
                raise ValueError("generic_capability_source_collision")
            sources[source.reference.code] = source
        raw["requirements"].append(
            {
                k: row[k]
                for k in ("id", "definition", "input_modalities", "basis_kind", "missing_reason")
            }
        )
        raw["groups"].append(
            dict(
                id="capability:" + row["id"],
                slots=[[row["id"]]],
                importance=dict(slo_image_research=1.0, comprehensive_research=1.0),
            )
        )
        additions.append(row["id"])
        if row.get("acquisition_allowed") and row["acquisition_method"] == "vision":
            if row["input_modalities"] != ["SLO"]:
                raise ValueError("generic_capability_original_vision_modality")
            dispatch.append(row["id"])
        elif row.get("acquisition_allowed"):
            raise ValueError("generic_capability_new_method_requires_original_owner")
    raw["version"] = "generic-capability-demand-1.0.0"
    material = seed.material.model_dump(mode="json")
    material["sources"] = [s.model_dump(mode="json") for s in sources.values()]
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    sourced = seed.model_copy(update=dict(material=validate(Material, material)))
    projected = rebuild(sourced, validate(type(seed.policy), raw), seed.material.observations)
    manifest = dict(
        schema_version=VERSION,
        case_code=seed.material.scope.case_code,
        target=seed.material.goal_code,
        run_id=seed.material.scope.run_id,
        selection_checksum=digest(snapshot),
        correspondence_checksum=digest(policy),
        original_state_checksum=digest(seed),
        projected_seed_checksum=digest(projected),
        retained_requirement_ids=sorted(originals),
        added_requirement_ids=additions,
        covered_by_existing_scope=links,
        linked_missing_requirement_ids=sorted(
            {r["existing"] for r in links if r["existing_missing_reason"] is not None}
        ),
        scope_links_are_observation_completion=False,
        dispatch_targets=dispatch,
        requirements_before=len(originals),
        requirements_after=len(projected.policy.requirements),
        original_items_checksum=digest(seed.policy.items),
        original_observations_checksum=digest(seed.material.observations),
        original_groups_checksum=digest(seed.policy.groups),
        new_observations=0,
        new_reviews=0,
        historical_scores_relabelled=False,
        independent_evidence_count_claimed=False,
    )
    return projected, json.loads(canonical(manifest))


class GenericCapabilityOwner:
    """Current original case, source and selection readers; no caller-made grant."""

    def __init__(self, campaign, *, own_claim=None):
        from packages.multimodal_research.material import MedicalSources

        from .generic_campaign import GenericCampaign
        from .generic_capability_selection import inspect
        from .generic_pending_review import original_phase

        if type(campaign) is not GenericCampaign:
            raise ValueError("generic_capability_original_campaign_owner_required")
        self.campaign = campaign
        self.own_claim = own_claim
        self.snapshot = inspect(campaign, own_claim=own_claim)
        _, old, _, _ = original_phase(campaign)
        self.previous_sources = old.material.sources
        self.medical = MedicalSources()
        self.policy = json.loads(Path(CORRESPONDENCE).read_bytes())

    def current(self):
        entries = _selection_reads.get()
        if entries is None or (id(self) not in entries and len(entries) >= 16):
            return self._read_current()
        binding = _selection_binding(self)
        if id(self) not in entries:
            value = self._read_current()
            if _selection_binding(self) != binding:
                raise ValueError("generic_capability_selection_owner_changed")
            entries[id(self)] = self, binding, deepcopy(value)
        owner, expected, value = entries[id(self)]
        if owner is not self or binding != expected:
            raise ValueError("generic_capability_selection_owner_changed")
        return deepcopy(value)

    def _read_current(self):
        from .generic_capability_selection import inspect

        if (
            inspect(self.campaign, own_claim=self.own_claim) != self.snapshot
            or json.loads(Path(CORRESPONDENCE).read_bytes()) != self.policy
        ):
            raise ValueError("generic_capability_current_selection_changed")
        return self.snapshot, self.medical.read()

    def project(self, seed):
        snapshot, medical = self.current()
        return project_demands(seed, snapshot, medical, self.previous_sources, self.policy)


class GenericCapabilityPhase(ComprehensivePhase):
    """Reuse original partial Vision and Review with a current demand revision."""

    def __init__(self, base, *, selection_owner, new_items_only=False, **kwargs):
        if (
            type(selection_owner) is not GenericCapabilityOwner
            or selection_owner.campaign.base is not base
        ):
            raise ValueError("generic_capability_phase_owner_binding")
        super().__init__(base, **kwargs)
        self.selection_owner = selection_owner
        if type(new_items_only) is not bool:
            raise ValueError("generic_capability_review_selection_contract")
        self.new_items_only = new_items_only

    def project(self, *, run_id, config):
        from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

        with historical_fact_read_scope(), _current_selection_reads():
            return self._project(run_id=run_id, config=config)

    def _project(self, *, run_id, config):
        self.original_seed, manifest = self.base.project(run_id=run_id, config=config)
        self.seed, self.capability_projection = self.selection_owner.project(self.original_seed)
        self.initial = self.project_current()
        self.capability_projection["phase_input_checksum"] = digest(self.initial)
        return self.initial, manifest

    def project_current(self):
        state = super().project_current()
        if self.target is not None:
            if self.target not in self.capability_projection["dispatch_targets"]:
                raise ValueError("generic_capability_target_not_selected")
            return state
        state, self.source_alias_manifest = alias_review(
            state, previous=self.selection_owner.previous_sources
        )
        if self.new_items_only:
            from .generic_pending_review import selected_state

            selected = set(self.capability_projection["dispatch_targets"]) & {
                item.id for item in state.policy.items
            }
            state = (
                selected_state(state, selected)
                if selected
                else rebuild(
                    state,
                    state.policy.model_copy(update=dict(items=())),
                    state.material.observations,
                )
            )
        return state

    def verify(self, state):
        from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

        with historical_fact_read_scope(), _current_selection_reads():
            self.base.verify(self.original_seed)
            seed, manifest = self.selection_owner.project(self.original_seed)
            if seed != self.seed or any(
                self.capability_projection.get(k) != v for k, v in manifest.items()
            ):
                raise ValueError("generic_capability_projection_changed")
            expected = self.project_current()
            if (
                expected != self.initial
                or state.policy != self.initial.policy
                or (state.material.sources != self.initial.material.sources)
            ):
                raise ValueError("generic_capability_phase_dependency_changed")
            scope = state.material.scope
            original = self.initial.material.scope
            for field in ("case_code", "run_id", "input_checksum", "visible_checksum"):
                if getattr(scope, field) != getattr(original, field):
                    raise ValueError("generic_capability_phase_scope")


def projection_record(evidence, state, run_id):
    """Persist with original phase input after current recheck, before dispatch."""
    if type(evidence) is not GenericCapabilityPhase:
        raise ValueError("generic_capability_phase_producer_required")
    evidence.verify(state)
    value = evidence.capability_projection
    if value["run_id"] != run_id or value["phase_input_checksum"] != digest(state):
        raise ValueError("generic_capability_phase_record_binding")
    return json.loads(canonical(value))
