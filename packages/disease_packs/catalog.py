"""Read exact declaration versions. Never enumerate or open dataset contents."""

from pathlib import Path

from packages.contracts import CaseBundle, VersionedReference

from .models import DatasetReference, DiseasePack, parse_pack


class DiseasePackCatalog:
    def __init__(self, packs: tuple[DiseasePack, ...], datasets: tuple[DatasetReference, ...]):
        self._packs = {}
        self._datasets = {}
        for values, model, destination in (
            (packs, DiseasePack, self._packs),
            (datasets, DatasetReference, self._datasets),
        ):
            for value in values:
                item = parse_pack(value) if model is DiseasePack else model.model_validate(value)
                ref = item.reference()
                key = (ref.code, ref.version)
                if key in destination:
                    raise ValueError("duplicate immutable declaration version")
                destination[key] = item.model_dump_json()
        for pack in packs:
            for ref in pack.definition.dataset_refs:
                dataset = self.dataset(ref)
                if dataset.synthetic != pack.definition.synthetic:
                    raise ValueError("synthetic pack must use synthetic DatasetReference only")

    @classmethod
    def from_directory(cls, directory: Path):
        # Only the specified configuration directory, never any DatasetReference.local_root.
        return cls(
            tuple(parse_pack(p.read_bytes()) for p in sorted(directory.glob("*.json"))),
            tuple(
                DatasetReference.model_validate_json(p.read_bytes())
                for p in sorted((directory / "datasets").glob("*.json"))
            ),
        )

    def _exact(self, reference, entries, model):
        ref = VersionedReference.model_validate(reference.model_dump(mode="python"))
        raw = entries[(ref.code, ref.version)]
        item = parse_pack(raw) if model is DiseasePack else model.model_validate_json(raw)
        if item.reference() != ref:
            raise ValueError("exact declaration checksum required; no latest fallback")
        return item

    def pack(self, reference):
        return self._exact(reference, self._packs, DiseasePack)

    def dataset(self, reference):
        return self._exact(reference, self._datasets, DatasetReference)

    def validate_planner_fields(self, dataset_ref, field_names, *, reviewed_derived_fields=()):
        """Validate a field contract, not case values; 07A owns redaction and its provenance."""
        policy = self.dataset(dataset_ref).visibility
        fields = set(field_names)
        if not fields <= set(policy.planner_visible):
            raise ValueError("hidden, owner-only or unknown Planner field")
        if not (fields & set(policy.derived_requires_review)) <= set(reviewed_derived_fields):
            raise ValueError("07A reviewed derivation required; raw notes are never redacted here")
        return tuple(sorted(fields))

    def validate_patient_relation(self, left_ref, right_ref):
        left, right = self.dataset(left_ref), self.dataset(right_ref)
        if left.code != right.code or "no_patient_linkage" in (
            left.patient_linkage,
            right.patient_linkage,
        ):
            raise ValueError("cross-dataset or tool-truth patient relationship forbidden")
        return "declaration_allows_within_dataset_only;07A_must_verify_actual_alignment"


def validate_conclusion(pack, conclusion):
    pack = parse_pack(pack)
    if type(conclusion) is not str or conclusion not in pack.definition.allowed_conclusions:
        raise ValueError("conclusion outside pack evidence-planning scope")
    return conclusion


def validate_prepared_case_contract(pack, case):
    """Check an upstream prepared v1 contract, without redacting or constructing a case."""
    pack = parse_pack(pack)
    case = CaseBundle.model_validate(case.model_dump(mode="python"))
    if case.report_uri is not None or case.extensions:
        raise ValueError("raw report or unregistered case extension is not Planner-visible")
    if any(goal.goal_code != pack.definition.target_node_code for goal in case.goals):
        raise ValueError("case requests a goal outside the pack")
    slots = pack.definition.evidence_slots
    observations = {slot.observation_code for slot in slots}
    artifacts = {
        code for slot in slots for code in (slot.input_artifact_code, slot.output_artifact_code)
    }
    if any(
        obs.concept_code not in observations or isinstance(obs.value, (dict, list))
        for obs in case.observations
    ):
        raise ValueError("unregistered or nested observation is not Planner-visible")
    if any(
        artifact.artifact_type_code not in artifacts or artifact.extensions
        for artifact in case.artifacts
    ):
        raise ValueError("unregistered artifact or extension is not Planner-visible")
    return case
