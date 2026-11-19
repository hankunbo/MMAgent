"""Selected training inputs, optional demographics/labels, original safe readers.

The official complete-dataset adapter stays unchanged. This versioned profile
retains source-row and image bindings without requiring unrelated images locally.
Unknown fields are quarantined, not automatically made visible to a model.
"""

from .adapters import FAIR_FIELDS, BaseAdapter, PreparedSample
from .reading import AdapterError, basename, csv_rows, grayscale_image, read_npz, read_source

SELECTION = "training_selected_input/1.0.0"


class FairVLMedRuntimeAdapter(BaseAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.dataset.code != "SOURCE:DATASET:FAIRVLMED":
            raise AdapterError("dataset_mismatch")

    def _index(self):
        rows, binding = csv_rows(
            self.root,
            "ReadMe/data_summary.csv",
            FAIR_FIELDS,
            required_fields=("filename", "use"),
        )
        seen, selected = set(), {}
        for row in rows:
            name = basename(row["filename"], ".npz")
            if name in seen:
                raise AdapterError("csv_duplicate_filename")
            seen.add(name)
            if row["use"] not in ("training", "validation", "test"):
                raise AdapterError("csv_split_invalid")
            if row["use"] == "training":
                selected[name] = row
        return selected, binding

    def candidates(self, *, count=20, split="training"):
        self.split(split)
        if type(count) is not int or not 1 <= count <= 50:
            raise AdapterError("candidate_count_invalid")
        rows, _ = self._index()
        return tuple(sorted(rows)[:count])

    def prepare(self, filename, *, split="training"):
        self.split(split)
        basename(filename, ".npz")
        rows, csv_binding = self._index()
        if filename not in rows:
            raise AdapterError("training_sample_not_found")
        row = rows[filename]
        raw, binding = read_source(self.root, "Dataset/Training/" + filename)
        arrays, metadata = read_npz(raw, runtime_profile=True)
        shape, pixels = arrays.pop("slo_fundus")
        return PreparedSample(
            self.dataset,
            self.pack,
            (csv_binding, binding),
            ((grayscale_image(shape, pixels), shape, "UNKNOWN", "SLO"),),
            dict(csv=row, npz=arrays, npz_metadata=metadata, selection_version=SELECTION),
            row.get("note"),
            source_images=(raw,),
        )


def factory(name, root):
    """Explicit registration; an unknown dataset never falls back to another reader."""
    from pathlib import Path

    from packages.disease_packs import DiseasePackCatalog
    from packages.disease_packs.models import DatasetReference, DiseasePack

    if name != "fairvlmed":
        raise AdapterError("runtime_dataset_not_registered")
    directory = Path(__file__).resolve().parents[2] / "configs/disease_packs"
    catalog = DiseasePackCatalog.from_directory(directory)
    dataset = DatasetReference.model_validate_json(
        (directory / "datasets/fairvlmed_1.0.0.json").read_bytes()
    )
    pack = DiseasePack.model_validate_json(
        (directory / "fairvlmed_glaucoma_image_note_evidence_v1_1.0.0.json").read_bytes()
    )
    return FairVLMedRuntimeAdapter(root, catalog, dataset.reference(), pack.reference())
