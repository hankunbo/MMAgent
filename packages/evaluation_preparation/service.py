"""Source inventory and bounded owner probes, never inference or holdout release."""

import csv
import io
from pathlib import Path

from packages.dataset_adapters import FairVLMedAdapter, ODIRAdapter
from packages.dataset_adapters.prepared import capture
from packages.dataset_adapters.reading import read_source
from packages.disease_packs import DiseasePackCatalog
from packages.disease_packs.models import DatasetReference, DiseasePack
from packages.grounding.models import digest
from packages.object_store import checksum
from packages.tool_benchmark import BenchmarkReader, manifest, papila_sample

from .models import EvaluationPreparation, Registry

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "fairvlmed": "https://huggingface.co/datasets/harvardairobotics/FairVLMed",
    "odir": "https://odir2019.grand-challenge.org/dataset/",
    "papila": "https://figshare.com/articles/dataset/PAPILA/14798004",
    "refuge2": "https://refuge.grand-challenge.org/Details/",
    "fairseg": "https://github.com/Harvard-Ophthalmology-AI-Lab/FairSeg",
}
ENVIRONMENTS = {
    "fairvlmed": "MMAGENT_FAIRVLMED_ROOT",
    "odir": "MMAGENT_ODIR_ROOT",
    "papila": "MMAGENT_PAPILA_ROOT",
    "refuge2": "MMAGENT_REFUGE2_ROOT",
    "fairseg": "MMAGENT_FAIRSEG_ROOT",
    "gamma": "MMAGENT_GAMMA_ROOT",
    "grape": "MMAGENT_GRAPE_ROOT",
}
LAYOUTS = {
    "fairvlmed": {
        s: (f"Dataset/{d}/*.npz",)
        for s, d in (("training", "Training"), ("validation", "Validation"), ("test", "Test"))
    },
    "odir": {
        "training": ("ODIR-5K/ODIR-5K/Training Images/*.jpg",),
        "test": ("ODIR-5K/ODIR-5K/Testing Images/*.jpg",),
    },
    "papila": {"unspecified": ("FundusImages/*.jpg", "ExpertsSegmentations/Contours/*.txt")},
    "refuge2": {
        s: (f"{s}/images/*.jpg", f"{s}/mask/*.bmp", f"{s}/mask/*.png")
        for s in ("train", "val", "test")
    },
    "fairseg": {"training": ("Training/*.npz",), "test": ("Test/*.npz",)},
}


def inventory(root, patterns):
    """Name/size metadata only; explicitly not a checksum of all image contents."""
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError("dataset_root_unsafe_or_missing")
    rows = {}
    for split, globs in patterns.items():
        values = []
        for pattern in globs:
            for p in sorted(root.glob(pattern)):
                if p.resolve() != p or not p.is_file():
                    raise ValueError("dataset_member_unsafe")
                size = p.stat().st_size
                if size <= 0:
                    raise ValueError("dataset_member_empty")
                values.append(dict(path=str(p.relative_to(root)), byte_size=size))
                if len(values) > 25000:
                    raise ValueError("dataset_inventory_limit")
        rows[split] = values
    return rows


def split_overlap(groups):
    """Explicit group IDs only; unknown identity may never count as an independent patient."""
    if any(g is None or g == "unknown" for values in groups.values() for g in values):
        raise ValueError("patient_identity_unresolved")
    seen = set()
    for values in groups.values():
        if seen.intersection(values):
            raise ValueError("split_group_overlap")
        seen.update(values)


def adapter_for(name, root):
    directory = ROOT / "configs/disease_packs"
    catalog = DiseasePackCatalog.from_directory(directory)
    dataset = DatasetReference.model_validate_json(
        (directory / "datasets" / (name + "_1.0.0.json")).read_bytes()
    )
    pack_name = (
        "fairvlmed_glaucoma_image_note_evidence_v1"
        if name == "fairvlmed"
        else "fundus_multidisease_triage_v1"
    )
    pack = DiseasePack.model_validate_json((directory / (pack_name + "_1.0.0.json")).read_bytes())
    cls = FairVLMedAdapter if name == "fairvlmed" else ODIRAdapter
    return cls(root, catalog, dataset.reference(), pack.reference())


def probe_adapter(adapter):
    """One filename/ID-order development sample; no label or note values leave the owner."""
    selected = adapter.candidates(count=1, split="training")
    if not selected:
        raise ValueError("no_development_sample")
    sample = adapter.prepare(selected[0], split="training")
    prepared = capture(sample)
    fields = sample.fields.get("npz", sample.fields.get("source_rows", [{}])[0])
    return dict(
        status="owner_prepared_input_verified",
        selection="training_identity_order_first",
        sample_id=selected[0],
        prepared_fingerprint=prepared.fingerprint(),
        identity_basis=sample.identity_basis,
        patient_group=sample.patient_hash if sample.identity_basis != "unknown" else None,
        images=[
            dict(dimensions=list(shape), laterality=eye, modality=modality, checksum=checksum(raw))
            for raw, shape, eye, modality in sample.images
        ],
        files=[f.model_dump(mode="json") for f in sample.files],
        observed_field_names=sorted(m["key"] for m in sample.fields["npz_metadata"])
        if "npz_metadata" in sample.fields
        else sorted(fields),
        prior_exposure="development_or_unknown_not_test",
        planner_payload_created=False,
        case_bundle_written=False,
    )


def build(environ):
    policy = EvaluationPreparation.model_validate_json(
        (ROOT / "configs/evaluation/preparation_v1.json").read_bytes()
    )
    entries, layouts = [], {}
    for name, variable in ENVIRONMENTS.items():
        entry = dict(
            dataset=name,
            local_root_env=variable,
            status="unavailable",
            source_url=SOURCES.get(name),
            source_basis="existing_shared_data_spec_not_new_web_verification",
            downloaded_at=None,
            source_version=None,
            license_snapshot=None,
            allowed_use="existing_DATA_001_research_only",
            prohibited_use=["clinical_decision", "cross_dataset_patient_fabrication"],
            clinical_review_status="not_evaluated",
            official_split="unverified",
            label_scope="unverified",
            metadata_gaps=[],
            deidentification_statement="no_raw_values_exported",
        )
        declared = ROOT / "configs/disease_packs/datasets" / (name + "_1.0.0.json")
        if declared.exists():
            d = DatasetReference.model_validate_json(declared.read_bytes())
            entry.update(
                dataset_code=d.code,
                reference=d.reference().model_dump(mode="json"),
                owner=d.owner,
                modalities=list(d.modalities),
                visibility=d.visibility.model_dump(mode="json"),
                known_leakage="withheld_fields_from_original_owner",
                source_version=d.version,
            )
        if not environ.get(variable):
            entry["metadata_gaps"] = ["root_not_configured"]
            entries.append(entry)
            continue
        if name not in LAYOUTS:
            entry.update(
                status="quarantine", metadata_gaps=["new_dataset_requires_13C_adapter_audit"]
            )
            entries.append(entry)
            continue
        root = Path(environ[variable])
        layout = inventory(root, LAYOUTS[name])
        layouts[name] = layout
        entry.update(
            status="metadata_registered",
            inventory_counts={k: len(v) for k, v in layout.items()},
            empty_partitions=[k for k, v in layout.items() if not v],
            layout_checksum=digest(layout),
            root_path_checksum=digest(str(root)),
            official_split="local_directory_names_not_independence_proof",
            metadata_gaps=["download_timestamp_not_available", "local_license_snapshot_not_bound"],
        )
        names = {
            k: {Path(v["path"]).name for v in rows if Path(v["path"]).suffix in (".jpg", ".npz")}
            for k, rows in layout.items()
        }
        entry["cross_split_duplicate_names"] = {
            a + ":" + b: len(names[a] & names[b]) for a in names for b in names if a < b
        }
        entry["patient_split_verified"] = False
        if name in ("fairvlmed", "odir"):
            entry["owner_probe"] = probe_adapter(adapter_for(name, root))
            entry["label_scope"] = (
                "dataset_outcome_not_node_truth"
                if name == "fairvlmed"
                else "patient_coarse_not_eye_truth"
            )
            rel = "ReadMe/data_summary.csv" if name == "fairvlmed" else "full_df.csv"
            raw, _ = read_source(root, rel, 33554432)
            entry["csv_header"] = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
            if name == "fairvlmed":
                entry["description_differences"] = dict(
                    npz_absent=["md", "tds", "note"],
                    note_location="CSV_restricted",
                    patient_and_laterality="unknown",
                )
        elif name == "papila":
            # Reuse the exact development identity already chosen by 06E.
            old = ROOT / "docs/acceptance/06e/real-preflight-final/benchmark-manifest.json"
            import json

            original = json.loads(old.read_text())
            selected = next(s for s in original["samples"] if s["split"] == "development")
            sample = papila_sample(root, selected["sample_id"], split="development")
            reader = BenchmarkReader({"PAPILA": root}, manifest((sample,)))
            entry["owner_probe"] = dict(
                status="06E_model_input_verified_no_truth_decode",
                input=reader.model_input("PAPILA", sample.sample_id).model_dump(mode="json"),
                prior_manifest_checksum=checksum(old.read_bytes()),
                prior_exposure="development",
            )
            entry["label_scope"] = "expert_contours_per_eye"
        elif name == "refuge2":
            entry.update(
                label_scope="image_mask_patient_linkage_unknown",
                owner_probe=dict(status="not_run", reason="06E_label_definition_not_bound"),
            )
        else:
            entry.update(
                status="quarantine",
                label_scope="unverified",
                metadata_gaps=[
                    "FairSeg_not_registered_in_original_06E_reader",
                    "mask_encoding_and_patient_linkage_need_confirmation",
                ],
            )
        entries.append(entry)
    bindings = {}
    for folder in (
        "packages/dataset_adapters",
        "packages/tool_benchmark",
        "configs/disease_packs/datasets",
    ):
        for path in sorted((ROOT / folder).glob("*")):
            if path.suffix in (".py", ".json"):
                bindings[str(path.relative_to(ROOT))] = checksum(path.read_bytes())
    registry = Registry(entries=tuple(entries), source_bindings=bindings, preparation=policy)
    return registry, layouts
