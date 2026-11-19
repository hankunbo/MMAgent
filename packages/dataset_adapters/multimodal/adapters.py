"""Dataset-specific decoding into one component contract, without clinical inference."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from packages.dataset_adapters.reading import AdapterError
from packages.disease_packs.models import seal
from packages.object_store import checksum
from packages.tool_benchmark.formats import jpeg_dimensions

from .formats import npz, sheets, source
from .models import CaseComponents, Component, Payload


@dataclass(frozen=True, repr=False)
class Prepared:
    root: str
    manifest: CaseComponents
    payload_bytes: tuple[tuple[str, bytes], ...]

    def recheck(self):
        value = prepare(
            self.manifest.dataset,
            self.root,
            self.manifest.sample,
            selection=self.manifest.selection,
            cutoff_visit=self.manifest.cutoff_visit,
            semantic_profile=(
                "harvard-gdp-v1"
                if self.manifest.adapter_version == "multimodal-adapters/1.1.0"
                else "grape-v1"
                if self.manifest.adapter_version == "multimodal-adapters/1.2.0"
                else None
            ),
        )
        if value != self:
            raise AdapterError("multimodal_preparation_changed")
        return value


class Builder:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.files = {}
        self.payloads = []
        self.contents = []
        self.components = []
        self.quarantine = set()

    def read(self, path):
        raw, binding = source(self.root, path)
        old = self.files.setdefault(path, binding)
        if old != binding:
            raise AdapterError("multimodal_source_changed")
        if sum(s.byte_size for s in self.files.values()) > 268435456:
            raise AdapterError("multimodal_case_size_limit")
        return raw

    def payload(self, id, raw, path, locator, media, projection):
        self.payloads.append(
            Payload(
                id=id,
                checksum=checksum(raw),
                byte_size=len(raw),
                media_type=media,
                source_path=path,
                source_locator=locator,
                projection=projection,
            )
        )
        self.contents.append((id, raw))
        return id

    def missing(self, id, kind, concept, modality, locator, *, reason="not_provided", **kw):
        self.components.append(
            Component(
                id=id,
                kind=kind,
                concept=concept,
                modality=modality,
                status="missing",
                reason=reason,
                source_locator=locator,
                **kw,
            )
        )

    def scalar(self, id, concept, value, path, locator, *, unit=None, basis=None, **kw):
        try:
            parsed = float(value) if value not in ("", "/", None) else None
        except (ValueError, TypeError):
            parsed = None
        valid = parsed is not None and math.isfinite(parsed)
        self.components.append(
            Component(
                id=id,
                kind="scalar",
                concept=concept,
                modality="structured_measurement",
                status="obtained" if valid else "missing",
                value=parsed if valid else None,
                unit=unit,
                unit_basis=basis,
                source_paths=(path,),
                source_locator=locator,
                reason=None if valid else "missing_or_nonfinite_measurement",
                axis_semantics="not_applicable",
                **kw,
            )
        )

    def image(self, id, path, *, modality="color_fundus", **kw):
        raw = self.read(path)
        shape = jpeg_dimensions(raw)
        payload = self.payload(id, raw, path, path, "image/jpeg", "source_image")
        self.components.append(
            Component(
                id=id,
                kind="image",
                concept=modality,
                modality=modality,
                status="obtained",
                shape=shape,
                dtype="uint8_encoded_jpeg",
                payload_ids=(payload,),
                source_paths=(path,),
                source_locator=path,
                semantic_limits=("jpeg_framing_checked_not_full_decode",),
                **kw,
            )
        )

    def finish(self, **kwargs):
        manifest = seal(
            CaseComponents,
            sources=tuple(self.files.values()),
            payloads=tuple(self.payloads),
            components=tuple(self.components),
            quarantined_fields=tuple(sorted(self.quarantine)),
            **kwargs,
        )
        return Prepared(str(self.root), manifest, tuple(self.contents))


def harvard(root, sample, *, semantic_profile=None):
    from .harvard_semantics import ADAPTER, PROFILE, fields

    if semantic_profile not in {None, PROFILE}:
        raise AdapterError("multimodal_semantic_profile")
    if not re.fullmatch(r"data_[0-9]{4}\.npz", sample):
        raise AdapterError("multimodal_sample_selector")
    b = Builder(root)
    policy = {"RNFLT": {"rnflt", "md", "tds", "age"}, "Bscan": {"bscans"}}
    for folder, allowed in policy.items():
        path = f"{folder}/{sample}"
        if not (b.root / path).exists():
            for key in sorted(allowed):
                kind = {
                    "rnflt": "map",
                    "bscans": "volume",
                    "tds": "vector",
                    "md": "scalar",
                    "age": "scalar",
                }[key]
                modality = {"rnflt": "OCT_RNFLT", "bscans": "OCT", "tds": "VF"}.get(
                    key, "structured_measurement"
                )
                b.missing(folder.lower() + ":" + key, kind, key, modality, path)
            continue
        arrays, names = npz(b.read(path), allowed)
        b.quarantine.update(set(names) - allowed - {"glaucoma", "progression"})
        for key in sorted(allowed):
            id = folder.lower() + ":" + key
            locator = path + "#" + key
            if key in ("age", "md"):
                value = arrays[key].scalar() if key in arrays else None
                qualified = fields(key) if semantic_profile else {}
                qualified.pop("axis_semantics", None)
                if "unit_basis" in qualified:
                    qualified["basis"] = qualified.pop("unit_basis")
                b.scalar(id, key, value, path, locator, **qualified)
                continue
            kind = {"rnflt": "map", "bscans": "volume", "tds": "vector"}[key]
            modality = {"rnflt": "OCT_RNFLT", "bscans": "OCT", "tds": "VF"}[key]
            if key not in arrays or not arrays[key].data:
                b.missing(id, kind, key, modality, locator, source_paths=(path,))
                continue
            a = arrays[key]
            expected = {"rnflt": (225, 225), "bscans": (200, 200, 300), "tds": (52,)}[key]
            if a.shape != expected or (key == "bscans" and a.dtype != "|u1"):
                raise AdapterError("multimodal_harvard_array_layout")
            nonfinite = key != "bscans" and any(not math.isfinite(v) for v in a.numbers())
            pid = b.payload(id, a.raw, path, locator, "application/x-npy", "selected_observation")
            b.components.append(
                Component(
                    id=id,
                    kind=kind,
                    concept=key,
                    modality=modality,
                    status="partial" if nonfinite else "obtained",
                    reason="nonfinite_values_preserved" if nonfinite else None,
                    shape=a.shape,
                    dtype=a.dtype,
                    payload_ids=(pid,),
                    source_paths=(path,),
                    source_locator=locator,
                    **(
                        fields(key)
                        if semantic_profile
                        else dict(
                            semantic_limits=("axes_units_and_missing_codes_require_qualification",)
                        )
                    ),
                )
            )
    if not b.files:
        raise AdapterError("multimodal_selected_case_missing")
    return b.finish(
        dataset="harvard_gdp",
        adapter_version=ADAPTER if semantic_profile else "multimodal-adapters/1.0.0",
        sample=sample,
        selection="development_preparation",
        task="glaucoma_detection",
        release_basis="harvardairobotics/Harvard-GDP@5358429aad56248d664e7d00c1f8a6a2e01dc655:local_snapshot",
        excluded_fields=("glaucoma", "progression"),
        limitations=(
            "sample_pairing_is_not_verified_patient_eye_visit_identity",
            "VF_label_derivation_audit_required_before_prediction",
            "raw_observation_available_does_not_mean_medically_qualified",
            "noncommercial_research_not_clinical_use",
        ),
    )


def gamma(root, sample):
    if not re.fullmatch(r"[0-9]{4}", sample):
        raise AdapterError("multimodal_sample_selector")
    b = Builder(root)
    prefix = f"training/multi-modality_images/{sample}"
    directory = b.root / prefix
    if not directory.is_dir() or directory.is_symlink():
        raise AdapterError("multimodal_selected_case_missing")
    photo = prefix + f"/{sample}.jpg"
    if (b.root / photo).exists():
        b.image("fundus", photo)
    else:
        b.missing("fundus", "image", "color_fundus", "color_fundus", photo)
    series = directory / sample
    entries = {}
    if series.is_symlink():
        raise AdapterError("multimodal_series_symlink")
    if series.exists():
        for p in series.iterdir():
            match = re.fullmatch(r"([0-9]+)_image\.jpg", p.name)
            if not match:
                b.quarantine.add("oct_unrecognized_member")
                continue
            index = int(match[1])
            if not 0 <= index < 256 or index in entries:
                raise AdapterError("multimodal_slice_index")
            entries[index] = p.relative_to(b.root).as_posix()
    payloads, paths, shapes = [], [], set()
    for index, path in sorted(entries.items()):
        raw = b.read(path)
        shapes.add(jpeg_dimensions(raw))
        payloads.append(b.payload(f"oct:{index}", raw, path, path, "image/jpeg", "source_image"))
        paths.append(path)
    if len(shapes) > 1:
        raise AdapterError("multimodal_slice_shape_mismatch")
    if not payloads:
        b.missing("oct", "image_series", "oct_bscans", "OCT", prefix + "/" + sample)
    else:
        complete = len(payloads) == 256
        b.components.append(
            Component(
                id="oct",
                kind="image_series",
                concept="oct_bscans",
                modality="OCT",
                status="obtained" if complete else "partial",
                reason=None if complete else "source_slices_missing",
                shape=(len(payloads), *next(iter(shapes))),
                payload_ids=tuple(payloads),
                source_paths=tuple(paths),
                source_locator=prefix + "/" + sample,
                axis_semantics="source_index_only",
                expected_members=256,
                member_indices=tuple(sorted(entries)),
                semantic_limits=("physical_spacing_and_scan_orientation_unqualified",),
            )
        )
    if not b.files:
        raise AdapterError("multimodal_selected_case_missing")
    return b.finish(
        dataset="gamma",
        sample=sample,
        selection="training",
        task="glaucoma_grading",
        release_basis="GAMMA/task1_gamma_grading:local_snapshot",
        excluded_fields=("non", "early", "mid_advanced", "evaluation_annotations"),
        limitations=(
            "local_training_directory_is_not_patient_identity",
            "grading_target_not_binary_score_thresholds",
            "reference_workbook_not_read_by_visible_adapter",
            "test_not_authorized_and_test_reference_not_locally_available",
        ),
    )


def column(n):
    value = ""
    while n:
        n, digit = divmod(n - 1, 26)
        value = chr(65 + digit) + value
    return value


def grape(root, sample, cutoff_visit, *, semantic_profile=None):
    from .grape_semantics import ADAPTER, PROFILE, fields

    if semantic_profile not in {None, PROFILE}:
        raise AdapterError("multimodal_semantic_profile")
    match = re.fullmatch(r"([1-9][0-9]{0,4}):(OD|OS)", sample)
    if not match or type(cutoff_visit) is not int or not 0 <= cutoff_visit <= 1000:
        raise AdapterError("multimodal_grape_selector_or_cutoff")
    subject, eye = match.groups()
    b = Builder(root)
    path = "VF_and_clinical_information.xlsx"
    tables = dict(sheets(b.read(path)))
    if set(tables) != {"Baseline", "Follow-up"}:
        raise AdapterError("multimodal_grape_sheets")
    selected, seen = [], set()
    for name, rows in tables.items():
        headers = {
            "A": "Subject Number",
            "B": "Laterality",
            "E": "IOP",
            **(
                {
                    "C": "Age",
                    "F": "CCT",
                    "L": "OCT RNFL thickness",
                    "Q": "Corresponding CFP",
                    "T": "VF",
                }
                if name == "Baseline"
                else {
                    "C": "Visit Number",
                    "D": "Interval Years",
                    "F": "Corresponding CFP",
                    "I": "VF",
                }
            ),
        }
        indexed = dict(rows)
        start = 20 if name == "Baseline" else 9
        if (
            not {1, 2} <= indexed.keys()
            or any(indexed[1].get(k) != v for k, v in headers.items())
            or any(indexed[2].get(column(start + i)) != str(i) for i in range(61))
            or (
                name == "Baseline"
                and any(
                    indexed[2].get(k) != v
                    for k, v in {"L": "Mean", "M": "S", "N": "N", "O": "I", "P": "T"}.items()
                )
            )
        ):
            raise AdapterError("multimodal_grape_headers")
        for number, cells in rows:
            if number <= 2 or cells.get("A") != subject or cells.get("B") != eye:
                continue
            visit = 0 if name == "Baseline" else int(cells["C"])
            if visit < 0 or (name == "Follow-up" and visit == 0) or visit in seen:
                raise AdapterError("multimodal_grape_duplicate_visit")
            seen.add(visit)
            if visit > cutoff_visit:
                continue
            selected.append((visit, name, number, cells))
    if not selected or 0 not in seen:
        raise AdapterError("multimodal_selected_case_missing")
    previous_interval = -1.0
    for visit, name, number, cells in sorted(selected):
        interval = 0.0 if visit == 0 else float(cells["D"])
        if not math.isfinite(interval) or interval < previous_interval:
            raise AdapterError("multimodal_grape_interval_order")
        previous_interval = interval
        common = dict(
            laterality="RIGHT" if eye == "OD" else "LEFT",
            visit=str(visit),
            interval_years=interval,
        )
        prefix = f"visit:{visit}:"
        cell_ref = path + f"#{name}!"
        # Explicit columns; diagnosis, total future visits and progression outcomes excluded.
        scalar_columns = (
            {"E": "iop"}
            if visit
            else {
                "C": "age",
                "E": "iop",
                "F": "cct",
                "L": "rnfl_mean",
                "M": "rnfl_superior",
                "N": "rnfl_nasal",
                "O": "rnfl_inferior",
                "P": "rnfl_temporal",
            }
        )
        for col, concept in scalar_columns.items():
            qualified = (
                fields(concept)
                if semantic_profile
                else dict(semantic_limits=("external_measurement_units_and_method_unqualified",))
            )
            basis = qualified.pop("unit_basis", None)
            b.scalar(
                prefix + concept,
                concept,
                cells.get(col),
                path,
                cell_ref + col + str(number),
                basis=basis,
                **qualified,
                **common,
            )
        start = 20 if visit == 0 else 9
        values = []
        for i in range(61):
            text = cells.get(column(start + i), "")
            try:
                value = float(text)
                value = value if math.isfinite(value) else None
            except (ValueError, TypeError):
                value = None
            values.append(value)
        raw = json.dumps(
            {"source_order": list(range(61)), "values": values}, allow_nan=False
        ).encode()
        complete = all(v is not None for v in values)
        if all(v is None for v in values):
            b.missing(
                prefix + "vf",
                "vector",
                "vf",
                "VF",
                cell_ref + str(number),
                **(fields("vf") if semantic_profile else {}),
                **common,
            )
        else:
            pid = b.payload(
                prefix + "vf",
                raw,
                path,
                cell_ref + str(number),
                "application/json",
                "visible_cells",
            )
            b.components.append(
                Component(
                    id=prefix + "vf",
                    kind="vector",
                    concept="vf",
                    modality="VF",
                    status="obtained" if complete else "partial",
                    reason=None if complete else "missing_or_nonfinite_points",
                    payload_ids=(pid,),
                    source_paths=(path,),
                    source_locator=cell_ref + str(number),
                    shape=(61,),
                    **(
                        fields("vf")
                        if semantic_profile
                        else dict(
                            axis_semantics="source_index_only",
                            semantic_limits=("test_pattern_point_mapping_and_units_unqualified",),
                        )
                    ),
                    **common,
                )
            )
        image_name = cells.get("Q" if visit == 0 else "F", "")
        if image_name in ("", "/"):
            b.missing(
                prefix + "fundus",
                "image",
                "color_fundus",
                "color_fundus",
                cell_ref + str(number),
                **common,
            )
        else:
            if not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.jpg)?", image_name):
                raise AdapterError("multimodal_grape_image_reference")
            image_name = image_name if image_name.endswith(".jpg") else image_name + ".jpg"
            image_path = "extracted/CFPs/" + image_name
            if not (b.root / image_path).exists():
                b.missing(
                    prefix + "fundus",
                    "image",
                    "color_fundus",
                    "color_fundus",
                    image_path,
                    reason="referenced_image_missing",
                    **common,
                )
            else:
                b.image(prefix + "fundus", image_path, **common)
    return b.finish(
        dataset="grape",
        adapter_version=ADAPTER if semantic_profile else "multimodal-adapters/1.0.0",
        sample=sample,
        selection="development_preparation",
        task="observed_change",
        cutoff_visit=cutoff_visit,
        release_basis="GRAPE/figshare-23674350:local_snapshot",
        patient_group="sha256:" + checksum(("GRAPE:subject:" + subject).encode())[7:],
        patient_identity_basis="explicit_dataset_id",
        excluded_fields=(
            "Progression Status",
            "Category of Glaucoma",
            "Total Visits",
            "Annotated Images",
            "json",
            "visits_after_cutoff",
        ),
        limitations=(
            "observed_change_is_not_future_progression_prediction",
            "two_eyes_share_patient_group",
            "clinical_rnfl_values_are_not_raw_OCT_volumes",
            "numeric_units_require_release_semantic_qualification",
        ),
    )


def prepare(dataset, root, sample, *, selection=None, cutoff_visit=None, semantic_profile=None):
    expected = {
        "harvard_gdp": "development_preparation",
        "gamma": "training",
        "grape": "development_preparation",
        "fairvlmed": "training",
    }
    if dataset not in expected or selection not in (None, expected[dataset]):
        raise AdapterError("multimodal_dataset_or_split_not_allowed")
    if dataset != "grape" and cutoff_visit is not None:
        raise AdapterError("multimodal_unexpected_cutoff")
    if semantic_profile and dataset not in {"harvard_gdp", "grape"}:
        raise AdapterError("multimodal_semantic_profile_dataset")
    if dataset == "harvard_gdp":
        return harvard(root, sample, semantic_profile=semantic_profile)
    if dataset == "gamma":
        return gamma(root, sample)
    if dataset == "fairvlmed":
        from .fairvlmed import prepare as selected_fairvlmed

        return selected_fairvlmed(root, sample)
    return grape(root, sample, cutoff_visit, semantic_profile=semantic_profile)
