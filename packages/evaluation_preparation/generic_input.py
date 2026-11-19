"""FairVLMed selection/configuration; no patient-specific source code or secret input."""

import json
import re
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.dataset_adapters.prepared import capture
from packages.dataset_adapters.reading import NPZ_KEYS
from packages.dataset_adapters.runtime_input import factory
from packages.grounding.models import digest
from packages.object_store import checksum

RUN_ROOT = Path(".local/fairvlmed-runs")
LEGACY_PROVIDER = "configs/agents/vectorengine_full_evidence_v1.json"
PROVIDER = "configs/agents/vectorengine_generic_evidence_v1.json"
RELAY_PROFILE = "relay-grounded-v1"
RELAY_PROVIDER = "configs/agents/fairvlmed_relay_grounded_v1.json"
SHAPE_PROFILE = "relay-grounded-shape-v1"
SHAPE_PROVIDER = "configs/agents/fairvlmed_relay_review_shape_v1.json"
ROUTED_PROFILE = "relay-grounded-shape-proxy-v1"
ROUTED_PROVIDER = "configs/agents/fairvlmed_relay_review_shape_proxy_v1.json"
ALTERNATE_PROFILE = "relay-grounded-shape-proxy-alternate-v1"
ALTERNATE_SELECTION = "configs/agents/relay_existing_model_selection_v1.json"
LOCAL_PROFILE = "relay-local-gpt6-shape-v1"
LOCAL_PROVIDER = "configs/agents/fairvlmed_relay_local_gpt6_review_v1.json"
ENVIRONMENT_PROFILE = "relay-environment-gpt6-shape-v1"
ENVIRONMENT_PROVIDER = "configs/agents/fairvlmed_relay_environment_gpt6_review_v1.json"
BOUND_ROUTE_VERSIONS = {"fairvlmed-run-input/1.10.0", "fairvlmed-run-input/1.11.0"}
PROFILE_INPUTS = {
    "fairvlmed-run-input/1.5.0": (RELAY_PROFILE, RELAY_PROVIDER),
    "fairvlmed-run-input/1.6.0": (SHAPE_PROFILE, SHAPE_PROVIDER),
    "fairvlmed-run-input/1.7.0": (ROUTED_PROFILE, ROUTED_PROVIDER),
    "fairvlmed-run-input/1.8.0": (ALTERNATE_PROFILE, ROUTED_PROVIDER),
    "fairvlmed-run-input/1.9.0": (ALTERNATE_PROFILE, ROUTED_PROVIDER),
    "fairvlmed-run-input/1.10.0": (LOCAL_PROFILE, LOCAL_PROVIDER),
    "fairvlmed-run-input/1.11.0": (ENVIRONMENT_PROFILE, ENVIRONMENT_PROVIDER),
}


def provider_config(spec):
    """Explicit public route for a new run; never reinterpret old provider bindings."""
    from packages.llm_provider.config import prompt, read_provider_config

    path = spec["provider_configuration"]
    alternate = spec["schema_version"] in {"fairvlmed-run-input/1.8.0", "fairvlmed-run-input/1.9.0"}
    if alternate != ("model_selection" in spec):
        raise ValueError("generic_model_selection_version")
    if spec["schema_version"] in PROFILE_INPUTS:
        profile, expected_path = PROFILE_INPUTS[spec["schema_version"]]
        if spec.get("provider_profile") != profile or path != expected_path:
            raise ValueError("generic_provider_profile_binding")
        config = read_provider_config(json.loads(Path(path).read_bytes()))
        if alternate:
            from packages.llm_provider.model_selection import apply_selection, load_selection

            if spec["model_selection"] != load_selection(ALTERNATE_SELECTION):
                raise ValueError("generic_model_selection_binding")
            config = apply_selection(config, spec["model_selection"])
        if digest(config) != spec.get("provider_configuration_checksum"):
            raise ValueError("generic_provider_configuration_changed")
        if spec["schema_version"] in BOUND_ROUTE_VERSIONS:
            # Bind the previously verified config, not an arbitrary caller override.
            from packages.llm_provider.model_selection import apply_selection, load_selection
            from packages.llm_provider.network_route import load_route

            original = apply_selection(
                read_provider_config(json.loads(Path(ROUTED_PROVIDER).read_bytes())),
                load_selection(ALTERNATE_SELECTION),
            ).model_dump(mode="json")
            original["transport"]["max_retries"] = 0
            original["network_route"] = load_route(
                "configs/agents/server_local_https_route_v1.json"
                if spec["schema_version"] == "fairvlmed-run-input/1.10.0"
                else "configs/agents/environment_https_route_v1.json"
            ).model_dump(mode="json")
            if config != read_provider_config(original):
                raise ValueError("generic_local_provider_delta")
        prompt(config, "review")
        if spec["schema_version"] == "fairvlmed-run-input/1.6.0":
            from .generic_review import review_configuration

            original = read_provider_config(json.loads(Path(RELAY_PROVIDER).read_bytes()))
            if config != review_configuration(original, SHAPE_PROFILE):
                raise ValueError("generic_provider_shape_delta")
        if spec["schema_version"] in {
            "fairvlmed-run-input/1.7.0",
            "fairvlmed-run-input/1.8.0",
            "fairvlmed-run-input/1.9.0",
        }:
            from packages.llm_provider.config import with_network_route

            original = read_provider_config(json.loads(Path(SHAPE_PROVIDER).read_bytes()))
            expected = with_network_route(original, config.network_route)
            if alternate:
                expected = apply_selection(expected, spec["model_selection"])
            if config != expected:
                raise ValueError("generic_provider_route_delta")
        return config
    return read_provider_config(json.loads(Path(path).read_bytes()))


def directory(key, *, root=RUN_ROOT):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", key):
        raise ValueError("generic_run_key_invalid")
    base = Path(root).resolve()
    result = base / key
    if result.is_symlink() or result.resolve().parent != base:
        raise ValueError("generic_run_path_invalid")
    return result


def inspect_input(dataset, root, sample, *, split="training"):
    snap = capture(factory(dataset, root).prepare(sample, split=split))
    metadata = {v["key"]: v for v in snap.fields["npz_metadata"]}
    fields = {}
    for name in sorted(NPZ_KEYS - {"slo_fundus"}):
        fields[name] = {
            "status": "missing" if name not in metadata else "available_withheld",
            "reason": "not_provided" if name not in metadata else "original_visibility_policy",
        }
    report = dict(
        schema_version="fairvlmed-input-check/1.0.0",
        dataset=dataset,
        sample=sample,
        split=split,
        preparation_version=snap.preparation_version,
        fingerprint=snap.fingerprint(),
        source_files=[v.model_dump(mode="json") for v in snap.files],
        image_checksum=checksum(snap.images[0][0]),
        image_shape=list(snap.images[0][1]),
        modality="SLO",
        fields=fields,
        unknown_fields=sorted(set(metadata) - NPZ_KEYS),
        patient_identity="unknown",
        independence="unproven",
        label_used=False,
        original_note_visible=False,
        model_requests=0,
        api_requests=0,
    )
    return snap, report


def prepare(
    *,
    dataset,
    root,
    sample,
    key,
    split="training",
    fairclip_configuration=None,
    evidence_profile=None,
    upstream_profile=None,
    report_profile=None,
    provider_profile=None,
    execution_lock_profile=None,
):
    _, checked = inspect_input(dataset, root, sample, split=split)
    out = directory(key)
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    spec = dict(
        schema_version="fairvlmed-run-input/1.1.0",
        owner_key_policy="content-bound-owner-key/1.0.0",
        dataset=dataset,
        root=str(Path(root).resolve()),
        sample=sample,
        split=split,
        key=key,
        checked=checked,
        provider_configuration=PROVIDER,
        fairclip_configuration=fairclip_configuration,
        max_requests=40,
        seconds=5400,
        tool_recipe="installed_slo/1.0.0",
        clinical_acceptance=False,
    )
    if evidence_profile is not None:
        from .unified_requirements import PROFILE, catalog

        if evidence_profile != PROFILE:
            raise ValueError("generic_evidence_profile")
        spec.update(
            schema_version="fairvlmed-run-input/1.2.0",
            evidence_profile=PROFILE,
            evidence_catalog_checksum=digest(catalog()),
            public_context_policy="fairvlmed-public-age/1.0.0",
        )
    if upstream_profile is not None:
        from packages.evidence_reuse.integration import PROFILE as UPSTREAM
        from packages.evidence_reuse.integration import implementation

        if upstream_profile != UPSTREAM or evidence_profile is None:
            raise ValueError("generic_upstream_requires_source_profile")
        spec.update(
            schema_version="fairvlmed-run-input/1.3.0",
            upstream_profile=UPSTREAM,
            upstream_implementation=implementation(),
        )
    if report_profile is not None:
        from .report_production import PROFILE as REPORT_PROFILE
        from .report_production import implementation as report_implementation

        if report_profile != REPORT_PROFILE or upstream_profile is None:
            raise ValueError("generic_report_requires_upstream_profile")
        spec.update(
            schema_version="fairvlmed-run-input/1.4.0",
            report_profile=REPORT_PROFILE,
            report_implementation=report_implementation(),
        )
    if provider_profile is not None:
        from packages.llm_provider.config import prompt, read_provider_config

        selections = {
            profile: (version, path)
            for version, (profile, path) in PROFILE_INPUTS.items()
            if version != "fairvlmed-run-input/1.9.0"
        }
        if provider_profile not in selections or report_profile is None:
            raise ValueError("generic_provider_requires_report_profile")
        version, selected_path = selections[provider_profile]
        config = read_provider_config(json.loads(Path(selected_path).read_bytes()))
        if version == "fairvlmed-run-input/1.8.0":
            from packages.llm_provider.model_selection import apply_selection, load_selection

            spec["model_selection"] = load_selection(ALTERNATE_SELECTION)
            config = apply_selection(config, spec["model_selection"])
        prompt(config, "review")
        spec.update(
            schema_version=version,
            provider_profile=provider_profile,
            provider_configuration=selected_path,
            provider_configuration_checksum=digest(config),
        )
        provider_config(spec)
        if version in {
            "fairvlmed-run-input/1.7.0",
            "fairvlmed-run-input/1.8.0",
            "fairvlmed-run-input/1.10.0",
            "fairvlmed-run-input/1.11.0",
        }:
            from packages.llm_provider.network_route import route_opener

            route_opener(config.network_route, config.transport.base_url)
    if execution_lock_profile is not None:
        from .generic_execution_lock import INPUT_VERSION, PROFILE, ROOT, implementation

        if execution_lock_profile != PROFILE or spec["schema_version"] not in {
            "fairvlmed-run-input/1.8.0",
            "fairvlmed-run-input/1.10.0",
            "fairvlmed-run-input/1.11.0",
        }:
            raise ValueError("generic_execution_lock_requires_selected_current_profile")
        spec.update(
            schema_version=(
                spec["schema_version"]
                if spec["schema_version"] in BOUND_ROUTE_VERSIONS
                else INPUT_VERSION
            ),
            execution_lock_profile=PROFILE,
            execution_lock_implementation=implementation(ROOT),
        )
    elif spec["schema_version"] in BOUND_ROUTE_VERSIONS:
        raise ValueError("generic_local_provider_requires_execution_lock")
    write_once(out / "input.json", spec)
    return spec


def load(key):
    value = json.loads((directory(key) / "input.json").read_bytes())
    if (
        value["schema_version"]
        not in {
            "fairvlmed-run-input/1.0.0",
            "fairvlmed-run-input/1.1.0",
            "fairvlmed-run-input/1.2.0",
            "fairvlmed-run-input/1.3.0",
            "fairvlmed-run-input/1.4.0",
            "fairvlmed-run-input/1.5.0",
            "fairvlmed-run-input/1.6.0",
            "fairvlmed-run-input/1.7.0",
            "fairvlmed-run-input/1.8.0",
            "fairvlmed-run-input/1.9.0",
            "fairvlmed-run-input/1.10.0",
            "fairvlmed-run-input/1.11.0",
        }
        or value["key"] != key
    ):
        raise ValueError("generic_input_version_or_key")
    from .generic_execution_lock import validate_profile

    validate_profile(value)
    if value["schema_version"] != "fairvlmed-run-input/1.0.0":
        if value.get("owner_key_policy") != "content-bound-owner-key/1.0.0":
            raise ValueError("generic_owner_key_policy")
    elif "owner_key_policy" in value:
        raise ValueError("generic_owner_key_policy")
    _, current = inspect_input(
        value["dataset"], value["root"], value["sample"], split=value["split"]
    )
    if current != value["checked"]:
        raise ValueError("generic_input_changed")
    if (
        value["provider_configuration"]
        != (
            LEGACY_PROVIDER
            if value["schema_version"] == "fairvlmed-run-input/1.0.0"
            else PROFILE_INPUTS[value["schema_version"]][1]
            if value["schema_version"] in PROFILE_INPUTS
            else PROVIDER
        )
        or value["max_requests"] != 40
        or value["seconds"]
        != (2700 if value["schema_version"] == "fairvlmed-run-input/1.0.0" else 5400)
        or value["tool_recipe"] != "installed_slo/1.0.0"
        or value["clinical_acceptance"] is not False
    ):
        raise ValueError("generic_configuration_unsupported")
    if value["schema_version"] in PROFILE_INPUTS:
        provider_config(value)
    elif any(k in value for k in ("provider_profile", "provider_configuration_checksum")):
        raise ValueError("generic_legacy_provider_extension")
    if value["schema_version"] in {
        "fairvlmed-run-input/1.2.0",
        "fairvlmed-run-input/1.3.0",
        "fairvlmed-run-input/1.4.0",
        "fairvlmed-run-input/1.5.0",
        "fairvlmed-run-input/1.6.0",
        "fairvlmed-run-input/1.7.0",
        "fairvlmed-run-input/1.8.0",
        "fairvlmed-run-input/1.9.0",
        "fairvlmed-run-input/1.10.0",
        "fairvlmed-run-input/1.11.0",
    }:
        from .unified_requirements import PROFILE, catalog

        if (
            value.get("evidence_profile") != PROFILE
            or value.get("evidence_catalog_checksum") != digest(catalog())
            or value.get("public_context_policy") != "fairvlmed-public-age/1.0.0"
        ):
            raise ValueError("generic_evidence_profile_changed")
    elif any(
        k in value
        for k in ("evidence_profile", "evidence_catalog_checksum", "public_context_policy")
    ):
        raise ValueError("generic_legacy_profile_extension")
    if value["schema_version"] in {
        "fairvlmed-run-input/1.3.0",
        "fairvlmed-run-input/1.4.0",
        "fairvlmed-run-input/1.5.0",
        "fairvlmed-run-input/1.6.0",
        "fairvlmed-run-input/1.7.0",
        "fairvlmed-run-input/1.8.0",
        "fairvlmed-run-input/1.9.0",
        "fairvlmed-run-input/1.10.0",
        "fairvlmed-run-input/1.11.0",
    }:
        from packages.evidence_reuse.integration import PROFILE as UPSTREAM
        from packages.evidence_reuse.integration import implementation

        from .generic_repair import implementation_matches

        if value.get("upstream_profile") != UPSTREAM or not implementation_matches(
            value, "upstream_implementation", implementation()
        ):
            raise ValueError("generic_upstream_implementation_changed")
    elif any(k in value for k in ("upstream_profile", "upstream_implementation")):
        raise ValueError("generic_legacy_upstream_extension")
    if value["schema_version"] in {"fairvlmed-run-input/1.4.0", *PROFILE_INPUTS}:
        from .generic_repair import implementation_matches
        from .report_production import PROFILE as REPORT_PROFILE
        from .report_production import implementation as report_implementation

        if value.get("report_profile") != REPORT_PROFILE or not implementation_matches(
            value, "report_implementation", report_implementation()
        ):
            raise ValueError("generic_report_implementation_changed")
    elif any(k in value for k in ("report_profile", "report_implementation")):
        raise ValueError("generic_legacy_report_extension")
    return value


def owner_key(spec, purpose, target=""):
    """Bound caller identifiers once; original owners retain their strict limits."""
    if purpose not in {"fairclip", "phase", "memory"}:
        raise ValueError("generic_owner_key_purpose")
    if spec["schema_version"] == "fairvlmed-run-input/1.0.0":
        if purpose == "fairclip":
            return "fairvlmed-generic:" + spec["key"] + ":fairclip"
        old = "generic-v1:" + spec["key"] + ":" + spec["sample"] + ":" + target
        return "memory:" + old if purpose == "memory" else old
    if (
        spec["schema_version"]
        not in {
            "fairvlmed-run-input/1.1.0",
            "fairvlmed-run-input/1.2.0",
            "fairvlmed-run-input/1.3.0",
            "fairvlmed-run-input/1.4.0",
            "fairvlmed-run-input/1.5.0",
            "fairvlmed-run-input/1.6.0",
            "fairvlmed-run-input/1.7.0",
            "fairvlmed-run-input/1.8.0",
            "fairvlmed-run-input/1.9.0",
            "fairvlmed-run-input/1.10.0",
            "fairvlmed-run-input/1.11.0",
        }
        or spec.get("owner_key_policy") != "content-bound-owner-key/1.0.0"
    ):
        raise ValueError("generic_owner_key_policy")
    return "fv:" + purpose + ":" + digest(dict(input=spec, purpose=purpose, target=target))[7:]


def register(spec, *, owner_factory=None):
    """Original 07 register/visible reader. No label/demographic application bypass."""
    from .case_execution import owners

    snap, checked = inspect_input(
        spec["dataset"], spec["root"], spec["sample"], split=spec["split"]
    )
    if checked != spec["checked"]:
        raise ValueError("generic_input_changed")
    runtime, objects, preparation = (owner_factory or owners)()
    sample = factory(spec["dataset"], spec["root"]).prepare(spec["sample"], split=spec["split"])
    if capture(sample).fingerprint() != snap.fingerprint():
        raise ValueError("generic_input_changed")
    registration_key = "fairvlmed-selected-v1:" + digest(
        dict(dataset=snap.dataset.reference(), input=snap.fingerprint())
    )
    result = preparation.register(sample, key=registration_key)
    visible = preparation.read_visible(result.visible.case.case_code)
    if (
        visible != result.visible
        or objects.read_bounded(visible.case.artifacts[0].uri, 4194304) != snap.images[0][0]
    ):
        raise ValueError("generic_input_readback")
    out = directory(spec["key"])
    binding = dict(
        schema_version="generic-case-binding/1.0.0",
        sample_id=spec["sample"],
        case_code=visible.case.case_code,
        image=visible.case.artifacts[0].model_dump(mode="json"),
        visible_checksum=visible.checksum,
        input_fingerprint=snap.fingerprint(),
        demographics={},
        demographics_status="withheld_by_original_visibility_policy",
        input_spec_checksum=digest(spec),
        execution_namespace=spec["key"],
        patient_independence="unproven",
    )
    if spec["schema_version"] in {
        "fairvlmed-run-input/1.2.0",
        "fairvlmed-run-input/1.3.0",
        "fairvlmed-run-input/1.4.0",
        "fairvlmed-run-input/1.5.0",
        "fairvlmed-run-input/1.6.0",
        "fairvlmed-run-input/1.7.0",
        "fairvlmed-run-input/1.8.0",
        "fairvlmed-run-input/1.9.0",
        "fairvlmed-run-input/1.10.0",
        "fairvlmed-run-input/1.11.0",
    }:
        from packages.dataset_adapters.public_context import register as register_context

        asset, context = register_context(
            preparation, sample, key=registration_key, policy=spec["public_context_policy"]
        )
        binding.update(
            schema_version="generic-case-binding/1.1.0",
            demographics={"age": context["age"]},
            demographics_status="explicit_age_only_projection",
            public_context_artifact=asset.model_dump(mode="json"),
        )
    if spec["schema_version"] in {"fairvlmed-run-input/1.9.0", *BOUND_ROUTE_VERSIONS}:
        binding["schema_version"] = "generic-case-binding/1.2.0"
    write_once(out / "binding.json", binding)
    write_once(out / "visible.json", visible)
    image = out / "input.png"
    if image.exists():
        if image.read_bytes() != snap.images[0][0]:
            raise ValueError("generic_image_changed")
    else:
        with image.open("xb") as stream:
            image.chmod(0o600)
            stream.write(snap.images[0][0])
    return out, binding, runtime, objects


def validate_registered(sample, registered):
    """An injected tuple is not authority: re-read the original input owner."""
    from db.runtime.repository.dataset_input import DatasetInputRepository
    from packages.dataset_adapters.service import VisibleInputReader

    out, binding, runtime, objects = registered
    value = VisibleInputReader(DatasetInputRepository(runtime), objects).read(binding["case_code"])
    if (
        binding["sample_id"] != sample
        or value.checksum != binding["visible_checksum"]
        or len(value.case.artifacts) != 1
        or value.case.artifacts[0].model_dump(mode="json") != binding["image"]
        or checksum((out / "input.png").read_bytes()) != value.case.artifacts[0].checksum
    ):
        raise ValueError("generic_registered_input_binding")
    if binding.get("schema_version") == "generic-case-binding/1.2.0":
        from apps.segman_resource_lock import verify_input

        from .generic_execution_lock import ROOT

        verify_input(ROOT, out, binding)
    if binding.get("schema_version") in {
        "generic-case-binding/1.1.0",
        "generic-case-binding/1.2.0",
    }:
        from packages.contracts import Artifact
        from packages.dataset_adapters.public_context import read

        context = read(
            runtime,
            objects,
            value,
            Artifact.model_validate_json(json.dumps(binding["public_context_artifact"])),
        )
        if context["input_fingerprint"] != binding["input_fingerprint"] or binding[
            "demographics"
        ] != {"age": context["age"]}:
            raise ValueError("generic_context_binding")
    return registered
