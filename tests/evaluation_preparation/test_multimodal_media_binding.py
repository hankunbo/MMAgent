"""A display alias never substitutes for the saved observation's acquisition identity."""

from types import SimpleNamespace

import pytest

from apps.vectorengine_case_materials import ref
from packages.grounding.models import digest
from packages.llm_provider.models import ImageBinding, ObservationInput
from packages.multimodal_research.report import media


@pytest.mark.parametrize("mutation", [None, "observation", "image"])
@pytest.mark.parametrize("baseline", ["none", "numeric", "conflicting_image"])
def test_shared_observation_display_alias_uses_exact_owner_binding(mutation, baseline):
    raw = b"synthetic-image"
    image = ImageBinding(
        reference=ref("image", "ASSET:SYNTHETIC", digest(raw.decode())),
        media_type="image/png",
        width=8,
        height=8,
        coordinate_system="pixel_xy_top_left",
        location=(0, 0, 8, 8),
        preprocessing=ref("preprocessing", "POLICY:SYNTHETIC", digest("synthetic")),
    )
    observed = ObservationInput(
        reference=ref("observation", "OBS:SYNTHETIC", digest("saved")),
        status="obtained",
        method="live_image_observation",
        unit="qualitative",
        value="original observation",
        image=image,
    )
    calls = []

    def view(target):
        calls.append(target)
        assert target == "original"
        current = image if mutation != "image" else image.model_copy(update={"width": 9})
        return (
            current,
            raw,
            dict(limits=[], transform="synthetic", inspected_fraction=1, modality="test"),
        )

    base = SimpleNamespace(
        observations={},
        view=view,
        ledger=[
            dict(id="original", component="input", definition=SimpleNamespace(examination="test"))
        ],
        manifest=SimpleNamespace(
            checksum=digest("input"),
            components=[SimpleNamespace(id="input", laterality="UNKNOWN", visit="unknown")],
        ),
    )
    campaign = SimpleNamespace(base=base, accepted_observations=lambda: {"original": observed})
    if baseline != "none":
        base.observations["original"] = observed.model_copy(
            update=dict(
                value="original numeric projection", image=None if baseline == "numeric" else image
            )
        )
    value = observed.model_dump(mode="json")
    if mutation == "observation":
        value["value"] = "changed after acceptance"
    report = dict(
        requirements=[dict(id="arbitrary-display-alias", observation=value)],
        result=dict(case_code="CASE:SYNTHETIC"),
    )
    if baseline == "conflicting_image":
        with pytest.raises(ValueError, match="multimodal_acquired_observation_conflict"):
            media(campaign, report)
    elif mutation:
        with pytest.raises(ValueError, match="multimodal_report_image_observation_binding"):
            media(campaign, report)
    else:
        result = media(campaign, report)
        assert list(result["images"]) == ["arbitrary-display-alias"]
        assert (
            result["images"]["arbitrary-display-alias"]["reference"] == value["image"]["reference"]
        )
        assert calls == ["original"]
