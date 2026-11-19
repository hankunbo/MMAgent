"""Unattempted requirements are not additional image payloads or observations."""

from types import SimpleNamespace

from packages.multimodal_research.report import media


def test_no_image_reads_for_missing_or_numeric_observations():
    def view(target):
        raise AssertionError("Unobserved planned image was packaged")

    campaign = SimpleNamespace(
        targets=("missing", "numeric"),
        base=SimpleNamespace(
            view=view, observations={}, manifest=SimpleNamespace(checksum="sha256:input")
        ),
        accepted_observations=lambda: {},
    )
    report = dict(
        result=dict(case_code="CASE:PUBLIC_TEST"),
        requirements=[
            dict(id="missing", observation=None),
            dict(id="numeric", observation=dict(value="0.5", image=None)),
        ],
    )
    assert media(campaign, report)["images"] == {}


def test_failed_review_retains_acquired_image_once():
    calls = []
    ref = SimpleNamespace(checksum="sha256:bytes", model_dump=lambda **kw: dict(code="ASSET:view"))
    image = SimpleNamespace(reference=ref, width=1, height=1)
    image.model_dump = lambda **kw: dict(reference=dict(code="ASSET:view"))
    observed = SimpleNamespace(image=image, model_dump=lambda **kw: dict(image=image.model_dump()))

    def view(target):
        calls.append(target)
        return (
            image,
            b"saved-bytes",
            dict(limits=[], transform="same", inspected_fraction=0.5, modality="test"),
        )

    campaign = SimpleNamespace(
        targets=("actual", "unattempted"),
        base=SimpleNamespace(
            observations={},
            view=view,
            manifest=SimpleNamespace(
                checksum="sha256:input",
                components=[SimpleNamespace(id="input", laterality="OD", visit="one")],
            ),
            ledger=[
                dict(id="actual", component="input", definition=SimpleNamespace(examination="Test"))
            ],
        ),
        accepted_observations=lambda: {"actual": observed},
    )
    report = dict(
        result=dict(case_code="CASE:PUBLIC_TEST"),
        requirements=[
            dict(id="actual", observation=None),
            dict(id="unattempted", observation=None),
        ],
        acquisition_observations=dict(actual=dict(image=dict(reference=dict(code="ASSET:view")))),
    )
    result = media(campaign, report)
    assert calls == ["actual"] and list(result["images"]) == ["actual"]
