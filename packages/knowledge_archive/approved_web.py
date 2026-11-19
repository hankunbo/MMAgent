"""Narrow source admission: user's explicit 2026-09-16 four-page EGS approval.

Not clinical publication, not blanket approval of WEB or the EGS domain.
The 1.1.0 archive pipeline and exact provider config pin this decision. A new
source/config requires a new reviewed policy, never caller-supplied approved=True.
"""

PROVIDER = "SOURCE:PROVIDER:EGS:GLAUCOMA_EDUCATION"
CONFIG_CHECKSUM = "sha256:d58d8f0e09b7cd875157215e3b5af71d664ce8dff436c5c36512d05a30912d51"
PAGES = {
    "EGS-HEMORRHAGE-2024": (
        "https://www.eugs.org/news/"
        "tip-not-every-peripapillary-hemorrhage-is-a-sign-for-glaucoma-2024-06-16-tip-of-the-month"
    ),
    "EGS-ONH-2019": (
        "https://www.eugs.org/news/clinical-examination-of-the-optic-nerve-head-2019-09-30-pearl"
    ),
    "EGS-RNFL-2017": (
        "https://eugs.org/news/oct-measurement-of-the-retinal-nerve-fibre-layer-is-useful-for-"
        "detecting-progression-in-early-glaucoma-2017-12-31-tip-of-the-month"
    ),
    "EGS-OCT-2020": (
        "https://eugs.org/news/optical-coherence-tomography-for-glaucoma-diagnosis-2020-12-30-pearl"
    ),
}


def admitted(hit, request):
    """Called only after archive's original DB/provider/fetch provenance checks."""
    discovery = hit.discovery
    if request.pipeline_version == "1.2.0":
        # User-approved 2026-10-05 risk-context extension. Exact single public
        # page/config only; no wider WEB approval and no medical publication.
        url = "https://www.nei.nih.gov/eye-health-information/eye-conditions-and-diseases/glaucoma"
        return bool(
            discovery
            and discovery.provider_code == "SOURCE:PROVIDER:NEI:GLAUCOMA_INFO"
            and discovery.provider_version == "1.0.0"
            and discovery.provider_record_id == "NEI-GLAUCOMA"
            and discovery.config_checksum
            == "sha256:df81464a9fa107635df44713c4cc0e0b49370c9e42886e42ccdc114673d5a4c2"
            and request.document.adapter == "web_fallback"
            and request.document.locator.url == url
            and request.bibliography.canonical_url == url
        )
    url = PAGES.get(discovery.provider_record_id) if discovery else None
    return bool(
        url
        and request.pipeline_version == "1.1.0"
        and discovery.provider_code == PROVIDER
        and discovery.provider_version == "1.1.0"
        and discovery.config_checksum == CONFIG_CHECKSUM
        and request.document.adapter == "web_fallback"
        and request.document.locator.url == url
        and request.bibliography.canonical_url == url
    )
