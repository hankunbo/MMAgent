from types import SimpleNamespace

import pytest

from packages.knowledge_archive.approved_web import CONFIG_CHECKSUM, PAGES, PROVIDER, admitted


def values():
    key, url = next(iter(PAGES.items()))
    hit = SimpleNamespace(
        discovery=SimpleNamespace(
            provider_record_id=key,
            provider_code=PROVIDER,
            provider_version="1.1.0",
            config_checksum=CONFIG_CHECKSUM,
        )
    )
    request = SimpleNamespace(
        pipeline_version="1.1.0",
        document=SimpleNamespace(adapter="web_fallback", locator=SimpleNamespace(url=url)),
        bibliography=SimpleNamespace(canonical_url=url),
    )
    return hit, request


def test_only_exact_user_approved_egs_content_admitted():
    assert admitted(*values())


@pytest.mark.parametrize(
    "field", ["provider_code", "provider_version", "provider_record_id", "config_checksum"]
)
def test_other_provider_page_or_config_stays_unapproved(field):
    hit, request = values()
    setattr(hit.discovery, field, "different")
    assert not admitted(hit, request)


@pytest.mark.parametrize("field", ["locator", "bibliography", "pipeline"])
def test_other_url_or_legacy_pipeline_stays_unapproved(field):
    hit, request = values()
    if field == "locator":
        request.document.locator.url += "?different"
    elif field == "bibliography":
        request.bibliography.canonical_url = "https://www.eugs.org/"
    else:
        request.pipeline_version = "1.0.0"
    assert not admitted(hit, request)
