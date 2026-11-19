import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from packages.knowledge_connectors import (
    ConnectorConfiguration,
    DeterministicFakeConnector,
    ProviderConfig,
    ProviderRegistry,
    SafeHTTPClient,
    load_configuration,
)
from packages.knowledge_connectors.fake import FakeClock, RecordedTransport


def test_six_provider_registry_and_one_disabled_web():
    configuration = load_configuration(Path("configs/knowledge_sources.toml"))
    registry = ProviderRegistry(
        configuration,
        lambda config: SafeHTTPClient(
            config,
            RecordedTransport(()),
            resolver=lambda host: ("93.184.216.34",),
            clock=FakeClock(),
        ),
    )
    assert len(registry.codes()) == 6
    assert len({p.provider_code for p in configuration.providers}) == 6
    web = registry.get("SOURCE:PROVIDER:WEB")
    assert not web.capabilities().search
    assert registry.get("SOURCE:PROVIDER:CROSSREF").capabilities().metadata_only
    assert registry.get("SOURCE:PROVIDER:NICE").capabilities().discovery_mode == "curated_catalog"


@pytest.mark.parametrize(
    "update",
    [
        {"enabled": 1},
        {"provider_version": "latest"},
        {"source_type": "WRONG"},
        {"api_key": "not permitted"},
        {"allowed_hosts": ("*.crossref.org",)},
        {"search_endpoint": "https://api.crossref.org/works?api_key=secret"},
        {"search_endpoint": "https://unexpected.example/works"},
        {"auth_mode": "bearer"},
        {"api_key_env": "bad-key-name"},
        {"adapter": "guideline"},
        {"homepage_url": "file:///tmp/x"},
    ],
)
def test_invalid_configuration_and_construct_bypass_rejected(configurations, update):
    config = configurations["crossref"]
    values = dict(vars(config))
    values.update(update)
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(values)
    invalid = config.model_copy(update=update)
    with pytest.raises(ValidationError):
        DeterministicFakeConnector(invalid, ())


@pytest.mark.parametrize(
    "update",
    [
        {"max_retries": True},
        {"max_response_bytes": 0},
        {"max_page_size": 101},
        {"concurrency": 0},
        {"rate_limit_per_minute": 0},
        {"max_redirects": 20},
        {"user_agent": "agent\r\nX: bad"},
        {"allowed_mime_types": ("*/*",)},
    ],
)
def test_http_budgets_are_strict_and_bounded(configurations, update):
    config = configurations["crossref"]
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(
            config.model_copy(update={"http": config.http.model_copy(update=update)})
        )


def test_duplicate_codes_and_multiple_fallbacks_rejected(configurations):
    config = configurations["crossref"]
    with pytest.raises(ValidationError):
        ConnectorConfiguration(config_version="1.0.0", providers=(config, config))
    web = configurations["web_fallback"]
    other = web.model_copy(update={"provider_code": "SOURCE:PROVIDER:OTHER_WEB"})
    with pytest.raises(ValidationError):
        ConnectorConfiguration(config_version="1.0.0", providers=(web, other))


def test_no_connector_database_archive_or_llm_dependency():
    forbidden = {"db", "apps", "agents", "psycopg", "openai", "subprocess"}
    for path in Path("packages/knowledge_connectors").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not ({a.name.split(".")[0] for a in node.names} & forbidden), path
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module.split(".")[0] not in forbidden, path
                assert not node.module.startswith("packages.object_store"), path
