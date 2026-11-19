"""One validated provider per stable code; no cross-provider query orchestration."""

from collections.abc import Callable

from .configuration import ConnectorConfiguration
from .connectors import CONNECTOR_TYPES
from .http import SafeHTTPClient
from .ports import KnowledgeSourceConnector, checked


class ProviderRegistry:
    def __init__(
        self, configuration: ConnectorConfiguration, client_factory: Callable[..., SafeHTTPClient]
    ):
        configuration = checked(ConnectorConfiguration, configuration)
        self.configuration = configuration
        self._connectors = {
            config.provider_code: CONNECTOR_TYPES[config.adapter](config, client_factory(config))
            for config in configuration.providers
        }

    def codes(self) -> tuple[str, ...]:
        return tuple(sorted(self._connectors))

    def get(self, provider_code: str) -> KnowledgeSourceConnector:
        if type(provider_code) is not str:
            raise TypeError("provider code must be a string")
        return self._connectors[provider_code]
