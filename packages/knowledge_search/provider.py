"""Immutable public configuration attestation; actual identities stay in 04B1 IO."""

from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, StringConstraints, model_validator

from packages.knowledge_connectors import ProviderConfig
from packages.knowledge_connectors.ports import (
    Checksum,
    ConfiguredIdentity,
    Version,
    checked,
    model_checksum,
)

Host = Annotated[
    str,
    StringConstraints(
        strict=True, max_length=253, pattern=r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,63}$"
    ),
]
EnvName = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Z][A-Z0-9_]{0,127}$")]


class ProviderConfigurationProof(ConfiguredIdentity):
    proof_version: Literal["provider-configuration/1.0.0"] = "provider-configuration/1.0.0"
    allowlist_version: Version
    allowed_hosts: tuple[Host, ...] = Field(min_length=1, max_length=64)
    locator_policy: Literal["doi-resolvers/1.0.0"] = "doi-resolvers/1.0.0"
    identity_policy: Literal["contact-env/2.0.0"]
    contact_environment: EnvName | None
    auth_environment_name: EnvName | None
    identity_configured: bool
    policy_version: Version
    policy_checksum: Checksum
    deployment_workers: Literal[1]

    @model_validator(mode="after")
    def canonical(self):
        if self.allowed_hosts != tuple(sorted(set(self.allowed_hosts))) or any(
            ".." in h for h in self.allowed_hosts
        ):
            raise ValueError("canonical provider allowlist required")
        if self.identity_configured != (self.contact_environment is not None):
            raise ValueError("configuration identity declaration mismatch")
        return self


def configuration_proof(config: ProviderConfig, policy) -> ProviderConfigurationProof:
    config = checked(ProviderConfig, config)
    return ProviderConfigurationProof(
        provider_code=config.provider_code,
        provider_version=config.provider_version,
        adapter=config.adapter,
        config_checksum=config.checksum(),
        allowlist_version=config.allowlist_version,
        allowed_hosts=tuple(sorted(config.allowed_hosts)),
        identity_policy=config.identity_policy,
        contact_environment=config.contact_email_env,
        auth_environment_name=config.api_key_env,
        identity_configured=config.contact_email_env is not None,
        policy_version=policy.policy_version,
        policy_checksum=model_checksum(policy),
        deployment_workers=policy.deployment_workers,
    )


def default_rate_policy():
    from pathlib import Path

    from .policy import load_rate_policy

    return load_rate_policy(Path(__file__).resolve().parents[2] / "configs/knowledge_search.toml")


def assert_identity(identity, proof):
    proof = ProviderConfigurationProof.model_validate(proof)
    if any(
        getattr(identity, field) != getattr(proof, field)
        for field in (
            "provider_code",
            "provider_version",
            "adapter",
            "config_checksum",
            "identity_policy",
            "contact_environment",
            "auth_environment_name",
            "identity_configured",
            "policy_version",
            "policy_checksum",
            "deployment_workers",
        )
    ):
        raise ValueError("registered provider configuration mismatch")


def discovery_url(value, proof):
    from .privacy import PrivacyError, canonical_doi, safe_url

    proof = ProviderConfigurationProof.model_validate(proof)
    resolvers = ("doi.org", "dx.doi.org")
    value = safe_url(value, allowed_hosts=(*proof.allowed_hosts, *resolvers))
    if urlsplit(value).hostname not in proof.allowed_hosts:
        if urlsplit(value).hostname not in resolvers:
            raise PrivacyError()
        canonical_doi(value)
    return value
