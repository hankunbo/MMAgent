"""Explicit platform Provider assembly. Legacy CLI grants are not platform grants."""

from packages.llm_provider.config import read_provider_config
from packages.llm_provider.provider import StructuredProvider
from packages.platform_security import EgressGuard
from packages.platform_security.egress import BoundEgressOwner


def platform_provider(config, *, egress_owner, **options):
    config = read_provider_config(config)
    if not isinstance(egress_owner, BoundEgressOwner):
        raise ValueError("platform_egress_owner_required")
    # Required for Fake too: offline tests exercise the exact platform control path.
    guard = EgressGuard(egress_owner, clock=options.get("clock"))
    return StructuredProvider(config, egress=guard, **options)
