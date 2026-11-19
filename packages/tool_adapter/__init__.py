"""Explicit synthetic research adapter stage; no catalog publication or execution."""

from .gates import FakeProtocol
from .models import AdapterManifest, AdapterPolicy, AdapterResult, SyntheticGrant
from .service import AdapterOnboarding, SyntheticAuthority

__all__ = [
    "AdapterManifest",
    "AdapterOnboarding",
    "AdapterPolicy",
    "AdapterResult",
    "FakeProtocol",
    "SyntheticAuthority",
    "SyntheticGrant",
]
