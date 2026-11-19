"""Synthetic offline extension interfaces; real activation remains unavailable."""

from .catalog import SyntheticCatalogOwner
from .future import SyntheticFutureRunOwner
from .models import CatalogFixture, ExtensionGrant, ExtensionPlan, ExtensionPolicy, ExtensionResult
from .service import ExtensionAuthority, OfflineExtension

__all__ = [
    "CatalogFixture",
    "ExtensionGrant",
    "ExtensionPlan",
    "ExtensionPolicy",
    "ExtensionResult",
    "ExtensionAuthority",
    "OfflineExtension",
    "SyntheticCatalogOwner",
    "SyntheticFutureRunOwner",
]
