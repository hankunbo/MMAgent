"""Minimal local front door; uninstalled domain capabilities remain explicitly unready."""

import os
from pathlib import Path

from apps.api.assembly import Dependencies
from apps.api.auth import LocalIdentities, Principal

from .runtime import Installation, Profile


def build():
    token = os.environ.get("MMAGENT_PLATFORM_TOKEN")
    if not token:
        raise ValueError("platform_local_identity_required")
    return Installation(
        Dependencies(LocalIdentities({token: Principal("SOURCE:LOCAL_READER", "read_only")})),
        Profile.model_validate_json(
            (Path(__file__).resolve().parents[2] / "configs/api/platform_v1.json").read_text()
        ),
    )
