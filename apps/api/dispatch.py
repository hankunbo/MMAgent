"""Explicit opt-in 12B port; no worker is started inside an HTTP application."""

from typing import Protocol

from packages.task_runtime.models import DispatchRequest as DispatchRequest
from packages.task_runtime.models import Receipt


class DispatchPort(Protocol):
    """Implemented by the controlled original-Task DispatchRepository when installed.

    A return requires a committed original Runtime Task and Audit, not just acceptance.
    UNKNOWN requires same-key reconciliation. No caller-provided executable payload.
    """

    def submit(self, request: DispatchRequest) -> Receipt: ...
