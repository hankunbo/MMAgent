"""Current server-installed owner authorizes exact projected bytes, never caller flags.

An owner must reconstruct current source/visibility/run qualifications on each read.
No historical approval, model claim or matching digest alone grants permission.
"""

from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import AwareDatetime, Field

from packages.grounding.models import digest, validate
from packages.llm_provider.models import Scope, Value
from packages.planner_control.models import Checksum, Reference


class Policy(Value):
    schema_version: Literal["platform-egress-policy/1.0.0"] = "platform-egress-policy/1.0.0"
    policy_id: str = Field(pattern=r"^POLICY:[A-Z0-9_]+$")
    purpose: Literal["research"] = "research"
    allowed_classes: tuple[Literal["public", "deidentified"], ...] = ("public", "deidentified")
    allow_private_with_approval: bool = False
    max_wire_bytes: int = Field(ge=1, le=16_777_216)
    active: bool = True


class EgressRequest(Value):
    schema_version: Literal["platform-egress-request/1.0.0"] = "platform-egress-request/1.0.0"
    scope: Scope
    purpose: Literal["research"] = "research"
    role: Literal["trace", "expert", "source", "vision", "review"]
    config_checksum: Checksum
    payload_checksum: Checksum
    wire_checksum: Checksum
    wire_bytes: int = Field(ge=1)
    image_checksums: tuple[Checksum, ...] = ()


class Approval(Value):
    schema_version: Literal["platform-egress-approval/1.0.0"] = "platform-egress-approval/1.0.0"
    reference: Reference
    request_checksum: Checksum
    policy_checksum: Checksum
    data_class: Literal["public", "deidentified", "private", "unknown"]
    visible_projection_approved: bool
    independent_private_approval: Reference | None = None
    expires_at: AwareDatetime
    active: bool
    source_references: tuple[Reference, ...] = ()


class CurrentOwner(Protocol):
    def read_policy(self) -> Policy: ...

    def read_current(self, request: EgressRequest) -> Approval:
        """Recheck source identities/use, visible projection, run and input via original owners."""
        ...


class EgressDenied(ValueError):
    def __init__(self):
        super().__init__("platform_egress_denied")


class BoundEgressOwner:
    """Assembly of existing visibility/source readers and a separate approval reader.

    read_approval is installed by the server, not passed in the HTTP request. It
    reads an exact approved request, never issues approval on lookup. check_run
    revalidates original Control/live permission and must raise on stale scope.
    """

    def __init__(
        self, *, read_policy, read_approval, materials, check_run, read_private_approval=None
    ):
        if (
            not all(callable(v) for v in (read_policy, read_approval, check_run))
            or materials is None
        ):
            raise ValueError("platform_egress_owner_required")
        self.read_policy = read_policy
        self._approval = read_approval
        self.materials = materials
        self.check_run = check_run
        self.read_private_approval = read_private_approval

    def read_current(self, request):
        self.check_run(request.scope)
        material = self.materials.read(request.scope)
        approval = validate(Approval, self._approval(request))
        if approval.data_class == "private":
            if (
                approval.independent_private_approval is None
                or self.read_private_approval is None
                or self.read_private_approval(approval.independent_private_approval, request)
                != approval.independent_private_approval
            ):
                raise EgressDenied()
        if material.scope != request.scope or approval.source_references != tuple(
            s.reference for s in material.sources
        ):
            raise EgressDenied()
        if request.image_checksums:
            from packages.object_store import checksum

            bindings = tuple(o.image for o in material.observations if o.image is not None)
            if request.image_checksums != tuple(
                checksum(self.materials.image(b)) for b in bindings
            ):
                raise EgressDenied()
        self.check_run(request.scope)
        return approval


class EgressGuard:
    def __init__(self, owner: CurrentOwner, *, clock=None):
        if owner is None:
            raise ValueError("platform_egress_owner_required")
        self.owner = owner
        self.clock = clock or (lambda: datetime.now(UTC))

    def check(self, request: EgressRequest):
        request = validate(EgressRequest, request)
        policy = validate(Policy, self.owner.read_policy())
        approval = validate(Approval, self.owner.read_current(request))
        # Catch policy changes during owner reconstruction without unbounded retry.
        if validate(Policy, self.owner.read_policy()) != policy:
            raise EgressDenied()
        allowed = approval.data_class in policy.allowed_classes or (
            approval.data_class == "private"
            and policy.allow_private_with_approval
            and approval.independent_private_approval is not None
        )
        if (
            not policy.active
            or not approval.active
            or not allowed
            or not approval.visible_projection_approved
            or approval.request_checksum != digest(request)
            or approval.policy_checksum != digest(policy)
            or approval.expires_at <= self.clock()
            or request.wire_bytes > policy.max_wire_bytes
        ):
            raise EgressDenied()
        return approval
