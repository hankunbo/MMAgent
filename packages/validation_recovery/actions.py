"""Filter freshly owner-authorized alternatives by installed fork ancestry."""

from packages.grounding.models import digest, seal, validate
from packages.planner_control.models import LegalActionSet, Phase

from .models import RecoveryPolicy


class NearestLegalActions:
    def __init__(self, source, policy):
        self.source, self.policy = source, validate(RecoveryPolicy, policy)

    def read(self, state, policy):
        if self.policy.source != next(r for r in state.binding.locks if r.role == "rule"):
            raise ValueError("recovery_route_rule_lock")
        current = validate(LegalActionSet, self.source.read(state, policy))
        routes = {r.branch: r.forks for r in self.policy.routes}
        failed = (
            state.selected.branch
            if state.phase == Phase.FALLBACK and state.selected
            else (state.attempted[-1] if state.attempted else None)
        )
        work = [a for a in current.actions if a.candidate and a.branch != failed]
        # A saved selection must still be revalidated, including its own prepare reservation.
        if state.phase in {Phase.PREPARE, Phase.EXECUTE, Phase.VERIFY, Phase.REVIEW}:
            work = [a for a in current.actions if a.candidate]
        elif failed is not None:
            if failed not in routes or any(a.branch not in routes for a in work):
                raise ValueError("recovery_fork_route_missing")

            def depth(a):
                count = 0
                for left, right in zip(routes[failed], routes[a.branch], strict=False):
                    if left != right:
                        break
                    count += 1
                return count

            if work:
                nearest = max(depth(a) for a in work)
                work = [a for a in work if depth(a) == nearest]
        kept = {a.id for a in work}
        values = {
            name: getattr(current, name)
            for name in LegalActionSet.model_fields
            if name != "checksum"
        }
        values["actions"] = tuple(a for a in current.actions if a.candidate is None or a.id in kept)
        values["source_checksums"] = tuple(
            sorted(set(current.source_checksums) | {digest(self.policy)})
        )
        return seal(LegalActionSet, **values)
