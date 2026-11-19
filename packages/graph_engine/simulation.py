"""Research dependency projection, no GraphDelta or accepted event production."""

from packages.evidence_scoring.engine import path_available


def removal_closure(state, *, evidence=(), resources=()):
    items = {i.id: i for i in state.policy.items}
    refs = {r.code: r for r in state.resources}
    if not set(evidence) <= items.keys() or not set(resources) <= refs.keys():
        raise ValueError("simulation_unknown_target")
    observations = {items[i].observation_code for i in evidence}
    removed = set(resources)
    observed = {o.reference.code: o.reference for o in state.material.observations}
    affected = {}
    while True:
        previous = (set(removed), set(observations))
        # A removed observation can itself be a downstream computation input.
        identities = {
            (observed[o].code, observed[o].version, observed[o].checksum) for o in observations
        }
        removed.update(
            r.code for r in refs.values() if (r.code, r.version, r.checksum) in identities
        )
        signatures = {(refs[r].role, refs[r].version, refs[r].checksum) for r in removed}
        removed.update(
            r.code for r in refs.values() if (r.role, r.version, r.checksum) in signatures
        )
        for item in items.values():
            if item.observation_code in observations:
                affected.setdefault(item.id, "removed_evidence_or_same_observation_alias")
            elif not path_available(item.dependency_paths, removed):
                affected[item.id] = "all_OR_supply_paths_blocked_by_required_resource"
                # Only invalidate the underlying observation if no recorded alias has supply.
                aliases = [p for p in items.values() if p.observation_code == item.observation_code]
                if all(not path_available(p.dependency_paths, removed) for p in aliases):
                    observations.add(item.observation_code)
        if previous == (removed, observations):
            return affected, removed
