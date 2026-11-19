"""Lifecycle mechanism, independent of unfrozen clinical publication policy."""

EDGES = {
    "candidate": {"reviewed", "rejected", "deprecated"},
    "reviewed": {"published", "rejected", "deprecated"},
    "published": {"deprecated"},
    "rejected": {"deprecated"},
    "deprecated": set(),
}


def transition(old, new):
    if new not in EDGES.get(old, set()):
        raise ValueError("memory_illegal_transition")
    if new == "published":
        raise ValueError("memory_publication_policy_unfrozen")
    return new
