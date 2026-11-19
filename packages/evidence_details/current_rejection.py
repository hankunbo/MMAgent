"""Explicit current-use refusals, distinct from corrupt objects or unknown commits.

Dependency owners can raise CurrentResearchRejected with a bounded reason. Legacy
owner codes below remain supported; arbitrary ValueError/DB errors must propagate.
"""

from packages.grounding.models import Token


class CurrentResearchRejected(ValueError):
    def __init__(self, reason):
        from pydantic import TypeAdapter

        self.reason = TypeAdapter(Token).validate_python(reason)
        super().__init__(self.reason)


def current_rejection(error):
    if isinstance(error, CurrentResearchRejected):
        return error.reason
    if type(error) is ValueError and str(error) in {
        "source_withdrawn",
        "source_version",
        "observation_version",
        "graph_reset",
        "policy_changed",
        "research_owner_origin_changed",
        "research_install_parent_changed",
        "historical_install_not_current",
    }:
        return str(error)
    return None
