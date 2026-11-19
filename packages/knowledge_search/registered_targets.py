"""Public task -> existing approved query vocabulary; no new search storage schema.

A search synonym is neither a clinical grading rule nor permission to relabel
observations. The original task owner retains the exact proposition identity.
"""

from .broad import broad_plan
from .targeted import targeted_plan

TARGETS = {
    "GOAL:GLAUCOMA_DETECTION": None,
    "GOAL:EARLY_GLAUCOMA": "GOAL:EARLY_GLAUCOMA",
    "GOAL:INTERMEDIATE_GLAUCOMA": "GOAL:MODERATE_GLAUCOMA",
    "GOAL:ADVANCED_GLAUCOMA": "GOAL:ADVANCED_GLAUCOMA",
}


def registered_plan(provider_codes, *, target, **kwargs):
    if target not in TARGETS:
        raise ValueError("unsupported_registered_public_proposition")
    query_target = TARGETS[target]
    if query_target is None:
        # Existing broad vocabulary already carries the glaucoma concept and
        # selected examination topics. No arbitrary patient text is accepted.
        return broad_plan(provider_codes, **kwargs)
    return targeted_plan(provider_codes, target=query_target, **kwargs)
