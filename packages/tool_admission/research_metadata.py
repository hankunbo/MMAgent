"""An exact owner-reviewed exception; never read requirements from a request.

The immutable reviewed manifest binds tool, implementation, catalog and review
versions. Its processing policy is revalidated by admission and frozen in plans.
All other tools and the formal entry retain the original strict metadata rules.
"""

from packages.fairclip_research.core import CONFIG, CONFIG_CHECKSUM, WEIGHT_SHA256
from packages.grounding.models import digest
from packages.integrations.medgraph.models import ReviewedDeploymentManifest

from .models import AdmissionError

FAIRCLIP_METADATA = {
    "version": "fairclip-single-image-metadata/1.0.0",
    "owner": "06E_DEPLOYMENT_REVIEWER",
    "use": "single_SLO_fixed_text_research_classification",
    "purposes": ["debug", "shadow"],
    "source_commit": CONFIG["source_commit"],
    "model_checksum": "sha256:" + WEIGHT_SHA256,
    "classification_checksum": CONFIG_CHECKSUM,
    "laterality": "not_consumed_not_required",
    "acquired_at": "not_consumed_not_required",
    "missing_facts": "preserve_UNKNOWN_and_null",
    "excluded_uses": ["bilateral_pairing", "longitudinal_comparison", "clinical_conclusion"],
}


def limited_metadata(manifest, candidate, purpose):
    """Only controlled, exact-version research deployment snapshots grant this scope."""
    if purpose is None or manifest is None:
        return False
    pre = manifest.preprocessing_policy or {}
    declaration = pre.get("metadata_requirements")
    if declaration is None:
        return False
    if (
        not isinstance(manifest, ReviewedDeploymentManifest)
        or manifest.review_ref.checksum != manifest.policy_checksum
        or declaration != FAIRCLIP_METADATA
        or purpose not in FAIRCLIP_METADATA["purposes"]
        or digest(pre.get("fixed_preprocess")) != digest(CONFIG["preprocess"])
        or (manifest.postprocessing_policy or {}).get("classification_checksum") != CONFIG_CHECKSUM
        or (manifest.tool_ref, manifest.implementation_ref)
        != (candidate.tool_ref, candidate.implementation_ref)
        or len(candidate.inputs) != 1
        or candidate.inputs[0].cardinality_min != 1
        or candidate.inputs[0].cardinality_max != 1
        or any(
            c.laterality.value not in {"UNKNOWN", "NOT_APPLICABLE"}
            for c in (candidate.output, candidate.inputs[0].constraint)
        )
        or candidate.intended_use.value != "research_only"
        or candidate.execution_policy.value != "research"
    ):
        raise AdmissionError("research_metadata_declaration_binding")
    return True
