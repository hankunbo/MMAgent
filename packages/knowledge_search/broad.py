"""Versioned, patient-free glaucoma discovery; preferences are not hard filters.

This is a deliberately broader *discovery* query, not a clinical decision or a
replacement for the historical v4 query template. Only fixed general concepts
can cross the connector boundary. No case text, observations or identifiers are
accepted by the builder.
"""

import json

from packages.knowledge_connectors.ports import digest

from .privacy import CATALOG

TOPICS = {
    "cup_disc_ratio": ("cup disc ratio", "cup-to-disc ratio"),
    "rim": ("neuroretinal rim", "rim thinning", "rim notching"),
    "rnfl": ("retinal nerve fiber layer", "rnfl"),
    "gcc": ("ganglion cell complex", "ganglion cell layer", "gcc"),
    "disc_hemorrhage": ("disc hemorrhage", "disc haemorrhage"),
    "disc_size": ("optic disc size", "optic disc area"),
    "progression": ("glaucoma progression", "longitudinal change"),
    "myopia": ("myopia", "myopic optic disc"),
    "vessels": ("retinal vessels", "retinal vascular density", "retinal microvasculature"),
    "tortuosity": ("vascular tortuosity", "vessel tortuosity"),
    "fractal": ("fractal dimension",),
    "visual_field": ("visual field", "perimetry"),
    "iop": ("intraocular pressure", "tonometry"),
    "cct": ("central corneal thickness", "pachymetry"),
    "gonioscopy": ("gonioscopy", "anterior chamber angle"),
    "slo": ("scanning laser ophthalmoscopy", "slo"),
    "oct": ("optical coherence tomography", "oct"),
    "ai": ("artificial intelligence", "deep learning", "machine learning"),
    "fairness": ("algorithmic fairness", "demographic bias", "health equity"),
    "age": ("age", "aging"),
    "family_history": ("family history", "heredity"),
    "differential": ("differential diagnosis", "glaucoma mimics"),
}
BROAD_CATALOG = {
    **CATALOG,
    "source_version": "2.0.0",
    "policy_version": "patient-free/4.0.0",
    "concepts": {
        "CONCEPT:DISEASE:GLAUCOMA": ["glaucoma"],
        **{"CONCEPT:GLAUCOMA:" + k.upper(): sorted(v) for k, v in TOPICS.items()},
    },
}
BROAD_CHECKSUM = digest(
    json.dumps(BROAD_CATALOG, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
)
MODE = "broad_terms_preferences_unfiltered"


def template_program():
    """Shared expression tree consumed by Python and the PostgreSQL validator."""
    groups = [
        "map",
        ["flatten", ["array", ["get", "p", "diseases"], ["get", "p", "concepts"]]],
        "c",
        [
            "concat",
            "(",
            [
                "join",
                [
                    "map",
                    [
                        "flatten",
                        ["array", ["array", ["get", "c", "label"]], ["get", "c", "synonyms"]],
                    ],
                    "t",
                    ["concat", '"', ["get", "t"], '"'],
                ],
                " OR ",
            ],
            ")",
        ],
    ]
    return {
        "version": "3.0.0",
        "mode": ["if", ["eq", ["get", "adapter"], "guideline"], "catalog_title_only", MODE],
        "texts": [
            "if",
            ["eq", ["get", "adapter"], "guideline"],
            ["array", "glaucoma"],
            [
                "array",
                [
                    "if",
                    ["eq", ["get", "adapter"], "crossref"],
                    [
                        "join",
                        [
                            "flatten",
                            [
                                "map",
                                [
                                    "flatten",
                                    ["array", ["get", "p", "diseases"], ["get", "p", "concepts"]],
                                ],
                                "c",
                                [
                                    "flatten",
                                    [
                                        "array",
                                        ["array", ["get", "c", "label"]],
                                        ["get", "c", "synonyms"],
                                    ],
                                ],
                            ],
                        ],
                        " ",
                    ],
                    ["join", groups, " AND "],
                ],
            ],
        ],
    }


def broad_plan(provider_codes, *, topics=(), limit=10, max_pages=1, timeout_ms=30000):
    """Build v5 discovery from public topic keys; no caller free-text field.

    Multiple topics are AND requirements; synonyms within each topic are OR.
    For broad coverage, call once per topic rather than ANDing all topics.
    Language/source-type/population fields describe preferences, NOT filtering.
    """
    from .models import CodedTerm, KnowledgeQueryPlan, PatientFreePolicy, ProviderBudget

    if not isinstance(topics, (tuple, list)) or any(t not in TOPICS for t in topics):
        raise ValueError("unknown_public_discovery_topic")
    if len(set(topics)) != len(topics):
        raise ValueError("duplicate_public_discovery_topic")
    return KnowledgeQueryPlan(
        query_plan_version="5.0.0",
        template_version="3.0.0",
        privacy=PatientFreePolicy(
            version="patient-free/4.0.0", checksum=BROAD_CHECKSUM, term_version="2.0.0"
        ),
        goal="GOAL:KNOWLEDGE:EVIDENCE",
        claim="CLAIM:KNOWLEDGE:ASSOCIATION",
        evidence_need="association",
        diseases=(
            CodedTerm(
                terminology_version="2.0.0", code="CONCEPT:DISEASE:GLAUCOMA", label="glaucoma"
            ),
        ),
        concepts=tuple(
            CodedTerm(
                terminology_version="2.0.0",
                code="CONCEPT:GLAUCOMA:" + t.upper(),
                label=TOPICS[t][0],
                synonyms=TOPICS[t][1:],
            )
            for t in topics
        ),
        population=("research cohort",),
        source_types=("guideline", "original_research", "systematic_review"),
        providers=tuple(
            ProviderBudget(
                provider_code=code,
                page_size=limit,
                max_hits=limit,
                max_pages=max_pages,
                max_requests=2 * max_pages + 2,
                timeout_ms=timeout_ms,
                max_retries=0,
            )
            for code in provider_codes
        ),
        total_max_hits=min(200, limit * len(provider_codes)),
        total_max_requests=min(256, (2 * max_pages + 2) * len(provider_codes)),
        total_timeout_ms=min(600000, max(60000, timeout_ms)),
        max_concurrency=min(3, len(provider_codes)),
    )
