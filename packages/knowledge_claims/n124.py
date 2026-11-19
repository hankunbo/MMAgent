"""One explicitly selected NICE snapshot/clause, never general language inference.

Content hashes identify immutable source bytes, not review or publication authority.
The repository must still load both current source chains before using this rule.
"""

from bs4 import BeautifulSoup

from packages.knowledge_index.fidelity import text as html_text
from packages.knowledge_index.models import ParsePolicy, canonical, digest
from packages.knowledge_index.parsers import parse
from packages.object_store import checksum

from .evidence import require
from .models import N124ValidationPolicyBinding, Span
from .n124_models import N124CandidateRequest, N124DraftOutput, N124StructuredOutput

RULE_VERSION = "nice-ng81-n124/1.0.0"
POLICY_VERSION = "claim-validation/3.0.0"
BODY = (
    "Obtain an optic nerve head image at diagnosis for baseline documentation "
    "(for example, a stereoscopic optic nerve head image or OCT)."
)
UPDATE = "[2009, amended 2017]"
QUOTE = BODY + " " + UPDATE
EXAMPLES = "(for example, a stereoscopic optic nerve head image or OCT)"
SCOPE = (
    "This guideline covers diagnosing and managing glaucoma in people aged 18 and over. "
    "It includes recommendations on testing and referral (case-finding) for chronic open-angle "
    "glaucoma and ocular hypertension and on effective diagnosis, treatment and reassessment "
    "to stop these conditions progressing."
)
URL = "https://www.nice.org.uk/guidance/ng81/chapter/Recommendations"
OVERVIEW_URL = "https://www.nice.org.uk/guidance/ng81"
ORIGINAL = "sha256:2bcf6cc2147ce8f986d58b99b86a164fd3cf4bd53f4e99771b45270f8abcdb26"
OVERVIEW_ORIGINAL = "sha256:500f3f1c8a1c032a40a09294aeca1bc3372493bc0f6a11888fdb144b72055890"
SPEC = {
    "version": POLICY_VERSION,
    "numeric_version": "n124-document-metadata/1.0.0",
    "polarity_version": "n124-exact-clause/1.0.0",
    "body": BODY,
    "update": UPDATE,
    "examples": EXAMPLES,
    "overview_scope": SCOPE,
    "sources": {URL: ORIGINAL, OVERVIEW_URL: OVERVIEW_ORIGINAL},
    "anchor": "ng81-1_2_4",
    "recommendation": "1.2.4",
    "required": (
        "complete_paragraph",
        "canonical_heading",
        "original_html_anchor",
        "reparse_both_originals",
        "current_source_bindings",
        "exact_all_fields",
        "external_scope_pending_review",
        "retain_age_min_18",
        "non_exhaustive_examples",
    ),
}
POLICY_CHECKSUM = digest(SPEC)
RULE_CONFIG = {
    "version": RULE_VERSION,
    "validation_checksum": POLICY_CHECKSUM,
    "privacy": "patient-free/3.0.0",
    "max_request_bytes": 32768,
    "max_json_depth": 12,
    "max_json_nodes": 2048,
    "execution": "local-no-model",
}
RULE_CONFIG_CHECKSUM = digest(RULE_CONFIG)


def policy_binding():
    return N124ValidationPolicyBinding(
        version=POLICY_VERSION,
        numeric_version=SPEC["numeric_version"],
        polarity_version=SPEC["polarity_version"],
        checksum=POLICY_CHECKSUM,
    )


def check_original(evidence, expected_checksum, url):
    require(
        evidence.source.content_checksum == expected_checksum
        and checksum(evidence.original) == expected_checksum,
        "n124_source_snapshot",
    )
    metadata = evidence.source.archive_metadata
    require(
        metadata is not None and metadata.bibliography.canonical_url == url,
        "n124_source_identity",
    )
    pp = ParsePolicy.model_validate_json(
        canonical(evidence.receipt["derivation_config"]["parse_policy"])
    )
    result = parse(evidence.original, "text/html", language=evidence.parsed.language, policy=pp)
    require(
        result.status == "parsed" and result.document == evidence.parsed, "n124_reparse_binding"
    )


def expected_output(evidence):
    """Derive exact fields from already controlled evidence, with no caller approval flags."""
    from .ingestion import source_binding

    scope = evidence.source_scope
    require(scope is not None, "n124_external_scope_required")
    check_original(evidence, ORIGINAL, URL)
    check_original(scope, OVERVIEW_ORIGINAL, OVERVIEW_URL)
    p = evidence.block.position
    require(
        p.kind == "paragraph"
        and evidence.text == evidence.block.text == QUOTE
        and evidence.chunk.derivation.position == p,
        "n124_structure_unsupported",
    )
    require(
        scope.text == scope.block.text == SCOPE
        and scope.block.position.kind == "paragraph"
        and scope.chunk.derivation.position == scope.block.position,
        "n124_scope_binding",
    )
    headings = [b for b in evidence.parsed.blocks if b.position.ordinal == p.ordinal - 1]
    require(
        len(headings) == 1
        and headings[0].text == "1.2.4"
        and headings[0].position.kind == "heading"
        and p.section_path
        == ("Glaucoma: diagnosis and management", "Recommendations", "1.2 Diagnosis", "1.2.4")
        and headings[0].position.section_path == p.section_path,
        "n124_heading_binding",
    )
    soup = BeautifulSoup(evidence.original, "html.parser")
    anchors = soup.find_all(id="ng81-1_2_4")
    require(len(anchors) == 1, "n124_anchor_binding")
    anchor = anchors[0]
    require(
        anchor.name == "article"
        and len(anchor.find_all("h4")) == 1
        and html_text(anchor.h4) == "1.2.4"
        and len(anchor.find_all("p")) == 1
        and html_text(anchor.p) == QUOTE
        and len(anchor.find_all("strong")) == 1
        and html_text(anchor.strong) == UPDATE
        and not anchor.find_all(["li", "table"]),
        "n124_anchor_binding",
    )

    def span(start, end):
        return Span(start=start, end=end)

    start = p.span_start
    age_start = scope.block.position.span_start + SCOPE.index("18")
    data = dict(
        output_version="nice-ng81-n124-output/1.0.0",
        claim_type="exam_requirement",
        statement=QUOTE,
        target_proposition=QUOTE,
        polarity="POSITIVE",
        uncertainty=(),
        recommendation_strength=(),
        subject="optic nerve head",
        population=(),
        disease_stage=("at diagnosis",),
        condition=("at diagnosis", "for baseline documentation"),
        intervention=(),
        exam=("optic nerve head image",),
        modality=(),
        outcome=("for baseline documentation",),
        time=("at diagnosis",),
        laterality=(),
        device=(),
        applicability=(),
        numbers=(),
        quote=QUOTE,
        span=span(start, p.span_end),
        confidence=1.0,
        metadata=dict(
            version="nice-ng81-n124-metadata/1.0.0",
            anchor="ng81-1_2_4",
            recommendation="1.2.4",
            recommendation_span=span(
                headings[0].position.span_start, headings[0].position.span_end
            ),
            semantic_text=BODY,
            semantic_span=span(start, start + len(BODY)),
            update_text=UPDATE,
            update_span=span(start + len(BODY) + 1, p.span_end),
            examples_text=EXAMPLES,
            examples_span=span(
                start + BODY.index(EXAMPLES), start + BODY.index(EXAMPLES) + len(EXAMPLES)
            ),
            source_scope=dict(
                source=source_binding(scope),
                quote=SCOPE,
                span=span(scope.block.position.span_start, scope.block.position.span_end),
                population_text="people aged 18 and over",
                age=dict(
                    value="18",
                    unit=None,
                    comparator=None,
                    quote="18",
                    span=span(age_start, age_start + 2),
                ),
                assessment="pending_review",
            ),
        ),
    )
    return N124DraftOutput.model_validate(data)


def validate_n124_output(value, evidence):
    value = N124DraftOutput.model_validate(dict(vars(value)))
    require(value == expected_output(evidence), "n124_exact_field_binding")
    return value


def build_n124_request(evidence, *, request_key, display_zh=None):
    """Public producer helper; ingest_candidate independently reloads all evidence."""
    from .ingestion import source_binding

    return N124CandidateRequest(
        schema_version="candidate-request/2.0.0",
        rule_version=RULE_VERSION,
        request_key=request_key,
        rule_config_checksum=RULE_CONFIG_CHECKSUM,
        source=source_binding(evidence),
        candidate=expected_output(evidence),
        display_zh=display_zh,
    )


def output_model(configuration):
    from .models import StructuredOutput

    return (
        N124StructuredOutput if configuration.version == "claim-config/4.0.0" else StructuredOutput
    )
