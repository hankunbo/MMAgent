"""B02 bounded offline candidate adapter. No fetch, SQL, model or executable rules."""

import json
import re

from pydantic import ValidationError

from packages.knowledge_index.models import canonical, digest
from packages.knowledge_search.privacy import PrivacyError, safe_text

from .evidence import Rejected, require, validate_output
from .models import CandidateRequest, CandidateSourceBinding, CandidateSubmission, StructuredOutput
from .n124_models import N124CandidateRequest, N124StructuredOutput

RULE_CONFIG = {
    "version": "exact-candidate/1.0.0",
    "validation": "claim-validation/2.0.0",
    "privacy": "patient-free/3.0.0",
    "scope": "complete-conditional-clause/1.0.0",
    "max_request_bytes": 32768,
    "max_json_depth": 12,
    "max_json_nodes": 2048,
    "execution": "local-no-model",
}
RULE_CONFIG_CHECKSUM = digest(RULE_CONFIG)


def parse_candidate_request(value):
    """Revalidate even constructed models; reject duplicate keys and bounded nesting."""
    try:
        if isinstance(value, CandidateRequest):
            value = (
                N124CandidateRequest
                if isinstance(value, N124CandidateRequest)
                else CandidateRequest
            ).model_validate(dict(vars(value)))
            raw = canonical(value)
        else:
            raw = value
        require(type(raw) is bytes and 0 < len(raw) <= 32768, "candidate_request_limit")

        def unique(pairs):
            result = {}
            for key, item in pairs:
                require(key not in result, "candidate_duplicate_key")
                result[key] = item
            return result

        data = json.loads(raw, object_pairs_hook=unique)
        pending, nodes = [(data, 0)], 0
        while pending:
            item, depth = pending.pop()
            nodes += 1
            require(depth <= 12 and nodes <= 2048, "candidate_structure_limit")
            if isinstance(item, dict):
                pending.extend((v, depth + 1) for v in item.values())
            elif isinstance(item, list):
                pending.extend((v, depth + 1) for v in item)
            elif isinstance(item, str):
                # Newlines/tabs are canonical table separators, not patient de-identification.
                safe_text(item.replace("\n", " ").replace("\t", " "))
        model, expected = CandidateRequest, RULE_CONFIG_CHECKSUM
        if isinstance(data, dict) and data.get("schema_version") == "candidate-request/2.0.0":
            from .n124 import RULE_CONFIG_CHECKSUM as n124_checksum

            model, expected = N124CandidateRequest, n124_checksum
        request = model.model_validate_json(canonical(data))
        require(request.rule_config_checksum == expected, "candidate_rule_config")
        return request
    except Rejected:
        raise
    except PrivacyError:
        raise Rejected("candidate_privacy") from None
    except (ValidationError, ValueError, TypeError, UnicodeError, RecursionError):
        raise Rejected("candidate_schema") from None


def source_binding(evidence):
    return CandidateSourceBinding.model_validate_json(
        canonical(
            {
                "index_key": evidence.receipt["key"],
                "source": evidence.source.source_ref.model_dump(mode="json"),
                "source_version": evidence.source.reference().model_dump(mode="json"),
                "document": evidence.document.reference().model_dump(mode="json"),
                "chunk": evidence.chunk.reference().model_dump(mode="json"),
                "original_checksum": evidence.source.content_checksum,
                "parse_checksum": evidence.document.content_checksum,
                "chunk_checksum": evidence.chunk.content_checksum,
                "parse_uri": evidence.document.object_uri,
                "fidelity_policy": evidence.chunk.derivation.fidelity_policy,
                "position": evidence.chunk.derivation.position.model_dump(mode="json"),
            }
        )
    )


def evidence_hit(request):
    return binding_hit(request.source, request.candidate.quote)


def binding_hit(s, quote):
    return {
        "index_key": s.index_key,
        **{
            key: getattr(s, field).model_dump(mode="json")
            for key, field in (
                ("source_ref", "source"),
                ("source_version_ref", "source_version"),
                ("document_ref", "document"),
                ("chunk_ref", "chunk"),
            )
        },
        "fidelity_policy": s.fidelity_policy,
        "position": s.position.model_dump(mode="json"),
        "parse_ref": {
            "uri": s.parse_uri,
            "checksum": s.parse_checksum,
            "block_ordinal": s.position.ordinal,
        },
        "snippet": quote,
        "snippet_span": {
            "coordinate_system": "canonical_unicode",
            "start": s.position.span_start,
            "end": s.position.span_end,
            "truncated": False,
        },
    }


def validate_candidate(request, evidence):
    require(request.source == source_binding(evidence), "candidate_source_binding")
    if isinstance(request, N124CandidateRequest):
        return validate_output(request.candidate, evidence, policy_version="claim-validation/3.0.0")
    require(not re.search(r"[?？]", evidence.text), "candidate_question_unsupported")
    require(
        not any(
            re.search(
                r"\b(?:adults?|children|patients|if|when|unless|only|before|after)\b|成人|儿童|患者|如果|仅当|除非",
                title,
                re.I,
            )
            and title not in evidence.text
            for title in evidence.chunk.derivation.position.section_path
        ),
        "candidate_context_unsupported",
    )
    value = validate_output(request.candidate, evidence)
    require(
        not re.search(r"\b(?:provided|except|whenever|before|after|during)\b", evidence.text, re.I),
        "candidate_condition_unsupported",
    )
    for match in re.finditer(
        r"\b(?:if|when|unless|only)\b(?:(?!\.(?!\d)|[,;\n]).)*|(?:如果|仅当|除非)[^。；\n]*",
        evidence.text,
        re.I,
    ):
        require(
            any(match.group() in term for term in (*value.condition, *value.applicability)),
            "candidate_condition_incomplete",
        )
    return value


def submission_for(request, producer):
    return CandidateSubmission(
        producer=producer,
        request_key=request.request_key,
        request_checksum=digest(request),
        source_binding_checksum=digest(request.source),
        display_zh=request.display_zh,
    )


def batch_identity(input_checksum, config_checksum, submission=None):
    if submission is not None:
        return digest(
            {
                "namespace": "candidate-ingestion/1.0.0",
                "producer": submission.producer,
                "request_key": submission.request_key,
            }
        )
    return digest({"input": input_checksum, "config": config_checksum})


def verify_submission(evidence, outputs, cfg):
    """04B6 reconstructs the exact request from stored Claim and current controlled evidence."""
    require(len(outputs) == 1, "candidate_submission_binding")
    sub = cfg.submission
    n124 = cfg.version == "claim-config/4.0.0"
    if n124:
        from .n124 import RULE_CONFIG_CHECKSUM as expected_checksum
    else:
        expected_checksum = RULE_CONFIG_CHECKSUM
    model = N124CandidateRequest if n124 else CandidateRequest
    extra = (
        {"schema_version": "candidate-request/2.0.0", "rule_version": "nice-ng81-n124/1.0.0"}
        if n124
        else {}
    )
    request = parse_candidate_request(
        model(
            **extra,
            request_key=sub.request_key,
            rule_config_checksum=expected_checksum,
            source=source_binding(evidence),
            candidate=outputs[0],
            display_zh=sub.display_zh,
        )
    )
    validate_candidate(request, evidence)
    require(sub == submission_for(request, sub.producer), "candidate_submission_binding")
    require(
        cfg.extractor_version == ("2.0.0" if n124 else "1.0.0")
        and cfg.extractor_config_checksum == expected_checksum,
        "candidate_rule_config",
    )
    return request


def rule_output(request):
    model = N124StructuredOutput if isinstance(request, N124CandidateRequest) else StructuredOutput
    return canonical(model(candidates=(request.candidate,)))
