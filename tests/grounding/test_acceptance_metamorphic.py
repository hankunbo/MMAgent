"""Independent 05C seeded AND/OR, mutation and externally retained result checks."""

import random

import pytest

from packages.grounding import evaluate_query, validate
from packages.grounding.query_models import QueryResult
from packages.grounding.validation import MappingPaths
from tests.grounding.acceptance_support import small_graph
from tests.grounding.query_support import updated


@pytest.mark.parametrize("seed", range(12))
def test_seeded_chain_and_or_permutation_and_single_invalid_input(seed):
    request, views, reader, widths = small_graph(seed)
    current = views()
    result = evaluate_query(request, current)
    assert result.status == "candidates" and len(result.candidates) == widths[0]
    assert result.enumeration == "complete" and not result.product_demands
    assert all(len(c.dependencies) == 2 for c in result.candidates)
    rng = random.Random(seed + 500)
    permuted = list(current)
    rng.shuffle(permuted)
    permuted = [
        validate(MappingPaths, {**v.model_dump(mode="json"), "paths": list(reversed(v.paths))})
        for v in permuted
    ]
    changed = updated(
        request,
        artifacts=list(reversed(request.artifacts)),
        environment=list(reversed(request.environment)),
    )
    assert evaluate_query(changed, permuted) == result == evaluate_query(request, current)
    invalid = request.model_dump(mode="json")["artifacts"]
    invalid[0]["qc_status"] = "FAIL"
    rejected = evaluate_query(updated(request, artifacts=invalid), current)
    assert not rejected.candidates and rejected.status == "gaps"
    assert "qc_rejected" in {f.reason for f in rejected.gaps}
    assert rejected.clinical_effect == "NONE"
    # One failed root leaves every independent OR root intact, including nested options.
    reader.disable(current[0].record.candidates[0].implementation_ref)
    remaining = evaluate_query(request, views())
    assert len(remaining.candidates) == widths[0] - 1
    assert remaining.candidates == result.candidates[1:]
    assert not remaining.suggestions


@pytest.mark.parametrize("boundary", ["json", "copy", "construct"])
def test_retained_result_legacy_version_cannot_be_relabelled_by_checksum(boundary):
    request, views, _, _ = small_graph(9)
    result = evaluate_query(request, views())
    fields = result.model_dump(mode="json")
    fields["schema_version"] = "grounding-query-result/1.0.0"
    from packages.grounding import digest

    fields.pop("checksum")
    fields["checksum"] = digest(fields)
    raw = (
        fields
        if boundary == "json"
        else (
            result.model_copy(update=fields)
            if boundary == "copy"
            else QueryResult.model_construct(**fields)
        )
    )
    with pytest.raises(ValueError):
        validate(QueryResult, raw)


@pytest.mark.parametrize(
    "field",
    [
        "catalog_ref",
        "mapping_refs",
        "release_ref",
        "template_ref",
        "graph_template_ref",
        "knowledge_refs",
        "policy",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "latest"])
def test_each_lock_dimension_requires_explicit_exact_version(field, mutation):
    request, _, _, _ = small_graph(0)
    fields = request.lock.model_dump(mode="json")
    if mutation == "missing":
        del fields[field]
    else:
        target = fields[field][0] if field.endswith("_refs") else fields[field]
        target["version"] = "latest"
    from packages.grounding.locks import GroundingLock
    from tests.grounding.test_models import reseal

    with pytest.raises(ValueError):
        reseal(GroundingLock, fields)


def test_formal_query_public_import_does_not_load_models_agents_or_medgraph():
    import subprocess
    import sys

    source = """
import sys
import db.runtime.repository.grounding_query
blocked = {'openai', 'torch', 'transformers', 'agents', 'medgraph', 'cv2', 'Seg_Model'}
assert not ({name.split('.')[0] for name in sys.modules} & blocked)
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("field,value", [("version", "0.9.0"), ("checksum", "sha256:" + "f" * 64)])
def test_same_artifact_code_with_wrong_type_version_or_checksum_is_rejected(field, value):
    request, views, _, _ = small_graph(1)
    artifacts = request.model_dump(mode="json")["artifacts"]
    artifacts[0]["artifact_ref"][field] = value
    result = evaluate_query(updated(request, artifacts=artifacts), views())
    assert result.status == "gaps" and not result.candidates
    assert "artifact_type_mismatch" in {f.reason for f in result.gaps}
