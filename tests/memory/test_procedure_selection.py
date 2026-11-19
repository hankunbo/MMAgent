"""Explicit metadata prefilter must not become cached Memory authorization."""

from types import SimpleNamespace

import pytest

from packages.memory.archive import WorkflowMemory

POLICY = "target-modality-prefilter/1.0.0"


def record(
    *,
    kind="tool_experience",
    case="CASE:SUPPORT",
    target="vertical_cdr",
    modality="SLO",
    tool="cdr_metrics",
    checksum="sha256:current",
):
    return SimpleNamespace(
        record=SimpleNamespace(
            kind=kind,
            case_code=case,
            detail=SimpleNamespace(
                target=target,
                modality=modality,
                steps=[
                    SimpleNamespace(
                        operation="measure",
                        tool=tool,
                        input_roles=("mask",),
                        output_roles=("ratio",),
                    )
                ],
            ),
        ),
        checksum=checksum,
        state="candidate",
    )


class Catalog:
    def __init__(self, *, keys=("matching",), records=None, failure=None):
        self.keys = keys
        self.records = records or {key: record() for key in keys}
        self.failure = failure
        self.calls = []

    def archive_keys(self):
        self.calls.append(("archive",))
        return ("unrelated",)

    def procedure_keys(self, **kwargs):
        self.calls.append(("prefilter", kwargs))
        return self.keys

    def read(self, key):
        self.calls.append(("read", key))
        if self.failure is not None:
            raise self.failure
        return self.records[key]


def suggest(memory, **kwargs):
    return memory.suggest(
        target="vertical_cdr",
        modality="SLO",
        legal_tools=("cdr_metrics",),
        query_case="CASE:QUERY",
        **kwargs,
    )


def test_prefilter_finds_matching_procedure_outside_global_prefix():
    catalog = Catalog()
    result = suggest(WorkflowMemory(catalog), candidate_policy=POLICY)
    assert catalog.calls == [
        ("prefilter", dict(targets=("vertical_cdr",), modality="SLO", limit=100)),
        ("read", "matching"),
    ]
    assert [row["key"] for row in result["matches"]] == ["matching"]
    assert result["candidate_selection"] == dict(
        policy=POLICY,
        candidate_limit=100,
        candidates_read=1,
        candidate_limit_reached=False,
        metadata_is_authorization=False,
    )
    hint = result["matches"][0]
    assert hint["execution_authorized"] is False and hint["clinical_evidence"] is False
    assert hint["reuse"] == "procedure_only_recompute_values"
    assert hint["publication_state"] == "candidate"
    assert "value" not in hint and "artifact" not in hint


def test_legacy_default_preserves_archive_catalog_and_result_shape():
    catalog = Catalog(records={"unrelated": record(kind="graph_snapshot")})
    result = suggest(WorkflowMemory(catalog))
    assert catalog.calls == [("archive",), ("read", "unrelated")]
    assert result == dict(matches=[], rejected=[], clinical_evidence=False)


@pytest.mark.parametrize(
    "change,reason",
    [
        (dict(case="CASE:QUERY"), "same_case_not_external_experience"),
        (dict(target="other"), "target_or_modality_mismatch"),
        (dict(modality="OCT"), "target_or_modality_mismatch"),
        (dict(tool="uninstalled"), "current_tool_not_admitted"),
    ],
)
def test_prefilter_metadata_does_not_replace_current_record_checks(change, reason):
    catalog = Catalog(records={"matching": record(**change)})
    result = suggest(WorkflowMemory(catalog), candidate_policy=POLICY)
    assert result["matches"] == []
    assert result["rejected"] == [dict(key="matching", reason=reason)]
    assert ("read", "matching") in catalog.calls


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("memory_source_withdrawn"),
        ValueError("memory_read_changed"),
        RuntimeError("source_database_unavailable"),
    ],
)
def test_original_owner_failures_remain_failures_not_empty_memory(failure):
    catalog = Catalog(failure=failure)
    with pytest.raises(type(failure), match=str(failure)):
        suggest(WorkflowMemory(catalog), candidate_policy=POLICY)
    assert catalog.calls[-1] == ("read", "matching")


def test_empty_prefilter_performs_no_record_read():
    catalog = Catalog(keys=())
    result = suggest(WorkflowMemory(catalog), candidate_policy=POLICY)
    assert result["matches"] == [] and result["rejected"] == []
    assert result["candidate_selection"]["candidates_read"] == 0
    assert len(catalog.calls) == 1


def test_second_call_revalidates_and_never_reuses_old_qualification():
    catalog = Catalog()
    memory = WorkflowMemory(catalog)
    first = suggest(memory, candidate_policy=POLICY)
    catalog.records["matching"] = record(checksum="sha256:changed")
    second = suggest(memory, candidate_policy=POLICY)
    assert first["matches"][0]["checksum"] != second["matches"][0]["checksum"]
    assert catalog.calls.count(("read", "matching")) == 2


def test_unknown_policy_rejected_before_any_owner_read():
    catalog = Catalog()
    with pytest.raises(ValueError, match="memory_candidate_policy"):
        suggest(WorkflowMemory(catalog), candidate_policy="unbounded-all")
    assert catalog.calls == []


def test_output_limit_does_not_bypass_current_checks_or_hide_catalog_bound():
    keys = tuple(f"matching-{i:03}" for i in range(100))
    catalog = Catalog(keys=keys)
    result = suggest(WorkflowMemory(catalog), candidate_policy=POLICY, limit=1)
    assert len(result["matches"]) == 1
    assert len([call for call in catalog.calls if call[0] == "read"]) == 100
    assert result["candidate_selection"]["candidate_limit_reached"] is True


@pytest.mark.parametrize("keys", [("same", "same"), ("",), tuple(str(i) for i in range(101))])
def test_invalid_or_unbounded_catalog_rejected_before_current_reads(keys):
    catalog = Catalog(keys=keys)
    with pytest.raises(ValueError, match="memory_candidate_catalog"):
        suggest(WorkflowMemory(catalog), candidate_policy=POLICY)
    assert len(catalog.calls) == 1 and catalog.calls[0][0] == "prefilter"
