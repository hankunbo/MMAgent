"""HTTP version dispatch and HTML use original synthetic graph/records/score owners."""

from types import SimpleNamespace

from apps.api.assembly import Dependencies
from apps.api.readers import AcceptedGraphReader
from packages.grounding.models import digest
from tests.api.support import HTTP, error, identities
from tests.path_graph_validation.support import accepted


def test_http_accepted_graph_records_scores_and_html_binding():
    h = accepted()
    reader = AcceptedGraphReader(h.f.acceptance, "path-source")
    receipt = h.f.acceptance.read(key="path-source")[-1]
    scope = h.parent.material.scope
    rt = SimpleNamespace(get_case_bundle=lambda code: SimpleNamespace(case_code=scope.case_code))
    deps = Dependencies(
        identities(cases=[scope.case_code]),
        runtime=rt,
        graphs={"case-evidence-graph/2.0.0": reader},
    )
    http = HTTP(deps)
    path = f"/cases/{scope.case_code}/graphs/{receipt['graph']['code']}"
    params = {"version": "case-evidence-graph/2.0.0"}
    before = list(h.f.documents.audit)
    versions = {
        "graph": "case-evidence-graph/2.0.0",
        "records": "case-observation-interpretation/2.1.0",
        "scores": "case-research-scoring/6.1.0",
    }
    for part, version in versions.items():
        result = http.get(path + "/" + part, params=params)
        assert result.status_code == 200, result.text
        assert result.json()["contract_version"] == version
        assert result.json()["data"]["schema_version"] == version
    result = http.get(path + "/html", params=params)
    assert result.status_code == 200, result.text
    value = reader.read(scope.case_code, receipt["graph"]["code"], params["version"])
    assert result.headers["x-source-checksum"] == digest(value)
    assert h.f.documents.audit == before
    error(
        http.get(path + "/graph", params={"version": "case-evidence-graph/99.0.0"}),
        "CAPABILITY_NOT_READY",
        503,
    )
    error(
        http.get(path.replace(scope.case_code, "CASE:OTHER") + "/records", params=params),
        "SCOPE_FORBIDDEN",
        403,
    )
    saved = h.f.documents.saved["accepted:path-source"]
    saved[2]["scores"]["checksum"] = digest("tampered")
    error(http.get(path + "/scores", params=params), "OWNER_FAILURE", 503)
