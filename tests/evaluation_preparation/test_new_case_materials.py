from types import SimpleNamespace

import pytest

from apps.vectorengine_single_case import load_config
from packages.evaluation_preparation.case_live import capability_basis
from packages.evaluation_preparation.case_materials import empty_model, template
from packages.evaluation_preparation.case_qc import components, geometry_quality
from packages.evaluation_preparation.direct_baseline import CONFIG
from packages.evidence_scoring.accepted import check_fixed_model
from tests.tool_execution.test_geometry_qc import mask_png


def png(pixels):
    return mask_png((3, 3), bytes(pixels))


def test_topology_not_disease_threshold():
    full = png([255] * 9)
    center = png([0, 0, 0, 0, 255, 0, 0, 0, 0])
    assert geometry_quality(full, center)["bbox_cdr_usable"]
    assert geometry_quality(full, png([0] * 9))["reasons"] == ["empty_cup"]
    disconnected = png([255, 0, 0, 0, 0, 0, 0, 0, 255])
    result = geometry_quality(full, disconnected)
    assert result["reasons"] == ["disconnected_cup"]
    assert result["masks_modified"] is False
    assert result["clinical_conclusion"] is None
    assert "cup_outside_disc" in geometry_quality(center, full)["reasons"]


def test_diagonal_eight_connected_and_empty():
    assert components([1, 0, 0, 1], (2, 2)) == [2]
    assert components([0] * 4, (2, 2)) == []


def test_only_structure_transferred_no_patient_answers():
    data = template()
    assert len(data["items"]) == 46
    assert all(
        set(row) == {"id", "node", "title", "kind", "role", "correlation_group", "dependence_block"}
        for row in data["items"]
    )
    row = next(row for row in data["items"] if row["role"] == "model")
    empty = empty_model(row, {}, [], [["missing"]])
    assert empty["judgment"]["value"] is None
    assert empty["weight"] is None and not empty["observation_available"]


def test_actual_provider_public_basis_and_changed_provider():
    config = load_config(CONFIG)
    assert capability_basis(config)["public_requests_repeated"] == 0
    with pytest.raises(ValueError, match="provider_capability_changed"):
        capability_basis(
            config.model_copy(
                update={"transport": config.transport.model_copy(update={"model": "changed"})}
            )
        )


def fixed_missing():
    row = next(row for row in template()["items"] if row["role"] == "model")
    fixed = empty_model(row, {}, [], [["missing"]])
    policy = SimpleNamespace(
        **{
            k: fixed[k]
            for k in (
                "id",
                "node",
                "kind",
                "role",
                "correlation_group",
                "dependence_block",
                "dependency_paths",
            )
        },
        applicable=True,
        fixed_model_row=fixed,
    )
    return policy, SimpleNamespace(status="unknown", value=None)


def test_missing_fixed_model_is_unscored_not_negative():
    policy, obs = fixed_missing()
    row = check_fixed_model(policy, obs)
    assert row.judgment.value is None and row.weight is None
    assert row.support == row.refutation == 0
    assert not row.observation_available


@pytest.mark.parametrize(
    "change",
    [
        "obtained_without_value",
        "ungradable",
        "out_of_scope",
        "not_applicable",
        "unknown_with_value",
        "nonmissing_rating",
    ],
)
def test_missing_fixed_model_does_not_weaken_other_guards(change):
    policy, obs = fixed_missing()
    if change == "obtained_without_value":
        obs.status = "obtained"
    if change == "ungradable":
        obs.status = "ungradable"
    if change == "out_of_scope":
        policy.node = "OTHER"
    if change == "not_applicable":
        policy.applicable = False
    if change == "unknown_with_value":
        obs.value = "invented output"
    if change == "nonmissing_rating":
        policy.fixed_model_row["observation"].update(value=0.5, missing_reason=None)
    with pytest.raises(ValueError):
        check_fixed_model(policy, obs)


def test_cli_serializes_summary_and_preserves_incomplete_exit(monkeypatch, capsys):
    import sys

    from apps import new_case_graph
    from packages.llm_provider.models import RequestSummary

    result = dict(
        complete=False,
        run_id="RUN:TEST",
        request_summary=RequestSummary(
            run_id="RUN:TEST",
            reservations=0,
            completed=0,
            pending_unknown=0,
            api_requests_observed=0,
            api_requests_unknown=0,
            simulated_attempts=0,
            usage_unknown=0,
        ),
    )
    monkeypatch.setattr(new_case_graph, "execute", lambda _: result)
    monkeypatch.setattr(sys, "argv", ["new_case_graph", "execute", "--sample", "data_00002.npz"])
    assert new_case_graph.main() == 2
    assert '"complete":false' in capsys.readouterr().out
