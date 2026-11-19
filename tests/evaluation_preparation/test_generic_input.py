"""Arbitrary synthetic IDs, no installed datasets/LLM/GPU/actual database."""

import json
from pathlib import Path

import pytest

from packages.dataset_adapters.reading import AdapterError
from packages.evaluation_preparation import generic_input as g
from tests.dataset_adapters.support import FAIR_FIELDS, csv_write, fair


def data(root, name):
    old, rows = fair(root)
    (root / "Dataset/Training/data_00000.npz").rename(root / "Dataset/Training" / name)
    rows[0]["filename"] = name
    csv_write(root / "ReadMe/data_summary.csv", FAIR_FIELDS, rows)
    return old


@pytest.mark.parametrize("name", ["arbitrary_A.npz", "data_06231.npz", "data_09999.npz"])
def test_any_indexed_training_identifier_no_two_case_history(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    data(tmp_path / "input", name)
    snap, checked = g.inspect_input("fairvlmed", tmp_path / "input", name)
    assert snap.runtime_profile and checked["image_shape"] == [2, 3]
    assert "SECRET" not in json.dumps(checked)
    spec = g.prepare(dataset="fairvlmed", root=tmp_path / "input", sample=name, key="synthetic")
    assert g.load("synthetic") == spec
    assert not list(tmp_path.rglob("binding.json"))
    assert not list(tmp_path.rglob("protocol.json"))
    assert (
        g.prepare(dataset="fairvlmed", root=tmp_path / "input", sample=name, key="synthetic")
        == spec
    )


def test_changed_source_and_same_key_conflict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data(tmp_path / "input", "arbitrary.npz")
    g.prepare(dataset="fairvlmed", root=tmp_path / "input", sample="arbitrary.npz", key="fixed")
    path = tmp_path / "input/ReadMe/data_summary.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="generic_input_changed"):
        g.load("fixed")


@pytest.mark.parametrize("key", ["../escape", "/root", "x/y", "", "a" * 65])
def test_safe_run_namespace(key):
    with pytest.raises(ValueError, match="run_key"):
        g.directory(key)


def test_run_symlink_not_followed(tmp_path):
    (tmp_path / "safe").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="run_path"):
        g.directory("safe", root=tmp_path)


def test_holdout_unknown_dataset_and_absent_input(tmp_path):
    data(tmp_path, "random.npz")
    for dataset, sample, split in [
        ("unknown", "random.npz", "training"),
        ("fairvlmed", "random.npz", "test"),
        ("fairvlmed", "absent.npz", "training"),
    ]:
        with pytest.raises(AdapterError):
            g.inspect_input(dataset, tmp_path, sample, split=split)


def test_generic_constructor_has_no_historical_campaign_dependency():
    import inspect

    from packages.evaluation_preparation.generic_campaign import GenericCampaign

    source = inspect.getsource(GenericCampaign.__init__)
    assert "super().__init__" not in source and "QualityCampaign" not in source
    c = object.__new__(GenericCampaign)
    c.spec = dict(key="test")
    assert c.prior()["reservations"] == 0
    assert c.phase_completed(dict(accepted=False, evidence_limited=True))
    assert not c.phase_completed(dict(accepted=False, evidence_limited=False))


def test_confirmed_failure_not_unknown(tmp_path):
    from packages.evaluation_preparation.generic_tools import confirmed_failure

    assert confirmed_failure("segman", tmp_path, "sha256:a") is None
    p = tmp_path / "segman-r03-process"
    p.mkdir()
    (tmp_path / "segman-r03-plan.json").write_text(json.dumps(dict(input_sha256="a")))
    (p / "process-claim.json").write_text(json.dumps(dict(argv=["owned"])))
    (p / "process-result.json").write_text(json.dumps(dict(argv=["owned"], exit_code=1)))
    assert confirmed_failure("segman", tmp_path, "sha256:a")["exit_code"] == 1
    with pytest.raises(ValueError, match="input_changed"):
        confirmed_failure("segman", tmp_path, "sha256:b")
    (p / "process-result.json").write_text(json.dumps(dict(argv=["owned"], exit_code=0)))
    assert confirmed_failure("segman", tmp_path, "sha256:a") is None


def test_cli_has_no_fixed_samples():
    import ast
    import inspect

    from apps.fairvlmed_case import main

    text = inspect.getsource(main)
    # Enumerating lock protocols must not be mistaken for hardcoded patient IDs.
    sample = [
        node
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--sample"
    ]
    assert len(sample) == 1 and not any(k.arg == "choices" for k in sample[0].keywords)
    assert "recover" in text and "export" in text
    assert Path("packages/dataset_adapters/schemas/BoundRegistration.schema.json").exists()
