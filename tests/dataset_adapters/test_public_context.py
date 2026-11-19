"""Age projection never leaks labels, original note, or source summaries."""

import json
from types import SimpleNamespace

import pytest

from packages.dataset_adapters.prepared import capture
from packages.dataset_adapters.public_context import project
from tests.dataset_adapters.support import fair
from tests.dataset_adapters.test_runtime_input import runtime_adapter


@pytest.mark.parametrize(
    "age,expected,status",
    [
        ("86", 86.0, "obtained"),
        ("0", 0.0, "obtained"),
        (None, None, "missing"),
        ("nan", None, "missing"),
        ("-1", None, "missing"),
        (True, None, "missing"),
        ("SECRET", None, "missing"),
    ],
)
def test_numeric_zero_missing_invalid_and_age_only(tmp_path, age, expected, status):
    old, _ = fair(tmp_path)
    sample = capture(runtime_adapter(old).prepare("data_00000.npz"))
    fields = sample.fields
    fields["npz"]["age"] = age
    snap = SimpleNamespace(
        dataset=sample.dataset,
        runtime_profile=True,
        fields=fields,
        fingerprint=sample.fingerprint,
        files=sample.files,
    )
    visible = SimpleNamespace(
        case=SimpleNamespace(
            case_code="CASE:TEST", artifacts=[SimpleNamespace(checksum="sha256:image")]
        ),
        checksum="sha256:visible",
    )
    value = project(snap, visible)
    assert value["age"]["value"] == expected and value["age"]["status"] == status
    assert value["interpretation"] is None and not value["hidden_fields_used"]
    assert "SECRET" not in json.dumps(value) and '"glaucoma"' not in json.dumps(value)
