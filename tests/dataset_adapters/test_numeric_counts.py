import math
import struct
from concurrent.futures import ThreadPoolExecutor

import pytest

from packages.dataset_adapters.multimodal import analysis, formats
from packages.dataset_adapters.multimodal.adapters import prepare
from packages.dataset_adapters.reading import AdapterError
from tests.dataset_adapters.support import npy as encoded
from tests.dataset_adapters.test_multimodal import gdp


@pytest.fixture(autouse=True)
def empty_counts():
    formats._numeric_counts.clear()
    yield
    formats._numeric_counts.clear()


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize(
    "dtype,code,values",
    [
        ("f4", "f", [0.0, -0.0, 1.0, -1.0, float("nan"), float("inf"), -float("inf"), 1e-30]),
        ("f8", "d", [0.0, -0.0, 1.0, -1.0, float("nan"), float("inf"), -float("inf"), 1e-300]),
        ("i4", "i", [0, -1, 2147483647, -2147483648]),
        ("i8", "q", [0, -1, 9223372036854775807, -9223372036854775808]),
        ("u1", "B", [0, 1, 128, 255]),
    ],
)
def test_exact_counts_match_original_iterator(endian, dtype, code, values):
    raw = encoded(
        (len(values),), endian + dtype, struct.pack(endian + str(len(values)) + code, *values)
    )
    value = formats.npy(raw)
    expected = (sum(math.isfinite(x) for x in value.numbers()), len(values))
    assert value.finite_counts() == expected
    assert formats.npy(raw).finite_counts() == expected


def test_content_and_dtype_distinguish_cache_and_empty_is_valid():
    finite = encoded((1,), "<f4", struct.pack("<f", 1.0))
    unknown = encoded((1,), "<f4", struct.pack("<f", float("nan")))
    assert formats.npy(finite).finite_counts() == (1, 1)
    assert formats.npy(unknown).finite_counts() == (0, 1)
    as_integer = encoded((1,), "<i4", formats.npy(unknown).data)
    assert formats.npy(as_integer).finite_counts() == (1, 1)
    assert formats.npy(encoded((0,), "<f8", b"")).finite_counts() == (0, 0)
    assert len(formats._numeric_counts) == 4


def test_cache_skips_only_numeric_iteration_not_payload_validation(monkeypatch):
    raw = encoded((2,), "<f8", struct.pack("<2d", 1.0, float("nan")))
    assert formats.npy(raw).finite_counts() == (1, 2)

    def forbidden(_):
        raise AssertionError("identical payload was recounted")

    monkeypatch.setattr(formats, "_count_finite", forbidden)
    assert formats.npy(raw).finite_counts() == (1, 2)
    with pytest.raises(AdapterError, match="npy_size"):
        formats.npy(raw[:-1])
    with pytest.raises(AdapterError, match="numeric_dtype"):
        formats.npy(encoded((1,), "|S1", b"a")).finite_counts()


def test_cache_is_metadata_only_bounded_and_thread_safe(monkeypatch):
    monkeypatch.setattr(formats, "_NUMERIC_COUNTS_LIMIT", 3)
    values = [formats.npy(encoded((1,), "<f8", struct.pack("<d", n))) for n in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda a: a.finite_counts(), values * 2)) == [(1, 1)] * 24
    assert len(formats._numeric_counts) == 3
    for key, result in formats._numeric_counts.items():
        assert isinstance(key[0], str) and isinstance(key[1], int)
        assert key[2].startswith("sha256:") and len(key[2]) == 71
        assert result == (1, 1)
        assert not any(isinstance(x, bytes) for x in key)


def test_whole_plan_exact_and_changed_payload_still_rejected(tmp_path):
    import importlib.util
    from pathlib import Path

    from packages.object_store import checksum

    baseline = Path(
        "docs/acceptance/capability-scope-text-installed-20261108/"
        "before/packages/dataset_adapters/multimodal/analysis.py"
    )
    assert checksum(baseline.read_bytes()) == (
        "sha256:2fe947f4ec3b573c91688071edc03de0b12cc0b01fcdebf5ba0cecbd5fd24ad3"
    )
    spec = importlib.util.spec_from_file_location(
        "packages.dataset_adapters.multimodal.numeric_audit_baseline", baseline
    )
    numeric_audit_baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(numeric_audit_baseline)

    sample = prepare("harvard_gdp", gdp(tmp_path, nonfinite=True), "data_0001.npz")
    payloads = dict(sample.payload_bytes)
    old = numeric_audit_baseline.build(sample.manifest, payloads).model_dump_json()
    assert analysis.build(sample.manifest, payloads).model_dump_json() == old
    assert analysis.build(sample.manifest, payloads).model_dump_json() == old
    key = next(iter(payloads))
    payloads[key] = payloads[key][:-1] + bytes([payloads[key][-1] ^ 1])
    with pytest.raises(AdapterError, match="payload_checksum"):
        analysis.build(sample.manifest, payloads)


def test_warm_counts_do_not_override_current_component_scope(tmp_path):
    sample = prepare("harvard_gdp", gdp(tmp_path), "data_0001.npz")
    payloads = dict(sample.payload_bytes)
    descriptors = {p.id: p for p in sample.manifest.payloads}
    component = next(c for c in sample.manifest.components if c.id == "rnflt:rnflt")
    warm = analysis.audit_component(component, payloads, descriptors)
    assert warm.acquisition == 1
    missing = component.model_copy(update=dict(status="missing", reason="current_input_missing"))
    absent = analysis.audit_component(missing, payloads, descriptors)
    assert absent.acquisition == 0 and absent.finite_values is None
    changed = component.model_copy(update=dict(shape=(1,)))
    with pytest.raises(AdapterError, match="array_binding"):
        analysis.audit_component(changed, payloads, descriptors)


def test_failed_computation_is_not_cached(monkeypatch):
    value = formats.npy(encoded((1,), "<f8", struct.pack("<d", 1.0)))
    calls = []

    def fail(values):
        calls.append(1)
        raise ValueError("injected_numeric_failure")

    monkeypatch.setattr(formats, "_count_finite", fail)
    for _ in range(2):
        with pytest.raises(ValueError, match="injected_numeric_failure"):
            value.finite_counts()
    assert calls == [1, 1]
    assert not formats._numeric_counts
