"""Pure decoding reuse must not become current-input authorization."""

import pytest

from packages.dataset_adapters.multimodal import adapters, formats
from packages.dataset_adapters.reading import AdapterError
from tests.dataset_adapters.test_multimodal import grape, workbook
from tests.dataset_adapters.test_multimodal_preparation import rewrite_workbook


@pytest.fixture(autouse=True)
def clear_decode_cache():
    formats._sheet_values.cache_clear()
    yield
    formats._sheet_values.cache_clear()


def test_repeat_decode_returns_detached_values_and_one_parse(tmp_path, monkeypatch):
    path = tmp_path / "book.xlsx"
    workbook(path, {"Sheet": [{"A": "first", "AA": "0"}]})
    raw = path.read_bytes()
    parse = formats._decode_sheets
    calls = []

    def counted(value):
        calls.append(len(value))
        yield from parse(value)

    monkeypatch.setattr(formats, "_decode_sheets", counted)
    first = list(formats.sheets(raw))
    first[0][1][0][1]["A"] = "caller mutation"
    first[0][1].append((2, {"B": "not source"}))
    assert list(formats.sheets(raw)) == [("Sheet", [(1, {"A": "first", "AA": "0"})])]
    assert calls == [len(raw)]


def test_changed_bytes_same_file_require_new_decode(tmp_path):
    path = tmp_path / "book.xlsx"
    workbook(path, {"Sheet": [{"A": "first"}]})
    first = path.read_bytes()
    assert list(formats.sheets(first))[0][1][0][1]["A"] == "first"
    workbook(path, {"Sheet": [{"A": "second"}]})
    assert list(formats.sheets(path.read_bytes()))[0][1][0][1]["A"] == "second"
    assert formats._sheet_values.cache_info().currsize == 1


def test_bad_workbook_not_cached(tmp_path):
    path = tmp_path / "bad.xlsx"
    workbook(path, {"Sheet": [{"A": "x"}]})
    rewrite_workbook(path, lambda k, v: v.replace(b"<is>", b"<f>1+1</f><is>"))
    for _ in range(2):
        with pytest.raises(AdapterError, match="formula"):
            list(formats.sheets(path.read_bytes()))
    assert formats._sheet_values.cache_info().currsize == 0


def test_current_prepared_owner_detects_changed_header_after_cached_parse(tmp_path):
    root = grape(tmp_path)
    old = adapters.prepare("grape", root, "1:OD", cutoff_visit=0)
    assert old.recheck() == old
    path = root / "VF_and_clinical_information.xlsx"
    rewrite_workbook(path, lambda k, v: v.replace(b">Mean<", b">Unknown<"))
    with pytest.raises(AdapterError, match="headers"):
        old.recheck()


def test_current_prepared_owner_rejects_symlink_even_with_cached_bytes(tmp_path):
    root = grape(tmp_path / "case")
    old = adapters.prepare("grape", root, "1:OD", cutoff_visit=0)
    path = root / "VF_and_clinical_information.xlsx"
    target = tmp_path / "original.xlsx"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(AdapterError, match="unsafe_path"):
        old.recheck()


def test_mutable_input_uses_original_decoder(tmp_path):
    path = tmp_path / "book.xlsx"
    workbook(path, {"Sheet": [{"A": "x"}]})
    assert list(formats.sheets(bytearray(path.read_bytes()))) == list(
        formats.sheets(path.read_bytes())
    )
    assert formats._sheet_values.cache_info().misses == 1
