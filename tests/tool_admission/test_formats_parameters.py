import struct

import pytest

from packages.tool_admission.inputs import image_header
from packages.tool_admission.parameters import validate_parameters
from tests.tool_admission.support import png


def test_png_npy_jpeg_bounded_headers():
    assert image_header(png(), 64) == ("image/png", (2, 3))
    body = b"{'descr': '|u1', 'fortran_order': False, 'shape': (2, 3)}\n"
    npy = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(body)) + body + b"\0" * 6
    assert image_header(npy, 1024) == ("application/x-npy", (2, 3))
    jpeg = (
        b"\xff\xd8\xff\xc0"
        + struct.pack(">H", 11)
        + b"\x08"
        + struct.pack(">HH", 2, 3)
        + b"\x01\x01\x11\x00"
    )
    assert image_header(jpeg, 1024) == ("image/jpeg", (2, 3))
    with pytest.raises(ValueError):
        image_header(npy[:-1], 1024)
    with pytest.raises(ValueError):
        image_header(npy.replace(b"|u1", b"|O1"), 1024)
    with pytest.raises(ValueError):
        image_header(npy, 16)


@pytest.mark.parametrize(
    "data", [b"", b"fake.png", b"\xff\xd8", b"\x89PNG\r\n\x1a\n", png()[:29] + b"bad!"]
)
def test_bad_headers(data):
    with pytest.raises(ValueError):
        image_header(data, 1024)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "additionalProperties": True, "properties": {}},
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"unused": {"$ref": "http://external"}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"unused": {"type": "string", "pattern": ".*"}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": ["missing"],
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"unused": {"type": "integer", "minimum": "zero"}},
        },
    ],
)
def test_unsupported_schema_never_silently_passes(schema):
    with pytest.raises(ValueError, match="schema_unsupported"):
        validate_parameters(schema, {})


def test_complete_schema_optional_defaults_and_nested_types():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["n"],
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 3},
            "optional": {
                "type": "array",
                "maxItems": 2,
                "items": {"type": "boolean"},
                "default": [],
            },
        },
    }
    value = {"n": 2}
    validate_parameters(schema, value)
    assert value == {"n": 2}
    for invalid in [{}, {"n": True}, {"n": 4}, {"n": 2, "optional": [1]}, {"n": 2, "unknown": 1}]:
        with pytest.raises(ValueError):
            validate_parameters(schema, invalid)
