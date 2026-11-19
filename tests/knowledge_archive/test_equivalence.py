import pytest

from packages.knowledge_archive.equivalence import ASSET_HOST, compare_html


def body(token="a", text="No clinical inference.", path="same", picture="body.png"):
    return (
        f'<html><head><meta name="csrf-token" content="{token}">'
        f'<meta property="og:image" content="https://{ASSET_HOST}/{path}?'
        f'X-Amz-Date={token}&amp;X-Amz-Signature={token}&amp;type=png"></head>'
        f'<body><h1>Synthetic source</h1><p>{text}</p><img src="{picture}"></body></html>'
    ).encode()


def test_only_known_presentation_nonces_ignored():
    proof = compare_html(body(), body(token="b"))
    assert proof["policy"] == "egs-presentation-nonce/1.0.0"
    assert len(proof["ignored_fields"]) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"text": "Clinical inference."},
        {"path": "different"},
        {"picture": "other.png"},
        {"text": "Number is 13 not 12."},
    ],
)
def test_text_numbers_or_asset_identity_changes_rejected(change):
    with pytest.raises(ValueError, match="not_equivalent"):
        compare_html(body(), body(**change))
