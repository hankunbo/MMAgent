import json
from types import SimpleNamespace

import pytest

from packages.evaluation_preparation.direct_baseline import body_for, dispatch
from packages.llm_provider.transport import Reply
from packages.object_store import checksum


def config():
    return SimpleNamespace(transport=SimpleNamespace(model="synthetic"))


class Fake:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def send(self, *args, **kwargs):
        self.calls += 1
        return self.reply


def test_direct_once_and_scope(tmp_path):
    payload = dict(
        support_score_0_100=30.0,
        direction="uncertain",
        observations=[],
        limitations=[],
        reasoning="synthetic",
    )
    transport = Fake(
        Reply(content=json.dumps(payload).encode(), api_requests=1, response_confirmed=True)
    )
    result = dispatch(dict(input="new"), config(), tmp_path, transport=transport)
    assert result["status"] == "completed" and transport.calls == 1
    assert dispatch(dict(input="new"), config(), tmp_path, transport=None) == result
    with pytest.raises(ValueError, match="request_conflict"):
        dispatch(dict(input="another"), config(), tmp_path, transport=transport)


@pytest.mark.parametrize(
    "reply", [Reply(error="timeout", api_requests=None), Reply(content=b"{}", api_requests=1)]
)
def test_direct_failure_never_retries(tmp_path, reply):
    transport = Fake(reply)
    result = dispatch({}, config(), tmp_path, transport=transport)
    assert result["status"] == "failed"
    assert dispatch({}, config(), tmp_path, transport=transport) == result
    assert transport.calls == 1


def test_direct_unknown_claim(tmp_path):
    (tmp_path / "claim.json").write_text("{}")
    transport = Fake(Reply())
    with pytest.raises(FileExistsError):
        dispatch({}, config(), tmp_path, transport=transport)
    assert transport.calls == 0


def test_direct_visible_whitelist():
    binding = dict(
        image=dict(checksum=checksum(b"image")),
        demographics=dict(age="53", note="secret-note", glaucoma="hidden"),
    )
    body = body_for(binding, b"image", config())
    text = json.dumps(body)
    assert "secret-note" not in text and "hidden" not in text
    assert "53.0" in text and "image_url" in text
    with pytest.raises(ValueError, match="image_binding"):
        body_for(binding, b"other", config())
