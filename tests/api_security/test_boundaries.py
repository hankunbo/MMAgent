from dataclasses import replace
from types import SimpleNamespace

import pytest

from apps.api.assembly import Dependencies
from apps.api.auth import LocalIdentities, Principal
from packages.grounding.models import canonical
from packages.knowledge_claims.ports import load_llm_config
from packages.object_store import LocalObjectStore, checksum
from tests.api.support import HTTP, error


@pytest.mark.parametrize(
    "role",
    [
        "researcher",
        "read_only",
        "clinical_reviewer",
        "offline_reviewer",
        "tool_admin",
        "restricted_agent",
    ],
)
def test_http_roles_never_trust_header_or_cross_case(role):
    identity = Principal("SOURCE:ACTOR", role, frozenset({"CASE:ONE"}))
    ids = LocalIdentities({"local-test-token": identity})
    case = SimpleNamespace(
        case_code="CASE:ONE", encounter_code="CASE:ENCOUNTER", goals=[], laterality="UNKNOWN"
    )
    runtime = SimpleNamespace(get_case_bundle=lambda _: case)
    h = HTTP(Dependencies(ids, runtime=runtime), "local-test-token")
    r = h.get("/cases/CASE:ONE")
    if role in {"tool_admin", "restricted_agent"}:
        error(r, "SCOPE_FORBIDDEN", 403)
    else:
        assert r.status_code == 200, r.text
        error(h.get("/cases/CASE:OTHER"), "SCOPE_FORBIDDEN", 403)
    forged = h.get(
        "/capabilities",
        headers={"Authorization": "Bearer local-test-token", "X-Role": "researcher"},
    )
    assert forged.json()["data"]["identity"]["role"] == role
    empty = HTTP(Dependencies(LocalIdentities({})))
    error(empty.get("/capabilities"), "AUTHENTICATION_REQUIRED", 401)
    with pytest.raises(ValueError, match="local_identity"):
        replace(identity, role="superuser")


@pytest.mark.parametrize(
    "uri",
    [
        "/etc/passwd",
        "file:///etc/passwd",
        "https://example.org/image.png",
        "object://knowledge/objects/../private",
        "object://knowledge/objects/%2e%2e",
    ],
)
def test_no_arbitrary_object_paths(tmp_path, uri):
    store = LocalObjectStore(tmp_path / "objects")
    with pytest.raises(ValueError, match="managed object URI"):
        store.read_bounded(uri, 1024)


def test_bounded_objects_checksum_size_and_symlink(tmp_path):
    store = LocalObjectStore(tmp_path / "store")
    uri = store.stage(b"synthetic")
    target = store.promote(uri, checksum(b"synthetic"))
    assert store.read_bounded(target, 1024) == b"synthetic"
    with pytest.raises(ValueError, match="exceeds read bound"):
        store.read_bounded(target, 1)
    other = LocalObjectStore(tmp_path / "other")
    other_uri = other.stage(b"outside")
    fake = "object://knowledge/objects/" + "0" * 32
    store._path(fake).symlink_to(other._path(other_uri))
    with pytest.raises(OSError):
        store.read_bounded(fake, 1024)
    # A replaced managed directory is also rejected.
    store._path(target).parent.rename(tmp_path / "moved")
    (store.root / "objects").symlink_to(tmp_path / "moved", target_is_directory=True)
    with pytest.raises(OSError):
        store.read_bounded(target, 1024)


def test_config_override_and_secret_free_errors(monkeypatch):
    secret = "SYNTHETIC_KEY_CANARY_12D"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    config = load_llm_config(environ={"MMAGENT_LLM_MODEL": "synthetic-model"})
    assert config.model == "synthetic-model" and secret not in canonical(config).decode()
    with pytest.raises(ValueError) as captured:
        load_llm_config(
            environ={"MMAGENT_LLM_BASE_URL": "https://user:" + secret + "@example.org/v1"}
        )
    assert secret not in str(captured.value)

    class Broken:
        def get_case_bundle(self, _):
            raise OSError({"nested": {"Authorization": secret}})

    h = HTTP(
        Dependencies(
            LocalIdentities(
                {"reader": Principal("SOURCE:READER", "read_only", frozenset({"CASE:ONE"}))}
            ),
            runtime=Broken(),
        ),
        "reader",
    )
    r = h.get("/cases/CASE:ONE")
    error(r, "OWNER_FAILURE", 503)
    assert secret not in r.text and "Authorization" not in r.text
