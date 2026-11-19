from types import SimpleNamespace

import pytest

from packages.evaluation_preparation import fairclip_execution as entry


def test_new_case_original_service_keyword_key_no_mask_no_api(tmp_path, monkeypatch):
    calls = []
    visible = SimpleNamespace(
        case=SimpleNamespace(
            case_code="CASE:NEW", artifacts=[SimpleNamespace(checksum="sha256:" + "a" * 64)]
        )
    )
    request = SimpleNamespace(scope=SimpleNamespace(case_code="CASE:NEW"))

    class Service:
        def run_registered(self, actual, *, key, recover_only):
            assert actual is request
            calls.append((key, recover_only))
            return dict(terminal=True, model_outputs=[], synthetic=True)

    monkeypatch.setattr(
        entry, "register", lambda sample: (tmp_path, dict(case_code="CASE:NEW"), object(), object())
    )
    monkeypatch.setattr(entry, "DatasetInputRepository", lambda r: r)
    monkeypatch.setattr(
        entry, "VisibleInputReader", lambda *a: SimpleNamespace(read=lambda _: visible)
    )
    monkeypatch.setattr(entry, "credential_boundary", lambda: object())
    monkeypatch.setattr(
        entry.deployment,
        "load_configuration",
        lambda _: dict(installed_bundle="bundle", checksum="config"),
    )
    monkeypatch.setattr(
        entry.deployment, "read_installed_bundle", lambda _: dict(research_review="review")
    )
    monkeypatch.setattr(
        entry.deployment,
        "read_record",
        lambda _: (
            b'{"deployment_declaration":{"tool_code":"T","version":"1","implementation_key":"I"}}'
        ),
    )
    monkeypatch.setattr(
        entry.deployment,
        "assemble",
        lambda *a, **kw: SimpleNamespace(
            service=Service(),
            requests=SimpleNamespace(register=lambda *a: request, for_key=lambda _: request),
        ),
    )
    monkeypatch.setattr(entry, "write_once", lambda *a: None)
    monkeypatch.setattr(entry, "digest", lambda _: "sha256:" + "b" * 64)
    entry.execute("data_00002.npz", recover_only=True)
    assert calls == [("13-source-driven-fairclip-v1:data_00002.npz", True)]
    assert not (tmp_path / "measurements.json").exists()


def test_explicit_fairclip_migration_profile_not_global_readiness_relaxation():
    required = entry.deployment.required_migrations("runtime", profile="installed_fairclip/1.0.0")
    assert len(required) == 13 and "0012_research_runtime_context.sql" in required
    assert "0013_tool_record_capacity.sql" in required
    assert "0022_research_dispatch.sql" not in required
    assert "0022_research_dispatch.sql" in entry.deployment.required_migrations("runtime")
    with pytest.raises(ValueError, match="research_deployment_profile"):
        entry.deployment.required_migrations("runtime", profile="skip_checks")


def test_fresh_probe_observes_environment_then_reuses_exact_proof(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from packages.grounding.models import validate
    from packages.research_runtime import fairclip_probe as module
    from packages.tool_admission.models import DeploymentProfile

    installed = json.loads(
        Path(".local/07-j-fairclip-shared-gpu-v1/installed-bundle.json").read_bytes()
    )
    owner = module.CurrentFairCLIPProfiles.__new__(module.CurrentFairCLIPProfiles)
    owner.proof_path = tmp_path / "probe.json"
    owner.current = lambda: installed
    calls = []
    for name in ("check_capabilities", "verify_review", "gpu_ready"):
        monkeypatch.setattr(
            module.TrustedLocalProcessRunner, name, lambda *a: calls.append("verified")
        )
    profile = validate(DeploymentProfile, installed["profile"])
    manifest = SimpleNamespace(checksum=profile.manifest_checksum)

    def observed(command, **kwargs):
        calls.append("subprocess")
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert "MMAGENT_VECTORENGINE_API_KEY" not in kwargs["env"]
        assert "-B" in command
        assert "import torch,numpy,PIL;" in command[-1]
        assert "import clip" not in command[-1] and "PIL,clip" not in command[-1]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                dict(versions=profile.runtime.dependencies, executable=command[0])
            ).encode(),
        )

    monkeypatch.setattr(module.subprocess, "run", observed)
    first = owner.read_probe(manifest, profile)
    assert owner.read_probe(manifest, profile) == first
    assert calls.count("subprocess") == 1
    assert first.observed_at.isoformat() != installed["probe"]["observed_at"]
    assert (first.expires_at - first.observed_at).total_seconds() == 2700
    with pytest.raises(ValueError, match="fairclip_current_probe_binding"):
        owner.read_probe(SimpleNamespace(checksum="sha256:" + "c" * 64), profile)
