"""Explicit synthetic fixture bytes and registry; no claim of real official identity."""

from packages.grounding.models import digest
from packages.tool_discovery import Provenance, RegisteredIdentity, SyntheticSource
from packages.tool_discovery.discovery import contract


def record(h, *, kind="local_contract", name="synthetic", **changes):
    data = dict(
        record_id=name,
        source_kind=kind,
        source_url="fixture://official/" + name,
        identity_ref="fixture:identity",
        repository="fixture://official/repository",
        repository_owner="synthetic-owner",
        model_id="synthetic-model",
        revision="fixture-v1",
        revision_kind="immutable_version",
        claimed_official=True,
        code_license="synthetic-test-license",
        weights_license="synthetic-test-license",
        code_checksum=digest("synthetic code bytes"),
        weights_checksum=digest("synthetic weight bytes"),
        model_card="Synthetic card; no medical evaluation",
        framework="fake",
        inference_entry="never-executed",
        semantics=contract(h.request.candidate.model_dump(mode="json")),
        input_shape=("fixture",),
        output_shape=("fixture",),
        preprocessing="synthetic identity",
        postprocessing="synthetic identity",
        memory_mb=32,
        requires_gpu=False,
        training_data="synthetic only",
        evaluation_data="synthetic declaration only",
        applicability="fixture contract only",
        limitations="not a medical model",
    )
    data.update(changes)
    return Provenance(**data)


def source(
    records, *, name="local", kind="local_contract", technical_failure=False, proof_record=None
):
    original = proof_record or (records[0] if records else None)
    proofs = (
        ()
        if original is None
        else (
            RegisteredIdentity(
                identity_ref="fixture:identity",
                repository="fixture://official/repository",
                repository_owner="synthetic-owner",
                model_id="synthetic-model",
                revision="fixture-v1",
                code_checksum=digest("synthetic code bytes"),
                weights_checksum=digest("synthetic weight bytes"),
                source_urls=tuple(r.source_url for r in records),
                official=True,
            ),
        )
    )
    return SyntheticSource(
        name, kind, records, identities=proofs, technical_failure=technical_failure
    )
