"""Read-only Fake catalog bridge: independent fixture, never an 11C promotion."""

from datetime import UTC, datetime

from packages.grounding.models import digest, validate
from packages.grounding.synthetic import SyntheticCatalogReader

from .models import CatalogFixture


class SyntheticCatalogOwner:
    def __init__(self, *, reader, fixture):
        if type(reader) is not SyntheticCatalogReader:
            raise ValueError("extension_synthetic_catalog_owner_required")
        self.reader = reader
        self.fixture = validate(CatalogFixture, fixture)

    def current(self, adapter_record):
        f = validate(CatalogFixture, self.fixture)
        if f.expires_at <= datetime.now(UTC):
            raise ValueError("extension_catalog_fixture_expired")
        if digest(adapter_record) != f.adapter_checksum:
            raise ValueError("extension_catalog_adapter_changed")
        snapshot = self.reader.read_snapshot(f.catalog_ref)
        safety = self.reader.current_safety(f.catalog_ref)
        tool = next(
            (t for t in snapshot.tools if t.implementation_ref == f.implementation_ref), None
        )
        if (
            not snapshot.synthetic
            or tool is None
            or not tool.enabled
            or tool.catalog_status != "active"
            or tool.implementation_ref in safety.disabled_implementations
        ):
            raise ValueError("extension_catalog_unavailable_or_terminal")
        semantics = adapter_record["plan"]["manifest"]["candidate"]["semantics"]
        if (
            [i.model_dump(mode="json") for i in tool.inputs] != semantics["inputs"]
            or f.target_output_port not in tool.output_ports
            or tool.outputs[tool.output_ports.index(f.target_output_port)].model_dump(mode="json")
            != semantics["output"]
        ):
            raise ValueError("extension_catalog_contract_or_mapping_mismatch")
        return dict(
            snapshot=snapshot.model_dump(mode="json"), safety=safety.model_dump(mode="json")
        )
