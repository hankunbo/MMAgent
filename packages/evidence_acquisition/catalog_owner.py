"""Adapter over original verified catalog/Knowledge readers, not a second repository."""

from packages.grounding.models import digest, validate
from packages.llm_provider.models import TaskDefinition
from packages.planner_control.models import Reference


class CatalogAcquisitionOwner:
    def __init__(self, reader):
        # RoundCaseEvidence already owns exact current SourceVersion/Chunk/object checks.
        self.reader = reader

    def _current_catalog(self):
        from apps.vectorengine_round_materials import load_catalog

        catalog = load_catalog(self.reader.catalog_path)
        if catalog != self.reader.catalog:
            raise ValueError("acquisition_catalog_changed")
        return catalog

    def read_source_ref(self, reference):
        self._current_catalog()
        sources = self.reader.read_sources()
        matches = [s for s in sources if s.reference == reference]
        # This whitelist is derived from actual catalog archive records, not code prefixes.
        archived_codes = {r["reference"]["code"] for r in self.reader.catalog["sources"]}
        if len(matches) != 1 or reference.code not in archived_codes:
            raise ValueError("acquisition_source_not_archived")
        return matches[0]

    def definition(self, target):
        from apps.vectorengine_round_materials import definitions

        catalog = self._current_catalog()
        rows = definitions(catalog)
        if target not in rows or not rows[target]["interpretation_installed"]:
            raise ValueError("acquisition_definition_not_installed")
        row = rows[target]
        sources = self.reader.read_sources()
        exact = {s.reference.code: s for s in sources}
        catalog_sources = {r["id"]: r["reference"]["code"] for r in catalog["sources"]}
        references = tuple(exact[catalog_sources[r["id"]]].reference for r in row["sources"])
        if not references:
            raise ValueError("acquisition_definition_reference_missing")
        body = dict(
            examination=row["examination"],
            applicability_conditions=row["applicability_conditions"],
            interpretation=row["interpretation_rule"],
            interpretation_status="installed",
            source_refs=[r.model_dump(mode="json") for r in references],
        )
        return validate(
            TaskDefinition,
            dict(
                **body,
                reference=Reference(
                    role="definition",
                    code="POLICY:ACQUISITION:" + target,
                    version="1.0.0",
                    checksum=digest(body),
                ).model_dump(mode="json"),
            ),
        )

    def read_definition(self, target, reference):
        definition = self.definition(target)
        if definition.reference != reference:
            raise ValueError("acquisition_definition_changed")
        return definition
