"""Display-only envelopes; business reference checks stay in the projection owner."""

import argparse
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Text = Annotated[str, Field(min_length=1)]


class Extensible(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")


class StructureNode(Extensible):
    id: Text
    type: Text
    title: Text


class StructureEdge(Extensible):
    source: Text
    target: Text
    relation: Text


class PresentationStructure(Extensible):
    schema_version: Literal["case-presentation-structure/1.0.0"]
    nodes: list[StructureNode] = Field(max_length=10000)
    edges: list[StructureEdge] = Field(max_length=50000)


class EvidenceSections(Extensible):
    schema_version: Literal["evidence-sections/1.0.0"]
    rationale: str | dict[str, Any] | None = None
    acquisition: str | dict[str, Any] | None = None
    reference: str | dict[str, Any] | None = None


class AssetItem(Extensible):
    id: Text
    kind: Text


class AssetLink(Extensible):
    node_id: Text
    asset_id: Text
    source_pointer: str
    source_value_checksum: Text


class EvidenceAssets(Extensible):
    schema_version: Literal["evidence-assets/1.0.0"]
    source_checksum: Text
    case_code: str | None = None
    input_checksum: str | None = None
    items: list[AssetItem] = Field(max_length=128)
    links: list[AssetLink] = Field(default_factory=list, max_length=50000)


class PresentationPacket(Extensible):
    schema_version: Literal["evidence-presentation/1.0.0"]
    scope: Literal["historical_read_only_not_current_authorization"]
    source_checksum: Text
    report_content_checksum: Text
    source_version: str
    sample: str
    nodes: list[StructureNode] = Field(max_length=10000)
    edges: list[StructureEdge] = Field(max_length=50000)
    navigation: list[StructureNode] = Field(max_length=10000)
    checks: dict[str, dict[str, Any]]
    scores: dict[str, Any]
    scores_checksum: Text
    counts: dict[str, int | None]
    confidence: float | None
    coverage: dict[str, Any]
    appendices: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    memory: dict[str, Any]


class DisplayExport(Extensible):
    schema_version: Literal["evidence-display-export/1.0.0"]
    packet_checksum: Text
    packet: PresentationPacket


def check(model, value):
    try:
        model.model_validate(value)
    except ValidationError as error:
        # Error values can contain raw text/bytes: report paths, not payloads.
        paths = ["/".join(map(str, e["loc"])) for e in error.errors(include_input=False)]
        raise ValueError(
            "presentation_contract:" + model.__name__ + ":" + ",".join(paths)
        ) from None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    models = [
        PresentationStructure,
        EvidenceSections,
        EvidenceAssets,
        PresentationPacket,
        DisplayExport,
    ]
    for model in models:
        path = Path(__file__).parent / "schemas" / (model.__name__ + ".json")
        expected = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        if args.write:
            path.write_text(expected)
        elif not path.exists() or path.read_text() != expected:
            raise SystemExit("presentation_schema_drift:" + model.__name__)
    print("5 display-only envelope schemas verified; reference checks are additional.")


if __name__ == "__main__":
    main()
