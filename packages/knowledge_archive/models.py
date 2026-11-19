"""04B3 bounded transient archive inputs and normalized bibliographic metadata."""

import re
import unicodedata
from datetime import date, datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from packages.knowledge_connectors.ports import (
    Checksum,
    FetchedDocument,
    Identifier,
    ShortText,
    StrictModel,
    Text,
    TransientResponse,
    Version,
    digest,
    model_checksum,
)
from packages.knowledge_search.privacy import safe_text


def normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", safe_text(value)).split())


def identity_text(value: str) -> str:
    return " ".join(re.findall(r"\w+", normalized_text(value).casefold()))


class Bibliography(StrictModel):
    """Metadata mapped by the trusted connector boundary from this fetch response.

    This is not a free-form patient input or a document parser. Optional fields
    distinguish absent metadata from an explicit correction/retraction notice.
    """

    title: Text | None = None
    authors: tuple[ShortText, ...] = Field(default=(), max_length=100)
    institutions: tuple[ShortText, ...] = Field(default=(), max_length=100)
    publication_date: ShortText | None = None
    language: ShortText | None = None
    source_type: Literal["GUIDELINE", "ARTICLE", "CONSENSUS", "WEB"] | None = None
    identifiers: tuple[Identifier, ...] = Field(default=(), max_length=16)
    canonical_url: Text | None = None
    abstract_permission: Literal["unknown", "metadata_only", "permitted", "restricted"] = "unknown"
    fulltext_urls: tuple[Text, ...] = Field(default=(), max_length=16)
    notice: Literal["normal", "correction", "retraction"] = "normal"

    @field_validator("title")
    @classmethod
    def title_text(cls, value):
        return normalized_text(value) if value is not None else None

    @field_validator("authors", "institutions")
    @classmethod
    def names(cls, values, info):
        # Author order is meaningful: the first author is part of fallback identity.
        result = tuple(dict.fromkeys(normalized_text(v) for v in values))
        return tuple(sorted(result)) if info.field_name == "institutions" else result

    @field_validator("publication_date")
    @classmethod
    def calendar_date(cls, value):
        if value is None:
            return None
        value = normalized_text(value)
        if not re.fullmatch(r"[0-9]{4}(-[0-9]{2}(-[0-9]{2})?)?", value):
            raise ValueError("publication date requires ISO year, month or day precision")
        date.fromisoformat(value + "-01" * (2 - value.count("-")))
        return value

    @field_validator("language")
    @classmethod
    def language_tag(cls, value):
        if value is None:
            return None
        value = normalized_text(value).lower().replace("_", "-")
        value = {"eng": "en", "english": "en", "zho": "zh", "chinese": "zh"}.get(value, value)
        if not re.fullmatch(r"[a-z]{2,3}(-[a-z0-9]{2,8})*", value):
            raise ValueError("bounded language tag required")
        return value


def material_checksum(bibliography, fetcher_version, pipeline_version):
    return digest(
        (
            "archive-normalization/1.0.0:"
            + model_checksum(bibliography)
            + fetcher_version
            + ":"
            + pipeline_version
        ).encode()
    )


class ArchiveMetadata(StrictModel):
    normalization_version: Literal["archive-normalization/1.0.0"] = "archive-normalization/1.0.0"
    bibliography: Bibliography
    metadata_checksum: Checksum
    fetcher_version: Version
    pipeline_version: Version
    etag: ShortText | None = None
    last_modified: datetime | None = None

    @model_validator(mode="after")
    def material_binding(self):
        if self.metadata_checksum != material_checksum(
            self.bibliography, self.fetcher_version, self.pipeline_version
        ):
            raise ValueError("archive material metadata checksum mismatch")
        if self.etag is not None:
            safe_text(self.etag)
        if self.last_modified is not None and (
            self.last_modified.tzinfo is None or self.last_modified.utcoffset() is None
        ):
            raise ValueError("Last-Modified must be timezone aware")
        return self


class FetchInput(StrictModel):
    """Hit reference is passed separately to the Repository public entry point."""

    document: FetchedDocument
    bibliography: Bibliography = Bibliography()
    metadata_response_checksum: Checksum
    discovery_responses: tuple[TransientResponse, ...] = Field(default=(), max_length=2)
    fetcher_version: Version = "1.0.0"
    pipeline_version: Version = "1.0.0"
    etag: ShortText | None = None
    last_modified: datetime | None = None

    @field_validator("etag")
    @classmethod
    def safe_etag(cls, value):
        return safe_text(value) if value is not None else None

    @field_validator("last_modified")
    @classmethod
    def aware_timestamp(cls, value):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Last-Modified must be timezone aware")
        return value
