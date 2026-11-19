"""Controlled local file URI and bounded byte/tree observations; never execute resources."""

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import Field, model_validator

from packages.grounding.models import Checksum, Value, check_seal, digest, seal

from .models import AdapterError, Text, parse


class LocalResourcePolicy(Value):
    version: str = "local-tree/1.0.0"
    allowed_roots: tuple[Text, ...] = Field(min_length=1, max_length=16)
    max_files: int = Field(default=4096, ge=1, le=100000)
    max_bytes: int = Field(default=1073741824, ge=1, le=10737418240)
    timeout_seconds: int = Field(default=30, ge=1, le=60)


class ResourceObservation(Value):
    uri: Text
    resolved_uri: Text
    kind: str
    checksum: Checksum
    byte_size: int
    entries: int
    policy_checksum: Checksum


def _lexical_path(uri):
    """Decode once without filesystem access; this alone grants no read authority."""
    if not isinstance(uri, str) or len(uri) > 4096 or re.search(r"%(?![0-9a-fA-F]{2})", uri):
        raise AdapterError("invalid_local_uri")
    parts = urlsplit(uri)
    if (
        parts.scheme != "file"
        or parts.netloc not in {"", "localhost"}
        or parts.query
        or parts.fragment
    ):
        raise AdapterError("invalid_local_uri")
    decoded = unquote(parts.path, encoding="utf-8", errors="strict")  # Exactly once.
    if "\x00" in decoded or "\\" in decoded or not decoded.startswith("/"):
        raise AdapterError("invalid_local_path")
    path = Path(decoded)
    if ".." in path.parts or "." in decoded.split("/"):
        raise AdapterError("local_path_traversal")
    return path


def local_path(uri, policy):
    policy = parse(LocalResourcePolicy, policy)
    path = _lexical_path(uri)
    # Check both lexical path and resolved symlink target against explicit roots.
    roots = tuple(Path(r).resolve(strict=True) for r in policy.allowed_roots)
    resolved = path.resolve(strict=True)
    if not any(path.is_relative_to(r) for r in roots) or not any(
        resolved.is_relative_to(r) for r in roots
    ):
        raise AdapterError("local_path_outside_roots")
    return path


def observe_resource(uri, policy):
    policy = parse(LocalResourcePolicy, policy)
    # visit checks current roots and symlinks before any file access, including
    # the first item. Do not resolve the same roots twice before that first visit.
    path = _lexical_path(uri)
    deadline = time.monotonic() + policy.timeout_seconds
    entries, total, files = [], 0, 0

    def check():
        if time.monotonic() > deadline:
            raise AdapterError("resource_timeout")

    def visit(item, relative, ancestors):
        nonlocal total, files
        check()
        files += 1
        if files > policy.max_files:
            raise AdapterError("resource_file_limit")
        local_path(item.as_uri(), policy)
        before = item.stat()
        identity = (before.st_dev, before.st_ino)
        link = os.readlink(item) if item.is_symlink() else None
        if item.is_dir():
            if identity in ancestors:
                raise AdapterError("resource_symlink_cycle")
            entries.append([relative, "directory", link])
            for child in sorted(item.iterdir(), key=lambda p: p.name):
                visit(child, relative + "/" + child.name, ancestors | {identity})
        elif item.is_file():
            h = hashlib.sha256()
            with item.open("rb") as stream:
                while block := stream.read(1048576):
                    check()
                    total += len(block)
                    if total > policy.max_bytes:
                        raise AdapterError("resource_byte_limit")
                    h.update(block)
            entries.append([relative, "file", link, before.st_size, "sha256:" + h.hexdigest()])
        else:
            raise AdapterError("resource_special_file")
        after = item.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise AdapterError("resource_changed_during_read")

    visit(path, ".", set())
    # A regular single file uses byte SHA256. Trees/symlinks bind names, kind and link text.
    checksum = entries[0][-1] if path.is_file() and not path.is_symlink() else digest(entries)
    return ResourceObservation(
        uri=uri,
        resolved_uri=path.resolve().as_uri(),
        kind="tree" if path.is_dir() else "file",
        checksum=checksum,
        byte_size=total,
        entries=files,
        policy_checksum=digest(policy),
    )


class ManifestResourceObservation(Value):
    resource_key: Text
    observation: ResourceObservation
    declared_checksum_matches: bool | None
    declared_size_matches: bool | None


class ManifestResources(Value):
    """A byte observation bound to a precise frozen deployment manifest, not authority."""

    checksum: Checksum
    manifest_checksum: Checksum
    catalog_checksum: Checksum
    implementation_checksum: Checksum
    observations: tuple[ManifestResourceObservation, ...] = Field(max_length=128)
    execution_authorized: Literal[False] = False

    @model_validator(mode="after")
    def integrity(self):
        check_seal(self)
        return self


def observe_manifest_resources(manifest, policy):
    from .models import parse_deployment

    manifest = parse_deployment(manifest)
    policy = parse(LocalResourcePolicy, policy)
    if manifest.execution_manifest is None:
        raise AdapterError("disabled_manifest_has_no_resource_observation")
    deadline = time.monotonic() + policy.timeout_seconds
    observations = []
    total_bytes = total_entries = 0
    for resource in manifest.execution_manifest.resources:
        observation = observe_resource(resource.uri, policy)
        total_bytes += observation.byte_size
        total_entries += observation.entries
        if (
            time.monotonic() > deadline
            or total_bytes > policy.max_bytes
            or total_entries > policy.max_files
        ):
            raise AdapterError("manifest_resource_budget")
        observations.append(
            ManifestResourceObservation(
                resource_key=resource.resource_key,
                observation=observation,
                declared_checksum_matches=(
                    observation.checksum == "sha256:" + resource.sha256
                    if resource.sha256 is not None
                    else None
                ),
                declared_size_matches=(
                    observation.byte_size == resource.byte_size
                    if resource.byte_size is not None
                    else None
                ),
            )
        )
    return seal(
        ManifestResources,
        manifest_checksum=manifest.checksum,
        catalog_checksum=manifest.catalog_ref.checksum,
        implementation_checksum=manifest.implementation_ref.checksum,
        observations=observations,
    )
