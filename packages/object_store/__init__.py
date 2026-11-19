"""Immutable local objects with explicit staging; domain records use object:// URIs."""

import hashlib
import os
import re
from pathlib import Path
from typing import Protocol
from uuid import uuid4


def checksum(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class ObjectStore(Protocol):
    def stage(self, data: bytes) -> str: ...
    def read(self, uri: str) -> bytes: ...
    def read_bounded(self, uri: str, max_bytes: int) -> bytes: ...
    def promote(self, uri: str, expected_checksum: str) -> str: ...
    def discard(self, uri: str) -> None: ...


class LocalObjectStore:
    """Service-owned root. Unique keys make rollback safe across concurrent writers.

    A crash before SQL commit can leave unreferenced objects; it cannot expose a
    committed reference to bytes that still need promotion. GC requires a DB
    reference inventory and an age/active-transaction fence (not implemented here).
    """

    def __init__(self, root: Path):
        self.root = root.resolve()
        for directory in ("staging", "objects"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def _path(self, uri: str) -> Path:
        if not re.fullmatch(r"object://knowledge/(staging|objects)/[0-9a-f]{32}", uri):
            raise ValueError("invalid managed object URI")
        return self.root / uri.removeprefix("object://knowledge/")

    def stage(self, data: bytes) -> str:
        if type(data) is not bytes or not 1 <= len(data) <= 4 * 1024 * 1024:
            raise ValueError("bounded nonempty object bytes required")
        uri = "object://knowledge/staging/" + uuid4().hex
        try:
            with self._path(uri).open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            self._path(uri).unlink(missing_ok=True)
            raise
        return uri

    def read(self, uri: str) -> bytes:
        return self._path(uri).read_bytes()

    def read_bounded(self, uri: str, max_bytes: int) -> bytes:
        """04B4 bounded read; reject symlinks and oversized stored objects."""
        if type(max_bytes) is not int or not 1 <= max_bytes <= 4 * 1024 * 1024:
            raise ValueError("invalid object read bound")
        path = self._path(uri)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        finally:
            os.close(directory)
        with os.fdopen(descriptor, "rb") as stream:
            import stat

            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
                raise ValueError("object exceeds read bound or is not a regular file")
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("object exceeds read bound")
        return data

    def promote(self, uri: str, expected_checksum: str) -> str:
        if "/staging/" not in uri or checksum(self.read(uri)) != expected_checksum:
            raise ValueError("staged object checksum mismatch")
        target = uri.replace("/staging/", "/objects/")
        os.rename(self._path(uri), self._path(target))
        try:
            with self._path(target).open("rb") as stream:
                os.fsync(stream.fileno())
            for directory in ("objects", "staging"):
                fd = os.open(self.root / directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except BaseException:
            self._path(target).unlink(missing_ok=True)
            raise
        return target

    def discard(self, uri: str) -> None:
        self._path(uri).unlink(missing_ok=True)
