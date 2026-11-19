"""Deterministic offline candidate Writer, with no execution authority."""

from .service import MemoryWriter, WriterUnavailable

__all__ = ["MemoryWriter", "WriterUnavailable"]
