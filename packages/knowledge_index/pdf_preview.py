"""Bounded PDF reading preview, separate from the unchanged 04B4 indexing parser.

Only text from selected physical pages is returned. No OCR, table interpretation,
annotation execution, link following, source archival or clinical publication.
"""

from __future__ import annotations

import json
import subprocess
import sys

MAX_BYTES = 4 * 1024 * 1024
MAX_PAGES = 200
VERSION = "pdf-page-preview/1.0.0"


def failure(reason):
    return {
        "status": "failed",
        "category": "parse",
        "parse_reason": reason,
        "text_extracted": False,
        "parser_version": VERSION,
    }


def preview(data, *, page_start=1, page_count=5, max_chars=4000):
    if (
        any(type(v) is not int for v in (page_start, page_count, max_chars))
        or not 1 <= page_start <= MAX_PAGES
        or not 1 <= page_count <= 10
        or not 1 <= max_chars <= 16000
    ):
        raise ValueError("invalid_pdf_preview_limits")
    if not data.startswith(b"%PDF-"):
        return failure("wrong_mime")
    if len(data) > MAX_BYTES:
        return failure("resource_limit")
    try:
        result = subprocess.run(
            [sys.executable, "-m", __name__, str(page_start), str(page_count), str(max_chars)],
            input=data,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return failure("resource_limit")
    if result.returncode or len(result.stdout) > 200000:
        return failure("resource_limit")
    try:
        return json.loads(result.stdout)
    except (ValueError, UnicodeError):
        return failure("parse_failed")


def extract(data, page_start, page_count, max_chars):
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data), strict=True)
    if reader.is_encrypted:
        return failure("encrypted")
    total = len(reader.pages)
    if total > MAX_PAGES:
        return failure("resource_limit")
    if page_start > total:
        return failure("page_out_of_range")
    last = min(total, page_start + page_count - 1)
    parts, pages = [], []
    used = 0
    for number in range(page_start, last + 1):
        text = reader.pages[number - 1].extract_text() or ""
        text = text[: max_chars + 1]
        part = f"[PDF page {number}]\n{text}" if text.strip() else ""
        pages.append(number)
        parts.append(part)
        used += len(part) + 2
        if used > max_chars:
            break
    combined = "\n\n".join(p for p in parts if p)
    if not combined.strip():
        return failure("no_text_ocr_unsupported")
    return {
        "status": "succeeded",
        "text_extracted": True,
        "parser_version": VERSION,
        "page_count": total,
        "requested_page_start": page_start,
        "requested_page_count": page_count,
        "pages_read": pages,
        "excerpt": combined[:max_chars],
        "excerpt_truncated": len(combined) > max_chars,
        "more_pages_available": pages[-1] < total,
        "representation": "page_text_preview_not_structured_index",
    }


def main():
    import logging
    import resource
    import socket

    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))

    def no_network(*args, **kwargs):
        raise OSError("network_disabled")

    socket.socket = no_network
    socket.create_connection = no_network
    logging.disable(logging.CRITICAL)
    try:
        data = sys.stdin.buffer.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            result = failure("resource_limit")
        else:
            result = extract(data, *(int(a) for a in sys.argv[1:]))
    except MemoryError:
        result = failure("resource_limit")
    except Exception:
        result = failure("parse_failed")
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
