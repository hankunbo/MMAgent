"""Bounded child process for untrusted document parsers. Internal protocol only."""

import base64
import io
import json
import re
import resource
import socket
import sys

from .fidelity import (
    BLOCKS,
    NUMERIC_TAGS,
    TableLimit,
    Unsupported,
    descendants,
    inline,
)
from .fidelity import (
    table as extract_table,
)
from .fidelity import (
    tag as node_tag,
)
from .fidelity import (
    text as extract_text,
)
from .models import (
    FIDELITY_VERSION,
    PARSER_VERSION,
    Block,
    ParsedDocument,
    ParsePolicy,
    ParseResult,
    Position,
    TableCapability,
    digest,
)


class Unsafe(ValueError):
    pass


class Limit(ValueError):
    pass


def no_network(*args, **kwargs):
    raise Unsafe("network disabled")


def jats_root(data):
    """Accept inert external JATS declarations, never a DTD or entity loader."""
    from defusedxml.ElementTree import DefusedXMLParser

    parser = DefusedXMLParser(forbid_dtd=False, forbid_entities=True, forbid_external=True)

    def declaration(name, sysid, pubid, internal):
        if (
            name != "article"
            or internal
            or not pubid
            or not pubid.startswith("-//NLM//DTD JATS ")
            or not sysid
            or not re.fullmatch(r"JATS-[A-Za-z0-9_.-]+\.dtd", sysid)
        ):
            raise Unsafe("unsupported JATS declaration")

    parser.parser.StartDoctypeDeclHandler = declaration
    parser.feed(data)
    root = parser.close()
    if root.tag != "article":
        raise Unsafe("JATS article required")
    return root


def citation_scripts(root):
    """Explicit JATS bibliography marks, not mathematical superscripts.

    Only leaf bibr xrefs to unique direct ref-list/ref targets are supported.
    Keep printed labels and separators exactly; never fetch or infer a citation.
    Archived XML is untouched: this changes only the parser's in-memory tree.
    """
    from collections import Counter

    ids = Counter(n.get("id") for n in root.iter() if n.get("id"))
    references = {
        n.get("id")
        for parent in root.iter()
        if parent.tag == "ref-list"
        for n in parent
        if n.tag == "ref" and n.get("id") and ids[n.get("id")] == 1
    }
    separator = r"[\s,;:()\[\]\-–—]*"
    for node in root.iter():
        if node.tag not in {"sup", "superscript"} or not len(node):
            continue
        if not re.fullmatch(separator, node.text or ""):
            raise Unsupported("numeric_markup_unsupported")
        text = node.text or ""
        for child in node:
            if (
                child.tag != "xref"
                or len(child)
                or set(child.attrib) != {"rid", "ref-type"}
                or child.get("ref-type") != "bibr"
                or child.get("rid") not in references
                or not child.text
                or not child.text.strip()
                or not re.fullmatch(separator, child.tail or "")
            ):
                raise Unsupported("numeric_markup_unsupported")
            text += child.text + (child.tail or "")
        if len(text) > 120:
            raise Unsupported("numeric_markup_unsupported")
        for child in list(node):
            node.remove(child)
        node.text = "cite:" + text


def run(data, mime, language, policy):
    raw = data.lstrip()
    if raw.startswith((b"PK", b"\x1f\x8b")):
        raise Unsafe("compressed container")
    blocks = []
    offset = 0
    headings = []
    title = ""

    def add(text, kind="paragraph", page=None, table="not_applicable", grid=None, capability=None):
        nonlocal offset, title
        text = text if grid is not None else text.strip()
        if not text.strip():
            return
        if offset + len(text) > policy.max_chars or len(blocks) >= policy.max_nodes:
            raise Limit()
        if kind == "heading" and not title:
            title = text[:512]
        position = Position(
            section_path=tuple(headings) or ("document",),
            kind=kind,
            ordinal=len(blocks) + 1,
            page=page,
            span_start=offset,
            span_end=offset + len(text),
            table_status=table,
            table_capability=capability,
        )
        blocks.append(Block(text=text, position=position, table=grid))
        offset += len(text) + 1

    if mime == "application/pdf":
        if not raw.startswith(b"%PDF-"):
            return ParseResult(status="failed", reason="wrong_mime")
        from pypdf import PdfReader
        from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject, StreamObject

        pdf = PdfReader(io.BytesIO(data), strict=True)
        if pdf.is_encrypted:
            return ParseResult(status="failed", reason="encrypted")
        if len(pdf.pages) > policy.max_pages:
            raise Limit()
        visited = set()
        expanded = 0
        count = 0

        def inspect(obj, depth=0):
            nonlocal expanded, count
            count += 1
            if count > policy.max_nodes or depth > policy.max_depth:
                raise Limit()
            if isinstance(obj, IndirectObject):
                key = (obj.idnum, obj.generation)
                if key in visited:
                    return
                visited.add(key)
                obj = obj.get_object()
            if isinstance(obj, DictionaryObject):
                if set(obj) & {
                    "/JS",
                    "/JavaScript",
                    "/Launch",
                    "/EmbeddedFiles",
                    "/EmbeddedFile",
                    "/XFA",
                    "/OpenAction",
                    "/AA",
                    "/F",
                    "/EF",
                    "/URI",
                }:
                    raise Unsafe()
                if isinstance(obj, StreamObject):
                    expanded += len(obj.get_data())
                    if expanded > policy.max_bytes * 4:
                        raise Limit()
                for key, value in obj.items():
                    if key not in {"/Parent", "/P"}:
                        inspect(value, depth + 1)
            elif isinstance(obj, ArrayObject):
                for value in obj:
                    inspect(value, depth + 1)

        inspect(pdf.trailer)
        for number, page in enumerate(pdf.pages, 1):
            headings[:] = [f"page {number}"]
            for paragraph in re.split(r"\n\s*\n", page.extract_text() or ""):
                add(
                    paragraph,
                    page=number,
                    table="unavailable",
                    capability=TableCapability(
                        status="unavailable", reason="pdf_grid_unsupported", numeric_evidence=False
                    ),
                )
        parser = "trusted-pdf"
    elif mime in ("application/xml", "text/xml"):
        if not raw.startswith(b"<") or re.search(
            rb"<\s*(?:html|!doctype\s+html)\b", raw[:512], re.I
        ):
            return ParseResult(status="failed", reason="wrong_mime")
        from defusedxml.ElementTree import fromstring

        root = (
            jats_root(data)
            if policy.version in {"xml-jats/1.0.0", "xml-jats-citations/1.0.0"}
            else fromstring(data, forbid_dtd=True, forbid_entities=True, forbid_external=True)
        )
        count = 0
        meaningful = {
            "p": "paragraph",
            "para": "paragraph",
            "list-item": "list",
            "table": "table",
            "caption": "caption",
            "title": "heading",
            "article-title": "heading",
            "label": "heading",
        }

        def walk(node, depth=0):
            nonlocal count
            count += 1
            if depth > policy.max_depth or count > policy.max_nodes:
                raise Limit()
            tag = node.tag.split("}")[-1].lower()
            if tag in NUMERIC_TAGS:
                raise Unsupported("numeric_markup_unsupported")
            if tag in {"include", "script", "object", "embed"} or any(
                k.split("}")[-1].lower() in {"href", "src"}
                and str(v).lower().startswith(("file:", "../"))
                for k, v in node.attrib.items()
            ):
                raise Unsafe()
            old = list(headings)
            if tag in {"sec", "section"}:
                heading = next(
                    (
                        extract_text(c)[:256]
                        for c in node
                        if c.tag.split("}")[-1] in {"title", "heading"}
                    ),
                    "section",
                )
                headings.append(heading)
            if tag in meaningful:
                if tag == "table":
                    body, grid, capability = extract_table(node)
                    add(body, "table", table=capability.status, grid=grid, capability=capability)
                else:
                    add(extract_text(node), meaningful[tag])
                # Walk descendants for safety/resource checks without duplicating their text.
                for child in node.iter():
                    if child is node:
                        continue
                    count += 1
                    if count > policy.max_nodes or child.tag.split("}")[-1].lower() in {
                        "include",
                        "script",
                        "object",
                        "embed",
                    }:
                        raise Unsafe()
            else:
                pending = node.text or ""
                for child in node:
                    child_tag = child.tag.split("}")[-1].lower()
                    if (
                        child_tag in meaningful
                        or child_tag in BLOCKS
                        or any(
                            n.tag.split("}")[-1].lower() in meaningful
                            or n.tag.split("}")[-1].lower() in BLOCKS
                            for n in child.iter()
                        )
                    ):
                        add(pending)
                        pending = ""
                        walk(child, depth + 1)
                    else:
                        pending += inline(child)
                    pending += child.tail or ""
                add(pending)
            headings[:] = old

        # Entire tree depth/security scan also covers descendants of semantic units.
        stack = [(root, 0)]
        seen = 0
        while stack:
            node, depth = stack.pop()
            seen += 1
            if depth > policy.max_depth or seen > policy.max_nodes:
                raise Limit()
            if node.tag.split("}")[-1].lower() in {"include", "script", "object", "embed"}:
                raise Unsafe()
            stack.extend((child, depth + 1) for child in node)
        if policy.version == "xml-jats-citations/1.0.0":
            citation_scripts(root)
        if policy.version in {"xml-scripts/1.0.0", "xml-jats/1.0.0", "xml-jats-citations/1.0.0"}:
            # New explicit canonical text only. Preserve the archived raw XML.
            # No formula inference, entity loading, unit conversion or silent
            # loss of numeric exponents. Nested/complex scripts remain unsupported.
            for node in root.iter():
                if node_tag(node) in {"sup", "sub", "superscript", "subscript"}:
                    if len(node) or not node.text or len(node.text) > 128:
                        raise Unsupported("numeric_markup_unsupported")
                    marker = "^" if node_tag(node) in {"sup", "superscript"} else "_"
                    node.text = marker + "{" + node.text + "}"
                    node.tag = "span"
        walk(root)
        parser = "structured-xml"
    else:
        if not re.search(rb"<(?:html|!doctype\s+html)\b", raw[:1024], re.I):
            return ParseResult(status="failed", reason="wrong_mime")
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(data, "html.parser")
        for node in list(
            soup.find_all(
                ["script", "style", "link", "iframe", "object", "embed", "svg", "form", "noscript"]
            )
        ):
            node.decompose()
        nodes = soup.find_all(True)
        if len(nodes) > policy.max_nodes:
            raise Limit()
        for node in nodes:
            if len(list(node.parents)) > policy.max_depth:
                raise Limit()
        title = extract_text(soup.title)[:512] if soup.title else ""

        def walk_html(node):
            if node.name in {"head", "title"}:
                return
            kind = {"p": "paragraph", "li": "list", "table": "table", "figcaption": "caption"}.get(
                node.name
            )
            if re.fullmatch(r"h[1-6]", node.name or ""):
                kind = "heading"
                level = int(node.name[1])
                headings[:] = headings[: level - 1] + [extract_text(node)[:256]]
            if kind == "table":
                body, grid, capability = extract_table(node)
                add(body, kind, table=capability.status, grid=grid, capability=capability)
            elif kind:
                add(extract_text(node), kind)
            else:
                from .fidelity import parts

                pending = ""
                for child in parts(node):
                    if isinstance(child, str):
                        pending += child
                    elif (
                        child.name in BLOCKS
                        or child.name in {"html", "head"}
                        or any(node_tag(n) in BLOCKS for n in descendants(child))
                    ):
                        add(pending)
                        pending = ""
                        walk_html(child)
                    else:
                        pending += inline(child)
                add(pending)

        walk_html(soup)
        parser = "structured-html"
    if not blocks:
        return ParseResult(status="no_text", reason="no_text_ocr_unsupported")
    return ParseResult(
        status="parsed",
        document=ParsedDocument(
            parser=parser,
            parser_version="2.3.0"
            if policy.version == "xml-jats-citations/1.0.0"
            else "2.2.0"
            if policy.version == "xml-jats/1.0.0"
            else "2.1.0"
            if policy.version == "xml-scripts/1.0.0"
            else PARSER_VERSION,
            fidelity_policy="text-fidelity/2.1.0"
            if policy.version in {"xml-scripts/1.0.0", "xml-jats/1.0.0", "xml-jats-citations/1.0.0"}
            else FIDELITY_VERSION,
            policy_checksum=digest(policy),
            language=language,
            title=title,
            blocks=tuple(blocks),
        ),
    )


def main():
    request = json.loads(sys.stdin.buffer.read(6_000_000))
    policy = ParsePolicy.model_validate_json(json.dumps(request["policy"]))
    resource.setrlimit(resource.RLIMIT_AS, (policy.memory_mb * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (policy.timeout_seconds,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    socket.socket = no_network
    socket.create_connection = no_network
    try:
        result = run(
            base64.b64decode(request["data"], validate=True),
            request["mime"],
            request["language"],
            policy,
        )
    except Unsupported as error:
        result = ParseResult(status="unsupported", reason=str(error))
    except (Limit, TableLimit, MemoryError):
        result = ParseResult(status="failed", reason="resource_limit")
    except Unsafe:
        result = ParseResult(status="failed", reason="unsafe_document")
    except Exception as error:
        from defusedxml.common import DefusedXmlException

        result = ParseResult(
            status="failed",
            reason="unsafe_document" if isinstance(error, DefusedXmlException) else "parse_failed",
        )
    output = result.model_dump_json()
    if len(output.encode()) > 4_000_000:
        output = ParseResult(status="failed", reason="resource_limit").model_dump_json()
    sys.stdout.write(output)


if __name__ == "__main__":
    main()
