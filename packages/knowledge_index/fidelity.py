"""Deterministic inline order and bounded ordinary grids; no semantic rewriting."""

from .models import TableCapability, TableCell, TableGrid, digest


class Unsupported(ValueError):
    pass


class TableLimit(ValueError):
    pass


NUMERIC_TAGS = {"sup", "sub", "superscript", "subscript", "math", "inline-formula", "disp-formula"}

BLOCKS = {
    "p",
    "para",
    "div",
    "section",
    "sec",
    "article",
    "body",
    "blockquote",
    "ul",
    "ol",
    "list",
    "li",
    "list-item",
    "table",
    "tr",
    "td",
    "th",
    "title",
    "article-title",
    "caption",
    "figcaption",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}


def tag(node):
    return (
        node.tag.split("}")[-1].lower()
        if hasattr(node, "tag") and isinstance(node.tag, str)
        else node.name
    )


def attributes(node):
    return node.attrib if isinstance(getattr(node, "tag", None), str) else node.attrs


def parts(node):
    if isinstance(getattr(node, "tag", None), str):
        yield node.text or ""
        for child in node:
            yield child
            yield child.tail or ""
    else:
        from bs4.element import Comment, NavigableString

        for child in node.children:
            if isinstance(child, Comment):
                continue
            yield str(child) if isinstance(child, NavigableString) else child


def children(node):
    return [part for part in parts(node) if not isinstance(part, str)]


def descendants(node):
    for child in children(node):
        yield child
        yield from descendants(child)


def inline(node, *, allow_tables=False):
    """Only structural tags introduce newlines; actual inline whitespace is retained."""
    name = tag(node)
    if name in NUMERIC_TAGS:
        raise Unsupported("numeric_markup_unsupported")
    if name == "br":
        return "\n"
    if not allow_tables and name != "table" and any(tag(n) == "table" for n in descendants(node)):
        raise Unsupported("embedded_table_unsupported")
    text = "".join(
        part if isinstance(part, str) else inline(part, allow_tables=allow_tables)
        for part in parts(node)
    )
    return "\n" + text + "\n" if name in BLOCKS else text


def text(node):
    return inline(node).strip()


def table(node):
    """Rectangular tr/(td|th) tables only. No inferred headers for td cells."""
    all_nodes = list(descendants(node))
    reason = None
    if any(tag(n) == "table" for n in all_nodes):
        reason = "nested_table_unsupported"
    elif any(
        str(attributes(n).get(a, "1")) != "1" for n in all_nodes for a in ("rowspan", "colspan")
    ):
        reason = "merged_cells_unsupported"
    elif any(attributes(n).get("headers") for n in all_nodes):
        reason = "explicit_header_ids_unsupported"
    elif any(attributes(n).get("scope", "") not in ("", "row", "col") for n in all_nodes):
        reason = "header_scope_unsupported"
    # Captions/colgroups and unexpected children cannot be silently discarded.
    elif any(tag(n) not in {"tr", "thead", "tbody", "tfoot"} for n in children(node)):
        reason = "table_layout_unsupported"
    rows = [n for n in all_nodes if tag(n) == "tr"]
    cells = [[n for n in children(row) if tag(n) in {"td", "th"}] for row in rows]
    if not reason and (
        not cells
        or not cells[0]
        or any(len(row) != len(cells[0]) for row in cells)
        or any(tag(n) not in {"td", "th"} for row in rows for n in children(row))
    ):
        reason = "nonrectangular_table_unsupported"
    if not reason:
        containers = [node, *[n for n in all_nodes if tag(n) in {"thead", "tbody", "tfoot", "tr"}]]
        for container in containers:
            allowed = (
                {"td", "th"}
                if tag(container) == "tr"
                else {"tr"}
                if container is not node
                else {"tr", "thead", "tbody", "tfoot"}
            )
            if any(
                (part.strip() if isinstance(part, str) else tag(part) not in allowed)
                for part in parts(container)
            ):
                reason = "table_layout_unsupported"
        if any(
            tag(n) in {"tr", "td", "th", "thead", "tbody", "tfoot"}
            for row in cells
            for cell in row
            for n in descendants(cell)
        ):
            reason = "table_layout_unsupported"
        if any(
            tag(cell) == "th" and not attributes(cell).get("scope") and r != 0 and c != 0
            for r, row in enumerate(cells)
            for c, cell in enumerate(row)
        ):
            reason = "header_position_unsupported"
    if reason:
        return (
            inline(node, allow_tables=True).strip(),
            None,
            TableCapability(status="unavailable", reason=reason, numeric_evidence=False),
        )
    if len(rows) > 64 or len(cells[0]) > 32 or sum(map(len, cells)) > 512:
        raise TableLimit()
    values = [[text(cell) for cell in row] for row in cells]
    body = "\n".join("\t".join(row) for row in values)
    if len(body) > 4096:
        raise TableLimit()
    grid_cells = []
    offset = 0
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            refs = []
            for hr, header_row in enumerate(cells):
                for hc, header in enumerate(header_row):
                    if tag(header) != "th" or (hr, hc) == (r, c):
                        continue
                    scope = attributes(header).get("scope", "")
                    if (scope == "col" or (not scope and hr == 0)) and hc == c and hr < r:
                        refs.append((hr, hc))
                    elif (scope == "row" or (not scope and hc == 0)) and hr == r and hc < c:
                        refs.append((hr, hc))
            value = values[r][c]
            grid_cells.append(
                TableCell(
                    row=r,
                    column=c,
                    text=value,
                    header=tag(cell) == "th",
                    headers=tuple(refs),
                    span_start=offset,
                    span_end=offset + len(value),
                )
            )
            offset += len(value) + 1
    grid = TableGrid(rows=len(rows), columns=len(cells[0]), cells=tuple(grid_cells))
    return (
        body,
        grid,
        TableCapability(
            status="structured",
            rows=grid.rows,
            columns=grid.columns,
            checksum=digest(grid),
            numeric_evidence=True,
        ),
    )
