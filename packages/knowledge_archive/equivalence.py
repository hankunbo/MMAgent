"""Conservative EGS HTML equivalence; no medical interpretation or HTTP access."""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from packages.knowledge_index.models import digest
from packages.knowledge_index.parsers import parse
from packages.object_store import checksum

POLICY = "egs-presentation-nonce/1.0.0"
ASSET_HOST = "microlensing-bitblox-production.s3.eu-south-1.amazonaws.com"


def html_proof(body):
    parsed = parse(body, "text/html", language="en")
    if parsed.status != "parsed" or not parsed.document.blocks:
        raise ValueError("equivalence_parse_failed")
    soup = BeautifulSoup(body, "html.parser")
    if len(soup.find_all("h1")) != 1:
        raise ValueError("equivalence_single_title_required")
    ignored = []
    # Only presentation metadata. Body images, links, scripts, numbers, polarity,
    # headings and all other attributes remain in the comparison.
    for tag in soup.select('head meta[name="csrf-token"]'):
        if "content" in tag.attrs:
            tag["content"] = "[csrf-token]"
            ignored.append("head.csrf-token.content")
    for selector, attribute in (
        ('head link[rel="apple-touch-icon"]', "href"),
        ('head meta[property="og:image"]', "content"),
    ):
        for tag in soup.select(selector):
            url = urlsplit(tag.get(attribute, ""))
            if url.scheme == "https" and url.netloc == ASSET_HOST:
                pairs = parse_qsl(url.query, keep_blank_values=True)
                kept = [(k, v) for k, v in pairs if k not in {"X-Amz-Date", "X-Amz-Signature"}]
                if len(kept) != len(pairs):
                    ignored.append(selector + ".signed_time")
                    tag[attribute] = urlunsplit(url._replace(query=urlencode(kept)))
    return {
        "policy": POLICY,
        "normalized_html_checksum": checksum(str(soup).encode()),
        "parsed_document_checksum": digest(parsed.document),
        "parser": parsed.document.parser,
        "parser_version": parsed.document.parser_version,
        "ignored_fields": sorted(ignored),
    }


def compare_html(left, right):
    a, b = html_proof(left), html_proof(right)
    if a != b:
        raise ValueError("archive_documents_not_equivalent")
    return a
