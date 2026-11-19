"""Finite claim validation policy 2.0.0, not a medical language inference engine.

Positions and exact surface forms are authoritative. Unsupported syntax fails
closed; absence of a recognized negative marker never establishes positivity.
"""

import re
from dataclasses import dataclass

from packages.knowledge_index.models import digest

POLICY_VERSION = "claim-validation/2.0.0"
NUMERIC_VERSION = "numeric-occurrence/1.0.0"
POLARITY_VERSION = "polarity-grammar/1.0.0"
HSPACE = r"[ \u00a0\u202f\u2009]*"
UNITS = (
    "%",
    "g",
    "kg",
    "mg",
    "μg",
    "µg",
    "ng",
    "L",
    "mL",
    "μL",
    "µL",
    "m",
    "cm",
    "mm",
    "μm",
    "µm",
    "nm",
    "mmHg",
    "Pa",
    "kPa",
    "s",
    "min",
    "h",
    "day",
    "days",
    "year",
    "years",
    "年",
    "岁",
    "U",
    "IU",
    "mol",
    "mmol",
    "mg/kg",
    "mg/kg/day",
    "mg/kg/d",
    "μg/kg/min",
    "µg/kg/min",
    "mmol/L",
    "mg/dL",
    "g/L",
    "μg/mL",
    "µg/mL",
    "mL/min",
    "mmHg/s",
    "μm/year",
    "µm/year",
)
COMPARATORS = ("<", "<=", ">", ">=", "=", "≤", "≥")
# These tokens can delimit a quantity in the supported whole-sentence grammar.
FOLLOWERS = ("and", "supports", "require", "for")
DECIMAL = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
NUMBER = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")
UNIT_TOKEN = r"[^\W\d_]+|%"
UNIT_EXPRESSION = rf"(?:{UNIT_TOKEN})(?:{HSPACE}/{HSPACE}(?:{UNIT_TOKEN}))*"
SUPPORTED_UNIT = (
    "(?:"
    + "|".join(
        (HSPACE + "/" + HSPACE).join(re.escape(part) for part in unit.split("/"))
        for unit in sorted(UNITS, key=len, reverse=True)
    )
    + ")"
)
MEASUREMENT = rf"(?:[<>=≤≥]+{HSPACE})?{DECIMAL}(?:{HSPACE}{SUPPORTED_UNIT})?"

# Anchored, finite English patterns. Subject/object catalogs are deliberately narrow.
# Negation and uncertainty have disjoint rules; their combination is unresolved.
POSITIVE = (
    r"OCT result is positive\.",
    r"OCT supports evaluation\.",
    r"OCT is recommended for (?:adults|children)\.",
    r"If adults have symptoms, consider OCT\.",
    rf"Pressure {MEASUREMENT}(?: supports evaluation)?\.",
    rf"Adults aged {DECIMAL} years require device A\.",
    rf"OCT (?:dose|value|count) is {MEASUREMENT}\.",
    rf"OCT values are {MEASUREMENT}(?: and {MEASUREMENT})*\.",
)
NEGATIVE = (
    r"OCT result is negative\.",
    r"OCT is not recommended for (?:adults|children)\.",
    r"OCT does not support (?:evaluation|exclusion)\.",
    r"OCT cannot support (?:evaluation|exclusion)\.",
    r"OCT is without evidence of disease\.",
)
MEDICALLY_UNCERTAIN = (
    r"OCT (?:may|might) support (?:evaluation|exclusion)\.",
    r"(?:Adults|Children) (?:may|might) benefit\.",
    r"OCT evidence is (?:uncertain|insufficient)\.",
    r"Insufficient evidence for OCT\.",
)
CHINESE = {
    "OCT结果为阳性。": "POSITIVE",
    "OCT结果为阴性。": "NEGATIVE",
    "OCT不能支持排除。": "NEGATIVE",
    "OCT未见异常。": "NEGATIVE",
    "OCT不推荐。": "NEGATIVE",
    "OCT可能支持评估。": "UNCERTAIN",
    "OCT证据不足。": "UNCERTAIN",
}
SPEC = {
    "version": POLICY_VERSION,
    "numeric_version": NUMERIC_VERSION,
    "polarity_version": POLARITY_VERSION,
    "units": UNITS,
    "comparators": COMPARATORS,
    "horizontal_space": HSPACE,
    "bare_followers": FOLLOWERS,
    "decimal": DECIMAL,
    "positive": POSITIVE,
    "negative": NEGATIVE,
    "medical_uncertainty": MEDICALLY_UNCERTAIN,
    "chinese": CHINESE,
    "table": {
        "columns": 2,
        "headers": ("Test", "Limit", "Dose", "Value"),
        "subjects": ("OCT", "Pressure"),
        "numeric_cells_only": True,
    },
    "unsupported": (
        "date",
        "identifier",
        "range",
        "scientific",
        "fraction",
        "thousands",
        "unicode_number",
        "leading_zero",
        "unknown_unit",
        "unit_in_other_cell",
        "mixed_polarity",
        "multiple_clauses",
        "double_negation",
        "negated_uncertainty",
    ),
}
POLICY_CHECKSUM = digest(SPEC)


class PolicyRejected(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise PolicyRejected(reason)


@dataclass(frozen=True)
class Occurrence:
    number_start: int
    number_end: int
    start: int
    end: int
    value: str
    unit: str | None
    comparator: str | None


def occurrences(text):
    """Parse complete supported quantities, preserving each numeric subspan identity."""
    require(
        not any(c.isnumeric() and c not in "0123456789" for c in text),
        "numeric_representation_unsupported",
    )
    require(
        not re.search(
            r"[0-9]\s*(?:[-–—−/,^]|\bto\b)\s*[0-9]|[0-9](?:\.[0-9]+)?[eE][+\-]?[0-9]"
            r"|\b(?:ID|No|number|type|stage|version|date)\.?\s*[:#]?\s*[0-9]"
            r"|[0-9]\s*(?:至|到|年[0-9])",
            text,
            re.I,
        ),
        "numeric_representation_unsupported",
    )
    result = []
    for match in NUMBER.finditer(text):
        ns, ne = match.span()
        require(
            re.fullmatch(DECIMAL, match.group()) is not None, "numeric_representation_unsupported"
        )
        require(
            ns == 0 or not (text[ns - 1].isalnum() or text[ns - 1] in "._/+-−"),
            "numeric_representation_unsupported",
        )
        require(
            not re.match(r"\.[0-9]|[0-9_^+\-−]", text[ne:]), "numeric_representation_unsupported"
        )
        left = text[:ns]
        operator = re.search(rf"([<>=≤≥≈~±]+){HSPACE}$", left)
        comparator = operator[1] if operator else None
        require(comparator is None or comparator in COMPARATORS, "numeric_comparator_unsupported")
        require(
            not re.search(
                r"(?:less than|greater than|at least|at most|about|approximately)\s*$", left, re.I
            ),
            "numeric_comparator_unsupported",
        )
        if operator:
            require(
                not re.search(r"[<>=≤≥≈~±]\s*$", left[: operator.start()]),
                "numeric_comparator_unsupported",
            )
        start = operator.start() if operator else ns
        end, unit = ne, None
        tail = text[ne:]
        space = re.match(HSPACE, tail).end()
        suffix = tail[space:]
        token = re.match(UNIT_EXPRESSION, suffix)
        if token and token.group().lower() not in FOLLOWERS:
            surface = token.group()
            normalized = re.sub(HSPACE + "/" + HSPACE, "/", surface)
            require(normalized in UNITS, "numeric_unit_unsupported")
            unit, end = surface, ne + space + token.end()
            # Never accept a known unit prefix of an unknown compound/annotation.
            require(not re.match(r"[A-Za-z0-9_µμ/%·^⁻−²³]", text[end:]), "numeric_unit_unsupported")
        elif suffix and not token and suffix[0] not in ".,;:!?。；，\n\t":
            require(False, "numeric_unit_unsupported")
        remainder = text[end:]
        require(
            not re.match(rf"{HSPACE}(?:/|·|\bper\b|\bsquared\b|\bcubed\b)", remainder),
            "numeric_unit_unsupported",
        )
        # A line/table separator cannot make a unit disappear from the next fragment.
        if re.match(rf"{HSPACE}[\n\t]", remainder):
            after = remainder.lstrip(" \u00a0\u202f\u2009\n\t")
            following = re.match(UNIT_EXPRESSION, after)
            require(
                following is None or following.group() not in UNITS,
                "numeric_unit_relation_unsupported",
            )
        result.append(Occurrence(ns, ne, start, end, match.group(), unit, comparator))
    return tuple(result)


def polarity(text, table=None):
    """Only a matched affirmative rule establishes POSITIVE; no fallback label."""
    if table is not None:
        rows = [[c.text for c in table.cells if c.row == r] for r in range(table.rows)]
        require(
            table.columns == 2
            and rows[0][0] == "Test"
            and rows[0][1] in {"Limit", "Dose", "Value"}
            and len(rows) > 1,
            "polarity_unresolved",
        )
        require(
            all(
                row[0] in {"OCT", "Pressure"} and re.fullmatch(MEASUREMENT, row[1])
                for row in rows[1:]
            ),
            "polarity_unresolved",
        )
        return "POSITIVE"
    if text in CHINESE:
        return CHINESE[text]
    labels = set()
    for label, patterns in (
        ("POSITIVE", POSITIVE),
        ("NEGATIVE", NEGATIVE),
        ("UNCERTAIN", MEDICALLY_UNCERTAIN),
    ):
        if any(re.fullmatch(pattern, text, re.I) for pattern in patterns):
            labels.add(label)
    require(len(labels) == 1, "polarity_unresolved")
    return labels.pop()
