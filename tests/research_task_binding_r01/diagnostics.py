"""Restricted synthetic-only projections, never report bodies, URIs or credentials."""

import json
import math
import struct
import zlib
from types import SimpleNamespace

from packages.case_score_ledger.models import ScoreEntry, ScoreEnvelope, TechnicalFact
from packages.case_score_ledger.service import CaseScoreService
from packages.grounding.models import canonical, digest, seal, validate
from packages.research_runtime.single_case import ResearchSingleCaseService
from packages.tool_execution.scalar_output import ScalarOutput
from tests.case_score_ledger.test_models import definition, source


def png(pixels):
    assert len(pixels) == 6

    def chunk(kind, raw):
        return struct.pack(">I", len(raw)) + kind + raw + struct.pack(">I", zlib.crc32(kind + raw))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0" + pixels[:3] + b"\0" + pixels[3:]))
        + chunk(b"IEND", b"")
    )


def scalar_value(value):
    wire = ScalarOutput(
        metric="logit", value=value, unit="logit", input_checksum=digest("synthetic-input")
    )
    return CaseScoreService._decode(
        SimpleNamespace(synthetic=True),
        wire.model_dump_json().encode(),
        "scalar-score-output/1.0.0",
        "value",
        definition(),
    )


def constructed_report(value):
    """CPU constructor probe only, NOT a persisted/authorized owner success chain."""
    src = source()
    entry = ScoreEntry(
        definition=definition(),
        target="GOAL:SYNTHETIC",
        status="available",
        value=value,
        missing_reason=None,
        source=src,
        calibration="uncalibrated",
    )
    ledger = seal(
        ScoreEnvelope,
        scope=src.scope,
        synthetic=True,
        policy_checksum=src.policy_checksum,
        entries=(entry,),
        technical=(
            TechnicalFact(
                key="probe:tool",
                status="SUCCEEDED",
                qc_statuses=("PASS",),
                required_dependencies=(),
                optional_dependencies=(),
            ),
        ),
        graph=None,
        graph_status="unavailable",
        original_report=None,
        run_status="COMPLETED_INSUFFICIENT_EVIDENCE",
    )
    state = dict(
        run_id=src.scope.run_id,
        binding=dict(execution_key="probe:tool", score_key="probe:score", plan_checksum=digest(1)),
        deadline="2026-09-26T00:00:00+00:00",
        technical_reason="research_output_without_medical_graph",
    )
    runtime = SimpleNamespace(
        run_status=lambda rid: "COMPLETED_INSUFFICIENT_EVIDENCE",
        record_lifecycle=lambda old, **kw: dict(old, **kw),
    )
    scores = SimpleNamespace(
        save=lambda rid, key: ledger,
        read=lambda rid, key: validate(ScoreEnvelope, json.loads(canonical(ledger))),
        policy_checksum=ledger.policy_checksum,
        objects=None,
    )
    driver = ResearchSingleCaseService(
        runtime=runtime,
        research=None,
        choice=None,
        bind_plan=None,
        config_checksum=digest("probe"),
        budget_seconds=60,
    )
    return driver._finish(state, scores), ledger


def restricted(value):
    result = dict(type=type(value).__name__, digest=digest(value))
    if isinstance(value, float):
        result.update(
            repr=repr(value),
            hex=value.hex(),
            negative_zero=value == 0 and math.copysign(1, value) < 0,
        )
    elif isinstance(value, str) and value.startswith("sha256:") and len(value) == 71:
        result["checksum"] = value
    elif value in ("SUCCEEDED", "COMPLETED_INSUFFICIENT_EVIDENCE"):
        result["status"] = value
    elif value is None or isinstance(value, (bool, int)):
        result["value"] = value
    return result


def differences(left, right, path="$"):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                result.append(
                    dict(path=path + "." + key, missing="left" if key not in left else "right")
                )
            else:
                result.extend(differences(left[key], right[key], path + "." + key))
        return result
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [
            d
            for i, (a, b) in enumerate(zip(left, right, strict=True))
            for d in differences(a, b, f"{path}[{i}]")
        ]
    if type(left) is type(right) and canonical(left) == canonical(right):
        return []
    return [
        dict(path=path, left=restricted(left), right=restricted(right), numeric_equal=left == right)
    ]


def binding_checks(report, state, definition):
    expected = dict(
        terminal=True,
        run_id=state["run_id"],
        run_status="COMPLETED_INSUFFICIENT_EVIDENCE",
        technical_status="SUCCEEDED",
        execution_key=state["binding"]["execution_key"],
        score_key=state["binding"]["score_key"],
        plan_checksum=definition["plan_checksum"],
        config_checksum=definition["config_checksum"],
        policy_checksum=definition["score_policy"],
        checksum=digest({k: v for k, v in report.items() if k != "checksum"}),
    )
    return {
        key: dict(
            actual=restricted(report.get(key)),
            expected=restricted(value),
            # Match the existing owner's truthiness/equality; separately expose types.
            matches=bool(report.get(key)) if key == "terminal" else report.get(key) == value,
            same_type=type(report.get(key)) is type(value),
        )
        for key, value in expected.items()
    }
