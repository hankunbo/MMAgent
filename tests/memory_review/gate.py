"""Frozen 10D PG-only terminal gate. No full suite, historical collector or live service."""

import difflib
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from tests.memory_review.capture import capture

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/acceptance/10d-offline-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def manifest():
    baseline = json.loads((EVIDENCE / "baseline.json").read_text())
    paths = subprocess.check_output(
        [
            "rg",
            "--files",
            "--hidden",
            "apps",
            "agents",
            "packages",
            "db",
            "configs",
            "tests",
            "Algorithm_Tool/mask_metrics",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    names = set(baseline) | {
        p
        for p in paths
        if "__pycache__" not in Path(p).parts and Path(p).suffix not in {".pyc", ".pyo"}
    }
    return {p: sha(ROOT / p) for p in sorted(names) if (ROOT / p).is_file()}


def freeze():
    target = EVIDENCE / "implementation-freeze.json"
    if target.exists():
        raise SystemExit("freeze already exists; never replace acceptance evidence")
    current = manifest()
    baseline = json.loads((EVIDENCE / "baseline.json").read_text())
    write(target, current)
    delta = dict(
        modified={
            p: dict(before=h, after=current[p])
            for p, h in baseline.items()
            if p in current and h != current[p]
        },
        added={p: h for p, h in current.items() if p not in baseline},
        removed=sorted(set(baseline) - set(current)),
    )
    write(EVIDENCE / "delta.json", delta)
    parts = []
    for p in delta["modified"]:
        before = EVIDENCE / "before" / p
        if before.exists():
            parts.extend(
                difflib.unified_diff(
                    before.read_text().splitlines(True),
                    (ROOT / p).read_text().splitlines(True),
                    fromfile="before/" + p,
                    tofile=p,
                )
            )
    (EVIDENCE / "implementation.diff").write_text("".join(parts))
    print(
        json.dumps(
            dict(frozen=len(current), modified=len(delta["modified"]), added=len(delta["added"]))
        )
    )


def main():
    os.chdir(ROOT)
    if sys.argv[1:] == ["--freeze"]:
        freeze()
        return 0
    if sys.argv[1:]:
        raise SystemExit("usage: python -m tests.memory_review.gate [--freeze]")
    expected = json.loads((EVIDENCE / "implementation-freeze.json").read_text())
    if manifest() != expected:
        raise SystemExit("10D implementation drift; do not run unfrozen tests")
    out = (
        ROOT
        / "docs/acceptance"
        / ("10d-pg-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    out.mkdir(exist_ok=False)
    write(out / "implementation-before.json", expected)
    examples = out / "examples"
    examples.mkdir()
    os.environ["MMAGENT_10D_EVIDENCE_DIR"] = str(examples)
    print("10D PG evidence: " + str(out), flush=True)
    code = capture(
        out,
        "targeted",
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/memory_review/test_postgres.py",
            "-vv",
            "-rA",
            "--tb=short",
        ],
    )
    current = manifest()
    write(
        out / "verification.json",
        dict(
            direct_exit=code,
            implementation_matches=current == expected,
            status="PG_result_requires_review",
            live_domain_acceptance="live_domain_acceptance_not_enabled",
            frozen_reference=str(EVIDENCE / "implementation-freeze.json"),
        ),
    )
    print("Evidence retained: " + str(out), flush=True)
    return code if code else (0 if current == expected else 2)


if __name__ == "__main__":
    raise SystemExit(main())
