"""Deterministically render or verify stage 01B schemas and examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from .example_data import VALID_EXAMPLES, invalid_state_example, version_mismatch_example
from .models import CONTRACT_MODEL_BY_TYPE

PACKAGE_ROOT = Path(__file__).resolve().parent
SCHEMA_ROOT = PACKAGE_ROOT / "schemas" / "v1"
EXAMPLE_ROOT = PACKAGE_ROOT / "examples" / "v1"


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def validate_example_matrix() -> None:
    """Prove every generated example is owned by and tested against its real model."""

    for contract_type, model in sorted(CONTRACT_MODEL_BY_TYPE.items()):
        model.model_validate(VALID_EXAMPLES[contract_type])
        for invalid in (
            invalid_state_example(contract_type),
            version_mismatch_example(contract_type),
        ):
            try:
                model.model_validate(invalid)
            except ValidationError:
                continue
            raise RuntimeError(f"{contract_type} generated negative example was accepted")


def rendered_files() -> dict[Path, str]:
    """Return the complete deterministic v1 file set without touching the filesystem."""

    validate_example_matrix()
    rendered: dict[Path, str] = {}
    for contract_type, model in sorted(CONTRACT_MODEL_BY_TYPE.items()):
        rendered[SCHEMA_ROOT / f"{contract_type}.schema.json"] = _json_text(
            model.model_json_schema(mode="validation")
        )
        rendered[EXAMPLE_ROOT / f"{contract_type}.valid.json"] = _json_text(
            VALID_EXAMPLES[contract_type]
        )
        rendered[EXAMPLE_ROOT / f"{contract_type}.invalid-state.json"] = _json_text(
            invalid_state_example(contract_type)
        )
        rendered[EXAMPLE_ROOT / f"{contract_type}.version-mismatch.json"] = _json_text(
            version_mismatch_example(contract_type)
        )
    return rendered


def write_rendered_files() -> None:
    """Write the canonical file set."""

    rendered = rendered_files()
    SCHEMA_ROOT.mkdir(parents=True, exist_ok=True)
    EXAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    for path, content in rendered.items():
        path.write_text(content, encoding="utf-8")
    print(f"wrote {len(rendered)} deterministic contract artifacts")


def check_rendered_files() -> None:
    """Fail when a committed schema/example differs from deterministic rendering."""

    rendered = rendered_files()
    errors: list[str] = []
    for path, expected in rendered.items():
        if not path.is_file():
            errors.append(f"missing: {path.relative_to(PACKAGE_ROOT)}")
        elif path.read_text(encoding="utf-8") != expected:
            errors.append(f"stale: {path.relative_to(PACKAGE_ROOT)}")

    expected_paths = set(rendered)
    for root in (SCHEMA_ROOT, EXAMPLE_ROOT):
        if root.is_dir():
            for path in root.glob("*.json"):
                if path not in expected_paths:
                    errors.append(f"unexpected: {path.relative_to(PACKAGE_ROOT)}")
    if errors:
        raise SystemExit("contract snapshot check failed:\n" + "\n".join(errors))
    print(f"verified {len(rendered)} deterministic contract artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write canonical artifacts")
    mode.add_argument("--check", action="store_true", help="verify committed artifacts")
    args = parser.parse_args()
    if args.write:
        write_rendered_files()
    else:
        check_rendered_files()


if __name__ == "__main__":
    main()
