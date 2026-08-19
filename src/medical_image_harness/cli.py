"""Command-line validation and fingerprint utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical_image_harness.provenance import sha256_file
from medical_image_harness.resources import skill_sha256
from medical_image_harness.schema import schema_text, validation_errors


def _validate(path: Path) -> int:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_non_finite_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"invalid JSON: {exc}")
        return 2
    errors = validation_errors(payload)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"valid: {path}")
    return 0


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _fingerprint() -> int:
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / "analysis-result.schema.json"
    schema_digest = sha256_file(schema_path) if schema_path.is_file() else "installed"
    print(
        json.dumps(
            {
                "skill_sha256": skill_sha256(),
                "schema_sha256": schema_digest,
                "schema_bytes": len(schema_text().encode("utf-8")),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="medical-image-harness")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a result JSON file")
    validate.add_argument("path", type=Path)
    commands.add_parser("fingerprint", help="print skill and schema fingerprints")
    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate(args.path)
    return _fingerprint()


if __name__ == "__main__":
    raise SystemExit(main())
