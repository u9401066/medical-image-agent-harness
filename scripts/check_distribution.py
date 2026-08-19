#!/usr/bin/env python3
"""Verify that built distributions contain the canonical skill and schema."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
REQUIRED = {
    "medical_image_harness/schemas/analysis-result.schema.json",
    "medical_image_harness/skills/medical-image-reading/SKILL.md",
    "medical_image_harness/skills/medical-image-reading/references/core-protocol.md",
    "medical_image_harness/skills/medical-image-reading/references/output-contract.md",
}


def main() -> int:
    wheels = sorted(DIST.glob("medical_image_agent_harness-*.whl"))
    if len(wheels) != 1:
        print(f"expected one built wheel, found {len(wheels)}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        missing = sorted(REQUIRED.difference(names))
        source_references = {
            "medical_image_harness/skills/medical-image-reading/references/" + path.name
            for path in (ROOT / ".agents/skills/medical-image-reading/references").glob("*.md")
        }
        missing.extend(sorted(source_references.difference(names)))
    if missing:
        print("built wheel is missing:\n" + "\n".join(missing), file=sys.stderr)
        return 1
    print("Built wheel contains the canonical skill and result schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
