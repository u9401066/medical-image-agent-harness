#!/usr/bin/env python3
"""Static Codex/Copilot, public-boundary, and skill-integrity checks."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".agents/skills/medical-image-reading/SKILL.md"
NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def frontmatter(path: Path, failures: list[str]) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.DOTALL)
    if not match:
        fail(f"{path.relative_to(ROOT)}: missing YAML frontmatter", failures)
        return {}, text
    data = yaml.safe_load(match.group(1))
    if not isinstance(data, dict):
        fail(f"{path.relative_to(ROOT)}: frontmatter is not a mapping", failures)
        return {}, match.group(2)
    return data, match.group(2)


def check_skill(failures: list[str]) -> None:
    data, body = frontmatter(SKILL, failures)
    name = data.get("name")
    description = data.get("description")
    if name != SKILL.parent.name or not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        fail("canonical skill name/path mismatch", failures)
    if not isinstance(description, str) or not 1 <= len(description) <= 1024:
        fail("canonical skill description must contain 1-1024 characters", failures)
    if len(body) > 30_000:
        fail("canonical skill body exceeds the conservative host limit", failures)
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", body):
        if "://" in target:
            continue
        resolved = (SKILL.parent / target).resolve()
        if SKILL.parent.resolve() not in resolved.parents or not resolved.is_file():
            fail(f"skill reference is missing or escapes its root: {target}", failures)
    duplicate_roots = [ROOT / ".github/skills", ROOT / ".claude/skills"]
    if any(path.exists() for path in duplicate_roots):
        fail("duplicate skill discovery directory present", failures)


def check_adapters(failures: list[str]) -> None:
    copilot = (ROOT / ".github/copilot-instructions.md").read_text(encoding="utf-8")
    if ".agents/skills/medical-image-reading/SKILL.md" not in copilot:
        fail("Copilot instructions do not route to the canonical skill", failures)
    agent_data, agent_body = frontmatter(
        ROOT / ".github/agents/medical-image-reader.agent.md", failures
    )
    if not agent_data.get("description") or len(agent_body) > 30_000:
        fail("Copilot custom agent metadata/body is invalid", failures)
    codex = tomllib.loads(
        (ROOT / ".codex/agents/medical-image-reviewer.toml").read_text(encoding="utf-8")
    )
    required = {"name", "description", "developer_instructions"}
    if not required.issubset(codex):
        fail("Codex custom agent lacks required fields", failures)
    if SKILL.stat().st_size + (ROOT / "AGENTS.md").stat().st_size > 32 * 1024:
        fail("bootstrap instructions exceed the conservative Codex budget", failures)


def check_schema(failures: list[str]) -> None:
    path = ROOT / "schemas/analysis-result.schema.json"
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        manifest = yaml.safe_load(
            (ROOT / "harness-manifest.yaml").read_text(encoding="utf-8")
        )
        if schema["properties"]["protocol_version"].get("const") != manifest.get(
            "protocol_version"
        ):
            fail("schema and harness manifest protocol versions differ", failures)
        if schema["properties"]["schema_version"].get("const") != manifest.get(
            "schema_version"
        ):
            fail("schema and harness manifest schema versions differ", failures)
    except Exception as exc:
        fail(f"invalid result schema: {exc}", failures)


def check_manifest(failures: list[str]) -> None:
    manifest = yaml.safe_load((ROOT / "harness-manifest.yaml").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        fail("harness manifest is not a mapping", failures)
        return
    canonical = manifest.get("canonical_skill")
    if canonical != SKILL.relative_to(ROOT).as_posix():
        fail("harness manifest canonical skill path is stale", failures)
    protocol_version = manifest.get("protocol_version")
    reference_files = [SKILL, *sorted((SKILL.parent / "references").glob("*.md"))]
    protocol_text = "\n".join(path.read_text(encoding="utf-8") for path in reference_files)
    if f"Protocol version: `{protocol_version}`" not in protocol_text:
        fail("harness manifest protocol version is not present in the protocol", failures)
    for invariant in manifest.get("mandatory_invariants", []):
        if not isinstance(invariant, str) or invariant not in protocol_text:
            fail(f"mandatory invariant is absent from canonical protocol: {invariant!r}", failures)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def check_public_boundary(failures: list[str]) -> None:
    policy = tomllib.loads((ROOT / "PUBLIC_SURFACE.toml").read_text(encoding="utf-8"))
    forbidden_suffixes = tuple(policy["forbidden_tracked_suffixes"])
    forbidden_paths = tuple(policy["forbidden_tracked_paths"])
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative.endswith(forbidden_suffixes) or any(
            part in forbidden_paths for part in path.relative_to(ROOT).parts
        ):
            fail(f"forbidden public artifact: {relative}", failures)
    prefixes = tuple(policy["forbidden_import_prefixes"])
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        for imported in imports:
            if imported.startswith(prefixes):
                fail(f"forbidden import {imported!r} in {path.relative_to(ROOT)}", failures)


def main() -> int:
    failures: list[str] = []
    check_skill(failures)
    check_adapters(failures)
    check_schema(failures)
    check_manifest(failures)
    check_public_boundary(failures)
    if failures:
        print("\n".join(f"ERROR: {failure}" for failure in failures), file=sys.stderr)
        return 1
    print("Codex/Copilot compatibility and public boundary checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
