"""Clinical rule-pack loader — the modular-update boundary.

Built-in clinical rules ship as cited data in :mod:`medical_image_harness.clinical_rules`.
This loader lets a deployment **update the rule set without touching code** by
dropping versioned YAML rule packs (``*.rules.yaml``) into a directory. When a
diagnostic guideline changes, edit/add a YAML rule there — no rebuild required.

A pack entry overrides a built-in rule with the same ``id`` (so you can tighten,
relax, or re-cite an existing rule) or adds a brand-new rule. Malformed entries
are skipped with a logged warning rather than crashing the harness, so a typo in
a rule pack can never take the whole agent down.

Infrastructure layer: owns YAML I/O; returns pure domain objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
import yaml

from medical_image_harness.clinical_rules import (
    ClinicalConsistencyEngine,
    ClinicalRule,
    ConditionError,
    RuleCondition,
    builtin_rules,
    group_by_modality,
)
from medical_image_harness.models import Severity

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)

# Only files matching this suffix are loaded, so an ``*.rules.yaml.example`` or
# other YAML in the directory is ignored.
RULE_PACK_GLOB = "*.rules.yaml"


def _parse_severity(value: object) -> Severity | None:
    if value is None or value == "":
        return None
    return Severity(str(value).strip().lower())


def _parse_condition(raw: dict) -> RuleCondition:
    values = raw.get("values") or ()
    if isinstance(values, str):
        values = (values,)
    return RuleCondition(
        field=str(raw["field"]),
        op=str(raw["op"]),
        values=tuple(str(v) for v in values),
        value=str(raw.get("value", "")),
    )


def _parse_rule(raw: dict) -> ClinicalRule:
    conditions = tuple(
        _parse_condition(c) for c in (raw.get("conditions") or [])
    )
    if not conditions:
        raise ConditionError(f"rule '{raw.get('id')}' has no conditions")
    # A rule pack may never ship an undocumented rule: a human-readable
    # ``description`` is mandatory so every override/addition stays auditable.
    description = str(raw.get("description", "")).strip()
    if not description:
        raise ConditionError(
            f"rule '{raw.get('id')}' missing required 'description' "
            "(needed for human audit)"
        )
    return ClinicalRule(
        id=str(raw["id"]),
        modality=str(raw["modality"]),
        description=description,
        conditions=conditions,
        message=str(raw["message"]),
        guideline=str(raw.get("guideline", "")),
        guideline_version=str(raw.get("guideline_version", "")),
        effective_date=str(raw.get("effective_date", "")),
        source_url=str(raw.get("source_url", "")),
        escalate_to=_parse_severity(raw.get("escalate_to")),
        require_review=bool(raw.get("require_review", True)),
    )


def load_rule_pack_dir(directory: Path) -> list[ClinicalRule]:
    """Load every ``*.rules.yaml`` rule from ``directory`` (recursively).

    Returns the parsed rules. Missing directory → empty list. Individual bad
    files or bad rules are skipped with a warning, never raised.
    """
    if not directory.is_dir():
        return []
    rules: list[ClinicalRule] = []
    for path in sorted(directory.glob(RULE_PACK_GLOB)):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError) as exc:
            logger.warning("clinical_rule_pack_unreadable", path=str(path), error=str(exc))
            continue
        entries = raw.get("rules", raw if isinstance(raw, list) else [])
        if not isinstance(entries, list):
            logger.warning("clinical_rule_pack_malformed", path=str(path))
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                rules.append(_parse_rule(entry))
            except (KeyError, ConditionError, ValueError) as exc:
                logger.warning(
                    "clinical_rule_skipped",
                    path=str(path),
                    rule_id=entry.get("id"),
                    error=str(exc),
                )
    return rules


def merge_rules(
    builtins: list[ClinicalRule], overrides: list[ClinicalRule]
) -> list[ClinicalRule]:
    """Overlay ``overrides`` onto ``builtins`` by rule id (override wins)."""
    by_id: dict[str, ClinicalRule] = {r.id: r for r in builtins}
    for rule in overrides:
        by_id[rule.id] = rule
    return list(by_id.values())


def build_clinical_engine(
    rule_pack_dir: Path | None = None,
) -> ClinicalConsistencyEngine:
    """Build the engine from built-ins overlaid with an optional rule-pack dir.

    ``rule_pack_dir`` is where deployments drop updated guideline rules; when it
    is ``None`` or empty the engine runs the built-in rules only.
    """
    builtins = list(builtin_rules())
    overrides = load_rule_pack_dir(rule_pack_dir) if rule_pack_dir else []
    if overrides:
        logger.info("clinical_rule_overrides_loaded", count=len(overrides))
    merged = merge_rules(builtins, overrides)
    return ClinicalConsistencyEngine(group_by_modality(merged))
