"""Canonical EKG lead names shared across layout consumers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from medical_image_harness.models import RegionRect

STANDARD_EKG_LEADS: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)

_CANONICAL_BY_FOLDED = {name.casefold(): name for name in STANDARD_EKG_LEADS}
_ROW_STRIP_EVIDENCE_METHOD = "local_black_ink_row_periodicity_v2"


def canonical_ekg_lead_name(value: object) -> str | None:
    """Return ``lead_<name>`` for common model and app lead-name spellings."""

    raw = str(value or "").strip()
    folded = raw.casefold()
    for prefix in ("lead_", "lead-", "lead "):
        if folded.startswith(prefix):
            raw = raw[len(prefix) :].strip()
            break
    compact = "".join(char for char in raw if not char.isspace() and char not in "_-")
    canonical = _CANONICAL_BY_FOLDED.get(compact.casefold())
    return f"lead_{canonical}" if canonical is not None else None


@dataclass(frozen=True)
class EkgLeadRegion:
    """One validated, visible lead region in normalized original-ROI space."""

    name: str
    bbox: RegionRect


@dataclass(frozen=True)
class EkgLeadInventory:
    """Typed parse result for an EKG layout declaration."""

    leads: tuple[EkgLeadRegion, ...]
    source_present: bool
    malformed_entries: int
    duplicate_names: tuple[str, ...]
    missing_names: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return (
            self.source_present
            and not self.malformed_entries
            and not self.duplicate_names
            and not self.missing_names
        )

    def by_name(self) -> dict[str, RegionRect]:
        return {lead.name: lead.bbox for lead in self.leads}

    def validation_warnings(self) -> list[str]:
        warnings: list[str] = []
        if not self.source_present:
            return ["EKG layout is missing a lead inventory"]
        if self.malformed_entries:
            warnings.append(
                "EKG layout has "
                f"{self.malformed_entries} malformed or hidden lead entr"
                f"{'y' if self.malformed_entries == 1 else 'ies'}"
            )
        if self.duplicate_names:
            warnings.append(
                "EKG layout has duplicate leads: "
                + ", ".join(self.duplicate_names)
            )
        if self.missing_names:
            warnings.append(
                "EKG layout is missing visible leads: "
                + ", ".join(self.missing_names)
            )
        return warnings


def parse_ekg_lead_inventory(layout: object) -> EkgLeadInventory:
    """Validate and normalize the model-declared standard 12-lead inventory."""

    raw_leads = layout.get("leads") if isinstance(layout, dict) else None
    if not isinstance(raw_leads, list):
        return EkgLeadInventory(
            leads=(),
            source_present=False,
            malformed_entries=0,
            duplicate_names=(),
            missing_names=tuple(f"lead_{name}" for name in STANDARD_EKG_LEADS),
        )

    parsed: list[EkgLeadRegion] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    malformed = 0
    for raw in raw_leads:
        if not isinstance(raw, dict) or raw.get("label_visible") is not True:
            malformed += 1
            continue
        name = canonical_ekg_lead_name(raw.get("name"))
        bbox = parse_normalized_region(raw.get("bbox"))
        if name is None or bbox is None:
            malformed += 1
            continue
        if name in seen:
            duplicates.add(name)
            continue
        seen.add(name)
        parsed.append(EkgLeadRegion(name=name, bbox=bbox))

    expected = tuple(f"lead_{name}" for name in STANDARD_EKG_LEADS)
    return EkgLeadInventory(
        leads=tuple(parsed),
        source_present=True,
        malformed_entries=malformed,
        duplicate_names=tuple(name for name in expected if name in duplicates),
        missing_names=tuple(name for name in expected if name not in seen),
    )


def normalize_ekg_row_strip_layout(
    layout: object,
    *,
    image_evidence: dict[str, object] | None = None,
) -> tuple[dict, bool]:
    """Repair a strongly evidenced 12-row inventory and clipped tail entries.

    This is deliberately narrow: the layout must explicitly claim 12 leads,
    contain at least 10 canonical labels in standard order across at least nine
    distinct vertical rows, span most of the image, and share nearly identical
    x origins. Six to nine declarations are accepted only when deterministic
    image evidence independently confirms 12 full-width black-ink rows. This
    evidence also permits a compact exact ``lead_order`` declaration with no
    model-authored geometry. This covers clipped/compact model JSON without
    turning a true 3x4 or partial capture into a full row strip.
    """
    if not isinstance(layout, dict):
        return {}, False
    raw_leads = layout.get("leads")
    image_confirmed = _confirmed_12_row_strip(image_evidence)
    compact_geometry_supported = _supports_compact_12_row_geometry(image_evidence)
    declared_format = str(layout.get("format") or "").casefold()
    compact_format = "".join(char for char in declared_format if char.isalnum())
    if "12lead" not in compact_format:
        return dict(layout), False

    expected = tuple(f"lead_{name}" for name in STANDARD_EKG_LEADS)
    raw_lead_order = layout.get("lead_order")
    compact_lead_order = (
        tuple(canonical_ekg_lead_name(name) for name in raw_lead_order)
        if isinstance(raw_lead_order, list)
        else ()
    )
    if (
        compact_geometry_supported
        and compact_lead_order == expected
        and (not isinstance(raw_leads, list) or not raw_leads)
    ):
        return _build_normalized_row_strip_layout(
            layout,
            recovered_count=len(compact_lead_order),
            image_confirmed=True,
        )

    minimum_leads = 6 if image_confirmed else 10
    if (
        not isinstance(raw_leads, list)
        or not minimum_leads <= len(raw_leads) <= 12
    ):
        return dict(layout), False
    # Pixel periodicity is not permission to invent a missing visibility field.
    # Keep the explicitly declared compact lead_order path above separate.
    if any(
        not isinstance(raw, dict)
        or type(raw.get("label_visible")) is not bool
        for raw in raw_leads
    ):
        return dict(layout), False
    declared_names = tuple(
        name
        for raw in raw_leads
        if isinstance(raw, dict)
        if (name := canonical_ekg_lead_name(raw.get("name"))) is not None
    )
    allow_degenerate_geometry = image_confirmed and declared_names == expected

    observations: list[tuple[float, str, float, float]] = []
    for raw in raw_leads:
        if not isinstance(raw, dict) or raw.get("label_visible") is not True:
            if image_confirmed:
                continue
            return dict(layout), False
        name = canonical_ekg_lead_name(raw.get("name"))
        values = _finite_bbox_values(raw.get("bbox"))
        if name is None or values is None:
            continue
        x, y, width, _height = values
        if x > 1.0 or y > 1.0 or width > 1.0:
            continue
        observations.append((y, name, x, width))

    observations.sort(key=lambda item: item[0])
    observed_names = tuple(item[1] for item in observations)
    y_values = [item[0] for item in observations]
    x_values = [item[2] for item in observations]
    widths = [item[3] for item in observations]
    minimum_distinct_rows = 6 if image_confirmed else 9
    expected_positions = tuple(
        expected.index(name) for name in observed_names if name in expected
    )
    geometry_is_plausible = (
        len(observations) >= minimum_leads
        and len(set(observed_names)) == len(observed_names)
        and len(expected_positions) == len(observed_names)
        and expected_positions == tuple(range(len(expected_positions)))
        and len({round(value, 3) for value in y_values}) >= minimum_distinct_rows
        and max(y_values) - min(y_values) >= 0.70
        and max(x_values) - min(x_values) <= 0.08
        and (len(observations) >= 10 or min(widths) >= 0.75)
    )
    if not geometry_is_plausible and not allow_degenerate_geometry:
        return dict(layout), False

    inventory = parse_ekg_lead_inventory(layout)
    geometry_needs_repair = (
        not geometry_is_plausible
        or len(raw_leads) != len(STANDARD_EKG_LEADS)
        or not inventory.complete
        or not widths
        or min(widths) < 0.75
        or "3x4" in declared_format
    )
    if not geometry_needs_repair:
        return dict(layout), False

    return _build_normalized_row_strip_layout(
        layout,
        recovered_count=len(declared_names),
        image_confirmed=image_confirmed,
    )


def _build_normalized_row_strip_layout(
    layout: dict,
    *,
    recovered_count: int,
    image_confirmed: bool,
) -> tuple[dict, bool]:
    expected = tuple(f"lead_{name}" for name in STANDARD_EKG_LEADS)
    row_height = 1.0 / len(STANDARD_EKG_LEADS)
    normalized = dict(layout)
    normalized["format"] = "12lead_12x1"
    normalized["leads"] = [
        {
            "name": name.removeprefix("lead_"),
            "label_visible": True,
            "bbox": [0.0, index * row_height, 1.0, row_height],
        }
        for index, name in enumerate(expected)
    ]
    notes = str(normalized.get("notes") or "").strip()
    repair_note = (
        "Lead geometry normalized from a detected 12-row strip "
        f"({recovered_count}/12 model lead declarations recovered)."
    )
    if image_confirmed:
        repair_note = f"{repair_note} Local row periodicity confirmed 12 rows."
    normalized["notes"] = f"{notes} {repair_note}".strip()
    return normalized, True


def _confirmed_12_row_strip(evidence: dict[str, object] | None) -> bool:
    if not isinstance(evidence, dict):
        return False
    return (
        evidence.get("method") == _ROW_STRIP_EVIDENCE_METHOD
        and evidence.get("status") == "ok"
        and evidence.get("is_12_row_strip") is True
        and evidence.get("detected_row_count") == 12
        and isinstance(evidence.get("consistent_gap_count"), int)
        and int(evidence["consistent_gap_count"]) >= 10
    )


def _supports_compact_12_row_geometry(
    evidence: dict[str, object] | None,
) -> bool:
    """Accept bounded detector misses only for an exact compact layout."""

    if _confirmed_12_row_strip(evidence):
        return True
    if not isinstance(evidence, dict):
        return False
    row_count = evidence.get("detected_row_count")
    gap_count = evidence.get("consistent_gap_count")
    period = evidence.get("median_row_period_normalized")
    span = evidence.get("vertical_span_ratio")
    minimum_gap_count = 6 if isinstance(row_count, int) and row_count <= 10 else 9
    return bool(
        evidence.get("method") == _ROW_STRIP_EVIDENCE_METHOD
        and evidence.get("status") == "ok"
        and isinstance(row_count, int)
        and not isinstance(row_count, bool)
        and 9 <= row_count <= 13
        and isinstance(gap_count, int)
        and not isinstance(gap_count, bool)
        and gap_count >= minimum_gap_count
        and isinstance(period, int | float)
        and not isinstance(period, bool)
        and 0.075 <= float(period) <= 0.09
        and isinstance(span, int | float)
        and not isinstance(span, bool)
        and float(span) >= 0.90
    )


def parse_normalized_region(value: object) -> RegionRect | None:
    """Parse strict normalized list- or object-shaped geometry."""

    values = _finite_bbox_values(value)
    if values is None:
        return None
    x, y, w, h = values
    if x < 0.0 or y < 0.0 or w <= 0.0 or h <= 0.0:
        return None
    if x + w > 1.0 + 1e-9 or y + h > 1.0 + 1e-9:
        return None
    try:
        return RegionRect(x=x, y=y, w=w, h=h)
    except ValueError:
        return None


def _finite_bbox_values(value: object) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        raw_values = tuple(value.get(key) for key in ("x", "y", "w", "h"))
    elif isinstance(value, list | tuple) and len(value) >= 4:
        raw_values = tuple(value[:4])
    else:
        return None
    try:
        values = tuple(float(item) for item in raw_values)
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        return None
    x, y, width, height = values
    if x < 0.0 or y < 0.0 or width < 0.0 or height < 0.0:
        return None
    return x, y, width, height
