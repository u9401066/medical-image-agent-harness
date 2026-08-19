"""Canonical EKG lead names shared across layout consumers."""

from __future__ import annotations

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
        if not isinstance(raw, dict) or raw.get("label_visible") is False:
            malformed += 1
            continue
        name = canonical_ekg_lead_name(raw.get("name"))
        bbox = _normalized_region(raw.get("bbox"))
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


def _normalized_region(value: object) -> RegionRect | None:
    if not isinstance(value, list | tuple) or len(value) < 4:
        return None
    try:
        x, y, w, h = (float(item) for item in value[:4])
    except (TypeError, ValueError):
        return None
    if x < 0.0 or y < 0.0 or w <= 0.0 or h <= 0.0:
        return None
    if x + w > 1.0 + 1e-9 or y + h > 1.0 + 1e-9:
        return None
    try:
        return RegionRect(x=x, y=y, w=w, h=h)
    except ValueError:
        return None
