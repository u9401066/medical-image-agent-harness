"""Modality profile registry — single source of truth for per-modality config.

This module collapses per-modality names, required observation axes, aliases,
and agent-skill routing into one :class:`ModalityProfile` and a
:class:`ModalityRegistry`, so the rest of the pipeline asks the registry instead
of hardcoding modality names.

The registry is config-extensible: ``config.yaml`` can override or add profiles
under a ``modalities:`` section, so new modality *data* (skill name, checklist
keys, icon, display name, model hint) needs no code change. The ``Modality`` enum
remains the canonical, type-checked key set for the known modalities.

The ``model_hint`` / ``backend`` fields are forward-looking hooks for routing a
modality to a different backing model or analyzer in the future; they are carried
through the registry but not yet acted upon.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


def _as_str_sequence(value: object) -> list[str]:
    """Coerce a config value into a list of strings.

    A bare string is treated as a single element (not iterated character by
    character — a common config footgun, e.g. ``checklist_keys: "midline_shift"``).
    ``None`` becomes an empty list; other iterables are stringified element-wise.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    return [str(value)]


@dataclass(frozen=True)
class ModalityProfile:
    """Everything the harness needs to know about one imaging modality."""

    key: str
    """Canonical key, matches ``Modality.value`` for known modalities (e.g. ``"EKG"``)."""

    display_name: str = ""
    """Human-facing label; falls back to ``key`` when empty."""

    icon: str = "📊"
    """Optional human-facing icon."""

    skill_name: str = ""
    """Host skill identifier; defaults to ``dicom-<key>-analysis``."""

    checklist_keys: frozenset[str] = frozenset()
    """Required systematic-checklist keys enforced by the output validator."""

    aliases: tuple[str, ...] = ()
    """Alternate keys/spellings that resolve to this profile (case-insensitive)."""

    model_hint: str = ""
    """Optional backing-model/backend routing hint (reserved for future use)."""

    supported: bool = True
    """Whether the modality is offered for analysis / appears in the cycle."""

    def resolved_skill_name(self) -> str:
        """Return the explicit skill name, or the ``dicom-<key>-analysis`` default."""
        return self.skill_name or f"dicom-{self.key.lower().replace('_', '-')}-analysis"

    def resolved_display_name(self) -> str:
        return self.display_name or self.key

    @classmethod
    def from_dict(cls, key: str, raw: dict) -> ModalityProfile:
        """Build a profile from a raw config mapping (``config.yaml`` entry)."""
        checklist = _as_str_sequence(raw.get("checklist_keys") or raw.get("checklist"))
        aliases = _as_str_sequence(raw.get("aliases"))
        return cls(
            key=key,
            display_name=str(raw.get("display_name", "")),
            icon=str(raw.get("icon", "📊")),
            skill_name=str(raw.get("skill_name", "")),
            checklist_keys=frozenset(checklist),
            aliases=tuple(aliases),
            model_hint=str(raw.get("model_hint", "")),
            supported=bool(raw.get("supported", True)),
        )

    def merged_with(self, override: ModalityProfile) -> ModalityProfile:
        """Overlay non-default fields from ``override`` onto this profile.

        Used so a config entry can tweak a single field (e.g. ``icon``) of a
        built-in profile without re-specifying everything.
        """
        return ModalityProfile(
            key=self.key,
            display_name=override.display_name or self.display_name,
            icon=override.icon if override.icon != "📊" else self.icon,
            skill_name=override.skill_name or self.skill_name,
            checklist_keys=override.checklist_keys or self.checklist_keys,
            aliases=override.aliases or self.aliases,
            model_hint=override.model_hint or self.model_hint,
            supported=override.supported,
        )


# 16-point systematic EKG checklist (spec §3.3).
_EKG_CHECKLIST = frozenset({
    "heart_rate", "rhythm", "regularity", "axis", "p_wave", "pr_interval",
    "qrs_duration", "qrs_morphology", "st_segment", "t_wave", "qtc_interval",
    "chamber_enlargement", "conduction", "av_block", "stemi_pattern", "ischemia",
})

# 10-point systematic CXR checklist — the attending radiologist "ABCDE + zones"
# read. Mirrors the EKG approach so a chest film also gets a structured safety
# net (each axis must be addressed, so pertinent negatives like "no pleural
# effusion" cannot be silently skipped).
_CXR_CHECKLIST = frozenset({
    "projection_quality", # projection, rotation, inspiration, exposure, coverage
    "airway",             # trachea midline / patent, carina
    "lungs",              # lung fields per zone: consolidation, opacity, nodule
    "pleura",             # effusion, pneumothorax
    "cardiac_silhouette", # heart size / cardiothoracic ratio
    "mediastinum",        # width, contour, masses
    "hila",               # hilar size/contour, lymphadenopathy
    "diaphragm",          # costophrenic angles, free air under diaphragm
    "bones",              # ribs, clavicles, spine, fractures
    "soft_tissue",        # subcutaneous emphysema, masses
    "lines_tubes",        # ETT, CVC, NG, chest tube positions
})

_CT_BRAIN_CHECKLIST = frozenset({
    "study_completeness",
    "extra_axial",
    "ventricles_cisterns",
    "midline_mass_effect",
    "parenchyma",
    "deep_gray",
    "posterior_fossa",
    "calvarium_soft_tissue",
    "visible_vessels_sinuses_orbits",
})

# Built-in profiles for the currently shipped modalities. New modalities can be
# added here, or — preferably — via the ``modalities:`` section of config.yaml.
_BUILTIN_PROFILES: tuple[ModalityProfile, ...] = (
    ModalityProfile(
        key="EKG",
        display_name="12-Lead EKG",
        icon="🫀",
        skill_name="dicom-ekg-analysis",
        checklist_keys=_EKG_CHECKLIST,
        aliases=("ECG", "EKG_12LEAD"),
    ),
    ModalityProfile(
        key="CXR",
        display_name="Chest X-Ray",
        icon="🫁",
        skill_name="dicom-cxr-analysis",
        checklist_keys=_CXR_CHECKLIST,
        aliases=("CHEST_XRAY", "CHEST_X_RAY"),
    ),
    ModalityProfile(
        key="CT_BRAIN",
        display_name="CT Brain",
        icon="🧠",
        skill_name="dicom-ct-brain-analysis",
        checklist_keys=_CT_BRAIN_CHECKLIST,
        aliases=("CTBRAIN", "BRAIN_CT"),
    ),
)


class ModalityRegistry:
    """Lookup table of :class:`ModalityProfile` keyed by modality key/alias."""

    def __init__(self, profiles: Iterable[ModalityProfile] = ()) -> None:
        self._by_key: dict[str, ModalityProfile] = {}
        self._index: dict[str, str] = {}  # normalized key/alias -> canonical key
        for profile in profiles:
            self.register(profile)

    @staticmethod
    def _norm(value: str) -> str:
        return value.strip().upper()

    def register(self, profile: ModalityProfile, *, merge: bool = False) -> None:
        """Add or replace a profile.

        With ``merge=True`` an existing profile of the same key is overlaid with
        the new profile's non-default fields (used for config overrides).
        """
        key = self._norm(profile.key)
        if merge and key in self._by_key:
            profile = self._by_key[key].merged_with(profile)
        self._by_key[key] = profile
        self._index[key] = key
        for alias in profile.aliases:
            self._index[self._norm(alias)] = key

    def get(self, key: str) -> ModalityProfile | None:
        canonical = self._index.get(self._norm(key))
        return self._by_key.get(canonical) if canonical else None

    def resolve(self, key: str) -> ModalityProfile:
        """Return the profile for ``key``, or a generic fallback profile.

        The fallback keeps the pipeline working for an unknown modality string
        instead of raising — it derives a default skill name from the key.
        """
        profile = self.get(key)
        if profile is not None:
            return profile
        return ModalityProfile(key=key)

    def keys(self) -> list[str]:
        return list(self._by_key.keys())

    def supported_keys(self) -> list[str]:
        return [k for k, p in self._by_key.items() if p.supported]

    def profiles(self) -> list[ModalityProfile]:
        return list(self._by_key.values())

    def __contains__(self, key: str) -> bool:
        return self._norm(key) in self._index

    def __iter__(self):
        return iter(self._by_key.values())


def default_registry() -> ModalityRegistry:
    """Return a fresh registry pre-populated with the built-in profiles."""
    return ModalityRegistry(_BUILTIN_PROFILES)


def build_registry(config_modalities: object = None) -> ModalityRegistry:
    """Build a registry from built-ins overlaid with optional config entries.

    ``config_modalities`` may be:
      * ``None`` — built-ins only.
      * a mapping ``{key: {field: value, ...}}``.
      * a list of mappings, each containing a ``key`` field.
    Config entries override matching built-in fields and add new modalities.
    """
    registry = default_registry()
    if not config_modalities:
        return registry

    entries: list[tuple[str, dict]] = []
    if isinstance(config_modalities, dict):
        for key, raw in config_modalities.items():
            if isinstance(raw, dict):
                entries.append((str(key), raw))
    elif isinstance(config_modalities, list):
        for raw in config_modalities:
            if isinstance(raw, dict) and raw.get("key"):
                entries.append((str(raw["key"]), raw))

    for key, raw in entries:
        registry.register(ModalityProfile.from_dict(key, raw), merge=True)
    return registry


# Process-wide active registry. ``__main__`` replaces this with a config-built
# registry at startup; components that are not explicitly handed a registry fall
# back to this one. Defaults to the built-ins so library/test usage works without
# any setup.
_ACTIVE_REGISTRY: ModalityRegistry = default_registry()


def get_active_registry() -> ModalityRegistry:
    return _ACTIVE_REGISTRY


def set_active_registry(registry: ModalityRegistry) -> None:
    global _ACTIVE_REGISTRY
    _ACTIVE_REGISTRY = registry
