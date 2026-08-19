"""Tests for the modality profile registry (single source of truth)."""

from __future__ import annotations

import pytest

from medical_image_harness.hooks import AnalyzeRequest, HookError
from medical_image_harness.input_validation import InputGuard
from medical_image_harness.models import (
    AnalysisResult,
    ChecklistItem,
    Finding,
    Modality,
    Severity,
)
from medical_image_harness.output_validation import OutputValidator
from medical_image_harness.profiles import (
    ModalityProfile,
    ModalityRegistry,
    build_registry,
    default_registry,
)

# ── ModalityProfile ──────────────────────────────────────────────────


class TestModalityProfile:
    def test_resolved_skill_name_explicit(self):
        p = ModalityProfile(key="EKG", skill_name="dicom-ekg-analysis")
        assert p.resolved_skill_name() == "dicom-ekg-analysis"

    def test_resolved_skill_name_default_from_key(self):
        p = ModalityProfile(key="CT_BRAIN")
        assert p.resolved_skill_name() == "dicom-ct-brain-analysis"

    def test_resolved_display_name_fallback_to_key(self):
        assert ModalityProfile(key="KUB").resolved_display_name() == "KUB"
        assert (
            ModalityProfile(key="KUB", display_name="Abdo X-Ray").resolved_display_name()
            == "Abdo X-Ray"
        )

    def test_from_dict_full(self):
        p = ModalityProfile.from_dict(
            "KUB",
            {
                "display_name": "KUB",
                "icon": "🩻",
                "skill_name": "dicom-kub-analysis",
                "checklist_keys": ["bowel_gas", "free_air"],
                "aliases": ["ABDOMINAL_XRAY"],
                "model_hint": "x-model",
                "supported": True,
            },
        )
        assert p.key == "KUB"
        assert p.icon == "🩻"
        assert p.checklist_keys == frozenset({"bowel_gas", "free_air"})
        assert p.aliases == ("ABDOMINAL_XRAY",)
        assert p.model_hint == "x-model"

    def test_from_dict_checklist_alias_field(self):
        p = ModalityProfile.from_dict("X", {"checklist": ["a", "b"]})
        assert p.checklist_keys == frozenset({"a", "b"})

    def test_from_dict_checklist_string_is_single_key(self):
        # A bare string must NOT be iterated character-by-character.
        p = ModalityProfile.from_dict("X", {"checklist_keys": "midline_shift"})
        assert p.checklist_keys == frozenset({"midline_shift"})

    def test_from_dict_aliases_string_is_single_alias(self):
        p = ModalityProfile.from_dict("X", {"aliases": "ECG"})
        assert p.aliases == ("ECG",)

    def test_merged_with_overlays_only_nondefault(self):
        base = ModalityProfile(
            key="EKG",
            display_name="12-Lead EKG",
            icon="🫀",
            skill_name="dicom-ekg-analysis",
            checklist_keys=frozenset({"rhythm"}),
        )
        override = ModalityProfile(key="EKG", icon="❤️")
        merged = base.merged_with(override)
        assert merged.icon == "❤️"  # overridden
        assert merged.display_name == "12-Lead EKG"  # preserved
        assert merged.skill_name == "dicom-ekg-analysis"  # preserved
        assert merged.checklist_keys == frozenset({"rhythm"})  # preserved

    def test_merged_with_default_icon_does_not_overwrite(self):
        base = ModalityProfile(key="EKG", icon="🫀")
        override = ModalityProfile(key="EKG", display_name="X")  # icon left as 📊
        merged = base.merged_with(override)
        assert merged.icon == "🫀"  # default 📊 override must not clobber
        assert merged.display_name == "X"

    def test_merged_with_supported_false_propagates(self):
        base = ModalityProfile(key="EKG", supported=True)
        override = ModalityProfile(key="EKG", supported=False)
        assert base.merged_with(override).supported is False


# ── ModalityRegistry ─────────────────────────────────────────────────


class TestModalityRegistry:
    def test_default_registry_has_builtins(self):
        reg = default_registry()
        assert "EKG" in reg
        assert "CXR" in reg
        assert "CT_BRAIN" in reg

    def test_get_is_case_insensitive(self):
        reg = default_registry()
        assert reg.resolve("ekg").key == "EKG"
        assert reg.resolve("  EkG ").key == "EKG"

    def test_alias_resolution(self):
        reg = default_registry()
        assert reg.resolve("ECG").key == "EKG"
        assert reg.resolve("chest_xray").key == "CXR"

    def test_resolve_unknown_returns_fallback(self):
        reg = default_registry()
        prof = reg.resolve("UNKNOWN_MOD")
        assert prof.key == "UNKNOWN_MOD"
        assert prof.resolved_skill_name() == "dicom-unknown-mod-analysis"

    def test_get_unknown_returns_none(self):
        assert default_registry().get("NOPE") is None

    def test_supported_keys_excludes_unsupported(self):
        reg = ModalityRegistry(
            [
                ModalityProfile(key="A"),
                ModalityProfile(key="B", supported=False),
            ]
        )
        assert reg.supported_keys() == ["A"]

    def test_register_merge_overlays_builtin(self):
        reg = default_registry()
        reg.register(ModalityProfile(key="EKG", icon="❤️"), merge=True)
        assert reg.resolve("EKG").icon == "❤️"
        # built-in skill name preserved through merge
        assert reg.resolve("EKG").resolved_skill_name() == "dicom-ekg-analysis"

    def test_alias_collision_last_registered_wins(self):
        # If a later profile claims an alias already used, the alias points at
        # the most recently registered owner (documented last-wins behavior).
        reg = ModalityRegistry([ModalityProfile(key="A", aliases=("X",))])
        reg.register(ModalityProfile(key="B", aliases=("X",)))
        assert reg.resolve("X").key == "B"
        # Canonical keys both remain reachable.
        assert reg.resolve("A").key == "A"
        assert reg.resolve("B").key == "B"


# ── build_registry (config extension) ────────────────────────────────


class TestBuildRegistry:
    def test_none_returns_builtins(self):
        reg = build_registry(None)
        assert "EKG" in reg and "CXR" in reg

    def test_dict_config_overrides_builtin(self):
        reg = build_registry({"EKG": {"icon": "💙"}})
        assert reg.resolve("EKG").icon == "💙"
        assert reg.resolve("EKG").resolved_display_name() == "12-Lead EKG"

    def test_dict_config_adds_new_modality(self):
        reg = build_registry(
            {
                "KUB": {
                    "display_name": "KUB",
                    "skill_name": "dicom-kub-analysis",
                    "checklist_keys": ["bowel_gas", "free_air"],
                    "aliases": ["ABDOMINAL_XRAY"],
                }
            }
        )
        assert "KUB" in reg
        assert reg.resolve("ABDOMINAL_XRAY").key == "KUB"
        assert reg.resolve("KUB").checklist_keys == frozenset({"bowel_gas", "free_air"})

    def test_list_config_adds_new_modality(self):
        reg = build_registry([{"key": "ECHO", "display_name": "Echocardiogram"}])
        assert reg.resolve("ECHO").resolved_display_name() == "Echocardiogram"

    def test_list_config_ignores_entries_without_key(self):
        reg = build_registry([{"display_name": "orphan"}])
        # Only built-ins remain.
        assert set(reg.keys()) == {"EKG", "CXR", "CT_BRAIN"}


# ── Registry injection into hooks ────────────────────────────────────


def _make_request(modality: Modality, regions: list[str]) -> AnalyzeRequest:
    return AnalyzeRequest(
        image_base64="A" * 5000,
        modality=modality,
        valid_regions=regions,
    )


def _make_cxr_result(checklist: dict[str, ChecklistItem]) -> AnalysisResult:
    return AnalysisResult(
        modality=Modality.CXR,
        summary="No acute findings",
        severity=Severity.NORMAL,
        findings=[
            Finding(
                id="f1",
                regions=["right_upper_lung"],
                label="Clear",
                detail="No infiltrate",
                severity=Severity.NORMAL,
            )
        ],
        checklist=checklist,
        analysis_time_ms=100,
        model_used="test",
    )


class TestRegistryInjectionIntoHooks:
    def test_input_guard_rejects_unsupported_modality(self):
        reg = ModalityRegistry([ModalityProfile(key="EKG")])  # CXR not registered
        guard = InputGuard(registry=reg)
        req = _make_request(Modality.CXR, ["right_upper_lung"])
        with pytest.raises(HookError):
            guard.pre_analyze(req)

    def test_input_guard_accepts_config_added_modality(self):
        reg = build_registry({"CXR": {}})
        guard = InputGuard(registry=reg)
        req = _make_request(Modality.CXR, ["right_upper_lung"])
        assert guard.pre_analyze(req) is req

    def test_output_validator_enforces_config_checklist_strict(self):
        reg = build_registry({"CXR": {"checklist_keys": ["lungs", "heart"]}})
        validator = OutputValidator(strict=True, registry=reg)
        req = _make_request(Modality.CXR, ["right_upper_lung"])
        result = _make_cxr_result(
            {"lungs": ChecklistItem(value="clear", status=Severity.NORMAL)}
        )
        with pytest.raises(HookError, match="heart"):
            validator.post_analyze(req, result)

    def test_output_validator_passes_with_full_config_checklist(self):
        reg = build_registry({"CXR": {"checklist_keys": ["lungs", "heart"]}})
        validator = OutputValidator(strict=True, registry=reg)
        req = _make_request(Modality.CXR, ["right_upper_lung"])
        result = _make_cxr_result(
            {
                "lungs": ChecklistItem(value="clear", status=Severity.NORMAL),
                "heart": ChecklistItem(value="normal", status=Severity.NORMAL),
            }
        )
        validated = validator.post_analyze(req, result)
        assert validated.summary == result.summary
