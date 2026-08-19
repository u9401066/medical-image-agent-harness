"""Unit tests for the EKG rhythm-strip refinement pass."""

from __future__ import annotations

import pytest

from medical_image_harness.models import (
    AnalysisResult,
    ChecklistItem,
    Finding,
    Modality,
    RegionRect,
    Severity,
)
from medical_image_harness.rhythm_strip import (
    RhythmStripRefiningAnalyzer,
    merge_rhythm_strip,
    refine_rhythm_strip,
    resolve_rhythm_strip_region,
)


def _result(
    *,
    modality: Modality = Modality.EKG,
    summary: str = "ok",
    severity: Severity = Severity.NORMAL,
    findings: list[Finding] | None = None,
    checklist: dict[str, ChecklistItem] | None = None,
    layout: dict | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        modality=modality,
        summary=summary,
        severity=severity,
        findings=findings or [],
        checklist=checklist or {},
        layout=layout or {},
    )


def test_resolve_rhythm_strip_region_from_layout() -> None:
    region = resolve_rhythm_strip_region(
        _result(layout={"rhythm_strip_bbox": [0.0, 0.8, 1.0, 0.18]})
    )
    assert region is not None
    assert region.y == 0.8
    assert region.h == 0.18


def test_resolve_rhythm_strip_region_none_when_absent_or_malformed() -> None:
    assert resolve_rhythm_strip_region(_result(layout={})) is None
    assert resolve_rhythm_strip_region(_result(layout={"rhythm_strip_bbox": None})) is None
    assert resolve_rhythm_strip_region(_result(layout={"rhythm_strip_bbox": [0, 1]})) is None
    assert (
        resolve_rhythm_strip_region(
            _result(layout={"rhythm_strip_bbox": [float("nan"), 0.8, 1.0, 0.1]})
        )
        is None
    )


def test_resolve_clamps_and_drops_degenerate() -> None:
    # y at the bottom edge leaves no height -> degenerate -> None.
    assert (
        resolve_rhythm_strip_region(_result(layout={"rhythm_strip_bbox": [0.0, 1.0, 1.0, 0.2]}))
        is None
    )
    # Out-of-range values are clamped into the unit square.
    region = resolve_rhythm_strip_region(
        _result(layout={"rhythm_strip_bbox": [-0.1, 0.8, 2.0, 0.1]})
    )
    assert region is not None
    assert region.x == 0.0
    assert region.w == 1.0


def test_merge_escalates_rhythm_axis_and_appends_finding() -> None:
    coarse = _result(
        checklist={"rhythm": ChecklistItem(value="sinus", status=Severity.NORMAL)},
        findings=[
            Finding(id="f1", regions=[], label="Sinus Rhythm", detail="", severity=Severity.NORMAL)
        ],
        severity=Severity.NORMAL,
    )
    strip = _result(
        checklist={"rhythm": ChecklistItem(value="atrial_fibrillation", status=Severity.WARNING)},
        findings=[
            Finding(
                id="s1",
                regions=[],
                label="Atrial Fibrillation",
                detail="irregularly irregular",
                severity=Severity.WARNING,
                bboxes=[RegionRect(0.1, 0.1, 0.2, 0.2)],
            )
        ],
        severity=Severity.WARNING,
    )
    merged = merge_rhythm_strip(coarse, strip, RegionRect(0.0, 0.8, 1.0, 0.2))
    assert merged.checklist["rhythm"].status is Severity.WARNING
    assert merged.severity is Severity.WARNING
    af = next(f for f in merged.findings if f.label == "Atrial Fibrillation")
    # Remapped into the bottom strip region (y >= 0.8).
    assert af.bboxes[0].y >= 0.8


def test_merge_never_downgrades_and_is_noop_when_nothing_added() -> None:
    coarse = _result(
        checklist={"rhythm": ChecklistItem(value="afib", status=Severity.CRITICAL)},
        severity=Severity.CRITICAL,
    )
    strip = _result(
        checklist={"rhythm": ChecklistItem(value="sinus", status=Severity.NORMAL)},
    )
    merged = merge_rhythm_strip(coarse, strip, RegionRect(0.0, 0.8, 1.0, 0.2))
    assert merged is coarse
    assert merged.checklist["rhythm"].status is Severity.CRITICAL


def test_merge_does_not_duplicate_existing_finding_label() -> None:
    coarse = _result(
        findings=[
            Finding(id="f1", regions=[], label="Atrial Fibrillation", detail="", severity=Severity.WARNING)
        ],
        severity=Severity.WARNING,
    )
    strip = _result(
        findings=[
            Finding(
                id="s1",
                regions=[],
                label="atrial fibrillation",
                detail="dup",
                severity=Severity.WARNING,
                bboxes=[RegionRect(0.1, 0.1, 0.2, 0.2)],
            )
        ],
        severity=Severity.WARNING,
    )
    merged = merge_rhythm_strip(coarse, strip, RegionRect(0.0, 0.8, 1.0, 0.2))
    assert sum(1 for f in merged.findings if f.label.lower() == "atrial fibrillation") == 1


async def test_refine_noop_for_non_ekg() -> None:
    calls: list[int] = []

    async def fake_analyze(img: str, modality: Modality, regions: list[str]) -> AnalysisResult:
        calls.append(1)
        return _result()

    result = _result(modality=Modality.CXR, layout={"rhythm_strip_bbox": [0.0, 0.8, 1.0, 0.2]})
    out = await refine_rhythm_strip(
        result,
        "img",
        analyze_fn=fake_analyze,
        cropper=lambda _img, _region: "crop",
        valid_regions=[],
    )
    assert out is result
    assert not calls


async def test_refine_noop_when_no_bbox_declared() -> None:
    calls: list[int] = []

    async def fake_analyze(img: str, modality: Modality, regions: list[str]) -> AnalysisResult:
        calls.append(1)
        return _result()

    result = _result(layout={})
    out = await refine_rhythm_strip(
        result,
        "img",
        analyze_fn=fake_analyze,
        cropper=lambda _img, _region: "crop",
        valid_regions=[],
    )
    assert out is result
    assert not calls


async def test_refine_merges_when_bbox_present() -> None:
    async def fake_analyze(img: str, modality: Modality, regions: list[str]) -> AnalysisResult:
        return _result(
            checklist={"av_block": ChecklistItem(value="first_degree", status=Severity.WARNING)},
            findings=[
                Finding(
                    id="s1",
                    regions=[],
                    label="First Degree AV Block",
                    detail="prolonged PR",
                    severity=Severity.WARNING,
                    bboxes=[RegionRect(0.1, 0.1, 0.1, 0.1)],
                )
            ],
            severity=Severity.WARNING,
        )

    coarse = _result(
        checklist={"av_block": ChecklistItem(value="absent", status=Severity.NORMAL)},
        severity=Severity.NORMAL,
        layout={"rhythm_strip_bbox": [0.0, 0.8, 1.0, 0.2]},
    )
    out = await refine_rhythm_strip(
        coarse,
        "img",
        analyze_fn=fake_analyze,
        cropper=lambda _img, _region: "crop",
        valid_regions=["rhythm_strip"],
    )
    assert out.checklist["av_block"].status is Severity.WARNING
    assert any(f.label == "First Degree AV Block" for f in out.findings)
    assert out.analysis_trace[-1]["stage"] == "rhythm_strip_refine"
    assert out.analysis_trace[-1]["crop_source"] == "source_image"


async def test_refining_analyzer_uses_original_source_image() -> None:
    class Inner:
        async def analyze(self, image, modality, valid_regions):
            raise AssertionError("source-aware entry point should be used")

        async def analyze_with_source_size(
            self,
            image,
            modality,
            valid_regions,
            **kwargs,
        ):
            assert image == "coarse"
            assert kwargs["source_image_base64"] == "original"
            return _result(
                layout={"rhythm_strip_bbox": [0.0, 0.8, 1.0, 0.2]},
                checklist={
                    "rhythm": ChecklistItem(value="sinus", status=Severity.NORMAL)
                },
            )

        async def chat(self, message):
            return message

        async def connect(self):
            return None

        async def disconnect(self):
            return None

        def is_connected(self):
            return True

    class Refiner(Inner):
        async def analyze(self, image, modality, valid_regions):
            assert image == "cropped-original"
            return _result(
                severity=Severity.WARNING,
                checklist={
                    "rhythm": ChecklistItem(
                        value="atrial_fibrillation",
                        status=Severity.WARNING,
                    )
                },
            )

    cropped_from: list[str] = []

    def cropper(image: str, _region: RegionRect) -> str:
        cropped_from.append(image)
        return "cropped-original"

    analyzer = RhythmStripRefiningAnalyzer(
        inner=Inner(),
        refinement_analyzer=Refiner(),
        cropper=cropper,
    )

    out = await analyzer.analyze_with_source_size(
        "coarse",
        Modality.EKG,
        ["rhythm_strip"],
        source_size_px=(2000, 1200),
        source_image_base64="original",
    )

    assert cropped_from == ["original"]
    assert out.checklist["rhythm"].value == "atrial_fibrillation"
    assert out.analysis_trace[-1]["stage"] == "rhythm_strip_refine"


async def test_refine_returns_coarse_on_analyze_failure() -> None:
    async def boom(img: str, modality: Modality, regions: list[str]) -> AnalysisResult:
        raise RuntimeError("gateway down")

    coarse = _result(layout={"rhythm_strip_bbox": [0.0, 0.8, 1.0, 0.2]})
    out = await refine_rhythm_strip(
        coarse,
        "img",
        analyze_fn=boom,
        cropper=lambda _img, _region: "crop",
        valid_regions=[],
    )
    assert out is coarse


async def test_refine_retries_transient_analysis_failure() -> None:
    attempts = 0

    async def flaky(img: str, modality: Modality, regions: list[str]) -> AnalysisResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("transient gateway timeout")
        return _result(
            severity=Severity.WARNING,
            findings=[
                Finding(
                    id="s1",
                    regions=["rhythm_strip"],
                    label="Atrial Fibrillation",
                    detail="irregularly irregular rhythm",
                    severity=Severity.WARNING,
                    bboxes=[RegionRect(0.1, 0.1, 0.2, 0.2)],
                )
            ],
        )

    coarse = _result(
        severity=Severity.NORMAL,
        layout={"rhythm_strip_bbox": [0.0, 0.8, 1.0, 0.2]},
    )
    out = await refine_rhythm_strip(
        coarse,
        "img",
        analyze_fn=flaky,
        cropper=lambda _img, _region: "crop",
        valid_regions=["rhythm_strip"],
        retry_attempts=1,
    )

    assert attempts == 2
    assert any(f.label == "Atrial Fibrillation" for f in out.findings)


async def test_refine_rejects_negative_retry_attempts() -> None:
    async def never_analyze(
        _image: str, _modality: Modality, _regions: list[str]
    ) -> AnalysisResult:
        raise AssertionError("should not be called")

    with pytest.raises(ValueError):
        await refine_rhythm_strip(
            _result(layout={"rhythm_strip_bbox": [0.0, 0.8, 1.0, 0.2]}),
            "img",
            analyze_fn=never_analyze,
            cropper=lambda _img, _region: "crop",
            valid_regions=[],
            retry_attempts=-1,
        )
