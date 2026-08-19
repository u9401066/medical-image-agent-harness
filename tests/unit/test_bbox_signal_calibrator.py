"""Tests for conservative EKG bbox alignment calibration."""

from __future__ import annotations

import base64
import io

from PIL import Image, ImageDraw

from medical_image_harness.bbox_calibration import BboxCalibrationHook
from medical_image_harness.bbox_signal_calibrator import calibrate_ekg_bboxes
from medical_image_harness.hooks import AnalyzeRequest
from medical_image_harness.models import (
    AnalysisResult,
    Finding,
    Modality,
    RegionRect,
    Severity,
)


def _image_base64(*, with_neighbor_signal: bool) -> str:
    image = Image.new("RGB", (400, 200), "white")
    if with_neighbor_signal:
        draw = ImageDraw.Draw(image)
        draw.line((160, 85, 235, 115), fill="black", width=5)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _result() -> AnalysisResult:
    return AnalysisResult(
        modality=Modality.EKG,
        summary="Possible ST-T change.",
        severity=Severity.WARNING,
        findings=[
            Finding(
                id="f1",
                regions=["lead_V2"],
                label="ST-T change",
                detail="Review localization.",
                severity=Severity.WARNING,
                bboxes=[RegionRect(0.2, 0.4, 0.2, 0.2)],
            )
        ],
        checklist={},
    )


def test_low_signal_box_snaps_only_to_stronger_local_waveform() -> None:
    original = _result()

    calibrated = calibrate_ekg_bboxes(
        _image_base64(with_neighbor_signal=True),
        original,
    )

    box = calibrated.findings[0].bboxes[0]
    assert box != original.findings[0].bboxes[0]
    assert any(row["status"] == "adjusted" for row in calibrated.analysis_trace)
    assert "signal-calibrated" in calibrated.findings[0].notes[0]


def test_blank_neighborhood_keeps_box_and_requires_review() -> None:
    original = _result()

    calibrated = calibrate_ekg_bboxes(
        _image_base64(with_neighbor_signal=False),
        original,
    )

    assert calibrated.findings[0].bboxes == original.findings[0].bboxes
    assert calibrated.review_required is True
    assert "Low-signal bbox" in calibrated.review_reasons[0]


def test_voltage_box_with_signal_expands_to_interpretable_waveform_context() -> None:
    image = Image.new("RGB", (1000, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.line((40, 350, 48, 410), fill="black", width=3)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    result = _result()
    result.findings[0] = Finding(
        id="f1",
        regions=["lead_V1"],
        label="Prominent V1 S-wave voltage",
        detail="Deep S-wave morphology.",
        severity=Severity.INFO,
        bboxes=[RegionRect(0.035, 0.49, 0.014, 0.068)],
        confidence="low",
        question="Does the full tracing satisfy LVH voltage criteria?",
    )

    calibrated = calibrate_ekg_bboxes(
        base64.b64encode(buffer.getvalue()).decode("ascii"),
        result,
    )

    box = calibrated.findings[0].bboxes[0]
    assert box.w * 1000 >= 128
    assert box.h * 720 >= 64
    assert any(
        row["status"] == "expanded_for_context" for row in calibrated.analysis_trace
    )


def test_bbox_region_is_reconciled_to_declared_lead_layout() -> None:
    image = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.line((80, 120, 150, 130), fill="black", width=4)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    result = _result()
    result.layout = {
        "leads": [
            {"name": "V4", "bbox": [0.0, 0.5, 1.0, 0.25]},
            {"name": "V5", "bbox": [0.0, 0.75, 1.0, 0.25]},
        ]
    }
    result.findings[0] = Finding(
        id="f1",
        regions=["lead_V5"],
        label="Possible ST-T change",
        detail="Review localization.",
        severity=Severity.INFO,
        bboxes=[RegionRect(0.2, 0.55, 0.2, 0.15)],
        confidence="low",
        question="Is this change present in V4?",
    )

    calibrated = calibrate_ekg_bboxes(
        base64.b64encode(buffer.getvalue()).decode("ascii"),
        result,
    )

    assert calibrated.findings[0].regions == ["lead_V4"]
    assert calibrated.findings[0].confidence == "low"
    assert "model named lead_V5" in calibrated.findings[0].question
    assert "box maps to lead_V4" in calibrated.findings[0].question
    assert calibrated.review_required is True
    assert any(
        row["status"] == "lead_region_reconciled" for row in calibrated.analysis_trace
    )


def test_single_pass_hook_calibrates_against_source_image() -> None:
    seen_images: list[str] = []

    def calibrator(image_base64: str, result: AnalysisResult) -> AnalysisResult:
        seen_images.append(image_base64)
        return result

    hook = BboxCalibrationHook(calibrator=calibrator)
    request = AnalyzeRequest(
        image_base64="coarse-image",
        modality=Modality.EKG,
        valid_regions=["lead_I"],
        metadata={"source_image_base64": "original-roi"},
    )
    result = _result()

    assert hook.post_analyze(request, result) is result
    assert seen_images == ["original-roi"]
