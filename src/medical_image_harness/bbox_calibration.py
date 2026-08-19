"""Post-analysis bbox calibration hook for single-pass interpretations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import structlog

from medical_image_harness.bbox_signal_calibrator import calibrate_ekg_bboxes
from medical_image_harness.hooks import AnalyzeHook, AnalyzeRequest

if TYPE_CHECKING:
    from medical_image_harness.models import AnalysisResult

logger = structlog.get_logger(__name__)


class BboxCalibrator(Protocol):
    def __call__(self, image_base64: str, result: AnalysisResult) -> AnalysisResult: ...


class BboxCalibrationHook(AnalyzeHook):
    """Calibrate model boxes against source pixels before clinical review.

    MultiPass performs this operation inside its coarse/refine/final workflow.
    This hook gives the single-pass path the same deterministic pixel and lead
    checks. Failure is advisory: the original result remains available.
    """

    def __init__(self, calibrator: BboxCalibrator = calibrate_ekg_bboxes) -> None:
        self._calibrator = calibrator

    def pre_analyze(self, request: AnalyzeRequest) -> AnalyzeRequest:
        return request

    def post_analyze(
        self,
        request: AnalyzeRequest,
        result: AnalysisResult,
    ) -> AnalysisResult:
        source = request.metadata.get("source_image_base64")
        image_base64 = (
            source if isinstance(source, str) and source else request.image_base64
        )
        try:
            return self._calibrator(image_base64, result)
        except Exception:
            logger.exception("bbox_calibration_hook_failed")
            return result
