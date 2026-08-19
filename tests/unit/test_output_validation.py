from copy import deepcopy

from medical_image_harness.hooks import AnalyzeRequest
from medical_image_harness.models import (
    AnalysisResult,
    Finding,
    Modality,
    RegionRect,
    Severity,
)
from medical_image_harness.output_validation import OutputValidator


def test_validator_normalizes_a_copy_and_preserves_raw_prediction() -> None:
    raw = AnalysisResult(
        modality=Modality.CXR,
        summary="No focal abnormality.",
        severity=Severity.NORMAL,
        findings=[
            Finding(
                id="normal",
                regions=["right_upper_lung"],
                label="Normal lung",
                detail="No visible opacity.",
                severity=Severity.NORMAL,
                bboxes=[RegionRect(0.1, 0.1, 0.2, 0.2)],
            )
        ],
        checklist={},
    )
    before = deepcopy(raw)
    normalized = OutputValidator().post_analyze(
        AnalyzeRequest(
            image_base64="fixture",
            modality=Modality.CXR,
            valid_regions=["right_upper_lung"],
        ),
        raw,
    )
    assert raw == before
    assert normalized is not raw
    assert normalized.findings[0].bboxes == []
    assert normalized.incomplete is True
