from medical_image_harness.evaluation import (
    EvaluationCase,
    ReferenceFinding,
    aggregate,
    intersection_over_union,
    score_case,
)
from medical_image_harness.models import (
    AnalysisResult,
    Finding,
    Modality,
    RegionRect,
    Severity,
)


def test_iou_uses_source_coordinate_geometry() -> None:
    assert intersection_over_union(
        RegionRect(0.0, 0.0, 0.5, 0.5),
        RegionRect(0.25, 0.25, 0.5, 0.5),
    ) == 1 / 7


def test_case_scoring_uses_explicit_alias_and_urgent_triage() -> None:
    case = EvaluationCase(
        id="synthetic-1",
        references=(
            ReferenceFinding(
                "pneumothorax",
                aliases=("pleural air",),
                bboxes=(RegionRect(0.7, 0.1, 0.2, 0.3),),
                urgent=True,
            ),
        ),
    )
    result = AnalysisResult(
        modality=Modality.CXR,
        summary="Urgent pleural abnormality.",
        severity=Severity.CRITICAL,
        findings=[
            Finding(
                id="f1",
                regions=["right_upper_lung"],
                label="Pleural air",
                detail="Visible pleural line candidate.",
                severity=Severity.CRITICAL,
                confidence="moderate",
                bboxes=[RegionRect(0.7, 0.1, 0.2, 0.3)],
            )
        ],
        checklist={},
    )
    metrics = score_case(case, result)
    assert metrics.sensitivity == 1.0
    assert metrics.precision == 1.0
    assert metrics.mean_localization_iou == 1.0
    assert metrics.urgent_detected is True
    assert metrics.urgent_triaged is True
    assert metrics.brier_score is not None
    assert aggregate([metrics])["urgent_detected"] == {"n": 1, "rate": 1}


def test_non_gradable_case_rewards_explicit_abstention() -> None:
    case = EvaluationCase(id="limited", references=(), gradable=False)
    result = AnalysisResult(
        modality=Modality.CT_BRAIN,
        summary="Single screenshot is not sufficient.",
        severity=Severity.INFO,
        findings=[],
        checklist={},
        incomplete=True,
        incomplete_reasons=["Incomplete volume."],
        review_required=True,
        review_reasons=["Full study required."],
    )
    assert score_case(case, result).appropriate_abstention is True
