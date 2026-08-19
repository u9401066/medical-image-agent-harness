from __future__ import annotations

import json
from copy import deepcopy

import pytest

from medical_image_harness.models import (
    AnalysisResult,
    ChecklistItem,
    Evidence,
    Finding,
    Modality,
    Observation,
    Polarity,
    RegionRect,
    Severity,
    VerificationStatus,
)
from medical_image_harness.provenance import InputProvenance
from medical_image_harness.schema import validation_errors
from medical_image_harness.study import ImageAsset, StudyManifest

SOURCE_HASH = "a" * 64
CXR_CHECKLIST = {
    "projection_quality",
    "airway",
    "lungs",
    "pleura",
    "cardiac_silhouette",
    "mediastinum",
    "hila",
    "diaphragm",
    "bones",
    "soft_tissue",
    "lines_tubes",
}
CT_BRAIN_CHECKLIST = {
    "study_completeness",
    "extra_axial",
    "ventricles_cisterns",
    "midline_mass_effect",
    "parenchyma",
    "deep_gray",
    "posterior_fossa",
    "calvarium_soft_tissue",
    "visible_vessels_sinuses_orbits",
}


def _payload() -> dict:
    return {
        "schema_version": "1.0.0",
        "protocol_version": "0.1.0",
        "modality": "CXR",
        "assessment_scope": "partial_study",
        "result_status": "research_draft",
        "summary": "Focal right basilar opacity requires review.",
        "summary_observation_ids": ["o1"],
        "severity": "warning",
        "observations": [
            {
                "id": "o1",
                "anatomy": "right lower lung",
                "finding": "focal opacity",
                "polarity": "present",
                "status": "supported",
                "claim_type": "descriptive_observation",
                "assessable": True,
                "laterality": "right",
                "temporal": "unknown",
                "evidence_ids": ["e1"],
                "question": "",
            }
        ],
        "evidence": [
            {
                "id": "e1",
                "kind": "source_region",
                "source_image_sha256": SOURCE_HASH,
                "description": "Visible focal increased density.",
                "bboxes": [
                    {
                        "x": 0.55,
                        "y": 0.55,
                        "w": 0.2,
                        "h": 0.2,
                        "coordinate_space": "source_image_normalized",
                        "verified": True,
                        "source_image_sha256": SOURCE_HASH,
                    }
                ],
                "source_ref": "image-1",
                "tool_name": "",
                "tool_version": "",
                "calibration_id": "",
            }
        ],
        "findings": [
            {
                "id": "f1",
                "label": "Focal opacity",
                "detail": "Visible at the right base.",
                "severity": "warning",
                "confidence": "moderate",
                "claim_type": "descriptive_observation",
                "regions": ["right_lower_lung"],
                "bboxes": [
                    {
                        "x": 0.55,
                        "y": 0.55,
                        "w": 0.2,
                        "h": 0.2,
                        "coordinate_space": "source_image_normalized",
                        "verified": True,
                        "source_image_sha256": SOURCE_HASH,
                    }
                ],
                "evidence_ids": ["e1"],
                "observation_ids": ["o1"],
                "question": "",
            }
        ],
        "checklist": {
            key: {
                "value": "not assessable in this synthetic fixture",
                "status": "info",
                "assessable": False,
            }
            for key in CXR_CHECKLIST
        }
        | {
            "lungs": {
                "value": "right basilar opacity",
                "status": "warning",
                "assessable": True,
                "evidence": "o1",
            }
        },
        "layout": {},
        "image_quality": {
            "adequacy": "limited",
            "issues": ["single view"],
            "detail": "Single AP view.",
            "views_present": ["AP"],
            "views_required": ["AP", "lateral"],
        },
        "next_steps": ["Review the source image and any prior study."],
        "incomplete": True,
        "incomplete_reasons": ["No lateral view."],
        "review_required": True,
        "review_reasons": ["Image finding needs clinician review."],
        "model_used": "fixture",
        "analysis_time_ms": 1,
        "input_provenance": {
            "source_image_sha256": SOURCE_HASH,
            "source_kind": "rendered_image",
            "deidentified": True,
            "transformations": [],
            "dataset_id": "synthetic",
            "case_id": "case-1",
        },
        "study_manifest": {
            "id": "study-1",
            "modality": "CXR",
            "complete": False,
            "assets": [
                {
                    "id": "image-1",
                    "sha256": SOURCE_HASH,
                    "source_kind": "rendered_image",
                    "deidentified": True,
                    "view": "AP",
                    "laterality": "",
                    "orientation": "upright",
                    "series_ref": "series-1",
                    "window": "",
                }
            ],
            "expected_views": ["AP", "lateral"],
            "expected_series": ["series-1"],
            "limitations": ["No lateral view."],
        },
        "analysis_trace": [
            {"stage": "intake", "status": "completed", "detail": "Input bound."},
            {
                "stage": "quality_gate",
                "status": "completed",
                "detail": "Single-view limitation recorded.",
            },
            {
                "stage": "blind_pass",
                "status": "completed",
                "detail": "Systematic pass recorded before optional tools.",
            },
            {
                "stage": "reconcile",
                "status": "completed",
                "detail": "Observation ledger reconciled.",
            },
            {
                "stage": "contract_validation",
                "status": "completed",
                "detail": "Contract checks passed.",
            },
            {
                "stage": "human_handoff",
                "status": "completed",
                "detail": "Draft prepared for authorized review.",
            },
        ],
    }


def _ct_screenshot_payload() -> dict:
    payload = deepcopy(_payload())
    payload["modality"] = "CT_BRAIN"
    payload["assessment_scope"] = "single_image_observation"
    payload["summary"] = "Single-image hyperattenuating focus requires review."
    payload["checklist"] = {
        key: {
            "value": "not assessable from one screenshot",
            "status": "info",
            "assessable": False,
        }
        for key in CT_BRAIN_CHECKLIST
    }
    payload["checklist"]["parenchyma"] = {
        "value": "visible hyperattenuating focus",
        "status": "warning",
        "assessable": True,
        "evidence": "o1",
    }
    payload["input_provenance"]["source_kind"] = "screenshot"
    payload["study_manifest"]["modality"] = "CT_BRAIN"
    payload["study_manifest"]["assets"][0]["source_kind"] = "screenshot"
    payload["study_manifest"]["expected_views"] = []
    payload["study_manifest"]["limitations"] = ["Single CT screenshot only."]
    return payload


def test_complete_payload_passes_schema_and_semantic_checks() -> None:
    assert validation_errors(_payload()) == []


def test_unresolved_observation_and_overflow_box_fail() -> None:
    payload = deepcopy(_payload())
    payload["findings"][0]["observation_ids"] = ["missing"]
    payload["evidence"][0]["bboxes"][0]["x"] = 0.9
    errors = validation_errors(payload)
    assert any("unresolved observation" in error for error in errors)
    assert any("exceeds source bounds" in error for error in errors)


def test_unbound_evidence_hash_fails() -> None:
    payload = deepcopy(_payload())
    payload["evidence"][0]["source_image_sha256"] = "b" * 64
    errors = validation_errors(payload)
    assert any("source hash is not in provenance" in error for error in errors)


def test_empty_ledgers_and_checklist_fail_closed() -> None:
    payload = deepcopy(_payload())
    payload["summary_observation_ids"] = []
    payload["observations"] = []
    payload["evidence"] = []
    payload["findings"] = []
    payload["checklist"] = {}

    errors = validation_errors(payload)

    assert any("summary_observation_ids" in error for error in errors)
    assert any("observations" in error for error in errors)
    assert any("evidence" in error for error in errors)
    assert any("checklist" in error for error in errors)


def test_finding_bbox_requires_binding_and_matching_evidence() -> None:
    payload = deepcopy(_payload())
    box = payload["findings"][0]["bboxes"][0]
    box["x"] = 0.9
    box["w"] = 0.5

    errors = validation_errors(payload)

    assert any("findings/f1/bboxes/0: box exceeds source bounds" in error for error in errors)
    assert any("box is not present in linked evidence" in error for error in errors)


def test_finding_bbox_source_hash_is_required() -> None:
    payload = deepcopy(_payload())
    del payload["findings"][0]["bboxes"][0]["source_image_sha256"]

    errors = validation_errors(payload)

    assert any("source_image_sha256" in error and "required" in error for error in errors)


def test_single_ct_screenshot_cannot_claim_complete_diagnostic_read() -> None:
    payload = deepcopy(_payload())
    payload["modality"] = "CT_BRAIN"
    payload["assessment_scope"] = "complete_study"
    payload["summary"] = "Definite acute intracranial hemorrhage."
    payload["observations"][0]["claim_type"] = "diagnostic_hypothesis"
    payload["findings"][0]["claim_type"] = "diagnostic_hypothesis"
    payload["findings"][0]["confidence"] = "high"
    payload["checklist"] = {
        key: {
            "value": "reported normal",
            "status": "normal",
            "assessable": False,
        }
        for key in CT_BRAIN_CHECKLIST
    }
    payload["image_quality"]["adequacy"] = "diagnostic"
    payload["image_quality"]["issues"] = []
    payload["incomplete"] = False
    payload["incomplete_reasons"] = []
    payload["review_required"] = False
    payload["review_reasons"] = []
    payload["input_provenance"]["source_kind"] = "screenshot"
    payload["study_manifest"]["modality"] = "CT_BRAIN"
    payload["study_manifest"]["complete"] = True
    payload["study_manifest"]["assets"][0]["source_kind"] = "screenshot"
    payload["study_manifest"]["expected_views"] = []
    payload["study_manifest"]["limitations"] = []

    errors = validation_errors(payload)

    assert any("CT screenshot must be a single-image observation" in error for error in errors)
    assert any("CT screenshot cannot be a complete study" in error for error in errors)
    assert any("incomplete study cannot be diagnostic" in error for error in errors)
    assert any("incomplete study must fail closed" in error for error in errors)
    assert any("review_required" in error for error in errors)
    assert any("permits only descriptive observations" in error for error in errors)
    assert any("cannot carry high diagnostic confidence" in error for error in errors)


def test_limited_ct_screenshot_rejects_diagnostic_claim_type_and_high_confidence() -> None:
    payload = _ct_screenshot_payload()
    assert validation_errors(payload) == []
    payload["summary"] = "Definite acute intracranial hemorrhage."
    payload["observations"][0]["claim_type"] = "diagnostic_hypothesis"
    payload["findings"][0]["claim_type"] = "diagnostic_hypothesis"
    payload["findings"][0]["confidence"] = "high"

    errors = validation_errors(payload)

    assert sum("permits only descriptive observations" in error for error in errors) == 2
    assert any("cannot carry high diagnostic confidence" in error for error in errors)


def test_missing_modality_checklist_axes_fail() -> None:
    payload = deepcopy(_payload())
    payload["checklist"] = {"lungs": payload["checklist"]["lungs"]}

    errors = validation_errors(payload)

    assert any("missing required axes" in error for error in errors)


def test_trace_must_be_unique_ordered_and_blind_before_tools() -> None:
    payload = deepcopy(_payload())
    payload["analysis_trace"].insert(
        2,
        {
            "stage": "independent_evidence",
            "status": "completed",
            "detail": "Tool ran too early.",
        },
    )
    payload["analysis_trace"][3]["status"] = "skipped"

    errors = validation_errors(payload)

    assert any("workflow stages are out of order" in error for error in errors)
    assert any("tools cannot precede a completed blind pass" in error for error in errors)


def test_non_finite_bbox_coordinates_fail_closed() -> None:
    payload = deepcopy(_payload())
    payload["evidence"][0]["bboxes"][0]["x"] = float("nan")
    payload["findings"][0]["bboxes"][0]["x"] = float("nan")

    errors = validation_errors(payload)

    assert any("box coordinates must be finite" in error for error in errors)


def test_typed_result_assembles_the_validated_canonical_contract() -> None:
    box = RegionRect(
        x=0.55,
        y=0.55,
        w=0.2,
        h=0.2,
        source_image_sha256=SOURCE_HASH,
        verified=True,
    )
    checklist = {
        key: ChecklistItem(
            value="not assessable in this synthetic fixture",
            status=Severity.INFO,
            assessable=False,
        )
        for key in CXR_CHECKLIST
    }
    checklist["lungs"] = ChecklistItem(
        value="right basilar opacity",
        status=Severity.WARNING,
        assessable=True,
        evidence="o1",
    )
    result = AnalysisResult(
        modality=Modality.CXR,
        summary="Focal right basilar opacity requires review.",
        severity=Severity.WARNING,
        findings=[
            Finding(
                id="f1",
                regions=["right_lower_lung"],
                label="Focal opacity",
                detail="Visible at the right base.",
                severity=Severity.WARNING,
                bboxes=[box],
                confidence="moderate",
                evidence_ids=["e1"],
                observation_ids=["o1"],
            )
        ],
        checklist=checklist,
        assessment_scope="partial_study",
        summary_observation_ids=["o1"],
        observations=[
            Observation(
                id="o1",
                anatomy="right lower lung",
                finding="focal opacity",
                polarity=Polarity.PRESENT,
                status=VerificationStatus.SUPPORTED,
                assessable=True,
                evidence_ids=["e1"],
                laterality="right",
                temporal="unknown",
            )
        ],
        evidence=[
            Evidence(
                id="e1",
                kind="source_region",
                source_image_sha256=SOURCE_HASH,
                description="Visible focal increased density.",
                bboxes=[box],
                source_ref="image-1",
            )
        ],
        input_provenance=InputProvenance(
            source_image_sha256=SOURCE_HASH,
            source_kind="rendered_image",
            deidentified=True,
            dataset_id="synthetic",
            case_id="case-1",
        ),
        study_manifest=StudyManifest(
            id="study-1",
            modality="CXR",
            assets=(
                ImageAsset(
                    id="image-1",
                    sha256=SOURCE_HASH,
                    modality="CXR",
                    source_kind="rendered_image",
                    view="AP",
                    orientation="upright",
                    series_ref="series-1",
                ),
            ),
            complete=False,
            expected_views=("AP", "lateral"),
            expected_series=("series-1",),
            limitations=("No lateral view.",),
        ),
        image_quality={
            "adequacy": "limited",
            "issues": ["single view"],
            "detail": "Single AP view.",
            "views_present": ["AP"],
            "views_required": ["AP", "lateral"],
        },
        next_steps=["Review the source image and any prior study."],
        incomplete=True,
        incomplete_reasons=["No lateral view."],
        review_required=True,
        review_reasons=["Image finding needs clinician review."],
        model_used="fixture",
        analysis_time_ms=1,
        workflow_events=deepcopy(_payload()["analysis_trace"]),
    )

    payload = result.to_contract_payload()

    assert payload["result_status"] == "research_draft"
    assert payload["study_manifest"]["assets"][0]["id"] == "image-1"
    assert json.loads(result.to_contract_json())["schema_version"] == "1.0.0"


def test_incomplete_typed_draft_cannot_masquerade_as_canonical_result() -> None:
    result = AnalysisResult(
        modality=Modality.CT_BRAIN,
        summary="Definite diagnosis from one screenshot.",
        severity=Severity.CRITICAL,
        findings=[],
        checklist={},
    )

    with pytest.raises(ValueError, match="invalid analysis result"):
        result.to_contract_payload()


def test_analysis_result_preserves_legacy_positional_field_order() -> None:
    result = AnalysisResult(
        Modality.CXR,
        "summary",
        Severity.NORMAL,
        [],
        {},
        123,
        "legacy-model",
    )

    assert result.analysis_time_ms == 123
    assert result.model_used == "legacy-model"
    assert result.schema_version == "1.0.0"


@pytest.mark.parametrize(
    "payload",
    [
        {"summary_observation_ids": None},
        {"findings": [{"observation_ids": None, "evidence_ids": None}]},
        {"evidence": [{"bboxes": None}]},
        {"study_manifest": {"assets": None}},
    ],
)
def test_malformed_payloads_report_errors_without_validator_crashes(payload) -> None:
    assert validation_errors(payload)
