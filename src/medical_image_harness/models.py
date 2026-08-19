"""Provider-neutral data contracts for image-reading research harnesses."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medical_image_harness.provenance import InputProvenance
    from medical_image_harness.study import StudyManifest


class Modality(Enum):
    """Built-in modalities; custom profiles can extend these string keys."""

    EKG = "EKG"
    CXR = "CXR"
    CT_BRAIN = "CT_BRAIN"
    AUTO = "auto"


class Severity(Enum):
    """Triage priority, deliberately separate from diagnostic certainty."""

    CRITICAL = "critical"
    WARNING = "warning"
    NORMAL = "normal"
    INFO = "info"


class Polarity(Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class VerificationStatus(Enum):
    SUPPORTED = "supported"
    POSSIBLE = "possible"
    CONTRADICTED = "contradicted"
    UNEVALUABLE = "unevaluable"


class ClaimType(Enum):
    """Whether text is pixel description or a study-level diagnostic hypothesis."""

    DESCRIPTIVE_OBSERVATION = "descriptive_observation"
    DIAGNOSTIC_HYPOTHESIS = "diagnostic_hypothesis"


@dataclass(frozen=True)
class RegionRect:
    """Rectangle in normalized source-image coordinates.

    Zero-area rectangles remain representable so evaluators can reject malformed
    model output without failing while parsing it. Clinical output validators must
    require positive width and height.
    """

    x: float
    y: float
    w: float
    h: float
    source_image_sha256: str = ""
    verified: bool = False
    coordinate_space: str = "source_image_normalized"

    def __post_init__(self) -> None:
        for attribute in ("x", "y", "w", "h"):
            value = getattr(self, attribute)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{attribute} must be in [0, 1], got {value}")
        if self.source_image_sha256:
            if len(self.source_image_sha256) != 64:
                raise ValueError("source_image_sha256 must be a 64-character digest")
            int(self.source_image_sha256, 16)
        if self.coordinate_space != "source_image_normalized":
            raise ValueError("RegionRect supports only normalized source coordinates")


@dataclass(frozen=True)
class Finding:
    """One visible abnormality or explicitly unresolved candidate."""

    id: str
    regions: list[str]
    label: str
    detail: str
    severity: Severity
    bboxes: list[RegionRect] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    confidence: str = ""
    question: str = ""
    source: str = "ai"
    evidence: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    observation_ids: list[str] = field(default_factory=list)
    claim_type: ClaimType = ClaimType.DESCRIPTIVE_OBSERVATION


@dataclass(frozen=True)
class ChecklistItem:
    """One observation axis in a modality-specific systematic read."""

    value: str
    status: Severity
    assessable: bool = True
    evidence: str = ""


@dataclass(frozen=True)
class Evidence:
    """One auditable source or tool-evidence record."""

    id: str
    kind: str
    source_image_sha256: str
    description: str
    bboxes: list[RegionRect] = field(default_factory=list)
    source_ref: str = ""
    tool_name: str = ""
    tool_version: str = ""
    calibration_id: str = ""


@dataclass(frozen=True)
class Observation:
    """Atomic claim kept separate from the final impression."""

    id: str
    anatomy: str
    finding: str
    polarity: Polarity
    status: VerificationStatus
    assessable: bool
    evidence_ids: list[str] = field(default_factory=list)
    laterality: str = ""
    temporal: str = ""
    question: str = ""
    claim_type: ClaimType = ClaimType.DESCRIPTIVE_OBSERVATION


@dataclass
class AnalysisResult:
    """Typed analysis draft that can be assembled into the canonical contract.

    Legacy adapters may populate only the original image-reading fields. A public,
    auditable result is complete only when :meth:`to_contract_payload` validates;
    missing provenance, ledger, study, or review fields therefore fail closed.
    """

    modality: Modality
    summary: str
    severity: Severity
    findings: list[Finding]
    checklist: dict[str, ChecklistItem]
    analysis_time_ms: int = 0
    model_used: str = ""
    image_quality: str | dict[str, object] = ""
    next_steps: list[str] = field(default_factory=list)
    incomplete: bool = False
    incomplete_reasons: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    zoom_hints: list[str] = field(default_factory=list)
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)
    layout: dict = field(default_factory=dict)
    analysis_trace: list[dict[str, object]] = field(default_factory=list)
    schema_version: str = "1.0.0"
    protocol_version: str = "0.1.0"
    assessment_scope: str = ""
    result_status: str = "research_draft"
    summary_observation_ids: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    input_provenance: InputProvenance | dict[str, object] | None = None
    study_manifest: StudyManifest | dict[str, object] | None = None
    workflow_events: list[dict[str, object]] = field(default_factory=list)

    def to_contract_payload(self, *, validate: bool = True) -> dict[str, object]:
        """Build the exact public JSON contract, optionally validating it."""

        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "modality": self.modality.value,
            "assessment_scope": self.assessment_scope,
            "result_status": self.result_status,
            "summary": self.summary,
            "summary_observation_ids": list(self.summary_observation_ids),
            "severity": self.severity.value,
            "observations": [_observation_payload(item) for item in self.observations],
            "evidence": [_evidence_payload(item) for item in self.evidence],
            "findings": [_finding_payload(item) for item in self.findings],
            "checklist": {
                key: {
                    "value": item.value,
                    "status": item.status.value,
                    "assessable": item.assessable,
                    **({"evidence": item.evidence} if item.evidence else {}),
                }
                for key, item in self.checklist.items()
            },
            "layout": dict(self.layout),
            "image_quality": self.image_quality,
            "next_steps": list(self.next_steps),
            "incomplete": self.incomplete,
            "incomplete_reasons": list(self.incomplete_reasons),
            "review_required": self.review_required,
            "review_reasons": list(self.review_reasons),
            "model_used": self.model_used,
            "analysis_time_ms": self.analysis_time_ms,
            "input_provenance": _record_payload(self.input_provenance),
            "study_manifest": _study_payload(self.study_manifest),
            "analysis_trace": [dict(event) for event in self.workflow_events],
        }
        if validate:
            from medical_image_harness.schema import validate_payload

            validate_payload(payload)
        return payload

    def to_contract_json(self, *, indent: int | None = 2) -> str:
        """Serialize a validated contract without non-standard numeric values."""

        return json.dumps(
            self.to_contract_payload(),
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )


@dataclass(frozen=True)
class UserRegionAnnotation:
    """Reviewer-authored context for one normalized region."""

    region: RegionRect
    question: str = ""
    answer: str = ""


def _record_payload(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return _json_value(value)
    return value


def _study_payload(value: object) -> object:
    payload = _record_payload(value)
    if not isinstance(payload, dict):
        return payload
    payload.pop("metadata", None)
    assets = payload.get("assets")
    if isinstance(assets, list):
        for asset in assets:
            if isinstance(asset, dict):
                asset.pop("modality", None)
                if asset.get("frame_number") is None:
                    asset.pop("frame_number", None)
    return payload


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def _bbox_payload(box: RegionRect) -> dict[str, object]:
    return {
        "x": box.x,
        "y": box.y,
        "w": box.w,
        "h": box.h,
        "coordinate_space": box.coordinate_space,
        "verified": box.verified,
        "source_image_sha256": box.source_image_sha256,
    }


def _finding_payload(finding: Finding) -> dict[str, object]:
    return {
        "id": finding.id,
        "label": finding.label,
        "detail": finding.detail,
        "severity": finding.severity.value,
        "confidence": finding.confidence or "not_assessable",
        "claim_type": finding.claim_type.value,
        "regions": list(finding.regions),
        "bboxes": [_bbox_payload(box) for box in finding.bboxes],
        "evidence_ids": list(finding.evidence_ids),
        "observation_ids": list(finding.observation_ids),
        "question": finding.question,
    }


def _observation_payload(observation: Observation) -> dict[str, object]:
    return {
        "id": observation.id,
        "anatomy": observation.anatomy,
        "finding": observation.finding,
        "polarity": observation.polarity.value,
        "status": observation.status.value,
        "claim_type": observation.claim_type.value,
        "assessable": observation.assessable,
        "laterality": observation.laterality,
        "temporal": observation.temporal,
        "evidence_ids": list(observation.evidence_ids),
        "question": observation.question,
    }


def _evidence_payload(evidence: Evidence) -> dict[str, object]:
    return {
        "id": evidence.id,
        "kind": evidence.kind,
        "source_image_sha256": evidence.source_image_sha256,
        "description": evidence.description,
        "bboxes": [_bbox_payload(box) for box in evidence.bboxes],
        "source_ref": evidence.source_ref,
        "tool_name": evidence.tool_name,
        "tool_version": evidence.tool_version,
        "calibration_id": evidence.calibration_id,
    }
