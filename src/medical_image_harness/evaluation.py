"""Transparent metrics for research evaluation of structured image reads.

These metrics compare model output with reviewer-authored reference concepts.
They do not establish clinical efficacy and must not be reported as such.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from statistics import mean

from medical_image_harness.models import AnalysisResult, Finding, RegionRect, Severity


def normalize_concept(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def intersection_over_union(left: RegionRect, right: RegionRect) -> float:
    left_x2, left_y2 = left.x + left.w, left.y + left.h
    right_x2, right_y2 = right.x + right.w, right.y + right.h
    width = max(0.0, min(left_x2, right_x2) - max(left.x, right.x))
    height = max(0.0, min(left_y2, right_y2) - max(left.y, right.y))
    intersection = width * height
    union = left.w * left.h + right.w * right.h - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class ReferenceFinding:
    """Reviewer-defined concept and optional source-image localization."""

    concept: str
    aliases: tuple[str, ...] = ()
    bboxes: tuple[RegionRect, ...] = ()
    urgent: bool = False

    @property
    def normalized_labels(self) -> frozenset[str]:
        return frozenset(
            normalize_concept(value) for value in (self.concept, *self.aliases)
        )


@dataclass(frozen=True)
class EvaluationCase:
    """One de-identified, reviewer-labelled evaluation case."""

    id: str
    references: tuple[ReferenceFinding, ...]
    gradable: bool = True
    subgroup: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchedFinding:
    reference: str
    prediction_id: str
    localization_iou: float | None


@dataclass(frozen=True)
class CaseMetrics:
    case_id: str
    true_positives: int
    false_positives: int
    false_negatives: int
    sensitivity: float | None
    precision: float | None
    mean_localization_iou: float | None
    urgent_detected: bool | None
    urgent_triaged: bool | None
    appropriate_abstention: bool | None
    brier_score: float | None
    matches: tuple[MatchedFinding, ...]


def _prediction_labels(finding: Finding) -> frozenset[str]:
    labels = {normalize_concept(finding.label)}
    labels.update(normalize_concept(value) for value in finding.evidence)
    return frozenset(label for label in labels if label)


def _localization_score(
    prediction: Finding,
    reference: ReferenceFinding,
) -> float | None:
    if not reference.bboxes:
        return None
    if not prediction.bboxes:
        return 0.0
    return max(
        intersection_over_union(predicted, expected)
        for predicted in prediction.bboxes
        for expected in reference.bboxes
    )


def _confidence_probability(finding: Finding) -> float | None:
    return {"low": 0.35, "moderate": 0.65, "high": 0.9}.get(
        finding.confidence.casefold()
    )


def score_case(case: EvaluationCase, result: AnalysisResult) -> CaseMetrics:
    """Score explicit finding labels with one-to-one greedy matching.

    Aliases are supplied by reviewers per concept; the scorer intentionally does
    not infer clinical synonyms from narrative text.
    """

    predictions = [
        finding
        for finding in result.findings
        if finding.severity is not Severity.NORMAL and finding.label.strip()
    ]
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        labels = _prediction_labels(prediction)
        for reference_index, reference in enumerate(case.references):
            if not labels.intersection(reference.normalized_labels):
                continue
            localization = _localization_score(prediction, reference)
            candidates.append(
                (localization if localization is not None else 1.0, prediction_index, reference_index)
            )
    candidates.sort(reverse=True)
    used_predictions: set[int] = set()
    used_references: set[int] = set()
    matches: list[MatchedFinding] = []
    brier_terms: list[float] = []
    for _, prediction_index, reference_index in candidates:
        if prediction_index in used_predictions or reference_index in used_references:
            continue
        used_predictions.add(prediction_index)
        used_references.add(reference_index)
        prediction = predictions[prediction_index]
        reference = case.references[reference_index]
        localization = _localization_score(prediction, reference)
        matches.append(
            MatchedFinding(reference.concept, prediction.id, localization)
        )
        probability = _confidence_probability(prediction)
        if probability is not None:
            brier_terms.append((probability - 1.0) ** 2)

    for prediction_index, prediction in enumerate(predictions):
        if prediction_index in used_predictions:
            continue
        probability = _confidence_probability(prediction)
        if probability is not None:
            brier_terms.append(probability**2)

    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(case.references) - true_positives
    sensitivity = (
        true_positives / len(case.references) if case.references else None
    )
    precision = true_positives / len(predictions) if predictions else (
        1.0 if not case.references else None
    )
    localizations = [
        match.localization_iou
        for match in matches
        if match.localization_iou is not None
    ]
    urgent_indices = {
        index for index, reference in enumerate(case.references) if reference.urgent
    }
    urgent_detected = (
        bool(urgent_indices.intersection(used_references)) if urgent_indices else None
    )
    urgent_triaged = (
        urgent_detected and result.severity is Severity.CRITICAL
        if urgent_indices
        else None
    )
    appropriate_abstention = None
    if not case.gradable:
        appropriate_abstention = result.incomplete and result.review_required

    return CaseMetrics(
        case_id=case.id,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        sensitivity=sensitivity,
        precision=precision,
        mean_localization_iou=mean(localizations) if localizations else None,
        urgent_detected=urgent_detected,
        urgent_triaged=urgent_triaged,
        appropriate_abstention=appropriate_abstention,
        brier_score=mean(brier_terms) if brier_terms else None,
        matches=tuple(matches),
    )


def bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float] | None:
    if not values:
        return None
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    generator = random.Random(seed)
    n = len(values)
    estimates = sorted(
        mean(values[generator.randrange(n)] for _ in range(n))
        for _ in range(samples)
    )
    alpha = (1.0 - confidence) / 2.0
    lower = estimates[max(0, math.floor(alpha * (samples - 1)))]
    upper = estimates[min(samples - 1, math.ceil((1.0 - alpha) * (samples - 1)))]
    return mean(values), lower, upper


def aggregate(metrics: list[CaseMetrics]) -> dict[str, object]:
    """Aggregate case-level values while preserving each metric denominator."""

    fields = (
        "sensitivity",
        "precision",
        "mean_localization_iou",
        "brier_score",
    )
    summary: dict[str, object] = {"cases": len(metrics)}
    for field_name in fields:
        values = [
            float(value)
            for metric in metrics
            if (value := getattr(metric, field_name)) is not None
        ]
        summary[field_name] = {
            "n": len(values),
            "bootstrap_mean_ci95": bootstrap_mean_ci(values),
        }
    for field_name in ("urgent_detected", "urgent_triaged", "appropriate_abstention"):
        values = [
            bool(value)
            for metric in metrics
            if (value := getattr(metric, field_name)) is not None
        ]
        summary[field_name] = {
            "n": len(values),
            "rate": mean(values) if values else None,
        }
    return summary
