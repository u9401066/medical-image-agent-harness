"""Multi-pass interpretation orchestrator (application layer).

Lets the agent look at a complex image more than once: a coarse first pass
finds candidate regions, then the orchestrator crops each non-normal region out
of the *original-resolution* ROI image and re-sends just that slice for a
closer, higher-effective-resolution look. Refined bounding boxes are mapped
back into the immutable source-image coordinate space.

Design constraints:
- Refined ``Finding.bboxes`` stay in normalized 0-1 source-image coordinates.
- The orchestrator uses ``VisionAnalyzerService`` plus an optional
  application-level ``RefinementAnalyzer`` capability; it is independent of
  any model provider, agent runtime, or plugin SDK.
- Privacy: every crop is a *subset* of the user-defined ROI, so capture is
  never widened beyond the ROI (a zoom crop can only shrink the region).
- DDD: this module decodes no images itself. Image slicing is delegated to an
  injected :class:`ImageCropper`, keeping PIL/numpy out of the application and
  domain layers.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import Protocol

import structlog

from medical_image_harness.ekg_layout import parse_ekg_lead_inventory
from medical_image_harness.models import (
    AnalysisResult,
    Finding,
    Modality,
    RegionRect,
    Severity,
)
from medical_image_harness.protocols import AnalyzerPort, VisionAnalyzerService

logger = structlog.get_logger(__name__)

# Findings worth a closer second look. Info findings are included after
# warning/critical so a low-confidence first pass can still be refined.
_ZOOMABLE_SEVERITIES: frozenset[Severity] = frozenset(
    {Severity.INFO, Severity.WARNING, Severity.CRITICAL}
)
_ZOOM_PRIORITY: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.WARNING: 1,
    Severity.INFO: 2,
}

# A tiny source signal remains resolution-limited, but a contextual crop can
# still improve model attention/patch allocation.  We therefore surface a
# manual-zoom hint below this lesion edge while still running refinement on a
# crop expanded to ``DEFAULT_MIN_REFINE_CROP_EDGE_PX`` when possible.
DEFAULT_MIN_ZOOM_SOURCE_EDGE_PX = 64
DEFAULT_MIN_REFINE_CROP_EDGE_PX = 256
DEFAULT_MAX_NORMAL_SAFETY_PROBES = 1
DEFAULT_MAX_LOCAL_CANDIDATE_AREA = 0.8
DEFAULT_MAX_EKG_SYSTEMATIC_PROBES = 2

_EKG_SYSTEMATIC_LEAD_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "limb_leads",
        frozenset(
            {"lead_I", "lead_II", "lead_III", "lead_aVR", "lead_aVL", "lead_aVF"}
        ),
    ),
    (
        "precordial_leads",
        frozenset(
            {"lead_V1", "lead_V2", "lead_V3", "lead_V4", "lead_V5", "lead_V6"}
        ),
    ),
)


class ImageCropper(Protocol):
    """Crops a normalized sub-region out of a base64 PNG image.

    Implemented by infrastructure (which owns PIL). ``region`` is expressed in
    normalized 0-1 coordinates relative to the *input* image. Implementations
    may upscale the crop so small lesions become legible; the returned image is
    still a base64 PNG. The crop must never extend outside the input image.
    """

    def __call__(self, image_base64: str, region: RegionRect) -> str: ...


class BboxCalibrator(Protocol):
    """Locally calibrate coarse boxes without making a diagnosis."""

    def __call__(
        self,
        image_base64: str,
        result: AnalysisResult,
    ) -> AnalysisResult: ...


class _RefinementAction(StrEnum):
    """A hypothesis-aware decision returned by a crop refinement turn."""

    CONFIRM = "confirm"
    REVISE = "revise"
    RETRACT = "retract"
    ADD = "add"


# Public contract alias. The implementation class stays private so the repo's
# wiring guard does not mistake this value enum for an application orchestrator.
RefinementAction = _RefinementAction


@dataclasses.dataclass(frozen=True)
class RefinementDelta:
    """One explicit change proposed by a crop refinement turn.

    ``finding.bboxes`` are crop-local normalized coordinates. ``CONFIRM`` may
    provide a finding to tighten detail/localization, but cannot change the
    coarse label or severity. ``REVISE`` may update all finding fields,
    including label and severity. ``RETRACT`` removes the target hypothesis,
    and ``ADD`` contributes a distinct finding discovered in the crop.
    """

    action: RefinementAction
    target_id: str = ""
    finding: Finding | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        targeted = {
            RefinementAction.CONFIRM,
            RefinementAction.REVISE,
            RefinementAction.RETRACT,
        }
        if self.action in targeted and not self.target_id.strip():
            raise ValueError(f"{self.action.value} requires target_id")
        if (
            self.action in {RefinementAction.REVISE, RefinementAction.ADD}
            and self.finding is None
        ):
            raise ValueError(f"{self.action.value} requires finding")


@dataclasses.dataclass(frozen=True)
class RefinementResult:
    """Structured output of one hypothesis-aware crop refinement turn."""

    deltas: tuple[RefinementDelta, ...] = ()


class RefinementAnalyzer(Protocol):
    """Optional analyzer capability for a true hypothesis-aware second turn.

    The crop image is supplied as ``image_base64``. ``hypothesis`` is the
    coarse finding being checked, or ``None`` for a bounded normal-case safety
    probe. ``crop_region`` is expressed in original ROI coordinates, while any
    returned finding bbox is relative to the crop image.
    """

    async def refine(
        self,
        image_base64: str,
        modality: Modality,
        valid_regions: list[str],
        *,
        hypothesis: Finding | None,
        crop_region: RegionRect,
    ) -> RefinementResult: ...


class ReportFinalizer(Protocol):
    """Optional capability that reconciles the full report after crop turns."""

    async def finalize(
        self,
        image_base64: str,
        modality: Modality,
        valid_regions: list[str],
        *,
        draft: AnalysisResult,
        refinement_trace: list[dict[str, object]],
    ) -> AnalysisResult: ...


@dataclasses.dataclass(frozen=True)
class _RefinementTarget:
    crop_region: RegionRect
    hypothesis: Finding | None
    key: str


def clamp_unit(value: float) -> float:
    """Clamp a scalar into the closed unit interval [0, 1]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _region_payload(region: RegionRect) -> dict[str, float]:
    return {
        "x": round(region.x, 6),
        "y": round(region.y, 6),
        "w": round(region.w, 6),
        "h": round(region.h, 6),
    }


def pad_region(region: RegionRect, pad: float) -> RegionRect:
    """Grow ``region`` outward by ``pad`` fraction of its own size per side.

    Padding gives the model surrounding context when it re-examines a tight
    bounding box. The result is clamped to stay inside the ROI [0, 1] frame.
    ``pad`` of 0 returns an equivalent region (clamped).
    """
    if pad < 0.0:
        raise ValueError(f"pad must be >= 0, got {pad}")
    if (
        pad == 0.0
        and region.x >= 0.0
        and region.y >= 0.0
        and region.w > 0.0
        and region.h > 0.0
        and region.x + region.w <= 1.0
        and region.y + region.h <= 1.0
    ):
        return region
    dx = region.w * pad
    dy = region.h * pad
    x0 = clamp_unit(region.x - dx)
    y0 = clamp_unit(region.y - dy)
    x1 = clamp_unit(region.x + region.w + dx)
    y1 = clamp_unit(region.y + region.h + dy)
    return RegionRect(x=x0, y=y0, w=clamp_unit(x1 - x0), h=clamp_unit(y1 - y0))


def remap_bbox(child: RegionRect, parent: RegionRect) -> RegionRect:
    """Map a bbox expressed relative to a crop back to global ROI coordinates.

    ``parent`` is the crop region in ROI coordinates; ``child`` is a bbox the
    model returned relative to that crop (its own 0-1 frame). The result is the
    bbox in the original ROI's 0-1 frame, clamped so ``x + w`` and ``y + h``
    never exceed 1.
    """
    if parent.w <= 0.0 or parent.h <= 0.0:
        raise ValueError("parent crop must have positive width and height")
    if child.w <= 0.0 or child.h <= 0.0:
        raise ValueError("child bbox must have positive width and height")

    # First clamp in the crop-local frame. Some model outputs keep each field
    # in [0, 1] but still overflow as x+w/y+h; those boxes must not spill
    # outside the parent crop after remapping.
    child_x = clamp_unit(child.x)
    child_y = clamp_unit(child.y)
    child_w = min(clamp_unit(child.w), 1.0 - child_x)
    child_h = min(clamp_unit(child.h), 1.0 - child_y)
    gx = clamp_unit(parent.x + child_x * parent.w)
    gy = clamp_unit(parent.y + child_y * parent.h)
    gw = clamp_unit(child_w * parent.w)
    gh = clamp_unit(child_h * parent.h)
    # Keep the box inside the unit square after clamping the origin.
    gw = min(gw, 1.0 - gx)
    gh = min(gh, 1.0 - gy)
    if gw <= 0.0 or gh <= 0.0:
        raise ValueError("remapped bbox must have positive width and height")
    return RegionRect(x=gx, y=gy, w=gw, h=gh)


def select_zoom_targets(
    result: AnalysisResult,
    *,
    max_targets: int,
) -> list[Finding]:
    """Pick non-normal findings that have a bbox and are worth a closer look.

    Critical findings are prioritized over warnings, then info findings; normal
    findings and findings without a bbox are skipped. At most ``max_targets``
    are returned.
    """
    if max_targets <= 0:
        return []
    candidates: list[Finding] = []
    for finding in result.findings:
        if finding.severity not in _ZOOMABLE_SEVERITIES:
            continue
        boxes = [
            region
            for bbox in finding.bboxes
            if (region := _clamp_region(bbox)) is not None
        ]
        if boxes:
            candidates.append(dataclasses.replace(finding, bboxes=boxes))
    # Critical first, then warning, then info; preserve original order in a tier.
    candidates.sort(key=lambda f: _ZOOM_PRIORITY[f.severity])
    return candidates[:max_targets]


def _clamp_region(region: RegionRect) -> RegionRect | None:
    """Clamp a candidate region to the unit square, dropping empty boxes."""
    if (
        region.x >= 0.0
        and region.y >= 0.0
        and region.w > 0.0
        and region.h > 0.0
        and region.x + region.w <= 1.0
        and region.y + region.h <= 1.0
    ):
        return region
    x0 = clamp_unit(region.x)
    y0 = clamp_unit(region.y)
    x1 = clamp_unit(region.x + region.w)
    y1 = clamp_unit(region.y + region.h)
    w = clamp_unit(x1 - x0)
    h = clamp_unit(y1 - y0)
    if w <= 0.0 or h <= 0.0:
        return None
    return RegionRect(x=x0, y=y0, w=w, h=h)


def covering_region(regions: list[RegionRect]) -> RegionRect:
    """Return the smallest normalized crop containing every valid child box."""
    valid = [region for item in regions if (region := _clamp_region(item)) is not None]
    if not valid:
        raise ValueError("at least one non-degenerate region is required")
    if len(valid) == 1:
        return valid[0]
    x0 = min(region.x for region in valid)
    y0 = min(region.y for region in valid)
    x1 = max(region.x + region.w for region in valid)
    y1 = max(region.y + region.h for region in valid)
    return RegionRect(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


def select_ekg_systematic_probe_regions(
    result: AnalysisResult,
    *,
    max_probes: int = DEFAULT_MAX_EKG_SYSTEMATIC_PROBES,
) -> list[tuple[str, RegionRect]]:
    """Build broad lead-group crops from the model-declared EKG layout.

    These probes are discovery turns, not diagnosis. They cover lead groups
    that a coarse finding crop may omit, while refusing malformed, sparse, or
    near-full-frame layout unions. No fixed screenshot coordinates are used.
    """
    if max_probes <= 0 or result.modality is not Modality.EKG:
        return []
    lead_regions = parse_ekg_lead_inventory(result.layout).by_name()

    probes: list[tuple[str, RegionRect]] = []
    for key, expected_names in _EKG_SYSTEMATIC_LEAD_GROUPS:
        regions = [
            lead_regions[name] for name in expected_names if name in lead_regions
        ]
        if len(regions) < 2:
            continue
        region = covering_region(regions)
        if region.w * region.h >= 0.65:
            continue
        probes.append((key, region))
    return probes[:max_probes]


def _overlap_fraction(region: RegionRect, other: RegionRect) -> float:
    x0 = max(region.x, other.x)
    y0 = max(region.y, other.y)
    x1 = min(region.x + region.w, other.x + other.w)
    y1 = min(region.y + region.h, other.y + other.h)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return intersection / max(region.w * region.h, 1e-9)


def _meaningful_local_regions(
    local_candidate_regions: list[RegionRect],
    *,
    max_candidate_area: float,
) -> list[RegionRect]:
    """Return bounded, non-degenerate candidates that are not near-full-frame."""
    if not 0.0 < max_candidate_area <= 1.0:
        raise ValueError("max_candidate_area must be in (0, 1]")

    regions: list[RegionRect] = []
    seen: set[tuple[float, float, float, float]] = set()
    for candidate in local_candidate_regions:
        region = _clamp_region(candidate)
        if region is None or region.w * region.h >= max_candidate_area:
            continue
        key = (region.x, region.y, region.w, region.h)
        if key in seen:
            continue
        seen.add(key)
        regions.append(region)
    return regions


def select_local_candidate_targets(
    result: AnalysisResult,
    local_candidate_regions: list[RegionRect],
    *,
    max_targets: int,
    max_candidate_area: float = DEFAULT_MAX_LOCAL_CANDIDATE_AREA,
) -> list[Finding]:
    """Build zoom targets from local candidates when the MLLM omitted bboxes.

    This path handles non-normal findings without usable coordinates. Normal
    reads use the separately bounded safety-probe selector below.
    """
    if max_targets <= 0 or not local_candidate_regions:
        return []
    has_non_normal_finding = any(
        f.severity in _ZOOMABLE_SEVERITIES for f in result.findings
    )
    if result.severity not in _ZOOMABLE_SEVERITIES and not has_non_normal_finding:
        return []

    regions = _meaningful_local_regions(
        local_candidate_regions,
        max_candidate_area=max_candidate_area,
    )[:max_targets]
    if not regions:
        return []

    unresolved = [
        f
        for f in result.findings
        if f.severity in _ZOOMABLE_SEVERITIES
        and not any(_clamp_region(bbox) is not None for bbox in f.bboxes)
    ]
    targets: list[Finding] = []
    for finding, region in zip(unresolved, regions, strict=False):
        targets.append(dataclasses.replace(finding, bboxes=[region]))
    return targets


def select_normal_safety_probe_regions(
    result: AnalysisResult,
    local_candidate_regions: list[RegionRect],
    *,
    max_probes: int = DEFAULT_MAX_NORMAL_SAFETY_PROBES,
    max_candidate_area: float = DEFAULT_MAX_LOCAL_CANDIDATE_AREA,
) -> list[RegionRect]:
    """Select a bounded set of local probes for an otherwise normal coarse read."""
    if max_probes <= 0 or not local_candidate_regions:
        return []
    if result.severity is not Severity.NORMAL:
        return []
    if any(f.severity in _ZOOMABLE_SEVERITIES for f in result.findings):
        return []
    return _meaningful_local_regions(
        local_candidate_regions,
        max_candidate_area=max_candidate_area,
    )[:max_probes]


def region_source_edge_px(region: RegionRect, source_size_px: tuple[int, int]) -> int:
    """Short edge of ``region`` measured in *captured* source pixels.

    ``source_size_px`` is the actual ``(width, height)`` of the ROI image that
    was captured from the screen (≤ the screen resolution). This is the real
    pixel budget a lesion occupies; it bounds how much a digital crop can ever
    show, since a screenshot has no detail beyond its own pixels.
    """
    src_w, src_h = source_size_px
    w_px = region.w * src_w
    h_px = region.h * src_h
    return int(min(w_px, h_px))


def needs_manual_zoom(
    region: RegionRect,
    source_size_px: tuple[int, int],
    *,
    min_source_edge_px: int = DEFAULT_MIN_ZOOM_SOURCE_EDGE_PX,
) -> bool:
    """True when ``region`` is too small in captured pixels for a digital zoom.

    Below ``min_source_edge_px`` captured pixels on the short edge, cropping the
    screenshot only upscales blur -- the user must zoom in their viewer and
    re-capture to gain real resolution.
    """
    return region_source_edge_px(region, source_size_px) < min_source_edge_px


def expand_crop_to_min_source_edge(
    region: RegionRect,
    source_size_px: tuple[int, int],
    *,
    min_crop_edge_px: int = DEFAULT_MIN_REFINE_CROP_EDGE_PX,
) -> RegionRect:
    """Expand a crop around its center to retain useful waveform context.

    The returned normalized rectangle never exceeds the source image. This is
    not synthetic upscaling: it sends real neighboring pixels from the original
    ROI so a tight lesion bbox does not become a context-free 20-pixel crop.
    """
    if min_crop_edge_px <= 0:
        return region
    src_w, src_h = source_size_px
    if src_w <= 0 or src_h <= 0:
        return region
    target_w = min(1.0, max(region.w, min_crop_edge_px / src_w))
    target_h = min(1.0, max(region.h, min_crop_edge_px / src_h))
    center_x = region.x + region.w / 2.0
    center_y = region.y + region.h / 2.0
    x = min(max(0.0, center_x - target_w / 2.0), 1.0 - target_w)
    y = min(max(0.0, center_y - target_h / 2.0), 1.0 - target_h)
    return RegionRect(x=x, y=y, w=target_w, h=target_h)


def build_manual_zoom_message(label: str, source_edge_px: int) -> str:
    """Traditional-Chinese hint asking the user to zoom in their viewer.

    Kept pure so the wording is unit-testable and the overlay just renders it.
    """
    name = label.strip() or "此區域"
    return (
        f"🔍 建議手動放大：「{name}」在目前截圖僅約 {source_edge_px}px，"
        "已達螢幕截圖解析度上限；請於 DICOM 檢視器中放大該區後重新截圖，"
        "以取得更清晰影像。"
    )


def _normalized_label(label: str) -> str:
    return " ".join(
        "".join(char if char.isalnum() else " " for char in label.casefold()).split()
    )


def _label_match_score(hypothesis: Finding, candidate: Finding) -> float:
    if hypothesis.id and hypothesis.id == candidate.id:
        return 100.0
    expected = _normalized_label(hypothesis.label)
    actual = _normalized_label(candidate.label)
    if not expected or not actual:
        return 0.0
    if expected == actual:
        return 50.0
    expected_tokens = set(expected.split())
    actual_tokens = set(actual.split())
    union = expected_tokens | actual_tokens
    overlap = len(expected_tokens & actual_tokens) / len(union)
    return 10.0 + overlap if overlap >= 0.6 else 0.0


def _select_legacy_match(
    hypothesis: Finding,
    candidates: list[Finding],
) -> Finding | None:
    """Resolve a legacy zoom finding without relying on response order."""
    scored = [(_label_match_score(hypothesis, item), item) for item in candidates]
    scored = [item for item in scored if item[0] > 0.0]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def _legacy_refinement_result(
    zoom: AnalysisResult,
    target: _RefinementTarget,
) -> RefinementResult:
    """Translate the old ``analyze(crop)`` API into explicit deltas safely."""
    candidates = [
        finding for finding in zoom.findings if finding.severity in _ZOOMABLE_SEVERITIES
    ]
    hypothesis = target.hypothesis
    if hypothesis is None:
        return RefinementResult(
            tuple(
                RefinementDelta(
                    action=RefinementAction.ADD,
                    finding=finding,
                    rationale="legacy normal-case safety probe",
                )
                for finding in candidates
            )
        )

    if not candidates and zoom.severity is Severity.NORMAL:
        return RefinementResult(
            (
                RefinementDelta(
                    action=RefinementAction.RETRACT,
                    target_id=hypothesis.id,
                    rationale="legacy crop read returned no abnormal finding",
                ),
            )
        )
    if not candidates:
        return RefinementResult()

    match = _select_legacy_match(hypothesis, candidates)
    deltas: list[RefinementDelta] = []
    if match is not None:
        action = RefinementAction.REVISE
        if (
            _normalized_label(match.label) == _normalized_label(hypothesis.label)
            and match.severity is hypothesis.severity
        ):
            action = RefinementAction.CONFIRM
        deltas.append(
            RefinementDelta(
                action=action,
                target_id=hypothesis.id,
                finding=match,
                rationale="legacy crop finding matched the coarse hypothesis",
            )
        )
    for candidate in candidates:
        if candidate is match:
            continue
        linked_id = candidate.id or "zoom_addition"
        if not linked_id.startswith(f"{hypothesis.id}_"):
            linked_id = f"{hypothesis.id}_{linked_id}"
        deltas.append(
            RefinementDelta(
                action=RefinementAction.ADD,
                finding=dataclasses.replace(candidate, id=linked_id),
                rationale="additional finding from legacy crop read",
            )
        )
    return RefinementResult(tuple(deltas))


def _remap_finding_boxes(
    finding: Finding,
    crop_region: RegionRect,
) -> Finding:
    boxes: list[RegionRect] = []
    for bbox in finding.bboxes:
        try:
            boxes.append(remap_bbox(bbox, crop_region))
        except ValueError:
            logger.warning(
                "Discarding invalid crop-local bbox",
                finding_id=finding.id,
            )
    return dataclasses.replace(finding, bboxes=boxes)


def _append_rationale(notes: list[str], rationale: str) -> list[str]:
    text = rationale.strip()
    if not text or text in notes:
        return list(notes)
    return [*notes, text]


def _unique_finding_id(findings: list[Finding], requested: str) -> str:
    base = requested.strip() or "refined_finding"
    existing = {finding.id for finding in findings}
    if base not in existing:
        return base
    suffix = 2
    while f"{base}_{suffix}" in existing:
        suffix += 1
    return f"{base}_{suffix}"


def apply_refinement_delta(
    findings: list[Finding],
    delta: RefinementDelta,
    *,
    crop_region: RegionRect,
    expected_target_id: str | None,
) -> list[Finding]:
    """Apply one crop-local delta within its target's mutation boundary."""
    if delta.action is not RefinementAction.ADD and (
        expected_target_id is None or delta.target_id != expected_target_id
    ):
        logger.warning(
            "Ignoring refinement delta outside its target boundary",
            action=delta.action.value,
            expected_target_id=expected_target_id,
            target_id=delta.target_id,
        )
        return findings

    if delta.action is RefinementAction.RETRACT:
        return [finding for finding in findings if finding.id != delta.target_id]

    payload = delta.finding
    if delta.action is RefinementAction.ADD:
        if payload is None or payload.severity is Severity.NORMAL:
            return findings
        mapped = _remap_finding_boxes(payload, crop_region)
        if not mapped.bboxes:
            logger.warning(
                "Ignoring added refinement finding without a valid bbox",
                finding_id=payload.id,
            )
            return findings
        mapped = dataclasses.replace(
            mapped,
            id=_unique_finding_id(findings, mapped.id),
            notes=_append_rationale(mapped.notes, delta.rationale),
            evidence=[],
            evidence_ids=[],
            observation_ids=[],
        )
        return [*findings, mapped]

    result: list[Finding] = []
    for current in findings:
        if current.id != delta.target_id:
            result.append(current)
            continue

        if payload is None:
            result.append(
                dataclasses.replace(
                    current,
                    notes=_append_rationale(current.notes, delta.rationale),
                )
            )
            continue

        mapped = _remap_finding_boxes(payload, crop_region)
        notes = list(current.notes)
        for note in mapped.notes:
            if note and note not in notes:
                notes.append(note)
        notes = _append_rationale(notes, delta.rationale)
        if delta.action is RefinementAction.CONFIRM:
            result.append(
                dataclasses.replace(
                    current,
                    detail=mapped.detail or current.detail,
                    bboxes=mapped.bboxes or current.bboxes,
                    regions=mapped.regions or current.regions,
                    notes=notes,
                    evidence=[],
                    evidence_ids=[],
                    observation_ids=[],
                )
            )
        else:
            result.append(
                dataclasses.replace(
                    current,
                    label=mapped.label or current.label,
                    detail=mapped.detail or current.detail,
                    severity=mapped.severity,
                    bboxes=mapped.bboxes or current.bboxes,
                    regions=mapped.regions or current.regions,
                    notes=notes,
                    evidence=[],
                    evidence_ids=[],
                    observation_ids=[],
                )
            )
    return result


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.NORMAL: 0,
    Severity.INFO: 1,
    Severity.WARNING: 2,
    Severity.CRITICAL: 3,
}


def _merged_severity(
    coarse: AnalysisResult,
    findings: list[Finding],
    *,
    allow_downgrade: bool,
) -> Severity:
    severity = max(
        (finding.severity for finding in findings),
        key=lambda item: _SEVERITY_RANK[item],
        default=Severity.NORMAL,
    )
    preserve_floor = coarse.review_required or not allow_downgrade
    if preserve_floor and _SEVERITY_RANK[coarse.severity] > _SEVERITY_RANK[severity]:
        return coarse.severity
    return severity


def reconcile_final_report(
    draft: AnalysisResult,
    final: AnalysisResult,
) -> AnalysisResult:
    """Accept narrative/checklist synthesis without surrendering grounded boxes."""

    def unique(*groups: list[str]) -> list[str]:
        return list(dict.fromkeys(item for group in groups for item in group if item))

    return dataclasses.replace(
        final,
        schema_version=draft.schema_version,
        protocol_version=draft.protocol_version,
        modality=draft.modality,
        assessment_scope=draft.assessment_scope,
        result_status=draft.result_status,
        summary=final.summary.strip() or draft.summary,
        summary_observation_ids=list(draft.summary_observation_ids),
        severity=draft.severity,
        observations=list(draft.observations),
        evidence=list(draft.evidence),
        input_provenance=draft.input_provenance,
        study_manifest=draft.study_manifest,
        findings=list(draft.findings),
        checklist=final.checklist or draft.checklist,
        analysis_time_ms=draft.analysis_time_ms + final.analysis_time_ms,
        model_used=final.model_used or draft.model_used,
        image_quality=final.image_quality or draft.image_quality,
        next_steps=final.next_steps or draft.next_steps,
        incomplete=draft.incomplete or final.incomplete,
        incomplete_reasons=unique(
            draft.incomplete_reasons,
            final.incomplete_reasons,
        ),
        validation_warnings=unique(
            draft.validation_warnings,
            final.validation_warnings,
        ),
        zoom_hints=unique(draft.zoom_hints, final.zoom_hints),
        review_required=draft.review_required or final.review_required,
        review_reasons=unique(draft.review_reasons, final.review_reasons),
        layout=dict(draft.layout),
        workflow_events=list(draft.workflow_events),
        analysis_trace=list(draft.analysis_trace),
    )


class MultiPassInterpreter:
    """Coarse -> crop -> hypothesis-aware refinement orchestrator."""

    def __init__(
        self,
        analyzer: AnalyzerPort,
        cropper: ImageCropper,
        *,
        bbox_calibrator: BboxCalibrator | None = None,
        max_zoom_targets: int = 3,
        zoom_padding: float = 0.15,
        min_zoom_source_edge_px: int = DEFAULT_MIN_ZOOM_SOURCE_EDGE_PX,
        zoom_retry_attempts: int = 1,
        max_normal_safety_probes: int = DEFAULT_MAX_NORMAL_SAFETY_PROBES,
        max_ekg_systematic_probes: int = DEFAULT_MAX_EKG_SYSTEMATIC_PROBES,
        max_local_candidate_area: float = DEFAULT_MAX_LOCAL_CANDIDATE_AREA,
    ) -> None:
        if max_zoom_targets < 0:
            raise ValueError("max_zoom_targets must be >= 0")
        if zoom_padding < 0.0:
            raise ValueError("zoom_padding must be >= 0")
        if min_zoom_source_edge_px < 0:
            raise ValueError("min_zoom_source_edge_px must be >= 0")
        if zoom_retry_attempts < 0:
            raise ValueError("zoom_retry_attempts must be >= 0")
        if max_normal_safety_probes < 0:
            raise ValueError("max_normal_safety_probes must be >= 0")
        if max_ekg_systematic_probes < 0:
            raise ValueError("max_ekg_systematic_probes must be >= 0")
        if not 0.0 < max_local_candidate_area <= 1.0:
            raise ValueError("max_local_candidate_area must be in (0, 1]")
        self._analyzer = analyzer
        self._cropper = cropper
        self._bbox_calibrator = bbox_calibrator
        self._max_zoom_targets = max_zoom_targets
        self._zoom_padding = zoom_padding
        self._min_zoom_source_edge_px = min_zoom_source_edge_px
        self._zoom_retry_attempts = zoom_retry_attempts
        self._max_normal_safety_probes = max_normal_safety_probes
        self._max_ekg_systematic_probes = max_ekg_systematic_probes
        self._max_local_candidate_area = max_local_candidate_area

    async def interpret(
        self,
        image_base64: str,
        modality: Modality,
        valid_regions: list[str],
        *,
        source_image_base64: str | None = None,
        source_size_px: tuple[int, int] | None = None,
        local_candidate_regions: list[RegionRect] | None = None,
    ) -> AnalysisResult:
        """Run the coarse pass, then optional zoom passes, and merge results.

        ``image_base64`` is the bounded coarse-pass image. When
        ``source_image_base64`` is provided, refinement crops are taken from
        that original-resolution ROI capture instead of the coarse-pass
        downscale. ``source_size_px`` is the ``(width, height)`` in pixels of
        that captured ROI image. A very small target receives a manual-zoom hint
        because cropping cannot invent source detail, but it is still refined
        using a bounded contextual crop from the original ROI. When ``None``
        (resolution unknown), every target is digitally zoomed as before.
        """
        coarse = await self._analyzer.analyze(image_base64, modality, valid_regions)
        if self._bbox_calibrator is not None:
            try:
                coarse = self._bbox_calibrator(
                    source_image_base64 or image_base64,
                    coarse,
                )
            except Exception:
                logger.warning("BBox signal calibration failed; keeping model boxes")
        runtime_trace = self._read_runtime_trace()
        calibration_trace = list(coarse.analysis_trace)
        trace: list[dict[str, object]] = [
            {
                "stage": "coarse",
                "status": "completed",
                "tool": "analyzer",
                "severity": coarse.severity.value,
                "finding_count": len(coarse.findings),
                **runtime_trace,
            }
        ]
        if local_candidate_regions:
            trace.append(
                {
                    "stage": "local_assist",
                    "status": "completed",
                    "tool": "local_signal_candidates",
                    "candidate_count": len(local_candidate_regions),
                    "regions": [
                        _region_payload(region)
                        for region in local_candidate_regions
                    ],
                }
            )
        trace.extend(calibration_trace)
        coarse.analysis_trace = trace

        model_findings = select_zoom_targets(
            coarse,
            max_targets=self._max_zoom_targets,
        )
        systematic_candidates = select_ekg_systematic_probe_regions(
            coarse,
            max_probes=self._max_ekg_systematic_probes,
        )
        systematic_budget = 0
        if systematic_candidates and self._max_zoom_targets > 0:
            if model_findings:
                desired = 1 if len(model_findings) >= 2 else 2
                systematic_budget = min(
                    desired,
                    len(systematic_candidates),
                    max(0, self._max_zoom_targets - 1),
                )
            elif local_candidate_regions:
                systematic_budget = min(
                    len(systematic_candidates),
                    max(0, self._max_zoom_targets - 1),
                )
            else:
                systematic_budget = min(
                    len(systematic_candidates),
                    self._max_zoom_targets,
                )

        specific_budget = self._max_zoom_targets - systematic_budget
        targets = [
            _RefinementTarget(
                crop_region=covering_region(finding.bboxes),
                hypothesis=finding,
                key=finding.id,
            )
            for finding in model_findings[:specific_budget]
        ]
        remaining = specific_budget - len(targets)
        if remaining > 0 and local_candidate_regions:
            local_targets = select_local_candidate_targets(
                coarse,
                local_candidate_regions,
                max_targets=remaining,
                max_candidate_area=self._max_local_candidate_area,
            )
            targets.extend(
                _RefinementTarget(
                    crop_region=covering_region(finding.bboxes),
                    hypothesis=finding,
                    key=finding.id,
                )
                for finding in local_targets
            )
        if not targets and remaining > 0 and local_candidate_regions:
            safety_regions = select_normal_safety_probe_regions(
                coarse,
                local_candidate_regions,
                max_probes=min(
                    remaining,
                    self._max_normal_safety_probes,
                ),
                max_candidate_area=self._max_local_candidate_area,
            )
            targets.extend(
                _RefinementTarget(
                    crop_region=region,
                    hypothesis=None,
                    key=f"normal_safety_probe_{index}",
                )
                for index, region in enumerate(safety_regions, start=1)
            )

        systematic_targets: list[_RefinementTarget] = []
        if systematic_budget:
            ranked_systematic = sorted(
                systematic_candidates,
                key=lambda item: max(
                    (
                        _overlap_fraction(item[1], target.crop_region)
                        for target in targets
                    ),
                    default=0.0,
                ),
            )
            for key, region in ranked_systematic:
                if any(
                    _overlap_fraction(region, target.crop_region) >= 0.85
                    for target in targets
                ):
                    continue
                systematic_targets.append(
                    _RefinementTarget(
                        crop_region=region,
                        hypothesis=None,
                        key=f"ekg_systematic_{key}",
                    )
                )
                if len(systematic_targets) >= systematic_budget:
                    break
            targets.extend(systematic_targets)
            if systematic_targets:
                trace.append(
                    {
                        "stage": "systematic_assist",
                        "status": "planned",
                        "tool": "ekg_layout_lead_group_probes",
                        "probes": [
                            {
                                "target_id": target.key,
                                "crop_region": _region_payload(target.crop_region),
                            }
                            for target in systematic_targets
                        ],
                    }
                )
        if not targets:
            return coarse

        zoom_hints: list[str] = []
        refinements: list[tuple[_RefinementTarget, RegionRect, RefinementResult]] = []
        completed_refinement_turn = False
        for target in targets:
            bbox = target.crop_region
            source_limited = source_size_px is not None and needs_manual_zoom(
                bbox,
                source_size_px,
                min_source_edge_px=self._min_zoom_source_edge_px,
            )
            if source_limited and source_size_px is not None:
                edge_px = region_source_edge_px(bbox, source_size_px)
                if target.hypothesis is not None:
                    logger.info(
                        "Region too small for digital zoom; suggesting manual zoom",
                        finding_id=target.hypothesis.id,
                        source_edge_px=edge_px,
                    )
                    zoom_hints.append(
                        build_manual_zoom_message(target.hypothesis.label, edge_px)
                    )
                trace.append(
                    {
                        "stage": "refine",
                        "status": "source_resolution_limited",
                        "tool": "source_resolution_gate",
                        "target_id": target.key,
                        "source_edge_px": edge_px,
                        "crop_region": _region_payload(bbox),
                    }
                )
            crop_region = pad_region(bbox, self._zoom_padding)
            if source_size_px is not None:
                crop_region = expand_crop_to_min_source_edge(
                    crop_region,
                    source_size_px,
                )
            try:
                crop_b64 = self._cropper(
                    source_image_base64 or image_base64,
                    crop_region,
                )
            except Exception:  # one bad zoom must not sink the whole pass
                logger.warning(
                    "Zoom crop failed; keeping coarse finding",
                    target=target.key,
                )
                trace.append(
                    {
                        "stage": "refine",
                        "status": "crop_failed",
                        "tool": "crop_region_base64",
                        "target_id": target.key,
                        "crop_region": _region_payload(crop_region),
                    }
                )
                continue
            refinement = await self._run_refinement(
                crop_b64,
                modality,
                valid_regions,
                target=target,
                crop_region=crop_region,
            )
            completed_refinement_turn = completed_refinement_turn or (
                refinement is not None
            )
            if refinement is not None and refinement.deltas:
                refinements.append((target, crop_region, refinement))
            trace.append(
                {
                    "stage": "refine",
                    "status": "completed" if refinement is not None else "failed",
                    "tool": "crop_region_base64",
                    "target_id": target.key,
                    "hypothesis": (
                        target.hypothesis.label
                        if target.hypothesis is not None
                        else target.key
                    ),
                    "crop_region": _region_payload(crop_region),
                    "crop_source": (
                        "original_roi" if source_image_base64 else "coarse_image"
                    ),
                    "decisions": [
                        {
                            "action": delta.action.value,
                            "target_id": delta.target_id,
                            "rationale": delta.rationale,
                            "finding": (
                                delta.finding.label
                                if delta.finding is not None
                                else ""
                            ),
                        }
                        for delta in (refinement.deltas if refinement else ())
                    ],
                    **self._read_runtime_trace(),
                }
            )

        can_finalize = callable(getattr(self._analyzer, "finalize", None))
        if not refinements and not zoom_hints and not (
            completed_refinement_turn and can_finalize
        ):
            return coarse
        merged = self._merge(coarse, refinements, zoom_hints)
        if self._bbox_calibrator is not None:
            try:
                merged = self._bbox_calibrator(
                    source_image_base64 or image_base64,
                    merged,
                )
            except Exception:
                logger.warning("Final bbox calibration failed; keeping merged result")
        if completed_refinement_turn and can_finalize:
            merged = await self._finalize_report(
                source_image_base64 or image_base64,
                modality,
                valid_regions,
                merged,
            )
        return merged

    async def _finalize_report(
        self,
        image_base64: str,
        modality: Modality,
        valid_regions: list[str],
        draft: AnalysisResult,
    ) -> AnalysisResult:
        finalize_method = getattr(self._analyzer, "finalize", None)
        if not callable(finalize_method):
            return draft
        trace = list(draft.analysis_trace)
        refinement_trace = [
            event
            for event in trace
            if event.get("stage") == "refine" and event.get("status") == "completed"
        ]
        try:
            final = await finalize_method(
                image_base64,
                modality,
                valid_regions,
                draft=draft,
                refinement_trace=refinement_trace,
            )
        except Exception as exc:
            logger.warning("Final report reconciliation failed", error=str(exc))
            trace.append(
                {
                    "stage": "finalize",
                    "status": "failed",
                    "tool": "report_finalizer",
                    "source": "original_roi",
                    "error_type": type(exc).__name__,
                }
            )
            reason = (
                "Final report reconciliation failed; review the narrative against "
                "the refined findings."
            )
            return dataclasses.replace(
                draft,
                analysis_trace=trace,
                review_required=True,
                review_reasons=list(dict.fromkeys([*draft.review_reasons, reason])),
            )

        reconciled = reconcile_final_report(draft, final)
        trace = list(reconciled.analysis_trace)
        trace.append(
            {
                "stage": "finalize",
                "status": "completed",
                "tool": "report_finalizer",
                "source": "original_roi",
                "finding_count": len(reconciled.findings),
                **self._read_runtime_trace(),
            }
        )
        return dataclasses.replace(reconciled, analysis_trace=trace)

    def _read_runtime_trace(self) -> dict[str, object]:
        trace_method = getattr(self._analyzer, "last_run_trace", None)
        if not callable(trace_method):
            return {}
        try:
            value = trace_method()
        except Exception:
            logger.warning("Analyzer runtime trace unavailable")
            return {}
        return value if isinstance(value, dict) else {}

    async def _run_refinement(
        self,
        crop_base64: str,
        modality: Modality,
        valid_regions: list[str],
        *,
        target: _RefinementTarget,
        crop_region: RegionRect,
    ) -> RefinementResult | None:
        refine_method = getattr(self._analyzer, "refine", None)
        for attempt in range(self._zoom_retry_attempts + 1):
            try:
                if callable(refine_method):
                    result = await refine_method(
                        crop_base64,
                        modality,
                        valid_regions,
                        hypothesis=target.hypothesis,
                        crop_region=crop_region,
                    )
                    if not isinstance(result, RefinementResult):
                        raise TypeError("refine() must return RefinementResult")
                    return result
                zoom = await self._analyzer.analyze(
                    crop_base64,
                    modality,
                    valid_regions,
                )
                return _legacy_refinement_result(zoom, target)
            except Exception:
                if attempt < self._zoom_retry_attempts:
                    logger.warning(
                        "Zoom refinement failed; retrying",
                        target=target.key,
                        attempt=attempt + 1,
                        max_retries=self._zoom_retry_attempts,
                    )
                else:
                    logger.warning(
                        "Zoom refinement failed; keeping coarse finding",
                        target=target.key,
                    )
        return None

    def _merge(
        self,
        coarse: AnalysisResult,
        refinements: list[tuple[_RefinementTarget, RegionRect, RefinementResult]],
        zoom_hints: list[str],
    ) -> AnalysisResult:
        """Apply explicit deltas and keep every remapped bbox safe for overlay."""
        merged = list(coarse.findings)
        allow_downgrade = False
        for target, crop_region, refinement in refinements:
            expected_target_id = (
                target.hypothesis.id if target.hypothesis is not None else None
            )
            for delta in refinement.deltas:
                if (
                    delta.action in {RefinementAction.REVISE, RefinementAction.RETRACT}
                    and expected_target_id is not None
                    and delta.target_id == expected_target_id
                ):
                    allow_downgrade = True
                merged = apply_refinement_delta(
                    merged,
                    delta,
                    crop_region=crop_region,
                    expected_target_id=expected_target_id,
                )
        return dataclasses.replace(
            coarse,
            findings=merged,
            severity=_merged_severity(
                coarse,
                merged,
                allow_downgrade=allow_downgrade,
            ),
            zoom_hints=[*coarse.zoom_hints, *zoom_hints],
        )


class MultiPassAnalyzer(VisionAnalyzerService):
    """Drop-in ``VisionAnalyzerService`` that runs a :class:`MultiPassInterpreter`.

    A host can adopt the coarse → crop → challenge → reconcile loop without
    changing its state machine. ``connect`` / ``chat`` / ``disconnect`` /
    ``is_connected`` delegate to the wrapped inner analyzer.
    """

    def __init__(
        self,
        inner: VisionAnalyzerService,
        interpreter: MultiPassInterpreter,
    ) -> None:
        self._inner = inner
        self._interpreter = interpreter

    async def analyze(
        self,
        image_base64: str,
        modality: Modality,
        valid_regions: list[str],
    ) -> AnalysisResult:
        return await self._interpreter.interpret(image_base64, modality, valid_regions)

    async def analyze_with_source_size(
        self,
        image_base64: str,
        modality: Modality,
        valid_regions: list[str],
        *,
        source_size_px: tuple[int, int] | None,
        source_image_base64: str | None = None,
        local_candidate_regions: list[RegionRect] | None = None,
    ) -> AnalysisResult:
        """Analyze with captured-image dimensions for resolution-aware zoom."""
        return await self._interpreter.interpret(
            image_base64,
            modality,
            valid_regions,
            source_image_base64=source_image_base64,
            source_size_px=source_size_px,
            local_candidate_regions=local_candidate_regions,
        )

    async def chat(self, message: str) -> str:
        return await self._inner.chat(message)

    async def connect(self) -> None:
        await self._inner.connect()

    async def disconnect(self) -> None:
        await self._inner.disconnect()

    def is_connected(self) -> bool:
        return self._inner.is_connected()
