"""Multi-pass interpretation orchestrator (application layer).

Lets the agent look at a complex image more than once: a coarse first pass
finds candidate regions, then the orchestrator crops each non-normal region out
of the *original-resolution* ROI image and re-sends just that slice for a
closer, higher-effective-resolution look. Refined bounding boxes are mapped
back into the global ROI coordinate space so the overlay can draw them in the
right place.

Design constraints (see AGENTS.md four cores):
- Core 1: refined ``Finding.bboxes`` stay in normalized 0-1 ROI coordinates.
- The orchestrator consumes provider-neutral analyzer capabilities; the host
  supplies transport/tool identity and its required checklist inventory.
- Privacy: every crop is a *subset* of the user-defined ROI, so capture is
  never widened beyond the ROI (a zoom crop can only shrink the region).
- DDD: this module decodes no images itself. Image slicing is delegated to an
  injected :class:`ImageCropper`, keeping PIL/numpy out of the application and
  domain layers.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

import structlog

from medical_image_harness.ekg_layout import (
    canonical_ekg_lead_name,
    normalize_ekg_row_strip_layout,
    parse_ekg_lead_inventory,
    parse_normalized_region,
)
from medical_image_harness.models import (
    AnalysisResult,
    ChecklistItem,
    Finding,
    Modality,
    RegionRect,
    Severity,
)
from medical_image_harness.profiles import get_active_registry
from medical_image_harness.protocols import (
    PENDING_MULTIPASS_REASON,
    AnalyzerPort,
    StageTools,
    VisionAnalyzerService,
)

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
DEFAULT_INITIAL_RESPONSE_SLA_SEC = 60.0
DEFAULT_FIRST_REFINEMENT_SLA_SEC = 100.0
DEFAULT_TOTAL_ANALYSIS_SLA_SEC = 180.0
DEFAULT_MAX_REFINEMENT_TURN_SEC = 55.0
# Finalization can make one bounded JSON-repair retry. Reserve enough time for
# both model turns while keeping the case under the 180-second outer deadline.
DEFAULT_MAX_FINALIZATION_TURN_SEC = 80.0
DEFAULT_FINALIZATION_RESERVE_SEC = 80.0
DEFAULT_MIN_FOLLOWUP_BUDGET_SEC = 8.0
_SLA_RETURN_BUFFER_SEC = 1.0
# ``asyncio.wait_for`` and the Gateway reader can race by a few milliseconds
# when a response and the timer become ready in the same event-loop tick.  A
# bounded grace is allowed only for already-running optional follow-up turns;
# the required initial response keeps its hard deadline.  The effective grace
# is additionally capped at 1% of the budget remaining when the turn starts.
_FOLLOWUP_COMPLETION_GRACE_SEC = 0.250
_MAX_HYPOTHESIS_COVERING_AREA = 0.12
_MAX_HYPOTHESIS_COVERING_HEIGHT = 0.45
_MAX_EKG_CONTEXT_CROP_AREA = 0.65
_MAX_EKG_CONTEXT_CROP_HEIGHT = 0.85
_MAX_EKG_FINDING_BOX_WIDTH = 0.35
_MAX_EKG_FINDING_BOX_HEIGHT = 0.30
_MAX_EKG_FINDING_BOX_AREA = 0.08

_T = TypeVar("_T")


class AnalysisSlaTimeout(TimeoutError):
    """A required analysis stage exhausted its absolute SLA budget."""

    def __init__(self, stage: str, *, budget_sec: float, elapsed_ms: int) -> None:
        self.stage = stage
        self.budget_sec = budget_sec
        self.elapsed_ms = elapsed_ms
        super().__init__(
            f"{stage} exceeded {budget_sec:g}s SLA after {elapsed_ms / 1000:.3f}s"
        )

    def audit_trace(self) -> dict[str, object]:
        """Return outcome metadata suitable for logs, never model reasoning."""
        return {
            "stage": "analysis_sla",
            "status": "required_stage_timeout",
            "timed_out_stage": self.stage,
            "budget_sec": self.budget_sec,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclasses.dataclass(frozen=True)
class _AnalysisDeadline:
    started_at: float
    clock: Callable[[], float]

    def elapsed_sec(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    def elapsed_ms(self) -> int:
        return int(self.elapsed_sec() * 1000)

    def remaining_sec(self, absolute_deadline_sec: float) -> float:
        return max(0.0, absolute_deadline_sec - self.elapsed_sec())


def _bounded_operation_timeout_sec(
    remaining_sec: float,
    completion_grace_sec: float = 0.0,
) -> float:
    """Add at most 1% scheduler grace without turning tiny budgets into retries."""

    remaining = max(0.0, remaining_sec)
    effective_grace = min(max(0.0, completion_grace_sec), remaining * 0.01)
    return remaining + effective_grace


_EKG_SYSTEMATIC_LEAD_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "precordial_leads",
        frozenset({"lead_V1", "lead_V2", "lead_V3", "lead_V4", "lead_V5", "lead_V6"}),
    ),
    (
        "limb_leads",
        frozenset(
            {"lead_I", "lead_II", "lead_III", "lead_aVR", "lead_aVL", "lead_aVF"}
        ),
    ),
)
_EKG_PRECORDIAL_LEADS: frozenset[str] = frozenset(
    {"lead_V1", "lead_V2", "lead_V3", "lead_V4", "lead_V5", "lead_V6"}
)
_WAVEFORM_RHYTHM_ATTENTION_TERMS: tuple[str, ...] = (
    "atrial fibrillation",
    "atrial flutter",
    "premature ventricular",
    "ventricular premature",
    "premature atrial",
    "atrial premature",
    "ectop",
    "av block",
    "atrioventricular block",
    "heart block",
    "prolonged pr",
    "long qt",
    "prolonged qt",
    "qt prolongation",
    "sinus bradycardia",
    "sinus tachycardia",
    "bradycardia",
    "atrial enlargement",
    "atrial abnormality",
    "undetermined rhythm",
    "junctional rhythm",
    "supraventricular tachycardia",
    "ventricular tachycardia",
    "rapid ventricular response",
)
_WAVEFORM_LIMB_ATTENTION_TERMS: tuple[str, ...] = (
    "axis deviation",
    "fascicular block",
    "left anterior fascicular",
    "left posterior fascicular",
    "inferior infarct",
)
_WAVEFORM_PRECORDIAL_ATTENTION_TERMS: tuple[str, ...] = (
    "bundle branch block",
    "intraventricular conduction",
    "septal infarct",
    "anterior infarct",
    "anteroseptal infarct",
    "infarct",
    "low voltage",
    "ventricular hypertrophy",
    "nonspecific st",
    "st abnormality",
    "st-t",
    "t wave abnormality",
    "t-wave abnormality",
    "repolarization",
    "st elevation",
    "st depression",
)
_EKG_TEMPORAL_CONTEXT_TERMS: tuple[str, ...] = (
    "rhythm",
    "atrial fibrillation",
    "atrial flutter",
    "premature",
    "pvc",
    "pac",
    "ectop",
    "bigeminy",
    "trigeminy",
    "couplet",
    "pause",
    "av block",
    "atrioventricular block",
    "heart block",
    "first-degree",
    "second-degree",
    "third-degree",
    "pr interval",
    "qt interval",
    "bradycard",
    "tachycard",
)
_EKG_MULTI_LEAD_CONTEXT_TERMS: tuple[str, ...] = (
    "bundle branch",
    "fascicular",
    "bifascicular",
    "trifascicular",
    "left anterior",
    "left posterior",
    "infarct",
    "ischemi",
    "st elevation",
    "st depression",
    "st-t",
    "repolarization",
    "t-wave",
    "t wave",
    "q wave",
    "q-wave",
    "r-wave progression",
    "r wave progression",
    "axis deviation",
    "low voltage",
    "hypertrophy",
    "lvh",
    "rvh",
    "strain",
)
_EKG_SYNCHRONIZED_EVENT_TERMS: tuple[str, ...] = (
    "broad complex",
    "broad-complex",
    "paced",
    "pacing",
    "pacemaker",
    "wide qrs",
    "wide-qrs",
    "wide complex",
    "wide-complex",
    "ventricular ectop",
    "synchronous",
    "synchronized",
    "artifact",
    "discordant",
)
_EKG_ROW_EVENT_MIN_WIDTH = 0.18
_EKG_ROW_EVENT_MAX_WIDTH = 0.35
_GENERIC_CRITICAL_LABELS: frozenset[str] = frozenset(
    {
        "abnormal finding",
        "abnormality",
        "acute abnormality",
        "critical finding",
        "critical observation",
        "finding",
        "lesion",
        "observation",
        "urgent finding",
    }
)
_CRITICAL_TEMPORAL_SUPPORT_TERMS: tuple[str, ...] = (
    *_EKG_TEMPORAL_CONTEXT_TERMS,
    "ventricular fibrillation",
    "wide complex",
    "wide-complex",
    "torsade",
)
_CRITICAL_TERRITORIAL_SUPPORT_TERMS: tuple[str, ...] = (
    "infarct",
    "ischemi",
    "st elevation",
    "st depression",
    "stemi",
    "wellens",
    "de winter",
    "hyperacute",
    "reciprocal",
    "coronary occlusion",
    "acute injury",
)
_CRITICAL_MORPHOLOGY_SUPPORT_TERMS: tuple[str, ...] = (
    "hyperkalemia",
    "hyperkalaemia",
    "peaked t",
    "tall t",
    "long qt",
    "prolonged qt",
    "qt prolongation",
    "brugada",
    "sodium channel",
    "bundle branch",
    "fascicular",
    "conduction",
    "wide qrs",
)
_CRITICAL_TRIAGE_REASON = (
    "Critical-first triage intentionally deferred unrelated lower-priority "
    "refinements and checklist exclusions so the time-critical image finding "
    "could be adjudicated first; native-study review remains required."
)
_CRITICAL_TRIAGE_UNASSESSED = "not_assessed_due_to_critical_triage"
_CRITICAL_TRIAGE_UNRESOLVED_REASON = (
    "Critical-first triage ended before every deferred checklist axis was "
    "assessed on the original study; the unresolved axes remain explicitly "
    "unassessed and the result remains incomplete for clinician review."
)
_CRITICAL_TRIAGE_UNRESOLVED_VALUE_MARKERS = (
    "not assessed",
    "not assessable",
    "unassessed",
    "cannot assess",
    "unable to assess",
    "insufficient to assess",
    "not evaluated",
    "indeterminate",
    "unknown",
    "deferred",
)
_CRITICAL_TRIAGE_BROAD_NEGATIVE_SUMMARY_MARKERS = (
    "normal ecg",
    "normal ekg",
    "normal study",
    "normal tracing",
    "within normal limits",
    "no acute abnormality",
    "no acute abnormalities",
    "no significant abnormality",
    "no significant abnormalities",
    "no abnormality",
    "no abnormalities",
    "all findings absent",
    "abnormalities absent",
    "unremarkable study",
    "unremarkable ecg",
    "unremarkable ekg",
)
_EKG_VENTRICULAR_RUN_CLAIM_MARKERS = (
    "ventricular tachycardia",
    "ventricular run",
    "nonsustained vt",
    "non sustained vt",
    "nsvt",
    "vt",
)
_EKG_NONASSERTIVE_CLAIM_MARKERS = (
    "possible",
    "possibly",
    "suspected",
    "suspicious for",
    "candidate",
    "concern for",
    "cannot exclude",
    "not excluded",
    "consider",
    "versus",
    " vs ",
    "insufficient",
    "unresolved",
    "not confirmed",
    "no confirmed",
    "cannot establish",
)
_EKG_MIN_VENTRICULAR_RUN_TIMESTAMPS = 3


class ImageCropper(Protocol):
    """Crops a normalized sub-region out of a base64 PNG image.

    Implemented by infrastructure (which owns PIL). ``region`` is expressed in
    normalized 0-1 coordinates relative to the *input* image. Implementations
    may upscale the crop so small lesions become legible; the returned image is
    still a base64 PNG. The crop must never extend outside the input image.
    """

    def __call__(self, image_base64: str, region: RegionRect) -> str: ...


class EkgRowStripDetector(Protocol):
    """Return bounded local geometry evidence for a base64 EKG image."""

    def __call__(self, image_base64: str) -> dict[str, object]: ...


class BboxCalibrator(Protocol):
    """Locally calibrate coarse boxes without making a diagnosis."""

    def __call__(
        self,
        image_base64: str,
        result: AnalysisResult,
    ) -> AnalysisResult: ...


class RefinementAction(StrEnum):
    """A hypothesis-aware decision returned by a crop refinement turn."""

    CONFIRM = "confirm"
    REVISE = "revise"
    RETRACT = "retract"
    ADD = "add"


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
    """Structured output of one hypothesis-aware crop refinement turn.

    ``unlocalized_missing_receipt_delta_indexes`` is non-model provenance.  The
    infrastructure client may populate it only after a parsed turn returned
    useful semantics but no bbox-validator receipt at all.  Unknown fields in
    model JSON are never mapped into this attribute, so model-authored notes
    cannot opt an ungrounded finding into the merge path.
    """

    deltas: tuple[RefinementDelta, ...] = ()
    unlocalized_missing_receipt_delta_indexes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        indexes = self.unlocalized_missing_receipt_delta_indexes
        if any(
            type(index) is not int or index < 0 or index >= len(self.deltas)
            for index in indexes
        ) or len(set(indexes)) != len(indexes):
            raise ValueError("invalid unlocalized refinement delta indexes")


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
        probe_id: str = "",
        crop_lead_regions: dict[str, RegionRect] | None = None,
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


def project_ekg_lead_regions_to_crop(
    layout: object,
    crop_region: RegionRect,
) -> dict[str, RegionRect]:
    """Project visible EKG lead rows into a crop-local coordinate frame.

    The model sees only the crop, so the original layout coordinates are easy
    to misread. This deterministic projection gives each visible lead's actual
    bounds in the attached crop without moving or diagnosing any waveform.
    """
    crop = _clamp_region(crop_region)
    if crop is None:
        return {}
    projected: dict[str, RegionRect] = {}
    for lead in parse_ekg_lead_inventory(layout).leads:
        x0 = max(crop.x, lead.bbox.x)
        y0 = max(crop.y, lead.bbox.y)
        x1 = min(crop.x + crop.w, lead.bbox.x + lead.bbox.w)
        y1 = min(crop.y + crop.h, lead.bbox.y + lead.bbox.h)
        if x1 <= x0 or y1 <= y0:
            continue
        projected_x = clamp_unit((x0 - crop.x) / crop.w)
        projected_y = clamp_unit((y0 - crop.y) / crop.h)
        projected_w = min(
            clamp_unit((x1 - x0) / crop.w),
            1.0 - projected_x,
        )
        projected_h = min(
            clamp_unit((y1 - y0) / crop.h),
            1.0 - projected_y,
        )
        if projected_w <= 0.0 or projected_h <= 0.0:
            continue
        projected[lead.name] = RegionRect(
            x=projected_x,
            y=projected_y,
            w=projected_w,
            h=projected_h,
        )
    return projected


def _refinement_probe_id(
    target_key: str,
    modality: Modality,
    crop_lead_regions: dict[str, RegionRect] | None,
) -> str:
    """Route every trusted precordial crop through the balanced lead review.

    The target key remains unchanged for merge/audit identity.  This probe id is
    prompt context only, so a hypothesis such as ``f1`` cannot accidentally
    bypass the precordial R/S-transition and ST-T cross-check merely because its
    label named a different candidate.
    """
    if (
        modality is not Modality.EKG
        or not crop_lead_regions
        or "precordial_leads" in target_key
    ):
        return target_key
    mapped_leads = {
        canonical
        for name in crop_lead_regions
        if (canonical := canonical_ekg_lead_name(name)) is not None
    }
    if mapped_leads & _EKG_PRECORDIAL_LEADS:
        return f"{target_key}_precordial_leads"
    return target_key


def select_zoom_targets(
    result: AnalysisResult,
    *,
    max_targets: int,
) -> list[Finding]:
    """Pick non-normal findings that can be routed to a useful closer look.

    Critical findings are prioritized over warnings, then info findings; normal
    findings are skipped. Unlocalized EKG hypotheses may use only a bounded
    semantic region from the model-declared lead layout. At most ``max_targets``
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
            continue
        if result.modality is Modality.EKG:
            contextual = _ekg_contextual_crop_strategy(finding, result.layout)
            if contextual is not None:
                candidates.append(dataclasses.replace(finding, bboxes=[contextual[0]]))
    # Critical first, then warning, then info; preserve original order in a tier.
    candidates.sort(key=lambda f: _ZOOM_PRIORITY[f.severity])
    return candidates[:max_targets]


def _is_specific_critical_candidate(finding: Finding) -> bool:
    label = " ".join(
        "".join(
            character if character.isalnum() else " "
            for character in finding.label.casefold()
        ).split()
    )
    return bool(
        finding.severity is Severity.CRITICAL
        and finding.id.strip()
        and label
        and finding.detail.strip()
        and label not in _GENERIC_CRITICAL_LABELS
    )


def select_critical_triage_candidates(result: AnalysisResult) -> list[Finding]:
    """Return structured, localizable critical findings in deterministic order.

    Top-level severity alone is not sufficient. A critical finding also needs
    a stable ID, a non-generic clinical label, concrete detail, and either a
    usable bbox or a bounded EKG layout route supplied by select_zoom_targets.
    This keeps vague urgency language from suppressing the normal systematic
    review path.
    """

    zoomable = select_zoom_targets(
        result,
        max_targets=max(1, len(result.findings)),
    )
    return [finding for finding in zoomable if _is_specific_critical_candidate(finding)]


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


def _layout_region(layout: object, key: str) -> RegionRect | None:
    if not isinstance(layout, dict):
        return None
    return parse_normalized_region(layout.get(key))


def _bounded_ekg_context_region(region: RegionRect) -> bool:
    return (
        region.w * region.h <= _MAX_EKG_CONTEXT_CROP_AREA
        and region.h <= _MAX_EKG_CONTEXT_CROP_HEIGHT
    )


def _trustworthy_ekg_horizontal_event_region(region: RegionRect) -> bool:
    """Return whether a box localizes an event in time, not only a lead row.

    A full-width lead or lead-group rectangle is valid spatial context, but it
    contains no evidence for choosing one horizontal time column.  In
    particular, layout routing can synthesize such a rectangle for an
    otherwise unlocalized hypothesis.  Treating that synthetic rectangle as a
    timed event on the next routing pass manufactured a fixed central crop.
    """

    return region.w <= _MAX_EKG_FINDING_BOX_WIDTH + 1e-9


def _ekg_contextual_crop_strategy(
    finding: Finding,
    layout: object,
) -> tuple[RegionRect, str] | None:
    """Choose a full temporal segment or bounded multi-lead context when needed."""
    text = f"{finding.label} {finding.detail}".casefold()
    lead_regions = parse_ekg_lead_inventory(layout).by_name()

    # A full-width 12-row strip shows every lead at the same horizontal time.
    # For intermittent pacing, wide-complex beats, or a suspected synchronous
    # artifact, a narrow full-height time slice preserves the cross-lead event
    # that a single-lead temporal crop destroys.  Restrict this strategy to the
    # locally verified row-strip geometry; columns in a 3x4 printout do not
    # share the same time axis.
    layout_format = str(layout.get("format") or "") if isinstance(layout, dict) else ""
    if (
        layout_format == "12lead_12x1"
        and len(lead_regions) >= 12
        and any(term in text for term in _EKG_SYNCHRONIZED_EVENT_TERMS)
        and finding.bboxes
    ):
        all_valid_boxes = [
            region
            for item in finding.bboxes
            if (region := _clamp_region(item)) is not None
        ]
        valid_boxes = [
            region
            for region in all_valid_boxes
            if _trustworthy_ekg_horizontal_event_region(region)
        ]
        if valid_boxes:
            event_span = covering_region(valid_boxes)
            if _unique_ekg_event_timestamp_count(valid_boxes) > 1:
                # Preserve all distinct event times and the beats between them.
                # A single-event crop cannot test a multi-event rhythm claim.
                width = min(1.0, max(_EKG_ROW_EVENT_MIN_WIDTH, event_span.w + 0.08))
                center_x = event_span.x + event_span.w / 2.0
                x = min(max(0.0, center_x - width / 2.0), 1.0 - width)
                return (
                    RegionRect(x=x, y=0.0, w=width, h=1.0),
                    "row_strip_multi_event_temporal_context",
                )
            focus = max(valid_boxes, key=lambda region: (region.w * region.h, region.w))
            width = min(
                _EKG_ROW_EVENT_MAX_WIDTH,
                max(_EKG_ROW_EVENT_MIN_WIDTH, focus.w * 2.0),
            )
            center_x = focus.x + focus.w / 2.0
            x = min(max(0.0, center_x - width / 2.0), 1.0 - width)
            return (
                RegionRect(x=x, y=0.0, w=width, h=1.0),
                "row_strip_cross_lead_temporal_context",
            )
        if all_valid_boxes:
            # The evidence says which lead row/group to inspect but does not
            # identify an event time.  Keep that horizontal context intact;
            # choosing its midpoint would be fabricated localization.
            return (
                covering_region(all_valid_boxes),
                "row_strip_unlocalized_full_width_context",
            )

    if any(term in text for term in _EKG_TEMPORAL_CONTEXT_TERMS):
        rhythm_strip = _layout_region(layout, "rhythm_strip_bbox")
        if rhythm_strip is not None and _bounded_ekg_context_region(rhythm_strip):
            return rhythm_strip, "declared_rhythm_strip"

        lead_ii = lead_regions.get("lead_II")
        if lead_ii is not None and _bounded_ekg_context_region(lead_ii):
            return lead_ii, "full_lead_II_temporal_context"

        for raw_name in finding.regions:
            canonical = canonical_ekg_lead_name(raw_name)
            candidate = lead_regions.get(canonical or "")
            if candidate is not None and _bounded_ekg_context_region(candidate):
                return candidate, "full_declared_lead_temporal_context"

    if any(term in text for term in _EKG_MULTI_LEAD_CONTEXT_TERMS):
        declared: list[RegionRect] = []
        seen: set[str] = set()
        for raw_name in finding.regions:
            canonical = canonical_ekg_lead_name(raw_name)
            if canonical is None or canonical in seen:
                continue
            seen.add(canonical)
            region = lead_regions.get(canonical)
            if region is not None:
                declared.append(region)
        if len(declared) >= 2:
            context = covering_region(declared)
            if _bounded_ekg_context_region(context):
                return context, "declared_multi_lead_context"
    return None


def select_hypothesis_crop_region(
    finding: Finding,
    *,
    modality: Modality,
    layout: object | None = None,
) -> RegionRect:
    """Select a useful hypothesis crop without losing required EKG context.

    Temporal EKG hypotheses need enough consecutive beats, while conduction and
    territorial patterns need their declared lead set. Other disjoint evidence
    stays local so a refinement turn still gains effective image resolution.
    """
    valid = [
        region for item in finding.bboxes if (region := _clamp_region(item)) is not None
    ]
    if not valid:
        raise ValueError("finding requires at least one non-degenerate bbox")
    covering = covering_region(valid)
    if modality is not Modality.EKG:
        return covering
    contextual = _ekg_contextual_crop_strategy(finding, layout)
    if contextual is not None:
        return contextual[0]
    if (
        covering.w * covering.h <= _MAX_HYPOTHESIS_COVERING_AREA
        and covering.h <= _MAX_HYPOTHESIS_COVERING_HEIGHT
    ):
        return covering
    return max(valid, key=lambda region: (region.w * region.h, region.w, region.h))


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


def _critical_candidate_text(findings: list[Finding]) -> str:
    return " ".join(
        f"{finding.label} {finding.detail} {finding.question or ''}".casefold()
        for finding in findings
    )


def select_ekg_critical_support_probe(
    result: AnalysisResult,
    critical_findings: list[Finding],
    planned_regions: list[RegionRect],
    *,
    systematic_candidates: list[tuple[str, RegionRect]] | None = None,
) -> tuple[str, RegionRect, str] | None:
    """Choose at most one non-redundant, mechanism-related EKG support crop."""

    if result.modality is not Modality.EKG or not critical_findings:
        return None
    text = _critical_candidate_text(critical_findings)
    temporal = any(term in text for term in _CRITICAL_TEMPORAL_SUPPORT_TERMS)
    territorial = any(term in text for term in _CRITICAL_TERRITORIAL_SUPPORT_TERMS)
    morphology = any(term in text for term in _CRITICAL_MORPHOLOGY_SUPPORT_TERMS)
    if not (temporal or territorial or morphology):
        return None

    generic = list(
        systematic_candidates
        if systematic_candidates is not None
        else select_ekg_systematic_probe_regions(
            result,
            max_probes=len(_EKG_SYSTEMATIC_LEAD_GROUPS),
        )
    )
    candidates: list[tuple[str, RegionRect]] = []
    reason: str
    if temporal:
        reason = "critical_temporal_crosscheck"
        rhythm_strip = _layout_region(result.layout, "rhythm_strip_bbox")
        if rhythm_strip is not None:
            candidates.append(("declared_rhythm_strip", rhythm_strip))
        lead_ii = parse_ekg_lead_inventory(result.layout).by_name().get("lead_II")
        if lead_ii is not None:
            candidates.append(("lead_II", lead_ii))
        candidates.extend(item for item in generic if item[0] == "limb_leads")
    else:
        reason = (
            "critical_territorial_reciprocal_crosscheck"
            if territorial
            else "critical_morphology_crosslead_check"
        )
        candidates.extend(
            sorted(
                generic,
                key=lambda item: max(
                    (
                        _overlap_fraction(item[1], planned)
                        for planned in planned_regions
                    ),
                    default=0.0,
                ),
            )
        )

    seen: set[RegionRect] = set()
    for key, region in candidates:
        if region in seen:
            continue
        seen.add(region)
        if any(
            _overlap_fraction(region, planned) >= 0.85 for planned in planned_regions
        ):
            continue
        return key, region, reason
    return None


def select_ekg_waveform_attention_probe_regions(
    result: AnalysisResult,
) -> list[tuple[str, RegionRect]]:
    """Route an uncalibrated waveform disagreement to a useful image crop.

    The external classifier never determines the diagnosis or severity here. Its
    highest-ranked labels may only choose where the analyzer looks next. Rhythm or
    ectopy routes lead II; axis/fascicular candidates route limb leads; and
    conduction/voltage/infarct/ST-T candidates route precordials. Image
    morphology still determines every conclusion.
    """
    if result.modality is not Modality.EKG:
        return []
    ranked_labels: list[str] = []
    for event in result.analysis_trace:
        if not isinstance(event, dict):
            continue
        audits = event.get("tool_audit")
        if not isinstance(audits, list):
            continue
        for audit in audits:
            if (
                not isinstance(audit, dict)
                or audit.get("tool") != "ecg_founder_analyze_waveform"
                or audit.get("status") != "ok"
            ):
                continue
            predictions = audit.get("predictions")
            if not isinstance(predictions, list):
                continue
            for prediction in predictions[:3]:
                if isinstance(prediction, dict):
                    label = str(prediction.get("label") or "").strip().casefold()
                    if label:
                        ranked_labels.append(label)
    lead_regions = parse_ekg_lead_inventory(result.layout).by_name()
    routed: list[tuple[str, RegionRect]] = []
    seen: set[str] = set()

    def add_group(key: str, names: tuple[str, ...]) -> None:
        if key in seen:
            return
        regions = [lead_regions[name] for name in names if name in lead_regions]
        if len(regions) < 2:
            return
        region = covering_region(regions)
        if region.w * region.h >= _MAX_EKG_CONTEXT_CROP_AREA:
            return
        seen.add(key)
        routed.append((key, region))

    for label in ranked_labels:
        if any(term in label for term in _WAVEFORM_RHYTHM_ATTENTION_TERMS):
            lead_ii = lead_regions.get("lead_II")
            if lead_ii is not None and "waveform_rhythm_lead_II" not in seen:
                seen.add("waveform_rhythm_lead_II")
                routed.append(("waveform_rhythm_lead_II", lead_ii))
        elif any(term in label for term in _WAVEFORM_LIMB_ATTENTION_TERMS):
            add_group(
                "waveform_attention_limb_leads",
                ("lead_I", "lead_II", "lead_III", "lead_aVR", "lead_aVL", "lead_aVF"),
            )
        elif any(term in label for term in _WAVEFORM_PRECORDIAL_ATTENTION_TERMS):
            add_group(
                "waveform_attention_precordial_leads",
                ("lead_V1", "lead_V2", "lead_V3", "lead_V4", "lead_V5", "lead_V6"),
            )
    return routed


def _waveform_rhythm_context(
    result: AnalysisResult,
) -> tuple[dict[str, object] | None, list[str]]:
    measurement: dict[str, object] | None = None
    ranked_labels: list[str] = []
    for event in result.analysis_trace:
        if not isinstance(event, dict):
            continue
        audits = event.get("tool_audit")
        if not isinstance(audits, list):
            continue
        for audit in audits:
            if (
                not isinstance(audit, dict)
                or audit.get("tool") != "ecg_founder_analyze_waveform"
                or audit.get("status") != "ok"
            ):
                continue
            predictions = audit.get("predictions")
            if isinstance(predictions, list):
                for prediction in predictions[:3]:
                    if isinstance(prediction, dict):
                        label = str(prediction.get("label") or "").strip().casefold()
                        if label:
                            ranked_labels.append(label)
            response = audit.get("response_evidence")
            candidate = (
                response.get("rhythm_measurement")
                if isinstance(response, dict)
                else None
            )
            if (
                isinstance(candidate, dict)
                and candidate.get("method") == "lead_II_qrs_energy_v1"
                and candidate.get("status") == "ok"
                and candidate.get("diagnostic_scope") == "rhythm_regularity_only"
            ):
                measurement = candidate
    return measurement, ranked_labels


def apply_ekg_waveform_rhythm_conflict_guard(
    result: AnalysisResult,
) -> AnalysisResult:
    """Keep quantified AF-vs-sinus disagreement from closing as normal.

    This guard does not diagnose atrial fibrillation. It requires two independent
    supporting signals: an AF/flutter label ranked in the waveform model's top
    three and deterministic lead-II R-R timing classified as irregular. The
    output remains an explicit conflict for native-ECG/human confirmation.
    """

    if result.modality is not Modality.EKG:
        return result
    measurement, ranked_labels = _waveform_rhythm_context(result)
    if measurement is None or measurement.get("regularity_signal") != "irregular":
        return result
    if not any(
        "atrial fibrillation" in label or "atrial flutter" in label
        for label in ranked_labels
    ):
        return result
    rr_cv = float(measurement.get("rr_cv") or 0.0)
    diff_fraction = float(
        measurement.get("successive_rr_diff_over_80ms_fraction") or 0.0
    )
    guard_detail = (
        "Deterministic lead-II R-peak timing is materially irregular "
        f"(R-R CV {rr_cv:.3f}; {diff_fraction:.0%} of successive R-R "
        "changes >=80 ms). This does not diagnose atrial fibrillation, but "
        "it conflicts with a regular-sinus visual conclusion and requires "
        "review on the native ECG."
    )
    question = (
        "On the native ECG, are P waves reproducibly present before every "
        "QRS, or is this atrial fibrillation/another irregular rhythm?"
    )
    rhythm_terms = (
        "atrial fibrillation",
        "atrial flutter",
        "irregular rhythm",
        "irregularly irregular",
    )
    existing_index = next(
        (
            index
            for index, item in enumerate(result.findings)
            if any(
                term in f"{item.label} {item.detail}".casefold()
                for term in rhythm_terms
            )
        ),
        None,
    )
    findings = list(result.findings)
    reconciled_finding_id = "waveform-rhythm-conflict"
    if existing_index is not None:
        existing = findings[existing_index]
        if (
            _SEVERITY_RANK[existing.severity] >= _SEVERITY_RANK[Severity.WARNING]
            and _SEVERITY_RANK[result.severity] >= _SEVERITY_RANK[Severity.WARNING]
        ):
            return result
        reconciled_finding_id = existing.id
        findings[existing_index] = dataclasses.replace(
            existing,
            severity=max(
                existing.severity,
                Severity.WARNING,
                key=lambda item: _SEVERITY_RANK[item],
            ),
            bboxes=list(existing.bboxes),
            notes=list(dict.fromkeys([*existing.notes, guard_detail])),
            confidence=existing.confidence or "moderate",
            question=existing.question or question,
        )
    else:
        findings.append(
            Finding(
                id="waveform-rhythm-conflict",
                regions=["lead_II"],
                label="Irregular rhythm; atrial fibrillation not excluded",
                detail=guard_detail,
                severity=Severity.WARNING,
                bboxes=[],
                confidence="moderate",
                question=question,
                notes=[
                    "The marker identifies the lead-II review region, not a "
                    "localized lesion or confirmed diagnosis."
                ],
                source="waveform_rhythm_conflict_guard",
            )
        )
    checklist = dict(result.checklist)
    checklist["rhythm"] = ChecklistItem(
        value="irregular rhythm; atrial fibrillation not excluded",
        status=Severity.WARNING,
    )
    checklist["regularity"] = ChecklistItem(
        value="irregular R-R timing",
        status=Severity.WARNING,
    )
    checklist["p_wave"] = ChecklistItem(
        value="not reliably established in rhythm-conflict review",
        status=Severity.INFO,
    )
    reason = (
        "Deterministic R-R timing and waveform-model ranking conflict with the "
        "visual regular-sinus conclusion."
    )
    trace = [
        *result.analysis_trace,
        {
            "stage": "waveform_rhythm_guardrail",
            "status": "conflict_escalated_for_review",
            "tool": "lead_II_qrs_energy_v1",
            "regularity_signal": "irregular",
            "rr_interval_count": measurement.get("rr_interval_count"),
            "rr_cv": rr_cv,
            "successive_rr_diff_over_80ms_fraction": diff_fraction,
            "waveform_ranked_rhythm_candidates": ranked_labels,
            "reconciled_finding_id": reconciled_finding_id,
            "diagnosis_forced": False,
        },
    ]
    summary = result.summary.strip()
    conflict_summary = (
        "Rhythm conflict: irregular R-R timing is present and atrial "
        "fibrillation is not excluded."
    )
    if "atrial fibrillation" not in summary.casefold():
        summary = f"{conflict_summary} {summary}".strip()
    return dataclasses.replace(
        result,
        summary=summary,
        severity=max(
            result.severity,
            Severity.WARNING,
            key=lambda item: _SEVERITY_RANK[item],
        ),
        findings=findings,
        checklist=checklist,
        incomplete=True,
        incomplete_reasons=list(dict.fromkeys([*result.incomplete_reasons, reason])),
        review_required=True,
        review_reasons=list(dict.fromkeys([*result.review_reasons, reason])),
        analysis_trace=trace,
        next_steps=list(
            dict.fromkeys(
                [
                    *result.next_steps,
                    "Confirm rhythm and P-wave relationships on the native ECG.",
                ]
            )
        ),
    )


def _overlap_fraction(region: RegionRect, other: RegionRect) -> float:
    x0 = max(region.x, other.x)
    y0 = max(region.y, other.y)
    x1 = min(region.x + region.w, other.x + other.w)
    y1 = min(region.y + region.h, other.y + other.h)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return intersection / max(region.w * region.h, 1e-9)


def _uncovered_hypothesis_regions(
    crop_region: RegionRect,
    hypothesis: Finding,
) -> list[RegionRect]:
    """Return coarse evidence boxes that a refinement crop cannot adjudicate."""
    tolerance = 1e-6
    uncovered: list[RegionRect] = []
    for raw_region in hypothesis.bboxes:
        region = _clamp_region(raw_region)
        if region is None:
            continue
        if (
            region.x < crop_region.x - tolerance
            or region.y < crop_region.y - tolerance
            or region.x + region.w > crop_region.x + crop_region.w + tolerance
            or region.y + region.h > crop_region.y + crop_region.h + tolerance
        ):
            uncovered.append(region)
    return uncovered


def _crop_covers_entire_roi(crop_region: RegionRect) -> bool:
    """Return whether a crop can adjudicate an otherwise unlocalized finding."""

    tolerance = 1e-6
    return (
        crop_region.x <= tolerance
        and crop_region.y <= tolerance
        and crop_region.x + crop_region.w >= 1.0 - tolerance
        and crop_region.y + crop_region.h >= 1.0 - tolerance
    )


def _target_crop_coverage(
    crop_region: RegionRect,
    hypothesis: Finding,
) -> tuple[bool, list[RegionRect], int]:
    """Return coverage proof for a targeted mutation of a coarse hypothesis.

    A bounded local candidate may be attached to a bbox-less finding purely to
    route a closer look.  That synthetic rectangle is not evidence that the
    crop covers the whole coarse hypothesis.  Such a finding can be mutated
    only after an original-ROI turn; localized hypotheses require every coarse
    bbox to be inside the crop.
    """

    source_regions = [
        region
        for raw_region in hypothesis.bboxes
        if (region := _clamp_region(raw_region)) is not None
    ]
    if not source_regions:
        return _crop_covers_entire_roi(crop_region), [], 0
    uncovered = _uncovered_hypothesis_regions(crop_region, hypothesis)
    return not uncovered, uncovered, len(source_regions)


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


def _merge_crop_notes(
    original_notes: list[str],
    crop_notes: list[str],
    rationale: str,
    crop_region: RegionRect,
) -> list[str]:
    """Retain the local evidence scope when merging into a study-level finding.

    For example, a refinement's "V1 is absent" describes the selected crop,
    not necessarily the original ROI. The scope comes from the actual crop
    supplied by the orchestrator, never from model-authored lead names. Raw
    rationale and precise coordinates remain in the refinement audit trace.
    """
    prefix = (
        "[Crop-only evidence; ROI "
        f"x={crop_region.x:.4f} y={crop_region.y:.4f} "
        f"w={crop_region.w:.4f} h={crop_region.h:.4f}] "
    )
    notes = list(original_notes)
    for text in [*crop_notes, rationale]:
        text = text.strip()
        if not text:
            continue
        scoped = prefix + text
        if scoped not in notes:
            notes.append(scoped)
    return notes


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
    allow_unlocalized_add: bool = False,
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
        if not mapped.bboxes and not allow_unlocalized_add:
            logger.warning(
                "Ignoring added refinement finding without a valid bbox",
                finding_id=payload.id,
            )
            return findings
        mapped = dataclasses.replace(
            mapped,
            id=_unique_finding_id(findings, mapped.id),
            notes=_merge_crop_notes([], mapped.notes, delta.rationale, crop_region),
            evidence=[], evidence_ids=[], observation_ids=[],
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
                    notes=_merge_crop_notes(
                        current.notes, [], delta.rationale, crop_region
                    ),
                )
            )
            continue

        mapped = _remap_finding_boxes(payload, crop_region)
        notes = _merge_crop_notes(
            current.notes, mapped.notes, delta.rationale, crop_region
        )
        if delta.action is RefinementAction.CONFIRM:
            result.append(
                dataclasses.replace(
                    current,
                    detail=mapped.detail or current.detail,
                    bboxes=mapped.bboxes or current.bboxes,
                    regions=mapped.regions or current.regions,
                    notes=notes,
                    evidence=[], evidence_ids=[], observation_ids=[],
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
                    evidence=[], evidence_ids=[], observation_ids=[],
                )
            )
    return result


def _normalize_discovery_delta(delta: RefinementDelta) -> RefinementDelta:
    """Enforce the non-urgent uncertainty contract for discovery-only crops.

    A reviewer question means the newly discovered diagnosis is unresolved.
    Critical triage findings retain their urgency, but a warning-level candidate
    cannot simultaneously be presented as an actionable diagnosis. The crop and
    marker remain available as an info/low-confidence item for expert review.
    """
    finding = delta.finding
    if (
        delta.action is RefinementAction.ADD
        and finding is not None
        and finding.severity is Severity.WARNING
        and finding.question.strip()
    ):
        return dataclasses.replace(
            delta,
            finding=dataclasses.replace(
                finding,
                severity=Severity.INFO,
                confidence="low",
            ),
        )
    return delta


def apply_unlocalized_ekg_grounding_guard(
    result: AnalysisResult,
) -> AnalysisResult:
    """Keep unlocalized EKG concerns visible without actionable marker claims."""

    if result.modality is not Modality.EKG:
        return result
    findings: list[Finding] = []
    events: list[dict[str, object]] = []
    for finding in result.findings:
        if (
            finding.severity not in {Severity.WARNING, Severity.CRITICAL}
            or finding.bboxes
        ):
            findings.append(finding)
            continue
        question = finding.question.strip() or (
            f"Can the reviewer confirm whether {finding.label or finding.id} is "
            "present on the native ECG? No tool-accepted tight image box remains."
        )
        note = (
            "No tool-accepted tight bbox remained after bounded crop/refine; "
            "the study-level triage severity is preserved separately."
        )
        findings.append(
            dataclasses.replace(
                finding,
                severity=Severity.INFO,
                confidence="low",
                question=question,
                notes=list(dict.fromkeys([*finding.notes, note])),
            )
        )
        events.append(
            {
                "stage": "grounding_guardrail",
                "status": "downgraded_unlocalized_actionable_finding",
                "tool": "bound_bbox_grounding_contract",
                "finding_id": finding.id,
                "severity_before": finding.severity.value,
                "severity_after": Severity.INFO.value,
                "study_severity_preserved": result.severity.value,
                "diagnosis_forced": False,
            }
        )
    if not events:
        return result
    reason = (
        "One or more EKG concerns lack a tool-accepted tight bbox and require "
        "confirmation on the native ECG."
    )
    return dataclasses.replace(
        result,
        findings=findings,
        incomplete=True,
        incomplete_reasons=list(dict.fromkeys([*result.incomplete_reasons, reason])),
        review_required=True,
        review_reasons=list(dict.fromkeys([*result.review_reasons, reason])),
        analysis_trace=[*result.analysis_trace, *events],
    )


def qualify_boxed_info_findings(result: AnalysisResult) -> AnalysisResult:
    """Preserve validated info markers while making uncertainty explicit."""

    findings: list[Finding] = []
    events: list[dict[str, object]] = []
    reasons = list(result.review_reasons)
    for finding in result.findings:
        if finding.severity is not Severity.INFO or not finding.bboxes:
            findings.append(finding)
            continue
        confidence_before = finding.confidence.strip()
        confidence = confidence_before or "low"
        question = finding.question.strip()
        if confidence == "low" and not question:
            question = (
                f"Can the reviewer confirm whether {finding.label or finding.id} "
                "is present in the highlighted source-image region?"
            )
        if confidence == confidence_before and question == finding.question.strip():
            findings.append(finding)
            continue
        note = (
            "The bounded info marker was retained with explicit uncertainty; "
            "its tool-accepted source-image coordinates were not changed."
        )
        findings.append(
            dataclasses.replace(
                finding,
                confidence=confidence,
                question=question,
                notes=list(dict.fromkeys([*finding.notes, note])),
            )
        )
        reason = (
            f"Boxed info finding requires confirmation: {finding.label or finding.id}."
        )
        if reason not in reasons:
            reasons.append(reason)
        events.append(
            {
                "stage": "boxed_info_guardrail",
                "status": "qualified_boxed_info_finding",
                "tool": "bounded_info_uncertainty_contract",
                "finding_id": finding.id,
                "confidence_before": confidence_before,
                "confidence_after": confidence,
                "question_added": not finding.question.strip() and bool(question),
                "coordinates_moved": False,
                "bbox_count": len(finding.bboxes),
            }
        )
    if not events:
        return result
    return dataclasses.replace(
        result,
        findings=findings,
        review_required=True,
        review_reasons=reasons,
        analysis_trace=[*result.analysis_trace, *events],
    )


def apply_ekg_overlay_bbox_guard(result: AnalysisResult) -> AnalysisResult:
    """Remove non-overlay and broad EKG boxes before final provenance binding."""

    if result.modality is not Modality.EKG:
        return result
    findings: list[Finding] = []
    events: list[dict[str, object]] = []
    reasons = list(result.review_reasons)
    for finding in result.findings:
        retained = [
            box
            for box in finding.bboxes
            if finding.severity is not Severity.NORMAL
            and box.w <= _MAX_EKG_FINDING_BOX_WIDTH
            and box.h <= _MAX_EKG_FINDING_BOX_HEIGHT
            and box.w * box.h <= _MAX_EKG_FINDING_BOX_AREA
        ]
        removed_count = len(finding.bboxes) - len(retained)
        if not removed_count:
            findings.append(finding)
            continue
        note = (
            "Non-overlay or broad lead-strip coordinates were removed before "
            "final binding; retained tight coordinates were not moved."
        )
        findings.append(
            dataclasses.replace(
                finding,
                bboxes=retained,
                notes=list(dict.fromkeys([*finding.notes, note])),
            )
        )
        reason = (
            f"Overlay geometry narrowed for {finding.label or finding.id}; "
            "verify any unlocalized concern on the native ECG."
        )
        if reason not in reasons:
            reasons.append(reason)
        events.append(
            {
                "stage": "ekg_overlay_bbox_guardrail",
                "status": "removed_non_tight_ekg_boxes",
                "tool": "tight_ekg_overlay_contract",
                "finding_id": finding.id,
                "removed_count": removed_count,
                "retained_count": len(retained),
                "coordinates_moved": False,
            }
        )
    if not events:
        return result
    return dataclasses.replace(
        result,
        findings=findings,
        review_required=True,
        review_reasons=reasons,
        analysis_trace=[*result.analysis_trace, *events],
    )


def reconcile_unavailable_ekg_rhythm_regions(
    result: AnalysisResult,
) -> AnalysisResult:
    """Replace an unavailable rhythm-strip tag with bbox-derived lead names."""

    if (
        result.modality is not Modality.EKG
        or _layout_region(result.layout, "rhythm_strip_bbox") is not None
    ):
        return result
    layout_leads = parse_ekg_lead_inventory(result.layout).by_name()
    findings: list[Finding] = []
    events: list[dict[str, object]] = []
    reasons = list(result.review_reasons)
    for finding in result.findings:
        if "rhythm_strip" not in finding.regions:
            findings.append(finding)
            continue
        geometric_leads = (
            list(
                dict.fromkeys(
                    name
                    for box in finding.bboxes
                    for name, lead in layout_leads.items()
                    if lead.x <= box.x + box.w / 2 <= lead.x + lead.w
                    and lead.y <= box.y + box.h / 2 <= lead.y + lead.h
                )
            )
            if finding.bboxes
            else []
        )
        regions = list(
            dict.fromkeys(
                [
                    *(region for region in finding.regions if region != "rhythm_strip"),
                    *geometric_leads,
                ]
            )
        )
        note = (
            "The layout has no distinct rhythm strip; the unavailable region tag "
            + (
                "was replaced with lead names derived from bbox centers."
                if geometric_leads
                else "was removed because no tight box supports a lead assignment."
            )
        )
        findings.append(
            dataclasses.replace(
                finding,
                regions=regions,
                notes=list(dict.fromkeys([*finding.notes, note])),
                confidence=finding.confidence or "low",
            )
        )
        reason = (
            f"Rhythm-strip region corrected for {finding.label or finding.id}; "
            "verify the geometry-derived lead assignment."
        )
        if reason not in reasons:
            reasons.append(reason)
        events.append(
            {
                "stage": "bbox_region_reconciliation",
                "status": (
                    "replaced_unavailable_rhythm_strip"
                    if geometric_leads
                    else "removed_unlocalized_rhythm_strip"
                ),
                "tool": "normalized_ekg_layout_geometry",
                "finding_id": finding.id,
                "removed_regions": ["rhythm_strip"],
                "geometry_regions": geometric_leads,
                "coordinates_moved": False,
            }
        )
    if not events:
        return result
    return dataclasses.replace(
        result,
        findings=findings,
        review_required=True,
        review_reasons=reasons,
        analysis_trace=[*result.analysis_trace, *events],
    )


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.NORMAL: 0,
    Severity.INFO: 1,
    Severity.WARNING: 2,
    Severity.CRITICAL: 3,
}
_FINAL_BBOX_COORDINATE_TOLERANCE = 1e-4
_EKG_STUDY_LEVEL_DEDUP_TERMS = (
    "rhythm",
    "sinus",
    "tachycard",
    "bradycard",
    "atrial fibrillation",
    "atrial flutter",
)
_CONFIDENCE_RANK = {"": 0, "low": 1, "moderate": 2, "high": 3}


def _ekg_study_level_finding_rank(finding: Finding) -> tuple[int, int, int, int]:
    regions = set(finding.regions)
    return (
        _SEVERITY_RANK[finding.severity],
        int(bool(regions & {"lead_II", "rhythm_strip"})),
        int(bool(finding.bboxes)),
        _CONFIDENCE_RANK.get(finding.confidence.casefold(), 0),
    )


def deduplicate_ekg_study_level_findings(result: AnalysisResult) -> AnalysisResult:
    """Retract exact duplicate EKG rate/rhythm findings before final binding."""

    if result.modality is not Modality.EKG:
        return result
    retained: list[Finding] = []
    by_label: dict[str, int] = {}
    events: list[dict[str, object]] = []
    for finding in result.findings:
        label_key = _normalized_label(finding.label)
        is_study_level = any(term in label_key for term in _EKG_STUDY_LEVEL_DEDUP_TERMS)
        existing_index = (
            by_label.get(label_key) if label_key and is_study_level else None
        )
        if existing_index is None:
            if label_key and is_study_level:
                by_label[label_key] = len(retained)
            retained.append(finding)
            continue

        existing = retained[existing_index]
        keep_new = _ekg_study_level_finding_rank(
            finding
        ) > _ekg_study_level_finding_rank(existing)
        kept = finding if keep_new else existing
        removed = existing if keep_new else finding
        if keep_new:
            retained[existing_index] = finding
        events.append(
            {
                "stage": "finding_deduplication",
                "status": "retracted_exact_study_level_duplicate",
                "tool": "ekg_semantic_duplicate_guard",
                "normalized_label": label_key,
                "retained_finding_id": kept.id,
                "retracted_finding_id": removed.id,
                "retained_bbox_count": len(kept.bboxes),
                "retracted_bbox_count": len(removed.bboxes),
                "coordinates_moved": False,
                "diagnosis_forced": False,
            }
        )
    if not events:
        return result
    return dataclasses.replace(
        result,
        findings=retained,
        analysis_trace=[*result.analysis_trace, *events],
    )


def _bbox_sequence_max_drift(
    draft_boxes: list[RegionRect],
    final_boxes: list[RegionRect],
) -> float | None:
    if len(draft_boxes) != len(final_boxes):
        return None
    return max(
        (
            abs(getattr(draft_box, axis) - getattr(final_box, axis))
            for draft_box, final_box in zip(draft_boxes, final_boxes, strict=True)
            for axis in ("x", "y", "w", "h")
        ),
        default=0.0,
    )


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
    *,
    required_checklist_keys: frozenset[str] | None = None,
    finalization_tool: str = "report_reconciliation",
) -> AnalysisResult:
    """Apply auditable final dispositions without surrendering grounded boxes."""

    def unique(*groups: list[str]) -> list[str]:
        return list(dict.fromkeys(item for group in groups for item in group if item))

    draft_by_id = {finding.id: finding for finding in draft.findings}
    if len(draft_by_id) != len(draft.findings) or "" in draft_by_id:
        raise ValueError("draft findings must have unique non-empty IDs")

    final_by_id = {finding.id: finding for finding in final.findings}
    if len(final_by_id) != len(final.findings) or "" in final_by_id:
        raise ValueError("final findings must have unique non-empty IDs")
    added_ids = sorted(set(final_by_id) - set(draft_by_id))
    if added_ids:
        raise ValueError("final report cannot add findings: " + ", ".join(added_ids))

    findings: list[Finding] = []
    disposition_trace: list[dict[str, object]] = []
    for draft_finding in draft.findings:
        final_finding = final_by_id.get(draft_finding.id)
        if final_finding is None:
            disposition_trace.append(
                {
                    "stage": "final_disposition",
                    "status": "retracted",
                    "tool": finalization_tool,
                    "source": "original_roi",
                    "finding_id": draft_finding.id,
                    "previous_label": draft_finding.label,
                    "previous_severity": draft_finding.severity.value,
                    "bbox_count": len(draft_finding.bboxes),
                    "geometry_locked": True,
                }
            )
            continue
        if final_finding.regions != draft_finding.regions:
            raise ValueError(
                f"final report cannot change regions for {draft_finding.id}"
            )
        bbox_drift = _bbox_sequence_max_drift(
            draft_finding.bboxes,
            final_finding.bboxes,
        )
        if bbox_drift is None or bbox_drift > _FINAL_BBOX_COORDINATE_TOLERANCE:
            raise ValueError(
                f"final report cannot change bboxes for {draft_finding.id}"
            )

        reconciled_finding = dataclasses.replace(
            final_finding,
            id=draft_finding.id,
            regions=list(draft_finding.regions),
            bboxes=list(draft_finding.bboxes),
            notes=list(draft_finding.notes),
            source=draft_finding.source,
            evidence=list(draft_finding.evidence),
            evidence_ids=list(draft_finding.evidence_ids),
            observation_ids=list(draft_finding.observation_ids),
        )
        revised_fields = [
            name
            for name in ("label", "detail", "severity", "confidence", "question", "claim_type")
            if getattr(reconciled_finding, name) != getattr(draft_finding, name)
        ]
        if revised_fields:
            reconciled_finding = dataclasses.replace(
                reconciled_finding, evidence=[], evidence_ids=[], observation_ids=[],
            )
        findings.append(reconciled_finding)
        disposition_trace.append(
            {
                "stage": "final_disposition",
                "status": "revised" if revised_fields else "retained",
                "tool": finalization_tool,
                "source": "original_roi",
                "finding_id": draft_finding.id,
                "previous_label": draft_finding.label,
                "final_label": reconciled_finding.label,
                "previous_severity": draft_finding.severity.value,
                "final_severity": reconciled_finding.severity.value,
                "revised_fields": revised_fields,
                "bbox_count": len(draft_finding.bboxes),
                "max_bbox_coordinate_drift": round(bbox_drift, 8),
                "coordinate_tolerance": _FINAL_BBOX_COORDINATE_TOLERANCE,
                "geometry_locked": True,
            }
        )

    severity_floor = max(
        (
            severity
            for severity in (
                *(finding.severity for finding in findings),
                *(item.status for item in final.checklist.values()),
            )
            if _SEVERITY_RANK[severity] >= _SEVERITY_RANK[Severity.WARNING]
        ),
        key=lambda item: _SEVERITY_RANK[item],
        default=Severity.NORMAL,
    )
    severity = max(
        final.severity,
        severity_floor,
        key=lambda item: _SEVERITY_RANK[item],
    )
    required_axes = (
        default_checklist_keys(draft.modality)
        if required_checklist_keys is None else required_checklist_keys
    )
    pending_resolved = (
        PENDING_MULTIPASS_REASON in draft.incomplete_reasons
        and bool(required_axes)
        and set(required_axes) <= final.checklist.keys()
        and PENDING_MULTIPASS_REASON not in final.incomplete_reasons
    )
    draft_reasons = [
        reason for reason in draft.incomplete_reasons
        if not (pending_resolved and reason == PENDING_MULTIPASS_REASON)
    ]
    draft_still_incomplete = draft.incomplete and (
        not pending_resolved or bool(draft_reasons)
    )
    incomplete_reasons = unique(
        draft_reasons,
        final.incomplete_reasons,
    )
    validation_warnings = unique(
        draft.validation_warnings,
        final.validation_warnings,
    )
    incomplete = (
        draft_still_incomplete or final.incomplete
        or bool(validation_warnings) or bool(incomplete_reasons)
    )
    host_review_required = bool(draft.input_provenance) and draft.review_required
    review_reasons = unique(
        draft.review_reasons if draft_still_incomplete or host_review_required else [],
        final.review_reasons,
    )
    if pending_resolved:
        disposition_trace.append({
            "stage": "workflow_progress",
            "status": "resolved",
            "reason": PENDING_MULTIPASS_REASON,
            "evidence": "accepted_final_report_with_all_required_checklist_axes",
            "clinical_limitations_removed": False,
        })

    return dataclasses.replace(
        final,
        schema_version=draft.schema_version,
        protocol_version=draft.protocol_version,
        assessment_scope=draft.assessment_scope,
        result_status=draft.result_status,
        input_provenance=draft.input_provenance,
        study_manifest=draft.study_manifest,
        observations=list(draft.observations),
        evidence=list(draft.evidence),
        workflow_events=list(draft.workflow_events),
        summary_observation_ids=(list(draft.summary_observation_ids)
            if not final.summary.strip() or final.summary.strip() == draft.summary else []),
        modality=draft.modality,
        summary=final.summary.strip() or draft.summary,
        severity=severity,
        findings=findings,
        checklist=final.checklist or draft.checklist,
        analysis_time_ms=draft.analysis_time_ms + final.analysis_time_ms,
        model_used=final.model_used or draft.model_used,
        image_quality=final.image_quality or draft.image_quality,
        next_steps=final.next_steps or draft.next_steps,
        incomplete=incomplete,
        incomplete_reasons=incomplete_reasons,
        validation_warnings=validation_warnings,
        zoom_hints=unique(draft.zoom_hints, final.zoom_hints),
        review_required=final.review_required or incomplete or host_review_required,
        review_reasons=review_reasons,
        layout=dict(draft.layout),
        analysis_trace=[
            *draft.analysis_trace,
            *final.analysis_trace,
            *disposition_trace,
        ],
    )


def complete_unassessed_checklist_fallback(
    result: AnalysisResult,
    *,
    reason: str,
    required_checklist_keys: frozenset[str] | None = None,
) -> AnalysisResult:
    """Keep a failed bounded final turn structurally honest and reviewable."""

    required = (
            default_checklist_keys(result.modality)
            if required_checklist_keys is None else required_checklist_keys
        )
    missing = sorted(required - set(result.checklist))
    if not missing:
        return result
    checklist = dict(result.checklist)
    for key in missing:
        checklist[key] = ChecklistItem(
            value="Not assessed because final reconciliation did not complete; review required.",
            status=Severity.INFO,
        )
    fallback_reason = (
        f"{reason} {len(missing)} checklist item(s) remain explicitly unassessed."
    )
    return dataclasses.replace(
        result,
        checklist=checklist,
        incomplete=True,
        incomplete_reasons=list(
            dict.fromkeys([*result.incomplete_reasons, fallback_reason])
        ),
        review_required=True,
        review_reasons=list(dict.fromkeys([*result.review_reasons, fallback_reason])),
        analysis_trace=[
            *result.analysis_trace,
            {
                "stage": "checklist_fallback",
                "status": "filled_unassessed_entries",
                "tool": "bounded_finalization_fallback",
                "missing_keys": missing,
                "clinical_status_inferred": False,
                "diagnosis_forced": False,
            },
        ],
    )


def _critical_ekg_evidence_axes(findings: list[Finding]) -> frozenset[str]:
    text = _critical_candidate_text(findings)
    axes: set[str] = set()
    if any(term in text for term in _CRITICAL_TEMPORAL_SUPPORT_TERMS):
        axes.update(
            {
                "heart_rate",
                "rhythm",
                "regularity",
                "p_wave",
                "pr_interval",
                "qrs_duration",
                "qrs_morphology",
                "qtc_interval",
                "av_block",
            }
        )
    if any(term in text for term in _CRITICAL_TERRITORIAL_SUPPORT_TERMS):
        axes.update(
            {
                "qrs_morphology",
                "st_segment",
                "t_wave",
                "stemi_pattern",
                "ischemia",
            }
        )
    if any(term in text for term in _CRITICAL_MORPHOLOGY_SUPPORT_TERMS):
        axes.update(
            {
                "qrs_duration",
                "qrs_morphology",
                "st_segment",
                "t_wave",
                "qtc_interval",
                "conduction",
            }
        )
    if any(
        term in text
        for term in (
            "av block",
            "heart block",
            "bundle branch",
            "conduction",
            "wide qrs",
            "wide complex",
        )
    ):
        axes.update(
            {
                "rhythm",
                "pr_interval",
                "qrs_duration",
                "qrs_morphology",
                "conduction",
                "av_block",
            }
        )
    return frozenset(axes)


def _critical_triage_axis_is_unresolved(item: ChecklistItem | None) -> bool:
    if item is None:
        return True
    normalized = item.value.strip().casefold().replace("_", " ")
    return not normalized or any(
        marker in normalized for marker in _CRITICAL_TRIAGE_UNRESOLVED_VALUE_MARKERS
    )


def _normalized_clinical_claim_text(value: str) -> str:
    return " ".join(
        "".join(character if character.isalnum() else " " for character in value)
        .casefold()
        .split()
    )


def _contains_broad_normal_or_absent_summary_claim(value: str) -> bool:
    """Detect an overall negative conclusion unsafe with deferred ECG axes."""

    normalized = _normalized_clinical_claim_text(value)
    if normalized in {"normal", "wnl", "absent", "unremarkable"}:
        return True
    padded = f" {normalized} "
    return any(
        f" {marker} " in padded
        for marker in _CRITICAL_TRIAGE_BROAD_NEGATIVE_SUMMARY_MARKERS
    )


def _contains_unassessed_axis_normal_claim(value: str) -> bool:
    """Detect a normal/absent assertion embedded in an unassessed axis value."""

    normalized = _normalized_clinical_claim_text(value)
    padded = f" {normalized} "
    return (
        " within normal limits " in padded
        or " wnl " in padded
        or " normal " in padded
        or " absent " in padded
        or " no abnormality " in padded
        or " no abnormalities " in padded
    )


def _contains_asserted_ventricular_run_claim(value: str) -> bool:
    """Return whether text makes an unqualified ventricular-run assertion."""

    # Negation is scoped to a mention, not the whole report. In particular,
    # "intervening beats exclude a consecutive ventricular run" is NOT an
    # assertion to relabel as a time-critical candidate (real desktop failure).
    for clause in re.split(r"[.;\n]+|\b(?:but|however|whereas)\b", value, flags=re.I):
        padded = f" {_normalized_clinical_claim_text(clause)} "
        if any(marker in padded for marker in _EKG_NONASSERTIVE_CLAIM_MARKERS):
            continue
        for marker in _EKG_VENTRICULAR_RUN_CLAIM_MARKERS:
            for mention in re.finditer(rf"\b{re.escape(marker)}\b", padded):
                prefix = padded[: mention.start()]
                suffix = padded[mention.end() :]
                negated_before = re.search(
                    r"\b(?:no|without|not|exclud(?:e|es|ed|ing))\s+"
                    r"(?:(?:a|an|the|any|clear|definite|evidence|of|consecutive|"
                    r"sustained|nonsustained)\s+)*$",
                    prefix,
                )
                negated_after = re.match(
                    r"\s+(?:(?:is|was|are|were)\s+)?"
                    r"(?:absent|excluded|not (?:present|seen|observed|demonstrated))\b",
                    suffix,
                )
                if not negated_before and not negated_after:
                    return True
    return False


def _unique_ekg_event_timestamp_count(boxes: list[RegionRect]) -> int:
    """Count clearly distinct horizontal EKG event intervals conservatively.

    Stacked 12-lead rows repeat one simultaneous cardiac event vertically.  A
    bbox contributes at most one timestamp, and horizontally overlapping boxes
    are one event regardless of how many lead rows repeat it. Broad context
    rectangles are excluded because they do not localize an event in time.
    """

    intervals = sorted(
        (
            (region.x, region.x + region.w)
            for raw_box in boxes
            if (region := _clamp_region(raw_box)) is not None
            and region.w <= _MAX_EKG_FINDING_BOX_WIDTH
            and region.h <= _MAX_EKG_FINDING_BOX_HEIGHT
            and region.w * region.h <= _MAX_EKG_FINDING_BOX_AREA
        ),
        key=lambda interval: (interval[0], interval[1]),
    )
    groups: list[tuple[float, float]] = []
    horizontal_tolerance = 0.005
    for start, end in intervals:
        if groups and start <= groups[-1][1] + horizontal_tolerance:
            previous_start, previous_end = groups[-1]
            groups[-1] = (previous_start, max(previous_end, end))
        else:
            groups.append((start, end))
    return len(groups)


def apply_ekg_ventricular_run_evidence_guard(
    result: AnalysisResult,
) -> AnalysisResult:
    """Qualify an asserted VT/run label when geometry shows fewer than 3 events.

    This is deliberately not a rhythm classifier. It neither declares artifact
    nor lowers study urgency. It only prevents one or two synchronized boxes
    (or a broad lead/context box) from being displayed as proof of a ventricular
    run, because the same timestamp repeated across leads is still one beat.
    """

    if result.modality is not Modality.EKG:
        return result
    layout_format = str(result.layout.get("format") or "").strip().casefold()
    if layout_format not in {"12lead_12x1", "12lead_rows"}:
        return result

    findings: list[Finding] = []
    events: list[dict[str, object]] = []
    for finding in result.findings:
        # Only constrain an explicit finding label. Free-text differential
        # prose is not a reliable positive classifier ("no pacing spike or
        # ventricular run" must not rewrite "Possible isolated ectopy").
        if not _contains_asserted_ventricular_run_claim(finding.label):
            findings.append(finding)
            continue
        unique_timestamps = _unique_ekg_event_timestamp_count(finding.bboxes)
        if unique_timestamps >= _EKG_MIN_VENTRICULAR_RUN_TIMESTAMPS:
            findings.append(finding)
            continue

        qualification = (
            f"Only {unique_timestamps} unique horizontal event timestamp(s) are "
            "localized; fewer than three cannot establish a ventricular run or "
            "ventricular tachycardia from bbox geometry alone."
        )
        original_claim = (
            "Original model claim retained for audit but not accepted as a "
            f"diagnosis: {finding.label} — {finding.detail}"
        )
        findings.append(
            dataclasses.replace(
                finding,
                label="Unresolved potentially time-critical ventricular rhythm candidate",
                detail=qualification,
                confidence="low",
                question=(
                    "Does the native ECG show at least three consecutive, unique "
                    "wide-complex event timestamps, and what is their mechanism?"
                ),
                notes=list(dict.fromkeys([*finding.notes, original_claim])),
            )
        )
        events.append(
            {
                "stage": "ekg_ventricular_run_guardrail",
                "status": "assertion_qualified_for_insufficient_unique_timestamps",
                "tool": "deterministic_bbox_timestamp_counter",
                "finding_id": finding.id,
                "unique_timestamp_count": unique_timestamps,
                "required_timestamp_count": _EKG_MIN_VENTRICULAR_RUN_TIMESTAMPS,
                "study_severity_preserved": result.severity.value,
                "finding_severity_preserved": finding.severity.value,
                "coordinates_moved": False,
                "diagnosis_forced": False,
            }
        )
    if not events:
        return result

    summary = result.summary
    if _contains_asserted_ventricular_run_claim(summary):
        summary = (
            "A potentially time-critical ventricular rhythm candidate remains "
            "unresolved because fewer than three unique event timestamps are "
            "localized; review the native ECG."
        )
    checklist = dict(result.checklist)
    for key, item in result.checklist.items():
        if _contains_asserted_ventricular_run_claim(item.value):
            checklist[key] = ChecklistItem(
                value=(
                    "unresolved ventricular rhythm candidate; fewer than three "
                    "unique event timestamps localized"
                ),
                status=item.status,
            )
    reason = (
        "An asserted ventricular run/VT had fewer than three uniquely localized "
        "horizontal event timestamps; synchronized lead repetitions were counted "
        "once and broad context boxes were not treated as timed events."
    )
    return dataclasses.replace(
        result,
        summary=summary,
        findings=findings,
        checklist=checklist,
        incomplete=True,
        incomplete_reasons=list(dict.fromkeys([*result.incomplete_reasons, reason])),
        review_required=True,
        review_reasons=list(dict.fromkeys([*result.review_reasons, reason])),
        analysis_trace=[*result.analysis_trace, *events],
        next_steps=list(
            dict.fromkeys(
                [
                    *result.next_steps,
                    "Confirm unique wide-complex event count and mechanism on the native ECG.",
                ]
            )
        ),
    )


def apply_critical_triage_guard(
    result: AnalysisResult,
    critical_findings: list[Finding],
    *,
    phase: str,
    required_checklist_keys: frozenset[str] | None = None,
) -> AnalysisResult:
    """Keep deferred checklist conclusions explicit after critical-first triage."""

    checklist = dict(result.checklist)
    deferred_axes: list[str] = []
    changed_axes: list[str] = []
    protected_axes: frozenset[str] = frozenset()
    if result.modality is Modality.EKG:
        required = (
            default_checklist_keys(result.modality)
            if required_checklist_keys is None else required_checklist_keys
        )
        protected_axes = _critical_ekg_evidence_axes(critical_findings)
        deferred_axes = sorted(required - protected_axes)
        for key in deferred_axes:
            item = checklist.get(key)
            if phase == "final_output" and item is not None:
                # The bounded final turn sees the original ROI and is the one
                # permitted place to resume axes deferred by critical-first
                # routing.  Preserve any explicit final assessment, including
                # a normal one; an existing sentinel remains unresolved.
                continue
            if item is not None and item.status is not Severity.NORMAL:
                continue
            checklist[key] = ChecklistItem(
                value=_CRITICAL_TRIAGE_UNASSESSED,
                status=Severity.INFO,
            )
            changed_axes.append(key)

    analysis_trace = list(result.analysis_trace)
    summary = result.summary
    if changed_axes:
        analysis_trace.append(
            {
                "stage": "critical_triage_checklist_guard",
                "status": "deferred_normal_entries_marked_unassessed",
                "tool": "critical_first_checklist_guard",
                "phase": phase,
                "critical_finding_ids": [finding.id for finding in critical_findings],
                "protected_axes": sorted(protected_axes),
                "deferred_checklist_axes": deferred_axes,
                "changed_axes": changed_axes,
                "clinical_status_inferred": False,
                "diagnosis_forced": False,
            }
        )
    incomplete_reasons = list(result.incomplete_reasons)
    review_reasons = list(result.review_reasons)
    incomplete = result.incomplete
    review_required = result.review_required
    deferred_review_resolved = False
    if phase == "final_output" and deferred_axes:
        unresolved_axes = [
            key
            for key in deferred_axes
            if _critical_triage_axis_is_unresolved(checklist.get(key))
        ]
        resumed_axes = sorted(set(deferred_axes) - set(unresolved_axes))
        retained_critical_ids = sorted(
            finding.id
            for finding in result.findings
            if finding.id in {item.id for item in critical_findings}
            and finding.severity is Severity.CRITICAL
        )
        cross_field_repairs: list[dict[str, str]] = []
        if unresolved_axes:
            if _contains_broad_normal_or_absent_summary_claim(summary):
                cross_field_repairs.append(
                    {
                        "field": "summary",
                        "reason": "broad_negative_claim_with_unassessed_axes",
                    }
                )
                summary = (
                    "At least one critical finding remains, but deferred ECG checklist "
                    "axes remain unassessed on the original study."
                    if retained_critical_ids
                    else "The critical candidate was not retained; deferred ECG "
                    "checklist axes remain unassessed on the original study."
                )
            for key in unresolved_axes:
                item = checklist.get(key)
                if item is None or not _contains_unassessed_axis_normal_claim(
                    item.value
                ):
                    continue
                checklist[key] = ChecklistItem(
                    value=_CRITICAL_TRIAGE_UNASSESSED,
                    status=Severity.INFO,
                )
                cross_field_repairs.append(
                    {
                        "field": f"checklist.{key}",
                        "reason": "normal_or_absent_claim_inside_unassessed_axis",
                    }
                )
        if unresolved_axes:
            # Clinical severity is never raised solely because a workflow axis
            # remains unassessed.  The explicit sentinel plus incomplete/review
            # flags carry that operational safety state without inventing an
            # abnormal image finding.
            incomplete_reasons = list(
                dict.fromkeys([*incomplete_reasons, _CRITICAL_TRIAGE_UNRESOLVED_REASON])
            )
            review_reasons = list(
                dict.fromkeys([*review_reasons, _CRITICAL_TRIAGE_UNRESOLVED_REASON])
            )
            incomplete = True
            review_required = True
        else:
            # The original-ROI final turn is the only stage allowed to close
            # axes skipped by critical-first crop planning.  Once it explicitly
            # assessed every deferred axis, remove the workflow-only limitation
            # instead of leaving a permanently incomplete otherwise-clean case.
            triage_reasons = {
                _CRITICAL_TRIAGE_REASON,
                _CRITICAL_TRIAGE_UNRESOLVED_REASON,
            }
            had_triage_incomplete_reason = any(
                reason in triage_reasons for reason in incomplete_reasons
            )
            had_triage_review_reason = any(
                reason in triage_reasons for reason in review_reasons
            )
            incomplete_reasons = [
                reason for reason in incomplete_reasons if reason not in triage_reasons
            ]
            review_reasons = [
                reason for reason in review_reasons if reason not in triage_reasons
            ]
            incomplete = bool(incomplete_reasons or result.validation_warnings) or (
                result.incomplete and not had_triage_incomplete_reason
            )
            review_required = (
                bool(review_reasons)
                or incomplete
                or (result.review_required and not had_triage_review_reason)
            )
            deferred_review_resolved = True
        if cross_field_repairs:
            analysis_trace.append(
                {
                    "stage": "critical_triage_cross_field_guard",
                    "status": "contradictory_negative_claims_suppressed",
                    "tool": "critical_first_cross_field_contract",
                    "repairs": cross_field_repairs,
                    "unresolved_axes": unresolved_axes,
                    "retained_critical_ids": retained_critical_ids,
                    "clinical_severity_changed": False,
                    "clinical_status_inferred": False,
                    "diagnosis_forced": False,
                }
            )
        analysis_trace.append(
            {
                "stage": "critical_triage_resume_guard",
                "status": (
                    "deferred_axes_incomplete_fail_safe"
                    if unresolved_axes
                    else "deferred_axes_resumed_on_original_study"
                ),
                "tool": "critical_first_deferred_axis_guard",
                "critical_candidate_ids": [finding.id for finding in critical_findings],
                "retained_critical_ids": retained_critical_ids,
                "resumed_axes": resumed_axes,
                "unresolved_axes": unresolved_axes,
                "clinical_severity_changed": False,
                "clinical_status_inferred": False,
                "diagnosis_forced": False,
            }
        )
    if not deferred_review_resolved:
        incomplete_reasons = list(
            dict.fromkeys([*incomplete_reasons, _CRITICAL_TRIAGE_REASON])
        )
        review_reasons = list(dict.fromkeys([*review_reasons, _CRITICAL_TRIAGE_REASON]))
        incomplete = True
        review_required = True
    return dataclasses.replace(
        result,
        summary=summary,
        checklist=checklist,
        incomplete=incomplete,
        incomplete_reasons=incomplete_reasons,
        review_required=review_required,
        review_reasons=review_reasons,
        analysis_trace=analysis_trace,
    )


def default_checklist_keys(modality: Modality) -> frozenset[str]:
    """Return public scientific checklist axes unless a host injects its policy."""
    return get_active_registry().resolve(modality.value).checklist_keys


class MultiPassInterpreter:
    """Coarse -> crop -> hypothesis-aware refinement orchestrator."""

    def __init__(
        self,
        analyzer: AnalyzerPort,
        cropper: ImageCropper,
        *,
        bbox_calibrator: BboxCalibrator | None = None,
        ekg_row_strip_detector: EkgRowStripDetector | None = None,
        max_zoom_targets: int = 3,
        zoom_padding: float = 0.15,
        min_zoom_source_edge_px: int = DEFAULT_MIN_ZOOM_SOURCE_EDGE_PX,
        zoom_retry_attempts: int = 1,
        max_normal_safety_probes: int = DEFAULT_MAX_NORMAL_SAFETY_PROBES,
        max_ekg_systematic_probes: int = DEFAULT_MAX_EKG_SYSTEMATIC_PROBES,
        max_local_candidate_area: float = DEFAULT_MAX_LOCAL_CANDIDATE_AREA,
        initial_response_sla_sec: float = DEFAULT_INITIAL_RESPONSE_SLA_SEC,
        first_refinement_sla_sec: float = DEFAULT_FIRST_REFINEMENT_SLA_SEC,
        total_analysis_sla_sec: float = DEFAULT_TOTAL_ANALYSIS_SLA_SEC,
        max_refinement_turn_sec: float = DEFAULT_MAX_REFINEMENT_TURN_SEC,
        max_finalization_turn_sec: float = DEFAULT_MAX_FINALIZATION_TURN_SEC,
        finalization_reserve_sec: float = DEFAULT_FINALIZATION_RESERVE_SEC,
        min_followup_budget_sec: float = DEFAULT_MIN_FOLLOWUP_BUDGET_SEC,
        clock: Callable[[], float] = time.monotonic,
        checklist_keys_for: Callable[[Modality], frozenset[str]] = default_checklist_keys,
        stage_tools: StageTools | None = None,
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
        if not 0.0 < initial_response_sla_sec < first_refinement_sla_sec:
            raise ValueError(
                "initial_response_sla_sec must be > 0 and below "
                "first_refinement_sla_sec"
            )
        if not first_refinement_sla_sec < total_analysis_sla_sec:
            raise ValueError(
                "first_refinement_sla_sec must be below total_analysis_sla_sec"
            )
        if max_refinement_turn_sec <= 0.0:
            raise ValueError("max_refinement_turn_sec must be > 0")
        if max_finalization_turn_sec <= 0.0:
            raise ValueError("max_finalization_turn_sec must be > 0")
        if not 0.0 <= finalization_reserve_sec < total_analysis_sla_sec:
            raise ValueError(
                "finalization_reserve_sec must be >= 0 and below the total SLA"
            )
        if min_followup_budget_sec <= 0.0:
            raise ValueError("min_followup_budget_sec must be > 0")
        self._analyzer = analyzer
        self._cropper = cropper
        self._bbox_calibrator = bbox_calibrator
        self._ekg_row_strip_detector = ekg_row_strip_detector
        self._max_zoom_targets = max_zoom_targets
        self._zoom_padding = zoom_padding
        self._min_zoom_source_edge_px = min_zoom_source_edge_px
        self._zoom_retry_attempts = zoom_retry_attempts
        self._max_normal_safety_probes = max_normal_safety_probes
        self._max_ekg_systematic_probes = max_ekg_systematic_probes
        self._max_local_candidate_area = max_local_candidate_area
        self._initial_response_sla_sec = float(initial_response_sla_sec)
        self._first_refinement_sla_sec = float(first_refinement_sla_sec)
        self._total_analysis_sla_sec = float(total_analysis_sla_sec)
        self._max_refinement_turn_sec = float(max_refinement_turn_sec)
        self._max_finalization_turn_sec = float(max_finalization_turn_sec)
        self._finalization_reserve_sec = float(finalization_reserve_sec)
        self._min_followup_budget_sec = float(min_followup_budget_sec)
        self._clock = clock
        self._checklist_keys_for = checklist_keys_for
        self._stage_tools = stage_tools if stage_tools is not None else StageTools()

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
        deadline = _AnalysisDeadline(self._clock(), self._clock)
        try:
            coarse_analyze = getattr(self._analyzer, "analyze_coarse", None)
            analyze_method = (
                coarse_analyze if callable(coarse_analyze) else self._analyzer.analyze
            )
            coarse = await self._await_until(
                lambda: analyze_method(
                    image_base64,
                    modality,
                    valid_regions,
                ),
                deadline=deadline,
                absolute_deadline_sec=self._initial_response_sla_sec,
            )
        except TimeoutError as exc:
            raise AnalysisSlaTimeout(
                "initial_response",
                budget_sec=self._initial_response_sla_sec,
                elapsed_ms=deadline.elapsed_ms(),
            ) from exc
        initial_response_ms = deadline.elapsed_ms()
        if coarse.modality is Modality.EKG:
            image_layout_evidence: dict[str, object] | None = None
            layout_detector = self._ekg_row_strip_detector
            if layout_detector is None:
                layout_detector_owner = getattr(
                    self._cropper,
                    "__self__",
                    self._cropper,
                )
                layout_detector = getattr(
                    layout_detector_owner,
                    "ekg_row_strip_evidence",
                    None,
                )
            if callable(layout_detector):
                try:
                    candidate_evidence = layout_detector(
                        source_image_base64 or image_base64
                    )
                    if isinstance(candidate_evidence, dict):
                        image_layout_evidence = candidate_evidence
                except Exception:
                    logger.warning(
                        "Local EKG row-strip detection failed; keeping model layout"
                    )
            normalized_layout, layout_repaired = normalize_ekg_row_strip_layout(
                coarse.layout,
                image_evidence=image_layout_evidence,
            )
            layout_trace = list(coarse.analysis_trace)
            if image_layout_evidence is not None:
                layout_trace.append(
                    {
                        "stage": "layout_signal_check",
                        "status": (
                            "confirmed_12_row_strip"
                            if image_layout_evidence.get("is_12_row_strip") is True
                            else "not_confirmed"
                        ),
                        "tool": str(
                            image_layout_evidence.get("method")
                            or "local_ekg_row_strip_detector"
                        ),
                        "evidence": image_layout_evidence,
                    }
                )
            if layout_repaired:
                coarse = dataclasses.replace(
                    coarse,
                    layout=normalized_layout,
                    validation_warnings=[
                        warning
                        for warning in coarse.validation_warnings
                        if not warning.startswith("EKG layout ")
                    ],
                    analysis_trace=[
                        *layout_trace,
                        {
                            "stage": "layout_normalization",
                            "status": "repaired_before_refinement",
                            "tool": "local_ekg_row_strip_normalizer",
                            "format": "12lead_12x1",
                            "lead_count": 12,
                        },
                    ],
                )
            elif image_layout_evidence is not None:
                coarse = dataclasses.replace(coarse, analysis_trace=layout_trace)
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
                "tool": self._stage_tools.coarse,
                "severity": coarse.severity.value,
                "finding_count": len(coarse.findings),
                "completed_ms": initial_response_ms,
                "absolute_deadline_ms": int(self._initial_response_sla_sec * 1000),
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
                        _region_payload(region) for region in local_candidate_regions
                    ],
                }
            )
        trace.extend(calibration_trace)
        coarse.analysis_trace = trace

        all_model_findings = select_zoom_targets(
            coarse,
            max_targets=max(1, len(coarse.findings)),
        )
        critical_triage_candidates = select_critical_triage_candidates(coarse)
        critical_triage_active = bool(
            critical_triage_candidates and self._max_zoom_targets > 0
        )
        model_findings = (
            critical_triage_candidates
            if critical_triage_active
            else all_model_findings[: self._max_zoom_targets]
        )
        waveform_attention_candidates = select_ekg_waveform_attention_probe_regions(
            coarse
        )
        generic_systematic_candidates = select_ekg_systematic_probe_regions(
            coarse,
            max_probes=max(
                self._max_ekg_systematic_probes,
                len(_EKG_SYSTEMATIC_LEAD_GROUPS),
            ),
        )
        systematic_candidates: list[tuple[str, RegionRect]] = []
        seen_systematic_regions: set[RegionRect] = set()
        for candidate in [
            *waveform_attention_candidates,
            *generic_systematic_candidates,
        ]:
            key, region = candidate
            if region in seen_systematic_regions:
                continue
            seen_systematic_regions.add(region)
            systematic_candidates.append((key, region))
        systematic_budget = 0
        if (
            not critical_triage_active
            and systematic_candidates
            and self._max_zoom_targets > 0
        ):
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
        targets: list[_RefinementTarget] = []
        for finding in model_findings[:specific_budget]:
            covering = covering_region(finding.bboxes)
            contextual_crop = (
                _ekg_contextual_crop_strategy(finding, coarse.layout)
                if coarse.modality is Modality.EKG
                else None
            )
            crop_region = select_hypothesis_crop_region(
                finding,
                modality=coarse.modality,
                layout=coarse.layout,
            )
            targets.append(
                _RefinementTarget(
                    crop_region=crop_region,
                    hypothesis=finding,
                    key=finding.id,
                )
            )
            if contextual_crop is not None:
                trace.append(
                    {
                        "stage": "crop_planning",
                        "status": "contextual_evidence_crop",
                        "tool": "ekg_context_crop_router",
                        "target_id": finding.id,
                        "strategy": contextual_crop[1],
                        "selected_region": _region_payload(crop_region),
                        "source_bbox_count": len(finding.bboxes),
                        "declared_regions": list(finding.regions),
                    }
                )
            elif crop_region != covering:
                trace.append(
                    {
                        "stage": "crop_planning",
                        "status": "reduced_disjoint_evidence_crop",
                        "tool": "local_crop_planner",
                        "target_id": finding.id,
                        "covering_region": _region_payload(covering),
                        "selected_region": _region_payload(crop_region),
                        "source_bbox_count": len(finding.bboxes),
                    }
                )
        remaining = specific_budget - len(targets)
        if not critical_triage_active and remaining > 0 and local_candidate_regions:
            local_targets = select_local_candidate_targets(
                coarse,
                local_candidate_regions,
                max_targets=remaining,
                max_candidate_area=self._max_local_candidate_area,
            )
            targets.extend(
                _RefinementTarget(
                    crop_region=select_hypothesis_crop_region(
                        finding,
                        modality=coarse.modality,
                        layout=coarse.layout,
                    ),
                    hypothesis=finding,
                    key=finding.id,
                )
                for finding in local_targets
            )
        if (
            not critical_triage_active
            and not targets
            and remaining > 0
            and local_candidate_regions
        ):
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
            targeted_ids = {
                target.hypothesis.id
                for target in targets
                if target.hypothesis is not None
            }
            untargeted_hypotheses = [
                finding for finding in model_findings if finding.id not in targeted_ids
            ]
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
            systematic_overlap_fallback = False
            for key, region in ranked_systematic:
                if any(
                    _overlap_fraction(region, target.crop_region) >= 0.85
                    for target in targets
                ):
                    continue
                paired_hypothesis = next(
                    (
                        finding
                        for finding in untargeted_hypotheses
                        if _overlap_fraction(
                            select_hypothesis_crop_region(
                                finding,
                                modality=coarse.modality,
                                layout=coarse.layout,
                            ),
                            region,
                        )
                        >= 0.6
                    ),
                    None,
                )
                if paired_hypothesis is not None:
                    untargeted_hypotheses.remove(paired_hypothesis)
                systematic_targets.append(
                    _RefinementTarget(
                        crop_region=region,
                        hypothesis=paired_hypothesis,
                        key=f"ekg_systematic_{key}",
                    )
                )
                if len(systematic_targets) >= systematic_budget:
                    break
            if not systematic_targets and ranked_systematic:
                key, region = ranked_systematic[0]
                paired_hypothesis = next(iter(untargeted_hypotheses), None)
                systematic_targets.append(
                    _RefinementTarget(
                        crop_region=region,
                        hypothesis=paired_hypothesis,
                        key=f"ekg_systematic_{key}",
                    )
                )
                systematic_overlap_fallback = True
            targets.extend(systematic_targets)
            if systematic_targets:
                trace.append(
                    {
                        "stage": "systematic_assist",
                        "status": (
                            "planned_overlap_fallback"
                            if systematic_overlap_fallback
                            else "planned"
                        ),
                        "tool": "ekg_layout_lead_group_probes",
                        "probes": [
                            {
                                "target_id": target.key,
                                "crop_region": _region_payload(target.crop_region),
                                "hypothesis_id": (
                                    target.hypothesis.id
                                    if target.hypothesis is not None
                                    else ""
                                ),
                            }
                            for target in systematic_targets
                        ],
                    }
                )
        if critical_triage_active:
            selected_critical_ids = [
                target.hypothesis.id
                for target in targets
                if target.hypothesis is not None
            ]
            support_probe_id = ""
            support_reason = ""
            if (
                coarse.modality is Modality.EKG
                and self._max_ekg_systematic_probes > 0
                and len(targets) < self._max_zoom_targets
            ):
                support = select_ekg_critical_support_probe(
                    coarse,
                    critical_triage_candidates,
                    [target.crop_region for target in targets],
                    systematic_candidates=generic_systematic_candidates,
                )
                if support is not None:
                    key, region, support_reason = support
                    support_target = _RefinementTarget(
                        crop_region=region,
                        hypothesis=None,
                        key=f"ekg_systematic_critical_support_{key}",
                    )
                    targets.append(support_target)
                    support_probe_id = support_target.key
                    trace.append(
                        {
                            "stage": "systematic_assist",
                            "status": "planned_critical_support",
                            "tool": "critical_mechanism_support_router",
                            "probes": [
                                {
                                    "target_id": support_target.key,
                                    "crop_region": _region_payload(region),
                                    "hypothesis_id": "",
                                    "reason": support_reason,
                                }
                            ],
                        }
                    )

            selected_id_set = set(selected_critical_ids)
            candidate_id_set = {finding.id for finding in critical_triage_candidates}
            overflow_ids = [
                finding.id
                for finding in critical_triage_candidates
                if finding.id not in selected_id_set
            ]
            skipped_lower_priority = [
                finding
                for finding in all_model_findings
                if finding.id not in candidate_id_set
                and finding.severity is not Severity.CRITICAL
            ]
            skipped_unstructured_critical_ids = [
                finding.id
                for finding in all_model_findings
                if finding.severity is Severity.CRITICAL
                and finding.id not in candidate_id_set
            ]
            deferred_checklist_axes: list[str] = []
            if coarse.modality is Modality.EKG:
                required_axes = self._checklist_keys_for(Modality.EKG)
                deferred_checklist_axes = sorted(
                    required_axes
                    - _critical_ekg_evidence_axes(critical_triage_candidates)
                )
            skipped_categories = {
                severity.value: sum(
                    finding.severity is severity for finding in skipped_lower_priority
                )
                for severity in (Severity.WARNING, Severity.INFO)
                if any(
                    finding.severity is severity for finding in skipped_lower_priority
                )
            }
            trace.append(
                {
                    "stage": "critical_triage",
                    "status": "activated",
                    "tool": "critical_first_refinement_planner",
                    "candidate_ids": [
                        finding.id for finding in critical_triage_candidates
                    ],
                    "selected_critical_ids": selected_critical_ids,
                    "support_probe_id": support_probe_id,
                    "support_reason": support_reason,
                    "skipped_lower_priority_ids": [
                        finding.id for finding in skipped_lower_priority
                    ],
                    "skipped_lower_priority_categories": skipped_categories,
                    "skipped_unstructured_critical_ids": (
                        skipped_unstructured_critical_ids
                    ),
                    "overflow_critical_ids": overflow_ids,
                    "deferred_checklist_axes": deferred_checklist_axes,
                    "max_refinement_turns": self._max_zoom_targets,
                    "planned_turn_count": len(targets),
                    "extra_turns_beyond_configured_budget": 0,
                    "diagnosis_forced": False,
                }
            )
        zoom_hints: list[str] = []
        refinements: list[tuple[_RefinementTarget, RegionRect, RefinementResult]] = []
        first_crop_created_ms: int | None = None
        first_refinement_completed_ms: int | None = None
        observed_refinement_sec = 0.0
        skipped_refinement_count = 0
        degradation_reasons: list[str] = []
        can_finalize = callable(getattr(self._analyzer, "finalize", None))
        refinement_absolute_limit = self._total_analysis_sla_sec - (
            self._finalization_reserve_sec if can_finalize else _SLA_RETURN_BUFFER_SEC
        )
        for target_index, target in enumerate(targets):
            first_refinement_pending = first_refinement_completed_ms is None
            stage_limit = (
                self._first_refinement_sla_sec
                if first_refinement_pending
                else refinement_absolute_limit
            )
            required_turn_sec = max(
                self._min_followup_budget_sec,
                observed_refinement_sec * 1.25,
            )
            if deadline.remaining_sec(stage_limit) < required_turn_sec:
                skipped_refinement_count += len(targets) - target_index
                status = (
                    "first_refinement_deadline_exhausted"
                    if first_refinement_pending
                    else "total_deadline_reserve_reached"
                )
                trace.append(
                    {
                        "stage": "refine",
                        "status": status,
                        "tool": "sla_deadline_controller",
                        "target_id": target.key,
                        "skipped_target_count": len(targets) - target_index,
                        "elapsed_ms": deadline.elapsed_ms(),
                        "minimum_start_budget_sec": round(required_turn_sec, 3),
                        "previous_refinement_sec": round(observed_refinement_sec, 3),
                    }
                )
                degradation_reasons.append(
                    "Crop refinement stopped at the analysis time budget; review "
                    "the unverified regions."
                )
                break
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
            crop_created_ms = deadline.elapsed_ms()
            if first_crop_created_ms is None:
                first_crop_created_ms = crop_created_ms
            crop_lead_regions = (
                project_ekg_lead_regions_to_crop(coarse.layout, crop_region)
                if modality is Modality.EKG
                else None
            )
            turn_limit = (
                stage_limit
                if first_refinement_pending
                else min(
                    stage_limit,
                    deadline.elapsed_sec() + self._max_refinement_turn_sec,
                )
            )
            turn_started_ms = deadline.elapsed_ms()
            turn_budget_ms = max(
                0,
                int(turn_limit * 1000) - turn_started_ms,
            )
            try:
                refinement = await self._run_refinement(
                    crop_b64,
                    modality,
                    valid_regions,
                    target=target,
                    crop_region=crop_region,
                    crop_lead_regions=crop_lead_regions,
                    deadline=deadline,
                    absolute_deadline_sec=turn_limit,
                )
            except TimeoutError:
                skipped_refinement_count += len(targets) - target_index
                trace.append(
                    {
                        "stage": "refine",
                        "status": "deadline_exceeded",
                        "tool": self._stage_tools.refinement,
                        "target_id": target.key,
                        "crop_region": _region_payload(crop_region),
                        "crop_created_ms": crop_created_ms,
                        "elapsed_ms": deadline.elapsed_ms(),
                        "absolute_deadline_sec": round(turn_limit, 3),
                        "turn_budget_ms": turn_budget_ms,
                        **self._read_runtime_trace(),
                    }
                )
                degradation_reasons.append(
                    "A crop refinement exceeded its bounded turn budget; review "
                    "the coarse finding and crop manually."
                )
                break
            if refinement is not None and first_refinement_completed_ms is None:
                first_refinement_completed_ms = deadline.elapsed_ms()
            if refinement is not None and refinement.deltas:
                refinements.append((target, crop_region, refinement))
            refinement_completed_ms = deadline.elapsed_ms()
            if refinement is not None:
                observed_refinement_sec = max(
                    observed_refinement_sec,
                    (refinement_completed_ms - turn_started_ms) / 1000,
                )
            absolute_deadline_ms = int(turn_limit * 1000)
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
                    "crop_created_ms": crop_created_ms,
                    "completed_ms": refinement_completed_ms,
                    "turn_started_ms": turn_started_ms,
                    "turn_budget_ms": turn_budget_ms,
                    "absolute_deadline_ms": absolute_deadline_ms,
                    "scheduler_grace_used_ms": max(
                        0, refinement_completed_ms - absolute_deadline_ms
                    ),
                    "crop_lead_regions": {
                        name: _region_payload(region)
                        for name, region in (crop_lead_regions or {}).items()
                    },
                    "decisions": [
                        {
                            "action": delta.action.value,
                            "target_id": delta.target_id,
                            "rationale": delta.rationale,
                            "finding": (
                                delta.finding.label if delta.finding is not None else ""
                            ),
                        }
                        for delta in (refinement.deltas if refinement else ())
                    ],
                    **self._read_runtime_trace(),
                }
            )

        merged = (
            self._merge(coarse, refinements, zoom_hints)
            if refinements or zoom_hints
            else coarse
        )
        if self._bbox_calibrator is not None and (refinements or zoom_hints):
            try:
                merged = self._bbox_calibrator(
                    source_image_base64 or image_base64,
                    merged,
                )
            except Exception:
                logger.warning("Final bbox calibration failed; keeping merged result")
        merged = reconcile_unavailable_ekg_rhythm_regions(merged)
        merged = apply_ekg_overlay_bbox_guard(merged)
        merged = qualify_boxed_info_findings(merged)
        merged = apply_unlocalized_ekg_grounding_guard(merged)
        merged = deduplicate_ekg_study_level_findings(merged)
        if critical_triage_active:
            merged = apply_critical_triage_guard(
                merged,
                critical_triage_candidates,
                phase="before_finalization",
                required_checklist_keys=self._checklist_keys_for(merged.modality),
            )
        if can_finalize:
            finalization_remaining = deadline.remaining_sec(
                self._total_analysis_sla_sec - _SLA_RETURN_BUFFER_SEC
            )
            if finalization_remaining >= self._min_followup_budget_sec:
                merged = await self._finalize_report(
                    source_image_base64 or image_base64,
                    modality,
                    valid_regions,
                    merged,
                    deadline=deadline,
                    absolute_deadline_sec=min(
                        self._total_analysis_sla_sec - _SLA_RETURN_BUFFER_SEC,
                        deadline.elapsed_sec() + self._max_finalization_turn_sec,
                    ),
                )
            else:
                reason = (
                    "Final narrative reconciliation was skipped to keep the "
                    "completed crop findings inside the total analysis budget."
                )
                degradation_reasons.append(reason)
                merged.analysis_trace.append(
                    {
                        "stage": "finalize",
                        "status": "deadline_reserve_exhausted",
                        "tool": "sla_deadline_controller",
                        "elapsed_ms": deadline.elapsed_ms(),
                    }
                )
                merged = complete_unassessed_checklist_fallback(
                    merged,
                    reason=reason,
                    required_checklist_keys=self._checklist_keys_for(merged.modality),
                )
        if critical_triage_active:
            merged = apply_critical_triage_guard(
                merged,
                critical_triage_candidates,
                phase="final_output",
                required_checklist_keys=self._checklist_keys_for(merged.modality),
            )
        return self._finish_with_sla(
            merged,
            deadline=deadline,
            initial_response_ms=initial_response_ms,
            first_crop_applicable=bool(targets),
            first_crop_created_ms=first_crop_created_ms,
            first_refinement_completed_ms=first_refinement_completed_ms,
            skipped_refinement_count=skipped_refinement_count,
            degradation_reasons=degradation_reasons,
        )

    async def _finalize_report(
        self,
        image_base64: str,
        modality: Modality,
        valid_regions: list[str],
        draft: AnalysisResult,
        *,
        deadline: _AnalysisDeadline,
        absolute_deadline_sec: float,
    ) -> AnalysisResult:
        finalize_method = getattr(self._analyzer, "finalize", None)
        if not callable(finalize_method):
            return draft
        trace = list(draft.analysis_trace)
        refinement_trace = [
            event
            for event in trace
            if (event.get("stage") == "refine" and event.get("status") == "completed")
            or (
                event.get("stage") == "refinement_guardrail"
                and event.get("tool") == "crop_coverage_guard"
                and event.get("decision_applied") is False
            )
        ]
        turn_started_ms = deadline.elapsed_ms()
        turn_budget_ms = max(
            0,
            int(absolute_deadline_sec * 1000) - turn_started_ms,
        )
        try:
            final = await self._await_until(
                lambda: finalize_method(
                    image_base64,
                    modality,
                    valid_regions,
                    draft=draft,
                    refinement_trace=refinement_trace,
                ),
                deadline=deadline,
                absolute_deadline_sec=absolute_deadline_sec,
                completion_grace_sec=_FOLLOWUP_COMPLETION_GRACE_SEC,
            )
            reconciled = reconcile_final_report(
                draft, final, required_checklist_keys=self._checklist_keys_for(modality),
                finalization_tool=self._stage_tools.finalize,
            )
        except Exception as exc:
            logger.warning("Final report reconciliation failed", error=str(exc))
            trace.append(
                {
                    "stage": "finalize",
                    "status": "failed",
                    "tool": self._stage_tools.finalize,
                    "source": "original_roi",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_ms": deadline.elapsed_ms(),
                    "turn_started_ms": turn_started_ms,
                    "turn_budget_ms": turn_budget_ms,
                    "absolute_deadline_ms": int(absolute_deadline_sec * 1000),
                    **self._read_runtime_trace(),
                }
            )
            reason = (
                "Final report reconciliation failed; review the narrative against "
                "the refined findings."
            )
            fallback = dataclasses.replace(
                draft,
                analysis_trace=trace,
                review_required=True,
                review_reasons=list(dict.fromkeys([*draft.review_reasons, reason])),
            )
            return complete_unassessed_checklist_fallback(
                fallback, reason=reason,
                required_checklist_keys=self._checklist_keys_for(modality),
            )

        trace = list(reconciled.analysis_trace)
        completed_ms = deadline.elapsed_ms()
        absolute_deadline_ms = int(absolute_deadline_sec * 1000)
        dispositions = [
            event for event in trace if event.get("stage") == "final_disposition"
        ]
        trace.append(
            {
                "stage": "finalize",
                "status": "completed",
                "tool": self._stage_tools.finalize,
                "source": "original_roi",
                "finding_count": len(reconciled.findings),
                "retained_count": sum(
                    event.get("status") == "retained" for event in dispositions
                ),
                "revised_count": sum(
                    event.get("status") == "revised" for event in dispositions
                ),
                "retracted_count": sum(
                    event.get("status") == "retracted" for event in dispositions
                ),
                "completed_ms": completed_ms,
                "turn_started_ms": turn_started_ms,
                "turn_budget_ms": turn_budget_ms,
                "absolute_deadline_ms": absolute_deadline_ms,
                "scheduler_grace_used_ms": max(0, completed_ms - absolute_deadline_ms),
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

    async def _await_until(
        self,
        operation: Callable[[], Awaitable[_T]],
        *,
        deadline: _AnalysisDeadline,
        absolute_deadline_sec: float,
        completion_grace_sec: float = 0.0,
    ) -> _T:
        remaining = deadline.remaining_sec(absolute_deadline_sec)
        if remaining <= 0.0:
            raise TimeoutError("analysis stage deadline exhausted")
        timeout = _bounded_operation_timeout_sec(remaining, completion_grace_sec)
        return await asyncio.wait_for(operation(), timeout=timeout)

    def _finish_with_sla(
        self,
        result: AnalysisResult,
        *,
        deadline: _AnalysisDeadline,
        initial_response_ms: int,
        first_crop_applicable: bool,
        first_crop_created_ms: int | None,
        first_refinement_completed_ms: int | None,
        skipped_refinement_count: int,
        degradation_reasons: list[str],
    ) -> AnalysisResult:
        result = apply_ekg_waveform_rhythm_conflict_guard(result)
        result = apply_ekg_ventricular_run_evidence_guard(result)
        result = reconcile_unavailable_ekg_rhythm_regions(result)
        result = apply_ekg_overlay_bbox_guard(result)
        result = qualify_boxed_info_findings(result)
        result = apply_unlocalized_ekg_grounding_guard(result)
        total_ms = deadline.elapsed_ms()
        degradation_reasons = list(degradation_reasons)
        if first_crop_applicable and first_refinement_completed_ms is None:
            degradation_reasons.append(
                "The first planned crop did not complete a detail read; review "
                "the marked region manually."
            )
        initial_met = initial_response_ms <= int(self._initial_response_sla_sec * 1000)
        first_refinement_met: bool | None = None
        if first_crop_applicable:
            first_refinement_met = (
                first_refinement_completed_ms is not None
                and first_refinement_completed_ms
                <= int(self._first_refinement_sla_sec * 1000)
            )
        total_met = total_ms <= int(self._total_analysis_sla_sec * 1000)
        if not total_met:
            degradation_reasons.append(
                "The analysis exceeded its total time budget; verify the report "
                "before clinical use."
            )
        trace = [
            event
            for event in result.analysis_trace
            if event.get("stage") != "analysis_sla"
        ]
        status = "completed"
        if not initial_met or first_refinement_met is False or not total_met:
            status = "degraded"
        if degradation_reasons:
            status = "degraded"
        trace.append(
            {
                "stage": "analysis_sla",
                "status": status,
                "budgets_sec": {
                    "initial_response": self._initial_response_sla_sec,
                    "first_crop_refinement": self._first_refinement_sla_sec,
                    "total": self._total_analysis_sla_sec,
                },
                "timings_ms": {
                    "initial_response": initial_response_ms,
                    "first_crop_created": first_crop_created_ms,
                    "first_crop_refinement": first_refinement_completed_ms,
                    "total": total_ms,
                },
                "met": {
                    "initial_response": initial_met,
                    "first_crop_refinement": first_refinement_met,
                    "total": total_met,
                },
                "first_crop_applicable": first_crop_applicable,
                "skipped_refinement_count": skipped_refinement_count,
                "degradation_reasons": list(dict.fromkeys(degradation_reasons)),
            }
        )
        if not degradation_reasons:
            result.analysis_time_ms = total_ms
            result.analysis_trace = trace
            return result
        reasons = list(
            dict.fromkeys([*result.incomplete_reasons, *degradation_reasons])
        )
        review_reasons = list(
            dict.fromkeys([*result.review_reasons, *degradation_reasons])
        )
        return dataclasses.replace(
            result,
            analysis_time_ms=total_ms,
            analysis_trace=trace,
            incomplete=True,
            incomplete_reasons=reasons,
            review_required=True,
            review_reasons=review_reasons,
        )

    async def _run_refinement(
        self,
        crop_base64: str,
        modality: Modality,
        valid_regions: list[str],
        *,
        target: _RefinementTarget,
        crop_region: RegionRect,
        crop_lead_regions: dict[str, RegionRect] | None,
        deadline: _AnalysisDeadline,
        absolute_deadline_sec: float,
    ) -> RefinementResult | None:
        refine_method = getattr(self._analyzer, "refine", None)
        for attempt in range(self._zoom_retry_attempts + 1):
            try:
                if callable(refine_method):
                    refinement_context: dict[str, object] = {}
                    if crop_lead_regions:
                        refinement_context["crop_lead_regions"] = crop_lead_regions
                    result = await self._await_until(
                        lambda context=refinement_context: refine_method(
                            crop_base64,
                            modality,
                            valid_regions,
                            hypothesis=target.hypothesis,
                            crop_region=crop_region,
                            probe_id=_refinement_probe_id(
                                target.key,
                                modality,
                                crop_lead_regions,
                            ),
                            **context,
                        ),
                        deadline=deadline,
                        absolute_deadline_sec=absolute_deadline_sec,
                        completion_grace_sec=_FOLLOWUP_COMPLETION_GRACE_SEC,
                    )
                    if not isinstance(result, RefinementResult):
                        raise TypeError("refine() must return RefinementResult")
                    return result
                zoom = await self._await_until(
                    lambda: self._analyzer.analyze(
                        crop_base64,
                        modality,
                        valid_regions,
                    ),
                    deadline=deadline,
                    absolute_deadline_sec=absolute_deadline_sec,
                    completion_grace_sec=_FOLLOWUP_COMPLETION_GRACE_SEC,
                )
                return _legacy_refinement_result(zoom, target)
            except TimeoutError:
                if (
                    attempt < self._zoom_retry_attempts
                    and deadline.remaining_sec(absolute_deadline_sec) > 0.05
                ):
                    logger.warning(
                        "Zoom refinement timed out; retrying within stage deadline",
                        target=target.key,
                        attempt=attempt + 1,
                        max_retries=self._zoom_retry_attempts,
                    )
                    continue
                raise
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
        trace = list(coarse.analysis_trace)
        allow_downgrade = False
        unlocalized_finding_ids: list[str] = []
        coarse_by_id = {finding.id: finding for finding in coarse.findings}
        for target, crop_region, refinement in refinements:
            expected_target_id = (
                target.hypothesis.id if target.hypothesis is not None else None
            )
            unlocalized_indexes = set(
                refinement.unlocalized_missing_receipt_delta_indexes
            )
            for delta_index, raw_delta in enumerate(refinement.deltas):
                delta = (
                    _normalize_discovery_delta(raw_delta)
                    if expected_target_id is None
                    or target.key.startswith("ekg_systematic_")
                    else raw_delta
                )
                semantic_only_missing_receipt = bool(
                    delta_index in unlocalized_indexes
                    and delta.finding is not None
                    and not delta.finding.bboxes
                )
                unlocalized_add = bool(
                    semantic_only_missing_receipt
                    and delta.action is RefinementAction.ADD
                )
                targeted_action = delta.action in {
                    RefinementAction.CONFIRM,
                    RefinementAction.REVISE,
                    RefinementAction.RETRACT,
                }
                targets_expected_hypothesis = bool(
                    targeted_action
                    and target.hypothesis is not None
                    and expected_target_id is not None
                    and delta.target_id == expected_target_id
                )
                if (
                    semantic_only_missing_receipt
                    and targets_expected_hypothesis
                    and delta.action
                    in {RefinementAction.CONFIRM, RefinementAction.REVISE}
                ):
                    # A targeted semantic payload with stripped coordinates
                    # cannot safely inherit the coarse boxes: those boxes refer
                    # to the old label/detail, while the new semantics have no
                    # turn-bound localization proof.  Keep the coarse finding
                    # byte-for-byte intact and expose the failed mutation only
                    # as audit/review state.
                    if any(item.id == delta.target_id for item in merged):
                        unlocalized_finding_ids.append(delta.target_id)
                        trace.append(
                            {
                                "stage": "bbox_receipt_guardrail",
                                "status": "semantic_only_missing_receipt",
                                "tool": "bound_bbox_grounding_contract",
                                "target_id": target.key,
                                "finding_id": delta.target_id,
                                "action": delta.action.value,
                                "decision_applied": False,
                                "coarse_finding_preserved": True,
                                "overlay_coordinates_asserted": False,
                                "diagnosis_forced": False,
                            }
                        )
                    continue

                target_has_complete_coverage = False
                if targets_expected_hypothesis:
                    source_hypothesis = coarse_by_id.get(
                        expected_target_id,
                        target.hypothesis,
                    )
                    (
                        target_has_complete_coverage,
                        uncovered,
                        source_bbox_count,
                    ) = _target_crop_coverage(crop_region, source_hypothesis)
                    if not target_has_complete_coverage:
                        status = {
                            RefinementAction.RETRACT: (
                                "partial_crop_retraction_blocked"
                            ),
                            RefinementAction.REVISE: "partial_crop_revision_blocked",
                            RefinementAction.CONFIRM: (
                                "partial_crop_confirmation_blocked"
                            ),
                        }[delta.action]
                        trace.append(
                            {
                                "stage": "refinement_guardrail",
                                "status": status,
                                "tool": "crop_coverage_guard",
                                "target_id": target.key,
                                "crop_region": _region_payload(crop_region),
                                "source_bbox_count": source_bbox_count,
                                "uncovered_bbox_count": len(uncovered),
                                "uncovered_bboxes": [
                                    _region_payload(region) for region in uncovered
                                ],
                                "source_was_unlocalized": source_bbox_count == 0,
                                "blocked_action": delta.action.value,
                                "decision_applied": False,
                                "severity_downgrade_authorized": False,
                                "evidence_interpretation": (
                                    "inconclusive_partial_crop_coverage"
                                ),
                            }
                        )
                        continue
                if delta is not raw_delta and delta.finding is not None:
                    trace.append(
                        {
                            "stage": "refinement_guardrail",
                            "status": "downgraded_unresolved_discovery",
                            "tool": "nonurgent_uncertainty_contract",
                            "target_id": target.key,
                            "finding_id": delta.finding.id,
                            "severity_before": raw_delta.finding.severity.value,
                            "severity_after": delta.finding.severity.value,
                        }
                    )
                before_count = len(merged)
                target_was_present = bool(
                    targets_expected_hypothesis
                    and any(item.id == delta.target_id for item in merged)
                )
                merged = apply_refinement_delta(
                    merged,
                    delta,
                    crop_region=crop_region,
                    expected_target_id=expected_target_id,
                    allow_unlocalized_add=unlocalized_add,
                )
                if (
                    target_was_present
                    and target_has_complete_coverage
                    and delta.action
                    in {RefinementAction.REVISE, RefinementAction.RETRACT}
                ):
                    allow_downgrade = True
                retained_id = ""
                if unlocalized_add and len(merged) > before_count:
                    retained_id = merged[-1].id
                if retained_id:
                    unlocalized_finding_ids.append(retained_id)
                    trace.append(
                        {
                            "stage": "bbox_receipt_guardrail",
                            "status": "semantic_only_missing_receipt",
                            "tool": "bound_bbox_grounding_contract",
                            "target_id": target.key,
                            "finding_id": retained_id,
                            "action": delta.action.value,
                            "decision_applied": True,
                            "overlay_coordinates_asserted": False,
                            "diagnosis_forced": False,
                        }
                    )
        merged_result = dataclasses.replace(
            coarse,
            findings=merged,
            severity=_merged_severity(
                coarse,
                merged,
                allow_downgrade=allow_downgrade,
            ),
            zoom_hints=[*coarse.zoom_hints, *zoom_hints],
            analysis_trace=trace,
        )
        if not unlocalized_finding_ids:
            return merged_result
        reason = (
            "A parsed crop refinement lacked an image/turn-bound bbox receipt; "
            "unlocalized additions remain coordinate-free and targeted mutations "
            "were not applied. Confirm localization on the native study."
        )
        return dataclasses.replace(
            merged_result,
            incomplete=True,
            incomplete_reasons=list(
                dict.fromkeys([*merged_result.incomplete_reasons, reason])
            ),
            review_required=True,
            review_reasons=list(dict.fromkeys([*merged_result.review_reasons, reason])),
        )


class MultiPassAnalyzer(VisionAnalyzerService):
    """Drop-in ``VisionAnalyzerService`` that runs a :class:`MultiPassInterpreter`.

    Runs the coarse → crop → refine loop for connected host analyzers. Session
    lifecycle and follow-up calls delegate to the injected analyzer; this module
    owns no capture, desktop UI, provider client or plugin runtime.
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
