"""EKG rhythm-strip refinement pass (application layer).

The coarse whole-image read often under-reports rhythm / P-wave / AV-conduction
findings because the rhythm strip is small in the down-scaled full image. This
pass crops the model-declared rhythm strip out of the original-resolution image
and re-reads just that strip, then merges the higher-confidence rhythm findings
back into the coarse result.

Design constraints (see AGENTS.md four cores):

- **General, not layout-assuming.** The strip region comes only from the
  model's Step-0 ``layout.rhythm_strip_bbox`` for THIS image. If the model did
  not declare one (single strip, partial, non-standard, unknown), this pass is
  a no-op -- it never guesses a fixed position.
- **Core 1.** Refined bboxes stay in normalized 0-1 ROI coordinates (remapped
  from the crop's own frame).
- **Escalate-only.** Rhythm checklist axes may be upgraded to a more severe
  reading but never downgraded, mirroring the clinical-safety rule.
- **DDD.** Image slicing and inference are injected; no PIL/numpy or network
  code lives here.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import structlog

from medical_image_harness.models import (
    AnalysisResult,
    Finding,
    Modality,
    RegionRect,
    Severity,
)
from medical_image_harness.multipass import pad_region, remap_bbox
from medical_image_harness.protocols import VisionAnalyzerService

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = structlog.get_logger(__name__)

# Checklist axes the rhythm strip is authoritative for.
RHYTHM_AXES: frozenset[str] = frozenset(
    {"heart_rate", "rhythm", "regularity", "p_wave", "pr_interval", "av_block"}
)

_SEVERITY_RANK: dict[Severity, int] = {
    Severity.NORMAL: 0,
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.CRITICAL: 2,
}
_ABNORMAL: frozenset[Severity] = frozenset({Severity.WARNING, Severity.CRITICAL})


def resolve_rhythm_strip_region(result: AnalysisResult) -> RegionRect | None:
    """Return the model-declared rhythm-strip region, or ``None``.

    General by design: the region comes only from the Step-0
    ``layout.rhythm_strip_bbox`` the model reported for THIS image, so a
    single-strip / partial / non-standard / unknown capture is never cropped on
    a guessed position. Accepts the ``[x, y, w, h]`` normalized array form and
    clamps it into the unit square, dropping degenerate strips.
    """
    layout = result.layout if isinstance(result.layout, dict) else {}
    raw = layout.get("rhythm_strip_bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x, y, w, h = float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        return None
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0 - x)
    h = min(max(h, 0.0), 1.0 - y)
    if w <= 0.0 or h <= 0.0:
        return None
    return RegionRect(x=x, y=y, w=w, h=h)


def _more_severe(a: Severity, b: Severity) -> Severity:
    return a if _SEVERITY_RANK[a] >= _SEVERITY_RANK[b] else b


def merge_rhythm_strip(
    coarse: AnalysisResult,
    strip: AnalysisResult,
    strip_region: RegionRect,
) -> AnalysisResult:
    """Merge rhythm findings/axes from a strip re-read into the coarse result.

    Conservative and escalate-only:

    - A rhythm checklist axis adopts the strip's reading only when the strip is
      *strictly more severe* (never downgraded).
    - Strip findings that are abnormal and not already present (by lowercased
      label) are appended, with bboxes remapped into global ROI coordinates.
    - Overall severity is escalated when an adopted axis/finding is more severe.

    Returns ``coarse`` unchanged when the strip adds nothing.
    """
    merged_checklist = dict(coarse.checklist)
    escalated: list[str] = []
    for axis in RHYTHM_AXES:
        strip_item = strip.checklist.get(axis)
        if strip_item is None:
            continue
        coarse_item = merged_checklist.get(axis)
        if coarse_item is None or (
            _SEVERITY_RANK[strip_item.status] > _SEVERITY_RANK[coarse_item.status]
        ):
            merged_checklist[axis] = strip_item
            if strip_item.status in _ABNORMAL:
                escalated.append(axis)

    existing_labels = {f.label.strip().lower() for f in coarse.findings}
    appended: list[Finding] = []
    for finding in strip.findings:
        if finding.severity not in _ABNORMAL:
            continue
        if finding.label.strip().lower() in existing_labels:
            continue
        remapped = [remap_bbox(b, strip_region) for b in finding.bboxes]
        appended.append(
            dataclasses.replace(
                finding,
                id=f"rhythm_{finding.id}" if finding.id else f"rhythm_{len(appended) + 1}",
                bboxes=remapped,
            )
        )

    if not escalated and not appended:
        return coarse

    new_severity = coarse.severity
    for finding in appended:
        new_severity = _more_severe(new_severity, finding.severity)
    for axis in escalated:
        new_severity = _more_severe(new_severity, merged_checklist[axis].status)

    return dataclasses.replace(
        coarse,
        severity=new_severity,
        checklist=merged_checklist,
        findings=[*coarse.findings, *appended],
    )


async def refine_rhythm_strip(
    result: AnalysisResult,
    image_base64: str,
    *,
    analyze_fn: Callable[[str, Modality, list[str]], Awaitable[AnalysisResult]],
    cropper: Callable[[str, RegionRect], str],
    valid_regions: list[str],
    padding: float = 0.05,
    retry_attempts: int = 1,
) -> AnalysisResult:
    """Crop the declared rhythm strip, re-read it, and merge rhythm findings.

    A no-op (returns ``result`` unchanged) when the modality is not EKG, when
    the model declared no rhythm-strip bbox, or when the strip re-read fails.
    That keeps non-standard / partial / single-strip captures safe: the pass
    only fires when a dedicated strip was explicitly localized by Step 0.
    """
    if retry_attempts < 0:
        raise ValueError("retry_attempts must be >= 0")
    if result.modality != Modality.EKG:
        return result
    region = resolve_rhythm_strip_region(result)
    if region is None:
        return result
    crop_region = pad_region(region, padding)
    try:
        crop_b64 = cropper(image_base64, crop_region)
    except Exception:
        logger.warning("rhythm_strip_crop_failed")
        return result
    strip: AnalysisResult | None = None
    for attempt in range(retry_attempts + 1):
        try:
            strip = await analyze_fn(crop_b64, Modality.EKG, valid_regions)
            if strip.summary.strip() or strip.findings:
                break
        except Exception:
            strip = None
        if attempt < retry_attempts:
            logger.warning(
                "rhythm_strip_analysis_retry",
                attempt=attempt + 1,
                max_retries=retry_attempts,
            )
    if strip is None or (not strip.summary.strip() and not strip.findings):
        logger.warning("rhythm_strip_analysis_failed")
        return result
    merged = merge_rhythm_strip(result, strip, crop_region)
    event: dict[str, object] = {
        "stage": "rhythm_strip_refine",
        "status": "completed" if merged is not result else "completed_no_change",
        "tool": "crop_region+analyzer",
        "crop_source": "source_image",
        "crop_region": {
            "x": crop_region.x,
            "y": crop_region.y,
            "w": crop_region.w,
            "h": crop_region.h,
        },
        "finding_count_before": len(result.findings),
        "finding_count_after": len(merged.findings),
    }
    return dataclasses.replace(
        merged,
        analysis_trace=[*merged.analysis_trace, event],
    )


class RhythmStripRefiningAnalyzer(VisionAnalyzerService):
    """Apply the shared rhythm-strip pass to a complete analyzer transaction.

    The desktop and evaluator can use the same application-layer implementation:
    the inner analyzer produces the full-image draft, then a separately injected
    analyzer re-reads only a model-localized strip from the original ROI.
    """

    def __init__(
        self,
        *,
        inner: VisionAnalyzerService,
        refinement_analyzer: VisionAnalyzerService,
        cropper: Callable[[str, RegionRect], str],
    ) -> None:
        self._inner = inner
        self._refinement_analyzer = refinement_analyzer
        self._cropper = cropper

    async def analyze(
        self,
        image_base64: str,
        modality: Modality,
        valid_regions: list[str],
    ) -> AnalysisResult:
        result = await self._inner.analyze(image_base64, modality, valid_regions)
        return await self._refine(
            result,
            image_base64,
            valid_regions=valid_regions,
        )

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
        analyze_with_source_size = getattr(
            self._inner,
            "analyze_with_source_size",
            None,
        )
        if callable(analyze_with_source_size):
            result = await analyze_with_source_size(
                image_base64,
                modality,
                valid_regions,
                source_size_px=source_size_px,
                source_image_base64=source_image_base64,
                local_candidate_regions=local_candidate_regions,
            )
        else:
            result = await self._inner.analyze(
                image_base64,
                modality,
                valid_regions,
            )
        return await self._refine(
            result,
            source_image_base64 or image_base64,
            valid_regions=valid_regions,
        )

    async def _refine(
        self,
        result: AnalysisResult,
        source_image_base64: str,
        *,
        valid_regions: list[str],
    ) -> AnalysisResult:
        return await refine_rhythm_strip(
            result,
            source_image_base64,
            analyze_fn=self._refinement_analyzer.analyze,
            cropper=self._cropper,
            valid_regions=valid_regions,
        )

    async def chat(self, message: str) -> str:
        return await self._inner.chat(message)

    async def connect(self) -> None:
        await self._inner.connect()

    async def disconnect(self) -> None:
        await self._inner.disconnect()

    def is_connected(self) -> bool:
        return self._inner.is_connected()
