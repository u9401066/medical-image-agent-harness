"""Deterministic EKG bbox signal calibration in original ROI coordinates."""

from __future__ import annotations

import base64
import dataclasses
import io

from PIL import Image

from medical_image_harness.ekg_layout import parse_ekg_lead_inventory
from medical_image_harness.models import (
    AnalysisResult,
    Modality,
    RegionRect,
    Severity,
)

_INK_THRESHOLD = 90
_LOW_SIGNAL_RATIO = 0.003
_MIN_IMPROVED_RATIO = 0.006
_MIN_IMPROVEMENT_MULTIPLIER = 3.0
_SEARCH_OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)
_DEFAULT_CONTEXT_SIZE_PX = (64, 64)
# QRS/voltage evidence needs horizontal pre/post-complex context.  Keeping the
# crop to roughly one lead row also avoids mixing adjacent sequential leads.
_VOLTAGE_CONTEXT_SIZE_PX = (128, 64)
_ST_T_CONTEXT_SIZE_PX = (96, 64)
_RHYTHM_CONTEXT_SIZE_PX = (180, 64)
_MAX_FINDING_BOX_WIDTH = 0.35
_MAX_FINDING_BOX_HEIGHT = 0.30
_MAX_FINDING_BOX_AREA = 0.08


def _pixel_box(
    region: RegionRect,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x0 = max(0, min(width, round(region.x * width)))
    y0 = max(0, min(height, round(region.y * height)))
    x1 = max(x0, min(width, round((region.x + region.w) * width)))
    y1 = max(y0, min(height, round((region.y + region.h) * height)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def _ink_ratio(gray: Image.Image, box: tuple[int, int, int, int]) -> float:
    crop = gray.crop(box)
    histogram = crop.histogram()
    ink = sum(histogram[:_INK_THRESHOLD])
    return ink / max(1, crop.width * crop.height)


def _region_payload(region: RegionRect) -> dict[str, float]:
    return {
        "x": round(region.x, 6),
        "y": round(region.y, 6),
        "w": round(region.w, 6),
        "h": round(region.h, 6),
    }


def _layout_leads(layout: object) -> list[tuple[str, RegionRect]]:
    return [
        (lead.name, lead.bbox)
        for lead in parse_ekg_lead_inventory(layout).leads
    ]


def _lead_at_box_center(
    box: RegionRect,
    layout_leads: list[tuple[str, RegionRect]],
) -> str | None:
    center_x = box.x + box.w / 2.0
    center_y = box.y + box.h / 2.0
    candidates = [
        (name, lead_box)
        for name, lead_box in layout_leads
        if lead_box.x <= center_x <= lead_box.x + lead_box.w
        and lead_box.y <= center_y <= lead_box.y + lead_box.h
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1].w * item[1].h)[0]


def _context_size_px(label: str, detail: str) -> tuple[int, int]:
    text = f"{label} {detail}".lower()
    if any(
        term in text
        for term in ("voltage", "qrs", "s-wave", "r-wave", "s wave", "r wave", "lvh")
    ):
        return _VOLTAGE_CONTEXT_SIZE_PX
    if any(
        term in text
        for term in ("st-t", "st segment", "st-segment", "t wave", "t-wave")
    ):
        return _ST_T_CONTEXT_SIZE_PX
    if any(term in text for term in ("rhythm", "bradycard", "tachycard", "ectopic")):
        return _RHYTHM_CONTEXT_SIZE_PX
    return _DEFAULT_CONTEXT_SIZE_PX


def _expand_to_context(
    region: RegionRect,
    *,
    image_size: tuple[int, int],
    minimum_px: tuple[int, int],
) -> RegionRect:
    width, height = image_size
    min_width, min_height = minimum_px
    target_width = min(1.0, max(region.w, min_width / max(1, width)))
    target_height = min(1.0, max(region.h, min_height / max(1, height)))
    center_x = region.x + region.w / 2.0
    center_y = region.y + region.h / 2.0
    x = min(max(0.0, center_x - target_width / 2.0), 1.0 - target_width)
    y = min(max(0.0, center_y - target_height / 2.0), 1.0 - target_height)
    return RegionRect(x=x, y=y, w=target_width, h=target_height)


def _best_signal_neighbor(
    gray: Image.Image,
    region: RegionRect,
) -> tuple[RegionRect, float, float, int] | None:
    width, height = gray.size
    original_box = _pixel_box(region, width, height)
    if original_box is None:
        return None
    base_ratio = _ink_ratio(gray, original_box)
    if base_ratio >= _LOW_SIGNAL_RATIO:
        return region, base_ratio, base_ratio, 0

    box_width = original_box[2] - original_box[0]
    box_height = original_box[3] - original_box[1]
    best_box = original_box
    best_ratio = base_ratio
    best_shift = 0
    for y_factor in _SEARCH_OFFSETS:
        for x_factor in _SEARCH_OFFSETS:
            x0 = round(original_box[0] + x_factor * box_width)
            y0 = round(original_box[1] + y_factor * box_height)
            x0 = max(0, min(width - box_width, x0))
            y0 = max(0, min(height - box_height, y0))
            candidate = (x0, y0, x0 + box_width, y0 + box_height)
            ratio = _ink_ratio(gray, candidate)
            shift = abs(x0 - original_box[0]) + abs(y0 - original_box[1])
            if ratio > best_ratio or (ratio == best_ratio and shift < best_shift):
                best_box = candidate
                best_ratio = ratio
                best_shift = shift

    improved = best_ratio >= max(
        _MIN_IMPROVED_RATIO,
        base_ratio * _MIN_IMPROVEMENT_MULTIPLIER,
    )
    if not improved or best_box == original_box:
        return region, base_ratio, best_ratio, 0
    x0, y0, x1, y1 = best_box
    return (
        RegionRect(
            x=x0 / width,
            y=y0 / height,
            w=(x1 - x0) / width,
            h=(y1 - y0) / height,
        ),
        base_ratio,
        best_ratio,
        best_shift,
    )


def calibrate_ekg_bboxes(
    image_base64: str,
    result: AnalysisResult,
) -> AnalysisResult:
    """Snap clearly blank EKG boxes locally or flag them for expert review."""
    if result.modality is not Modality.EKG:
        return result
    try:
        raw = base64.b64decode(image_base64, validate=True)
        gray = Image.open(io.BytesIO(raw)).convert("L")
    except Exception:
        return result

    findings = []
    trace = list(result.analysis_trace)
    layout_leads = _layout_leads(result.layout)
    review_reasons = list(result.review_reasons)
    review_required = result.review_required
    for finding in result.findings:
        if finding.severity is Severity.NORMAL or not finding.bboxes:
            findings.append(finding)
            continue
        calibrated_boxes: list[RegionRect] = []
        notes = list(finding.notes)
        for box_index, box in enumerate(finding.bboxes, start=1):
            if (
                box.w > _MAX_FINDING_BOX_WIDTH
                or box.h > _MAX_FINDING_BOX_HEIGHT
                or box.w * box.h > _MAX_FINDING_BOX_AREA
            ):
                calibrated_boxes.append(box)
                reason = (
                    f"Broad lead-strip bbox for {finding.label or finding.id}; "
                    "refinement must replace it with local evidence boxes."
                )
                if reason not in review_reasons:
                    review_reasons.append(reason)
                notes.append(
                    "Broad lead-strip bbox; not accepted as final localization."
                )
                review_required = True
                trace.append(
                    {
                        "stage": "bbox_calibration",
                        "status": "too_broad_for_finding",
                        "tool": "local_ekg_signal_calibrator",
                        "finding_id": finding.id,
                        "bbox_index": box_index,
                        "original": _region_payload(box),
                    }
                )
                continue
            calibrated = _best_signal_neighbor(gray, box)
            if calibrated is None:
                calibrated_boxes.append(box)
                continue
            candidate, before_ratio, after_ratio, shift_px = calibrated
            if candidate != box:
                note = (
                    f"BBox {box_index} signal-calibrated locally "
                    f"({before_ratio:.3%} -> {after_ratio:.3%} ink)."
                )
                notes.append(note)
                trace.append(
                    {
                        "stage": "bbox_calibration",
                        "status": "adjusted",
                        "tool": "local_ekg_signal_calibrator",
                        "finding_id": finding.id,
                        "bbox_index": box_index,
                        "original": _region_payload(box),
                        "calibrated": _region_payload(candidate),
                        "ink_ratio_before": round(before_ratio, 6),
                        "ink_ratio_after": round(after_ratio, 6),
                        "shift_manhattan_px": shift_px,
                    }
                )
            if after_ratio >= _LOW_SIGNAL_RATIO:
                contextual = _expand_to_context(
                    candidate,
                    image_size=gray.size,
                    minimum_px=_context_size_px(finding.label, finding.detail),
                )
                calibrated_boxes.append(contextual)
                if contextual != candidate:
                    notes.append(
                        f"BBox {box_index} expanded to preserve interpretable "
                        "waveform context."
                    )
                    trace.append(
                        {
                            "stage": "bbox_calibration",
                            "status": "expanded_for_context",
                            "tool": "local_ekg_signal_calibrator",
                            "finding_id": finding.id,
                            "bbox_index": box_index,
                            "original": _region_payload(candidate),
                            "calibrated": _region_payload(contextual),
                            "minimum_context_px": list(
                                _context_size_px(finding.label, finding.detail)
                            ),
                        }
                    )
            else:
                calibrated_boxes.append(candidate)
                if before_ratio < _LOW_SIGNAL_RATIO:
                    reason = (
                        f"Low-signal bbox for {finding.label or finding.id}; "
                        "no confident local alignment correction was found."
                    )
                    if reason not in review_reasons:
                        review_reasons.append(reason)
                    notes.append(
                        "Low-signal bbox; verify alignment on the source image."
                    )
                    review_required = True
                    trace.append(
                        {
                            "stage": "bbox_calibration",
                            "status": "review_required",
                            "tool": "local_ekg_signal_calibrator",
                            "finding_id": finding.id,
                            "bbox_index": box_index,
                            "original": _region_payload(box),
                            "ink_ratio": round(before_ratio, 6),
                        }
                    )
        declared_regions = list(finding.regions)
        confidence = finding.confidence
        question = finding.question
        bbox_regions = list(
            dict.fromkeys(
                lead
                for box in calibrated_boxes
                if (lead := _lead_at_box_center(box, layout_leads)) is not None
            )
        )
        reconciled_regions = declared_regions
        if bbox_regions and not set(bbox_regions).issubset(declared_regions):
            reconciled_regions = bbox_regions
            reason = (
                f"BBox/lead mismatch for {finding.label or finding.id}; regions were "
                "reconciled to the declared EKG layout."
            )
            if reason not in review_reasons:
                review_reasons.append(reason)
            notes.append(
                "BBox lead regions reconciled from "
                f"{', '.join(declared_regions) or '(none)'} to {', '.join(bbox_regions)}."
            )
            confidence = "low"
            localization_question = (
                "Localization conflict: the model named "
                f"{', '.join(declared_regions) or '(no lead)'}, but its box maps to "
                f"{', '.join(bbox_regions)}. Which lead and finding are correct?"
            )
            question = (
                f"{question.rstrip()} {localization_question}"
                if question.strip()
                else localization_question
            )
            review_required = True
            trace.append(
                {
                    "stage": "bbox_calibration",
                    "status": "lead_region_reconciled",
                    "tool": "local_ekg_signal_calibrator",
                    "finding_id": finding.id,
                    "declared_regions": declared_regions,
                    "bbox_regions": bbox_regions,
                }
            )
        findings.append(
            dataclasses.replace(
                finding,
                bboxes=calibrated_boxes,
                regions=reconciled_regions,
                notes=list(dict.fromkeys(notes)),
                confidence=confidence,
                question=question,
            )
        )
    return dataclasses.replace(
        result,
        findings=findings,
        analysis_trace=trace,
        review_required=review_required,
        review_reasons=review_reasons,
    )
