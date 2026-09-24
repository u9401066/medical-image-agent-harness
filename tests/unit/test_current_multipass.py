"""Unit tests for the multi-pass interpretation orchestrator.

Covers the pure coordinate maths (clamp / pad / remap), zoom-target selection,
and the coarse -> crop -> refine orchestration including the privacy invariant
that a zoom crop only ever shrinks the captured region.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from medical_image_harness.models import (
    AnalysisResult,
    ChecklistItem,
    Finding,
    Modality,
    RegionRect,
    Severity,
)
from medical_image_harness.multipass import (
    AnalysisSlaTimeout,
    MultiPassAnalyzer,
    MultiPassInterpreter,
    RefinementAction,
    RefinementDelta,
    RefinementResult,
    _bounded_operation_timeout_sec,
    apply_critical_triage_guard,
    apply_ekg_overlay_bbox_guard,
    apply_ekg_ventricular_run_evidence_guard,
    apply_ekg_waveform_rhythm_conflict_guard,
    apply_refinement_delta,
    apply_unlocalized_ekg_grounding_guard,
    build_manual_zoom_message,
    clamp_unit,
    covering_region,
    deduplicate_ekg_study_level_findings,
    expand_crop_to_min_source_edge,
    needs_manual_zoom,
    pad_region,
    project_ekg_lead_regions_to_crop,
    qualify_boxed_info_findings,
    reconcile_final_report,
    reconcile_unavailable_ekg_rhythm_regions,
    region_source_edge_px,
    remap_bbox,
    select_critical_triage_candidates,
    select_ekg_critical_support_probe,
    select_ekg_systematic_probe_regions,
    select_ekg_waveform_attention_probe_regions,
    select_hypothesis_crop_region,
    select_zoom_targets,
)
from medical_image_harness.profiles import get_active_registry
from medical_image_harness.protocols import VisionAnalyzerService


def _result(findings: list[Finding]) -> AnalysisResult:
    return AnalysisResult(
        modality=Modality.CXR,
        summary="test",
        severity=Severity.WARNING if findings else Severity.NORMAL,
        findings=findings,
        checklist={"x": ChecklistItem(value="ok", status=Severity.NORMAL)},
    )


def _finding(
    fid: str,
    severity: Severity,
    bbox: RegionRect | None,
    *,
    label: str = "lesion",
    detail: str = "",
) -> Finding:
    return Finding(
        id=fid,
        regions=[],
        label=label,
        detail=detail,
        severity=severity,
        bboxes=[bbox] if bbox else [],
    )


def _ekg_row_layout_result(findings: list[Finding]) -> AnalysisResult:
    names = [
        "lead_I",
        "lead_II",
        "lead_III",
        "lead_aVR",
        "lead_aVL",
        "lead_aVF",
        "lead_V1",
        "lead_V2",
        "lead_V3",
        "lead_V4",
        "lead_V5",
        "lead_V6",
    ]
    result = AnalysisResult(
        modality=Modality.EKG,
        summary="test",
        severity=Severity.WARNING if findings else Severity.NORMAL,
        findings=findings,
        checklist={"rhythm": ChecklistItem(value="sinus", status=Severity.NORMAL)},
    )
    result.layout = {
        "format": "12lead_rows",
        "leads": [
            {
                "name": name,
                "label_visible": True,
                "bbox": [0.0, index / 12, 1.0, 1 / 12],
            }
            for index, name in enumerate(names)
        ],
    }
    return result


def _complete_ekg_checklist(
    *, status: Severity = Severity.NORMAL
) -> dict[str, ChecklistItem]:
    keys = get_active_registry().resolve(Modality.EKG.value).checklist_keys
    return {
        key: ChecklistItem(value=f"explicitly assessed {key}", status=status)
        for key in keys
    }


# ── clamp_unit ───────────────────────────────────────────────────────


def test_followup_completion_grace_is_bounded_and_opt_in() -> None:
    assert _bounded_operation_timeout_sec(48.5) == pytest.approx(48.5)
    assert _bounded_operation_timeout_sec(48.5, 0.250) == pytest.approx(48.75)
    assert _bounded_operation_timeout_sec(0.020, 0.250) == pytest.approx(0.0202)


class TestClampUnit:
    def test_passthrough(self):
        assert clamp_unit(0.5) == 0.5

    def test_below_zero(self):
        assert clamp_unit(-0.3) == 0.0

    def test_above_one(self):
        assert clamp_unit(1.4) == 1.0


# ── pad_region ───────────────────────────────────────────────────────


class TestPadRegion:
    def test_grows_by_fraction_of_size(self):
        region = RegionRect(x=0.4, y=0.4, w=0.2, h=0.2)
        padded = pad_region(region, 0.5)  # +/- 0.1 per side
        assert padded.x == pytest.approx(0.3)
        assert padded.y == pytest.approx(0.3)
        assert padded.w == pytest.approx(0.4)
        assert padded.h == pytest.approx(0.4)

    def test_clamped_to_roi_frame(self):
        region = RegionRect(x=0.0, y=0.0, w=0.2, h=0.2)
        padded = pad_region(region, 1.0)
        assert padded.x == 0.0  # cannot go negative
        assert padded.y == 0.0
        assert padded.x + padded.w <= 1.0 + 1e-9

    def test_negative_pad_rejected(self):
        with pytest.raises(ValueError):
            pad_region(RegionRect(x=0.1, y=0.1, w=0.1, h=0.1), -0.1)


# ── remap_bbox ───────────────────────────────────────────────────────


class TestRemapBbox:
    def test_full_crop_bbox_is_identity(self):
        parent = RegionRect(x=0.0, y=0.0, w=1.0, h=1.0)
        child = RegionRect(x=0.25, y=0.5, w=0.1, h=0.2)
        out = remap_bbox(child, parent)
        assert out.x == pytest.approx(0.25)
        assert out.y == pytest.approx(0.5)
        assert out.w == pytest.approx(0.1)
        assert out.h == pytest.approx(0.2)

    def test_center_of_crop_maps_into_crop(self):
        # Crop occupies the bottom-right quarter of the ROI.
        parent = RegionRect(x=0.5, y=0.5, w=0.5, h=0.5)
        # A bbox centered in the crop.
        child = RegionRect(x=0.4, y=0.4, w=0.2, h=0.2)
        out = remap_bbox(child, parent)
        # Global center should be 0.5 + 0.5*0.5 = 0.75
        assert (out.x + out.w / 2) == pytest.approx(0.75)
        assert (out.y + out.h / 2) == pytest.approx(0.75)
        # Width scales by the crop width.
        assert out.w == pytest.approx(0.1)

    def test_result_stays_in_unit_square(self):
        parent = RegionRect(x=0.8, y=0.8, w=0.2, h=0.2)
        child = RegionRect(x=0.9, y=0.9, w=0.3, h=0.3)
        out = remap_bbox(child, parent)
        assert 0.0 <= out.x <= 1.0
        assert 0.0 <= out.y <= 1.0
        assert out.x + out.w <= 1.0 + 1e-9
        assert out.y + out.h <= 1.0 + 1e-9

    def test_child_bbox_overflow_is_clamped_to_parent_crop(self):
        parent = RegionRect(x=0.2, y=0.2, w=0.3, h=0.3)
        child = RegionRect(x=0.8, y=0.8, w=0.5, h=0.5)
        out = remap_bbox(child, parent)
        assert out.x == pytest.approx(0.44)
        assert out.y == pytest.approx(0.44)
        assert out.w == pytest.approx(0.06)
        assert out.h == pytest.approx(0.06)
        assert out.x + out.w <= parent.x + parent.w + 1e-9
        assert out.y + out.h <= parent.y + parent.h + 1e-9


class TestCoveringRegion:
    def test_covers_all_boxes_for_one_multi_lead_hypothesis(self):
        region = covering_region(
            [
                RegionRect(0.1, 0.2, 0.2, 0.1),
                RegionRect(0.4, 0.6, 0.3, 0.2),
            ]
        )

        assert region.x == pytest.approx(0.1)
        assert region.y == pytest.approx(0.2)
        assert region.w == pytest.approx(0.6)
        assert region.h == pytest.approx(0.6)

    def test_rejects_no_valid_regions(self):
        with pytest.raises(ValueError):
            covering_region([])

    def test_disjoint_hypothesis_uses_largest_local_box(self):
        finding = Finding(
            id="lvh",
            regions=["lead_aVL", "lead_V5"],
            label="Left ventricular hypertrophy",
            detail="Voltage criteria",
            severity=Severity.WARNING,
            bboxes=[
                RegionRect(0.06, 0.34, 0.22, 0.08),
                RegionRect(0.06, 0.88, 0.24, 0.08),
            ],
        )

        selected = select_hypothesis_crop_region(finding, modality=Modality.EKG)

        assert selected == finding.bboxes[1]

    def test_temporal_ekg_hypothesis_uses_declared_rhythm_strip(self):
        finding = Finding(
            id="pvc",
            regions=["lead_II", "rhythm_strip"],
            label="Frequent premature ventricular complexes",
            detail="Wide premature beats with compensatory pauses",
            severity=Severity.WARNING,
            bboxes=[RegionRect(0.12, 0.22, 0.08, 0.05)],
        )
        result = _ekg_row_layout_result([finding])
        result.layout["rhythm_strip_bbox"] = [0.02, 0.78, 0.96, 0.18]

        selected = select_hypothesis_crop_region(
            finding,
            modality=Modality.EKG,
            layout=result.layout,
        )

        assert selected == RegionRect(0.02, 0.78, 0.96, 0.18)

    def test_temporal_ekg_hypothesis_falls_back_to_full_lead_ii(self):
        finding = Finding(
            id="avb",
            regions=["lead_II"],
            label="First-degree AV block",
            detail="Prolonged PR relationship",
            severity=Severity.WARNING,
            bboxes=[RegionRect(0.2, 0.09, 0.05, 0.03)],
        )
        result = _ekg_row_layout_result([finding])

        selected = select_hypothesis_crop_region(
            finding,
            modality=Modality.EKG,
            layout=result.layout,
        )

        assert selected == RegionRect(0.0, 1 / 12, 1.0, 1 / 12)

    def test_row_strip_wide_complex_uses_cross_lead_time_slice(self):
        finding = Finding(
            id="intermittent_wide_qrs",
            regions=["lead_I", "lead_II", "lead_V1", "lead_V6"],
            label="Intermittent wide QRS with possible pacing",
            detail="Synchronized discordant complexes recur across stacked leads",
            severity=Severity.WARNING,
            bboxes=[RegionRect(0.38, 0.08, 0.06, 0.05)],
        )
        result = _ekg_row_layout_result([finding])
        result.layout["format"] = "12lead_12x1"

        selected = select_hypothesis_crop_region(
            finding,
            modality=Modality.EKG,
            layout=result.layout,
        )

        assert (selected.x, selected.y, selected.w, selected.h) == pytest.approx(
            (0.32, 0.0, 0.18, 1.0)
        )

    def test_row_strip_full_width_evidence_never_manufactures_central_time_crop(
        self,
    ):
        full_lead = RegionRect(0.0, 1 / 12, 1.0, 1 / 12)
        finding = Finding(
            id="possible_demand_pacing",
            regions=["lead_II"],
            label="Possible demand-paced rhythm",
            detail="Intermittent synchronized wide complexes require review.",
            severity=Severity.WARNING,
            bboxes=[full_lead],
        )
        result = _ekg_row_layout_result([finding])
        result.layout["format"] = "12lead_12x1"

        selected = select_hypothesis_crop_region(
            finding,
            modality=Modality.EKG,
            layout=result.layout,
        )

        assert selected == full_lead
        assert selected != RegionRect(0.325, 0.0, 0.35, 1.0)

    def test_row_strip_broad_unlocalized_box_retains_its_horizontal_context(self):
        broad_context = RegionRect(0.1, 0.0, 0.8, 0.25)
        finding = Finding(
            id="possible_pacing",
            regions=["lead_I", "lead_II", "lead_III"],
            label="Possible demand pacing",
            detail="Synchronized wide complexes need mechanism review.",
            severity=Severity.WARNING,
            bboxes=[broad_context],
        )
        result = _ekg_row_layout_result([finding])
        result.layout["format"] = "12lead_12x1"

        selected = select_hypothesis_crop_region(
            finding,
            modality=Modality.EKG,
            layout=result.layout,
        )

        assert selected == broad_context

    def test_row_strip_local_event_crop_tracks_horizontal_shift_metamorphically(
        self,
    ):
        shift = 0.45
        left_box = RegionRect(0.12, 0.08, 0.06, 0.05)
        right_box = dataclasses.replace(left_box, x=left_box.x + shift)

        def route(box: RegionRect) -> RegionRect:
            finding = Finding(
                id="intermittent_wide_qrs",
                regions=["lead_I", "lead_II", "lead_III"],
                label="Intermittent wide QRS with possible pacing",
                detail="Synchronized event across stacked leads.",
                severity=Severity.WARNING,
                bboxes=[box],
            )
            result = _ekg_row_layout_result([finding])
            result.layout["format"] = "12lead_12x1"
            return select_hypothesis_crop_region(
                finding,
                modality=Modality.EKG,
                layout=result.layout,
            )

        left_crop = route(left_box)
        right_crop = route(right_box)

        assert right_crop.x - left_crop.x == pytest.approx(shift)
        assert right_crop.w == pytest.approx(left_crop.w)
        assert left_crop.x + left_crop.w / 2 == pytest.approx(
            left_box.x + left_box.w / 2
        )
        assert right_crop.x + right_crop.w / 2 == pytest.approx(
            right_box.x + right_box.w / 2
        )

    def test_standard_grid_wide_complex_does_not_assume_shared_time_axis(self):
        finding = Finding(
            id="wide_qrs",
            regions=["lead_V1", "lead_V6"],
            label="Wide QRS complexes",
            detail="Possible bundle branch conduction pattern",
            severity=Severity.WARNING,
            bboxes=[RegionRect(0.52, 0.12, 0.08, 0.06)],
        )
        result = _ekg_row_layout_result([finding])
        result.layout["format"] = "12lead_3x4"

        selected = select_hypothesis_crop_region(
            finding,
            modality=Modality.EKG,
            layout=result.layout,
        )

        assert selected != RegionRect(x=0.47, y=0.0, w=0.18, h=1.0)
        assert selected.h < 1.0

    def test_multi_lead_conduction_hypothesis_uses_declared_lead_context(self):
        finding = Finding(
            id="rbbb",
            regions=["lead_V1", "lead_V2", "lead_V6"],
            label="Right bundle branch block",
            detail="Wide QRS with terminal right-precordial forces",
            severity=Severity.WARNING,
            bboxes=[
                RegionRect(0.1, 0.52, 0.08, 0.04),
                RegionRect(0.1, 0.94, 0.08, 0.04),
            ],
        )
        result = _ekg_row_layout_result([finding])

        selected = select_hypothesis_crop_region(
            finding,
            modality=Modality.EKG,
            layout=result.layout,
        )

        assert selected == RegionRect(0.0, 0.5, 1.0, 0.5)


class TestCropLeadProjection:
    def test_projects_visible_stacked_leads_into_crop_coordinates(self):
        result = _ekg_row_layout_result([])
        crop = RegionRect(x=0.0, y=0.5, w=1.0, h=0.5)

        projected = project_ekg_lead_regions_to_crop(result.layout, crop)

        assert list(projected) == [
            "lead_V1",
            "lead_V2",
            "lead_V3",
            "lead_V4",
            "lead_V5",
            "lead_V6",
        ]
        assert projected["lead_V1"].y == pytest.approx(0.0)
        assert projected["lead_V4"].y == pytest.approx(0.5)
        assert projected["lead_V6"].y == pytest.approx(5 / 6)
        assert projected["lead_V6"].h == pytest.approx(1 / 6)

    def test_clamps_full_crop_projection_roundoff_to_unit_interval(self):
        result = _ekg_row_layout_result([])
        result.layout["leads"] = [
            {
                "name": "II",
                "label_visible": True,
                "bbox": [0.1, 0.1, 0.7, 0.2],
            }
        ]
        crop = RegionRect(x=0.1, y=0.1, w=0.7, h=0.2)

        projected = project_ekg_lead_regions_to_crop(result.layout, crop)

        assert projected["lead_II"] == RegionRect(0.0, 0.0, 1.0, 1.0)


class TestEkgSystematicProbeRegions:
    def test_uses_declared_lead_layout_to_cover_limb_and_precordial_groups(self):
        result = _ekg_row_layout_result([])

        probes = select_ekg_systematic_probe_regions(result)

        assert [key for key, _region in probes] == [
            "precordial_leads",
            "limb_leads",
        ]
        assert probes[0][1].y == pytest.approx(0.5)
        assert probes[0][1].h == pytest.approx(0.5)
        assert probes[1][1].y == pytest.approx(0.0)
        assert probes[1][1].h == pytest.approx(0.5)

    def test_accepts_unprefixed_real_model_lead_names(self):
        result = _ekg_row_layout_result([])
        for lead in result.layout["leads"]:
            lead["name"] = lead["name"].removeprefix("lead_")

        probes = select_ekg_systematic_probe_regions(result)

        assert [key for key, _region in probes] == [
            "precordial_leads",
            "limb_leads",
        ]

    def test_accepts_case_and_separator_variants(self):
        result = _ekg_row_layout_result([])
        variants = [
            "lead i",
            "LEAD-II",
            "iii",
            "AVR",
            "aVl",
            "avf",
            "v 1",
            "V-2",
            "v3",
            "V4",
            "v5",
            "V6",
        ]
        for lead, name in zip(result.layout["leads"], variants, strict=True):
            lead["name"] = name

        probes = select_ekg_systematic_probe_regions(result)

        assert [key for key, _region in probes] == [
            "precordial_leads",
            "limb_leads",
        ]

    def test_rejects_sparse_or_non_ekg_layout(self):
        result = _ekg_row_layout_result([])
        result.layout = {
            "leads": [{"name": "lead_I", "label_visible": True, "bbox": [0, 0, 1, 0.1]}]
        }
        assert select_ekg_systematic_probe_regions(result) == []
        assert select_ekg_systematic_probe_regions(_result([])) == []

    def test_waveform_rhythm_candidate_routes_attention_to_lead_ii(self):
        result = _ekg_row_layout_result([])
        result.analysis_trace = [
            {
                "stage": "coarse",
                "tool_audit": [
                    {
                        "tool": "ecg_founder_analyze_waveform",
                        "status": "ok",
                        "predictions": [
                            {"label": "ATRIAL FIBRILLATION"},
                            {"label": "ABNORMAL ECG"},
                            {"label": "SINUS RHYTHM"},
                        ],
                    }
                ],
            }
        ]

        probes = select_ekg_waveform_attention_probe_regions(result)

        assert probes == [
            ("waveform_rhythm_lead_II", RegionRect(0.0, 1 / 12, 1.0, 1 / 12))
        ]

    def test_low_rank_rhythm_does_not_redirect_top_three_st_probe(self):
        result = _ekg_row_layout_result([])
        result.analysis_trace = [
            {
                "tool_audit": [
                    {
                        "tool": "ecg_founder_analyze_waveform",
                        "status": "ok",
                        "predictions": [
                            {"label": "NORMAL SINUS RHYTHM"},
                            {"label": "NORMAL ECG"},
                            {"label": "NONSPECIFIC ST ABNORMALITY"},
                            {"label": "ATRIAL FIBRILLATION"},
                        ],
                    }
                ]
            }
        ]

        assert select_ekg_waveform_attention_probe_regions(result) == [
            (
                "waveform_attention_precordial_leads",
                RegionRect(0.0, 0.5, 1.0, 0.5),
            )
        ]

    def test_top_ranked_ectopy_routes_attention_to_lead_ii(self):
        result = _ekg_row_layout_result([])
        result.analysis_trace = [
            {
                "tool_audit": [
                    {
                        "tool": "ecg_founder_analyze_waveform",
                        "status": "ok",
                        "predictions": [
                            {"label": "PREMATURE VENTRICULAR COMPLEXES"},
                            {"label": "SINUS RHYTHM"},
                            {"label": "NORMAL SINUS RHYTHM"},
                        ],
                    }
                ]
            }
        ]

        assert select_ekg_waveform_attention_probe_regions(result) == [
            ("waveform_rhythm_lead_II", RegionRect(0.0, 1 / 12, 1.0, 1 / 12))
        ]

    @pytest.mark.parametrize(
        "label",
        [
            "LONG QT",
            "FIRST DEGREE AV BLOCK",
            "SINUS BRADYCARDIA",
            "LEFT ATRIAL ENLARGEMENT",
        ],
    )
    def test_ranked_temporal_candidate_routes_attention_to_lead_ii(self, label):
        result = _ekg_row_layout_result([])
        result.analysis_trace = [
            {
                "tool_audit": [
                    {
                        "tool": "ecg_founder_analyze_waveform",
                        "status": "ok",
                        "predictions": [{"label": label}],
                    }
                ]
            }
        ]

        assert select_ekg_waveform_attention_probe_regions(result) == [
            ("waveform_rhythm_lead_II", RegionRect(0.0, 1 / 12, 1.0, 1 / 12))
        ]

    def test_ranked_axis_candidate_routes_attention_to_limb_leads(self):
        result = _ekg_row_layout_result([])
        result.analysis_trace = [
            {
                "tool_audit": [
                    {
                        "tool": "ecg_founder_analyze_waveform",
                        "status": "ok",
                        "predictions": [
                            {"label": "ABNORMAL ECG"},
                            {"label": "LEFT AXIS DEVIATION"},
                            {"label": "LEFT BUNDLE BRANCH BLOCK"},
                        ],
                    }
                ]
            }
        ]

        probes = select_ekg_waveform_attention_probe_regions(result)

        assert probes == [
            ("waveform_attention_limb_leads", RegionRect(0.0, 0.0, 1.0, 0.5)),
            (
                "waveform_attention_precordial_leads",
                RegionRect(0.0, 0.5, 1.0, 0.5),
            ),
        ]

    def test_ranked_low_voltage_candidate_routes_attention_to_precordials(self):
        result = _ekg_row_layout_result([])
        result.analysis_trace = [
            {
                "tool_audit": [
                    {
                        "tool": "ecg_founder_analyze_waveform",
                        "status": "ok",
                        "predictions": [
                            {"label": "LOW VOLTAGE QRS"},
                            {"label": "SINUS RHYTHM"},
                        ],
                    }
                ]
            }
        ]

        assert select_ekg_waveform_attention_probe_regions(result) == [
            (
                "waveform_attention_precordial_leads",
                RegionRect(0.0, 0.5, 1.0, 0.5),
            )
        ]

    @pytest.mark.parametrize(
        "label",
        [
            "NONSPECIFIC T WAVE ABNORMALITY",
            "NONSPECIFIC ST-T ABNORMALITY",
            "REPOLARIZATION ABNORMALITY",
        ],
    )
    def test_ranked_st_t_candidate_routes_attention_to_precordials(self, label):
        result = _ekg_row_layout_result([])
        result.analysis_trace = [
            {
                "tool_audit": [
                    {
                        "tool": "ecg_founder_analyze_waveform",
                        "status": "ok",
                        "predictions": [
                            {"label": "ABNORMAL ECG"},
                            {"label": label},
                            {"label": "SINUS RHYTHM"},
                        ],
                    }
                ]
            }
        ]

        assert select_ekg_waveform_attention_probe_regions(result) == [
            (
                "waveform_attention_precordial_leads",
                RegionRect(0.0, 0.5, 1.0, 0.5),
            )
        ]


class TestEkgWaveformRhythmConflictGuard:
    @staticmethod
    def _candidate(
        *,
        regularity_signal: str = "irregular",
        labels: tuple[str, ...] = ("ATRIAL FIBRILLATION", "ABNORMAL ECG"),
    ) -> AnalysisResult:
        result = _ekg_row_layout_result([])
        result.summary = "Normal sinus rhythm; no acute abnormality."
        result.analysis_trace = [
            {
                "stage": "coarse",
                "tool_audit": [
                    {
                        "tool": "ecg_founder_analyze_waveform",
                        "status": "ok",
                        "predictions": [
                            {"label": label, "probability": 0.9} for label in labels
                        ],
                        "response_evidence": {
                            "rhythm_measurement": {
                                "method": "lead_II_qrs_energy_v1",
                                "lead": "II",
                                "status": "ok",
                                "diagnostic_scope": "rhythm_regularity_only",
                                "rr_interval_count": 6,
                                "rr_cv": 0.108,
                                "successive_rr_diff_over_80ms_fraction": 0.417,
                                "regularity_signal": regularity_signal,
                            }
                        },
                    }
                ],
            }
        ]
        return result

    def test_escalates_dual_signal_conflict_without_forcing_diagnosis(self):
        guarded = apply_ekg_waveform_rhythm_conflict_guard(self._candidate())

        finding = guarded.findings[-1]
        assert guarded.severity is Severity.WARNING
        assert guarded.incomplete is True
        assert guarded.review_required is True
        assert "atrial fibrillation is not excluded" in guarded.summary.casefold()
        assert finding.id == "waveform-rhythm-conflict"
        assert finding.source == "waveform_rhythm_conflict_guard"
        assert finding.bboxes == []
        assert guarded.checklist["rhythm"].status is Severity.WARNING
        guard_trace = guarded.analysis_trace[-1]
        assert guard_trace["stage"] == "waveform_rhythm_guardrail"
        assert guard_trace["diagnosis_forced"] is False

        grounded = apply_unlocalized_ekg_grounding_guard(guarded)
        assert grounded.severity is Severity.WARNING
        assert grounded.findings[-1].severity is Severity.INFO
        assert grounded.findings[-1].confidence == "low"

    def test_upgrades_existing_localized_rhythm_finding_without_duplicate(self):
        candidate = self._candidate()
        marker = RegionRect(0.08, 0.08, 0.28, 0.09)
        candidate.findings = [
            _finding(
                "rhythm1",
                Severity.INFO,
                marker,
                label="Irregular rhythm; atrial fibrillation not excluded",
                detail="Irregular R-R intervals on the lead-II crop.",
            )
        ]
        candidate.severity = Severity.INFO

        guarded = apply_ekg_waveform_rhythm_conflict_guard(candidate)

        assert len(guarded.findings) == 1
        assert guarded.findings[0].id == "rhythm1"
        assert guarded.findings[0].severity is Severity.WARNING
        assert guarded.findings[0].bboxes == [marker]
        assert guarded.analysis_trace[-1]["reconciled_finding_id"] == "rhythm1"

    @pytest.mark.parametrize(
        ("regularity_signal", "labels"),
        [
            ("regular", ("ATRIAL FIBRILLATION", "ABNORMAL ECG")),
            ("irregular", ("NORMAL SINUS RHYTHM", "NORMAL ECG")),
        ],
    )
    def test_requires_both_quantified_irregularity_and_ranked_rhythm_signal(
        self,
        regularity_signal: str,
        labels: tuple[str, ...],
    ):
        candidate = self._candidate(
            regularity_signal=regularity_signal,
            labels=labels,
        )

        assert apply_ekg_waveform_rhythm_conflict_guard(candidate) is candidate


class TestEkgVentricularRunEvidenceGuard:
    @pytest.mark.parametrize(
        "detail",
        [
            "Two separate early broad complexes; intervening intrinsic beats "
            "exclude a consecutive ventricular run.",
            "No ventricular tachycardia is visible.",
            "Without evidence of ventricular tachycardia.",
            "Ventricular tachycardia is absent.",
            "VT was not observed.",
            "Ventricular run excluded by intervening intrinsic beats.",
            "Cannot exclude ventricular tachycardia.",
            "No unequivocal pacing spike or ventricular run.",
        ],
    )
    def test_negated_or_uncertain_run_is_not_promoted_to_new_candidate(self, detail):
        finding = _finding(
            "ectopy",
            Severity.WARNING,
            RegionRect(0.2, 0.1, 0.05, 0.04),
            label="Intermittent abnormal broad complexes",
            detail=detail,
        )
        result = _ekg_row_layout_result([finding])
        result.layout["format"] = "12lead_12x1"

        assert apply_ekg_ventricular_run_evidence_guard(result) is result

    @pytest.mark.parametrize(
        "detail",
        [
            "No ectopy. Ventricular tachycardia is present.",
            "No atrial fibrillation, but ventricular tachycardia is present.",
            "VT excluded in the first segment; ventricular run in the second.",
        ],
    )
    def test_narrative_differential_does_not_reclassify_a_nonvt_label(self, detail):
        finding = _finding(
            "rhythm",
            Severity.CRITICAL,
            RegionRect(0.2, 0.1, 0.05, 0.04),
            label="Wide-complex rhythm",
            detail=detail,
        )
        result = _ekg_row_layout_result([finding])
        result.layout["format"] = "12lead_12x1"

        guarded = apply_ekg_ventricular_run_evidence_guard(result)

        assert guarded is result
        assert guarded.findings[0].severity is Severity.CRITICAL

    def test_same_synchronized_event_across_leads_counts_once_and_qualifies_claim(
        self,
    ):
        boxes = [
            RegionRect(0.20, 0.02, 0.04, 0.04),
            RegionRect(0.20, 0.18, 0.04, 0.04),
            RegionRect(0.205, 0.52, 0.04, 0.04),
        ]
        finding = Finding(
            id="vt",
            regions=["lead_I", "lead_III", "lead_V1"],
            label="Ventricular tachycardia",
            detail="Synchronized wide-complex run.",
            severity=Severity.CRITICAL,
            bboxes=boxes,
        )
        result = _ekg_row_layout_result([finding])
        result.layout["format"] = "12lead_12x1"
        result.summary = "Ventricular tachycardia."
        result.severity = Severity.CRITICAL
        result.checklist["rhythm"] = ChecklistItem(
            value="ventricular tachycardia",
            status=Severity.CRITICAL,
        )

        guarded = apply_ekg_ventricular_run_evidence_guard(result)

        guarded_finding = guarded.findings[0]
        assert guarded_finding.label == (
            "Unresolved potentially time-critical ventricular rhythm candidate"
        )
        assert guarded_finding.bboxes == boxes
        assert guarded_finding.severity is Severity.CRITICAL
        assert guarded.severity is Severity.CRITICAL
        assert "ventricular tachycardia." not in guarded.summary.casefold()
        assert "unresolved ventricular rhythm candidate" in (
            guarded.checklist["rhythm"].value
        )
        assert guarded.incomplete is True
        assert guarded.review_required is True
        event = guarded.analysis_trace[-1]
        assert event["unique_timestamp_count"] == 1
        assert event["required_timestamp_count"] == 3
        assert event["study_severity_preserved"] == "critical"
        assert event["diagnosis_forced"] is False

    def test_three_distinct_tight_timestamps_do_not_trigger_minimum_count_guard(self):
        finding = Finding(
            id="vt",
            regions=["lead_II"],
            label="Ventricular tachycardia",
            detail="Three localized wide-complex events.",
            severity=Severity.CRITICAL,
            bboxes=[
                RegionRect(0.10, 0.09, 0.04, 0.04),
                RegionRect(0.35, 0.09, 0.04, 0.04),
                RegionRect(0.60, 0.09, 0.04, 0.04),
            ],
        )
        result = _ekg_row_layout_result([finding])
        result.layout["format"] = "12lead_12x1"

        assert apply_ekg_ventricular_run_evidence_guard(result) is result

    def test_broad_context_boxes_never_count_as_timed_events(self):
        finding = Finding(
            id="run",
            regions=["lead_II", "lead_V1"],
            label="VT",
            detail="Wide-complex run.",
            severity=Severity.CRITICAL,
            bboxes=[
                RegionRect(0.0, 0.08, 1.0, 0.08),
                RegionRect(0.0, 0.50, 1.0, 0.08),
            ],
        )
        result = _ekg_row_layout_result([finding])
        result.layout["format"] = "12lead_12x1"

        guarded = apply_ekg_ventricular_run_evidence_guard(result)

        assert guarded.analysis_trace[-1]["unique_timestamp_count"] == 0
        assert guarded.findings[0].label.startswith("Unresolved")


# ── select_zoom_targets ──────────────────────────────────────────────


class TestSelectZoomTargets:
    def test_separate_broad_events_keep_intervening_beats_and_all_leads(self):
        finding = Finding(
            id="events",
            label="Intermittent abnormal broad complexes",
            detail="Two separate abnormal complexes with intervening intrinsic beats.",
            severity=Severity.WARNING,
            regions=["lead_II", "lead_V2"],
            bboxes=[
                RegionRect(0.07, 0.09, 0.06, 0.07),
                RegionRect(0.62, 0.59, 0.06, 0.08),
            ],
        )
        result = _ekg_row_layout_result([finding])
        result.layout["format"] = "12lead_12x1"

        crop = select_hypothesis_crop_region(
            finding,
            modality=Modality.EKG,
            layout=result.layout,
        )

        assert crop.x <= 0.07
        assert crop.x + crop.w >= 0.68
        assert crop.y == 0.0
        assert crop.h == 1.0

    def test_skips_normal_but_includes_info_after_abnormal(self):
        box = RegionRect(x=0.1, y=0.1, w=0.1, h=0.1)
        res = _result(
            [
                _finding("a", Severity.NORMAL, box),
                _finding("i", Severity.INFO, box),
                _finding("w", Severity.WARNING, box),
            ]
        )
        assert [t.id for t in select_zoom_targets(res, max_targets=3)] == ["w", "i"]

    def test_skips_findings_without_bbox(self):
        res = _result([_finding("a", Severity.CRITICAL, None)])
        assert select_zoom_targets(res, max_targets=3) == []

    def test_routes_unlocalized_ekg_temporal_finding_to_lead_ii(self):
        res = _ekg_row_layout_result(
            [
                _finding(
                    "rhythm",
                    Severity.CRITICAL,
                    None,
                    label="Tachyarrhythmia",
                )
            ]
        )

        targets = select_zoom_targets(res, max_targets=1)

        assert [target.id for target in targets] == ["rhythm"]
        assert targets[0].bboxes == [RegionRect(0.0, 1 / 12, 1.0, 1 / 12)]

    def test_critical_prioritized_over_warning(self):
        box = RegionRect(x=0.1, y=0.1, w=0.1, h=0.1)
        res = _result(
            [
                _finding("w", Severity.WARNING, box),
                _finding("c", Severity.CRITICAL, box),
            ]
        )
        targets = select_zoom_targets(res, max_targets=1)
        assert [t.id for t in targets] == ["c"]

    def test_respects_max_targets(self):
        box = RegionRect(x=0.1, y=0.1, w=0.1, h=0.1)
        res = _result([_finding(str(i), Severity.WARNING, box) for i in range(5)])
        assert len(select_zoom_targets(res, max_targets=2)) == 2

    def test_zero_max_targets(self):
        box = RegionRect(x=0.1, y=0.1, w=0.1, h=0.1)
        res = _result([_finding("a", Severity.CRITICAL, box)])
        assert select_zoom_targets(res, max_targets=0) == []


class TestCriticalTriagePlanning:
    def test_requires_specific_structured_localizable_finding(self):
        box = RegionRect(0.1, 0.1, 0.2, 0.2)
        result = _result(
            [
                _finding(
                    "specific",
                    Severity.CRITICAL,
                    box,
                    label="Tension pneumothorax",
                    detail="Pleural line with mediastinal shift.",
                ),
                _finding(
                    "generic",
                    Severity.CRITICAL,
                    box,
                    label="Critical finding.",
                    detail="Urgent abnormality.",
                ),
                _finding(
                    "empty-detail",
                    Severity.CRITICAL,
                    box,
                    label="Acute process",
                ),
                _finding(
                    "unlocalized",
                    Severity.CRITICAL,
                    None,
                    label="Large effusion",
                    detail="Possible tension physiology.",
                ),
            ]
        )
        result.severity = Severity.CRITICAL

        candidates = select_critical_triage_candidates(result)

        assert [finding.id for finding in candidates] == ["specific"]

    def test_temporal_support_prefers_nonredundant_lead_ii(self):
        critical = _finding(
            "vt",
            Severity.CRITICAL,
            RegionRect(0.0, 0.75, 1.0, 0.08),
            label="Ventricular tachycardia",
            detail="Synchronized wide-complex run.",
        )
        result = _ekg_row_layout_result([critical])

        support = select_ekg_critical_support_probe(
            result,
            [critical],
            [critical.bboxes[0]],
        )

        assert support == (
            "lead_II",
            RegionRect(0.0, 1 / 12, 1.0, 1 / 12),
            "critical_temporal_crosscheck",
        )

    def test_hyperkalemia_support_is_morphology_not_territorial(self):
        critical = _finding(
            "hyperk",
            Severity.CRITICAL,
            RegionRect(0.2, 0.55, 0.2, 0.08),
            label="Possible hyperkalemia",
            detail="Tall peaked T waves with QRS widening.",
        )
        result = _ekg_row_layout_result([critical])

        support = select_ekg_critical_support_probe(
            result,
            [critical],
            [critical.bboxes[0]],
        )

        assert support is not None
        assert support[2] == "critical_morphology_crosslead_check"

    def test_guard_marks_only_deferred_normal_axes_unassessed(self):
        critical = _finding(
            "stemi",
            Severity.CRITICAL,
            RegionRect(0.1, 0.6, 0.2, 0.1),
            label="Anterior STEMI pattern",
            detail="Contiguous ST elevation in V2-V4.",
        )
        result = _ekg_row_layout_result([critical])
        result.checklist = {
            "rhythm": ChecklistItem(value="sinus", status=Severity.NORMAL),
            "st_segment": ChecklistItem(value="elevated", status=Severity.CRITICAL),
        }

        guarded = apply_critical_triage_guard(
            result,
            [critical],
            phase="unit_test",
        )

        assert guarded.checklist["st_segment"].status is Severity.CRITICAL
        assert guarded.checklist["rhythm"].value == (
            "not_assessed_due_to_critical_triage"
        )
        assert guarded.checklist["rhythm"].status is Severity.INFO
        assert guarded.incomplete is True
        assert guarded.review_required is True

    def test_final_guard_preserves_explicitly_resumed_normal_axes(self):
        critical = _finding(
            "vt",
            Severity.CRITICAL,
            RegionRect(0.1, 0.1, 0.2, 0.05),
            label="Ventricular tachycardia",
            detail="Possible synchronized wide-complex run.",
        )
        result = _ekg_row_layout_result([])
        result.severity = Severity.INFO
        result.checklist = _complete_ekg_checklist()

        guarded = apply_critical_triage_guard(
            result,
            [critical],
            phase="final_output",
        )

        assert guarded.severity is Severity.INFO
        assert all(
            item.value.startswith("explicitly assessed")
            for item in guarded.checklist.values()
        )
        event = guarded.analysis_trace[-1]
        assert event["stage"] == "critical_triage_resume_guard"
        assert event["status"] == "deferred_axes_resumed_on_original_study"
        assert event["unresolved_axes"] == []
        assert event["clinical_status_inferred"] is False
        assert guarded.incomplete is False
        assert guarded.incomplete_reasons == []
        assert guarded.review_required is False
        assert guarded.review_reasons == []

    def test_unresolved_axes_suppress_broad_normal_claim_with_retained_critical(self):
        critical = _finding(
            "vt",
            Severity.CRITICAL,
            RegionRect(0.1, 0.1, 0.2, 0.05),
            label="Ventricular tachycardia",
            detail="Possible synchronized wide-complex run.",
        )
        result = _ekg_row_layout_result([critical])
        result.severity = Severity.CRITICAL
        result.summary = "Normal ECG with no acute abnormality."
        result.checklist = _complete_ekg_checklist()
        result.checklist["axis"] = ChecklistItem(
            value="not_assessed_due_to_critical_triage; otherwise WNL",
            status=Severity.INFO,
        )

        guarded = apply_critical_triage_guard(
            result,
            [critical],
            phase="final_output",
        )

        assert guarded.summary == (
            "At least one critical finding remains, but deferred ECG checklist axes "
            "remain unassessed on the original study."
        )
        assert guarded.checklist["axis"] == ChecklistItem(
            value="not_assessed_due_to_critical_triage",
            status=Severity.INFO,
        )
        repair = next(
            event
            for event in guarded.analysis_trace
            if event.get("stage") == "critical_triage_cross_field_guard"
        )
        assert repair["retained_critical_ids"] == ["vt"]
        assert {item["field"] for item in repair["repairs"]} == {
            "summary",
            "checklist.axis",
        }

    def test_resumed_axes_clear_only_triage_limitation_not_other_review_reason(self):
        critical = _finding(
            "vt",
            Severity.CRITICAL,
            RegionRect(0.1, 0.1, 0.2, 0.05),
            label="Possible ventricular tachycardia",
            detail="Synchronized wide-complex candidate.",
        )
        draft = apply_critical_triage_guard(
            _ekg_row_layout_result([critical]),
            [critical],
            phase="before_finalization",
        )
        other_reason = "Image calibration remains unavailable."
        final = dataclasses.replace(
            draft,
            checklist=_complete_ekg_checklist(),
            incomplete_reasons=[*draft.incomplete_reasons, other_reason],
            review_reasons=[*draft.review_reasons, other_reason],
        )

        guarded = apply_critical_triage_guard(
            final,
            [critical],
            phase="final_output",
        )

        assert guarded.incomplete is True
        assert guarded.incomplete_reasons == [other_reason]
        assert guarded.review_required is True
        assert guarded.review_reasons == [other_reason]

    def test_final_guard_flags_unresolved_deferred_axes_without_diagnosis(self):
        critical = _finding(
            "vt",
            Severity.CRITICAL,
            RegionRect(0.1, 0.1, 0.2, 0.05),
            label="Ventricular tachycardia",
            detail="Possible synchronized wide-complex run.",
        )
        draft = apply_critical_triage_guard(
            _ekg_row_layout_result([critical]),
            [critical],
            phase="before_finalization",
        )
        downgraded = dataclasses.replace(
            draft,
            severity=Severity.INFO,
            findings=[],
        )

        guarded = apply_critical_triage_guard(
            downgraded,
            [critical],
            phase="final_output",
        )

        assert guarded.severity is Severity.INFO
        event = guarded.analysis_trace[-1]
        assert event["status"] == "deferred_axes_incomplete_fail_safe"
        assert event["retained_critical_ids"] == []
        assert event["unresolved_axes"]
        assert event["clinical_severity_changed"] is False
        assert event["diagnosis_forced"] is False

    def test_retracted_critical_cannot_close_unassessed_axes_as_normal_or_absent(self):
        critical = _finding(
            "vt",
            Severity.CRITICAL,
            RegionRect(0.1, 0.1, 0.2, 0.05),
            label="Ventricular tachycardia",
            detail="Possible synchronized wide-complex run.",
        )
        draft = apply_critical_triage_guard(
            _ekg_row_layout_result([critical]),
            [critical],
            phase="before_finalization",
        )
        contradictory_checklist = dict(draft.checklist)
        contradictory_checklist["axis"] = ChecklistItem(
            value="not_assessed_due_to_critical_triage; otherwise WNL and absent",
            status=Severity.INFO,
        )
        retracted = dataclasses.replace(
            draft,
            summary="Normal ECG; all findings absent.",
            severity=Severity.INFO,
            findings=[],
            checklist=contradictory_checklist,
        )

        guarded = apply_critical_triage_guard(
            retracted,
            [critical],
            phase="final_output",
        )

        assert guarded.summary == (
            "The critical candidate was not retained; deferred ECG checklist axes "
            "remain unassessed on the original study."
        )
        assert guarded.checklist["axis"] == ChecklistItem(
            value="not_assessed_due_to_critical_triage",
            status=Severity.INFO,
        )
        repair = next(
            event
            for event in guarded.analysis_trace
            if event.get("stage") == "critical_triage_cross_field_guard"
        )
        assert repair["status"] == "contradictory_negative_claims_suppressed"
        assert {item["field"] for item in repair["repairs"]} == {
            "summary",
            "checklist.axis",
        }
        assert repair["clinical_status_inferred"] is False

    @pytest.mark.parametrize(
        "value",
        [
            "not_assessable",
            "Cannot assess from this image",
            "indeterminate",
            "unknown",
        ],
    )
    def test_final_guard_treats_nonassessable_semantics_as_unresolved(self, value):
        critical = _finding(
            "vt",
            Severity.CRITICAL,
            RegionRect(0.1, 0.1, 0.2, 0.05),
            label="Ventricular tachycardia",
            detail="Possible synchronized wide-complex run.",
        )
        result = _ekg_row_layout_result([])
        result.checklist = _complete_ekg_checklist()
        result.checklist["axis"] = ChecklistItem(value=value, status=Severity.INFO)

        guarded = apply_critical_triage_guard(
            result,
            [critical],
            phase="final_output",
        )

        event = guarded.analysis_trace[-1]
        assert event["status"] == "deferred_axes_incomplete_fail_safe"
        assert "axis" in event["unresolved_axes"]
        assert guarded.severity is result.severity


class TestRefinementDeltaContract:
    def test_targeted_actions_require_target_id(self):
        with pytest.raises(ValueError):
            RefinementDelta(RefinementAction.RETRACT)

    def test_add_requires_finding(self):
        with pytest.raises(ValueError):
            RefinementDelta(RefinementAction.ADD)

    def test_zero_size_addition_is_rejected(self):
        finding = _finding(
            "new",
            Severity.WARNING,
            RegionRect(0.2, 0.2, 0.0, 0.2),
        )
        out = apply_refinement_delta(
            [],
            RefinementDelta(RefinementAction.ADD, finding=finding),
            crop_region=RegionRect(0.1, 0.1, 0.4, 0.4),
            expected_target_id=None,
        )
        assert out == []

    def test_model_note_cannot_authorize_an_unlocalized_addition(self):
        finding = Finding(
            id="new",
            regions=["right_lung"],
            label="Semantic concern",
            detail="Model supplied no usable geometry.",
            severity=Severity.INFO,
            bboxes=[],
            notes=[
                "Semantic-only refinement retained because no image/turn-bound "
                "bbox validation receipt was observed."
            ],
        )

        out = apply_refinement_delta(
            [],
            RefinementDelta(RefinementAction.ADD, finding=finding),
            crop_region=RegionRect(0.1, 0.1, 0.4, 0.4),
            expected_target_id=None,
        )

        assert out == []

    @pytest.mark.parametrize("indexes", [(-1,), (1,), (0, 0), (True,)])
    def test_unlocalized_provenance_rejects_invalid_delta_indexes(self, indexes):
        delta = RefinementDelta(
            RefinementAction.ADD,
            finding=_finding("new", Severity.INFO, None),
        )

        with pytest.raises(ValueError, match="invalid unlocalized"):
            RefinementResult(
                (delta,),
                unlocalized_missing_receipt_delta_indexes=indexes,
            )

    def test_overflowing_child_is_clamped_inside_crop_and_roi(self):
        finding = _finding(
            "new",
            Severity.WARNING,
            RegionRect(0.8, 0.8, 0.5, 0.5),
        )
        crop = RegionRect(0.7, 0.7, 0.3, 0.3)
        out = apply_refinement_delta(
            [],
            RefinementDelta(RefinementAction.ADD, finding=finding),
            crop_region=crop,
            expected_target_id=None,
        )

        bbox = out[0].bboxes[0]
        assert bbox.w > 0.0
        assert bbox.h > 0.0
        assert bbox.x + bbox.w <= crop.x + crop.w
        assert bbox.y + bbox.h <= crop.y + crop.h
        assert bbox.x + bbox.w <= 1.0
        assert bbox.y + bbox.h <= 1.0


# ── MultiPassInterpreter orchestration ───────────────────────────────


class _FakeAnalyzer(VisionAnalyzerService):
    """Returns a scripted result per call; records the images it received."""

    def __init__(self, results: list[AnalysisResult]) -> None:
        self._results = list(results)
        self.images: list[str] = []

    async def analyze(self, image_base64, modality, valid_regions):
        self.images.append(image_base64)
        return self._results.pop(0)

    async def chat(self, message):  # pragma: no cover - unused here
        return ""

    async def connect(self):  # pragma: no cover
        return None

    async def disconnect(self):  # pragma: no cover
        return None

    def is_connected(self):  # pragma: no cover
        return True


class _FlakyZoomAnalyzer(VisionAnalyzerService):
    """Returns the coarse result, then fails before a successful zoom read."""

    def __init__(
        self,
        coarse: AnalysisResult,
        zoom: AnalysisResult,
        failures_before_success: int,
    ) -> None:
        self._coarse = coarse
        self._zoom = zoom
        self._failures_before_success = failures_before_success
        self.calls = 0
        self.zoom_calls = 0

    async def analyze(self, image_base64, modality, valid_regions):
        self.calls += 1
        if self.calls == 1:
            return self._coarse
        self.zoom_calls += 1
        if self.zoom_calls <= self._failures_before_success:
            raise TimeoutError("transient gateway timeout")
        return self._zoom

    async def chat(self, message):  # pragma: no cover - unused here
        return ""

    async def connect(self):  # pragma: no cover
        return None

    async def disconnect(self):  # pragma: no cover
        return None

    def is_connected(self):  # pragma: no cover
        return True


class _HypothesisAwareAnalyzer(_FakeAnalyzer):
    """Uses the optional refine capability and records its turn context."""

    def __init__(
        self,
        coarse: AnalysisResult,
        refinements: list[RefinementResult],
    ) -> None:
        super().__init__([coarse])
        self._refinements = list(refinements)
        self.refine_calls: list[dict] = []

    async def refine(
        self,
        image_base64,
        modality,
        valid_regions,
        *,
        hypothesis,
        crop_region,
        probe_id="",
        crop_lead_regions=None,
    ):
        self.refine_calls.append(
            {
                "image": image_base64,
                "modality": modality,
                "valid_regions": valid_regions,
                "hypothesis": hypothesis,
                "crop_region": crop_region,
                "probe_id": probe_id,
                "crop_lead_regions": crop_lead_regions,
            }
        )
        return self._refinements.pop(0)


class _FinalizingAnalyzer(_HypothesisAwareAnalyzer):
    def __init__(
        self,
        coarse: AnalysisResult,
        refinements: list[RefinementResult],
        final: AnalysisResult,
    ) -> None:
        super().__init__(coarse, refinements)
        self._final = final
        self.finalize_calls: list[dict] = []

    async def finalize(
        self,
        image_base64,
        modality,
        valid_regions,
        *,
        draft,
        refinement_trace,
    ):
        self.finalize_calls.append(
            {
                "image": image_base64,
                "modality": modality,
                "valid_regions": valid_regions,
                "draft": draft,
                "refinement_trace": refinement_trace,
            }
        )
        return self._final


class _FailingFinalizingAnalyzer(_HypothesisAwareAnalyzer):
    async def finalize(
        self,
        image_base64,
        modality,
        valid_regions,
        *,
        draft,
        refinement_trace,
    ):
        raise TimeoutError("bounded final turn expired")


class _RecordingCropper:
    """Fake cropper: records crop regions, returns a marker string."""

    def __init__(self) -> None:
        self.regions: list[RegionRect] = []
        self.images: list[str] = []

    def __call__(self, image_base64: str, region: RegionRect) -> str:
        self.images.append(image_base64)
        self.regions.append(region)
        return f"crop::{region.x:.3f},{region.y:.3f}"


class _RowEvidenceCropper(_RecordingCropper):
    def ekg_row_strip_evidence(self, _image_base64: str) -> dict[str, object]:
        return {
            "method": "local_black_ink_row_periodicity_v2",
            "status": "ok",
            "is_12_row_strip": True,
            "detected_row_count": 12,
            "consistent_gap_count": 10,
        }


class _SlowCoarseAnalyzer(_FakeAnalyzer):
    async def analyze(self, image_base64, modality, valid_regions):
        await asyncio.sleep(0.03)
        return await super().analyze(image_base64, modality, valid_regions)


def test_final_report_can_revise_clinical_fields_with_locked_geometry() -> None:
    box = RegionRect(0.2, 0.2, 0.2, 0.2)
    draft_finding = dataclasses.replace(
        _finding("f1", Severity.CRITICAL, box, detail="possible acute pattern"),
        regions=["lead_V2"],
        source="crop_refine",
    )
    draft = _result([draft_finding])
    final_finding = dataclasses.replace(
        draft_finding,
        label="Benign repolarization variant",
        detail="Concave morphology without convincing reciprocal change.",
        severity=Severity.INFO,
        confidence="moderate",
    )
    final = _result([final_finding])
    final.severity = Severity.INFO
    final.summary = "No convincing acute ischemic pattern."

    result = reconcile_final_report(draft, final)

    assert result.severity is Severity.INFO
    assert result.findings[0].label == "Benign repolarization variant"
    assert result.findings[0].bboxes == [box]
    assert result.findings[0].regions == ["lead_V2"]
    assert result.findings[0].source == "crop_refine"
    disposition = result.analysis_trace[-1]
    assert disposition["status"] == "revised"
    assert disposition["geometry_locked"] is True


@pytest.mark.parametrize(
    "limitation", ["", "Lead V1 missing", "Preliminary triage; review pending"]
)
def test_final_report_resolves_only_exact_workflow_marker(limitation):
    from medical_image_harness.profiles import get_active_registry
    from medical_image_harness.protocols import (
        PENDING_MULTIPASS_REASON,
    )

    draft = _result([])
    draft.incomplete = True
    draft.incomplete_reasons = [
        PENDING_MULTIPASS_REASON,
        *([limitation] if limitation else []),
    ]
    final = _result([])
    final.checklist = {
        key: ChecklistItem(value="assessed", status=Severity.NORMAL)
        for key in get_active_registry().resolve(final.modality.value).checklist_keys
    }
    result = reconcile_final_report(draft, final)
    assert result.incomplete_reasons == ([limitation] if limitation else [])
    assert result.incomplete is bool(limitation)
    assert result.analysis_trace[-1]["stage"] == "workflow_progress"
    assert result.analysis_trace[-1]["clinical_limitations_removed"] is False


def test_incomplete_final_checklist_cannot_resolve_workflow_marker():
    from medical_image_harness.protocols import (
        PENDING_MULTIPASS_REASON,
    )

    draft = _result([])
    draft.incomplete = True
    draft.incomplete_reasons = [PENDING_MULTIPASS_REASON]
    final = _result([])
    final.checklist = {"rhythm": ChecklistItem(value="sinus", status=Severity.NORMAL)}
    result = reconcile_final_report(draft, final)
    assert result.incomplete
    assert result.incomplete_reasons == [PENDING_MULTIPASS_REASON]


def test_final_limitations_remain_after_workflow_resolution():
    from medical_image_harness.profiles import get_active_registry
    from medical_image_harness.protocols import (
        PENDING_MULTIPASS_REASON,
    )

    draft = _result([])
    draft.incomplete = True
    draft.incomplete_reasons = [PENDING_MULTIPASS_REASON]
    final = _result([])
    final.incomplete = True
    final.incomplete_reasons = ["Lead V6 absent"]
    final.checklist = {
        key: ChecklistItem(value="not_assessable", status=Severity.INFO)
        for key in get_active_registry().resolve(final.modality.value).checklist_keys
    }
    result = reconcile_final_report(draft, final)
    assert result.incomplete
    assert result.incomplete_reasons == ["Lead V6 absent"]


def test_final_report_rejects_added_ids_and_moved_geometry() -> None:
    box = RegionRect(0.2, 0.2, 0.2, 0.2)
    draft = _result([_finding("f1", Severity.WARNING, box)])

    with pytest.raises(ValueError, match="cannot add findings"):
        reconcile_final_report(
            draft,
            _result([_finding("invented", Severity.WARNING, box)]),
        )

    with pytest.raises(ValueError, match="cannot change bboxes"):
        reconcile_final_report(
            draft,
            _result(
                [
                    _finding(
                        "f1",
                        Severity.WARNING,
                        RegionRect(0.3, 0.2, 0.2, 0.2),
                    )
                ]
            ),
        )


def test_final_report_normalizes_bounded_decimal_rounding_to_draft_geometry() -> None:
    draft_box = RegionRect(0.123456, 0.234567, 0.2, 0.1)
    final_box = RegionRect(0.1235, 0.2346, 0.2, 0.1)
    draft = _result([_finding("f1", Severity.WARNING, draft_box)])
    final = _result([_finding("f1", Severity.WARNING, final_box)])

    result = reconcile_final_report(draft, final)

    assert result.findings[0].bboxes == [draft_box]
    disposition = result.analysis_trace[-1]
    assert disposition["status"] == "retained"
    assert disposition["max_bbox_coordinate_drift"] == pytest.approx(0.000044)
    assert disposition["geometry_locked"] is True


class _CoarseAwareAnalyzer(_FakeAnalyzer):
    def __init__(self, results: list[AnalysisResult]) -> None:
        super().__init__(results)
        self.coarse_calls = 0

    async def analyze(self, image_base64, modality, valid_regions):
        raise AssertionError("full analysis should not be used for the coarse turn")

    async def analyze_coarse(self, image_base64, modality, valid_regions):
        self.coarse_calls += 1
        self.images.append(image_base64)
        return self._results.pop(0)


class _SlowRefinementAnalyzer(_HypothesisAwareAnalyzer):
    async def refine(self, *args, **kwargs):
        await asyncio.sleep(0.03)
        return await super().refine(*args, **kwargs)


@pytest.mark.asyncio
class TestMultiPassInterpreter:
    async def test_observed_crop_duration_reserves_time_for_finalization(
        self, monkeypatch
    ):
        # Test scheduling policy with the injected clock, not a 20 ms Windows
        # timer race. Real timeout/cancellation behavior has separate tests.
        observed_time = [0.0]
        monkeypatch.setattr(
            "medical_image_harness.multipass._SLA_RETURN_BUFFER_SEC",
            0.1,
        )

        class SlowFinalizingAnalyzer(_FinalizingAnalyzer):
            async def refine(self, *args, **kwargs):
                observed_time[0] += 3.0
                return await super().refine(*args, **kwargs)

        coarse = _result(
            [
                _finding("first", Severity.WARNING, RegionRect(0.1, 0.1, 0.1, 0.1)),
                _finding("second", Severity.WARNING, RegionRect(0.7, 0.7, 0.1, 0.1)),
            ]
        )
        analyzer = SlowFinalizingAnalyzer(
            coarse,
            [RefinementResult(), RefinementResult()],
            coarse,
        )
        interpreter = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            max_zoom_targets=2,
            initial_response_sla_sec=2.0,
            first_refinement_sla_sec=6.0,
            total_analysis_sla_sec=15.0,
            finalization_reserve_sec=10.0,
            min_followup_budget_sec=0.1,
            clock=lambda: observed_time[0],
        )

        result = await interpreter.interpret("img", Modality.CXR, [])

        assert len(analyzer.refine_calls) == 1
        assert len(analyzer.finalize_calls) == 1
        skipped = next(
            event
            for event in result.analysis_trace
            if event.get("status") == "total_deadline_reserve_reached"
        )
        assert skipped["previous_refinement_sec"] == 3.0
        assert skipped["minimum_start_budget_sec"] == 3.75

    async def test_prefers_compact_coarse_capability_when_available(self):
        analyzer = _CoarseAwareAnalyzer([_result([])])
        interp = MultiPassInterpreter(analyzer, _RecordingCropper())

        result = await interp.interpret("img", Modality.CXR, [])

        assert analyzer.coarse_calls == 1
        assert analyzer.images == ["img"]
        assert result.severity is Severity.NORMAL

    async def test_initial_response_deadline_is_hard_and_auditable(self):
        analyzer = _SlowCoarseAnalyzer([_result([])])
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            initial_response_sla_sec=0.01,
            first_refinement_sla_sec=0.02,
            total_analysis_sla_sec=0.05,
            finalization_reserve_sec=0.01,
            min_followup_budget_sec=0.001,
        )

        with pytest.raises(AnalysisSlaTimeout) as captured:
            await interp.interpret("img", Modality.CXR, [])

        assert captured.value.stage == "initial_response"
        assert captured.value.audit_trace()["status"] == "required_stage_timeout"

    async def test_first_crop_timeout_returns_coarse_result_for_review(self):
        box = RegionRect(0.2, 0.2, 0.2, 0.2)
        coarse = _result([_finding("f1", Severity.WARNING, box)])
        analyzer = _SlowRefinementAnalyzer(
            coarse,
            [RefinementResult()],
        )
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            zoom_retry_attempts=0,
            initial_response_sla_sec=0.01,
            first_refinement_sla_sec=0.02,
            total_analysis_sla_sec=0.06,
            max_refinement_turn_sec=0.02,
            finalization_reserve_sec=0.01,
            min_followup_budget_sec=0.001,
        )

        result = await interp.interpret("img", Modality.CXR, [])

        receipt = result.analysis_trace[-1]
        assert receipt["stage"] == "analysis_sla"
        assert receipt["met"]["initial_response"] is True
        assert receipt["met"]["first_crop_refinement"] is False
        assert receipt["met"]["total"] is True
        assert receipt["timings_ms"]["first_crop_created"] is not None
        assert result.incomplete is True
        assert result.review_required is True

    async def test_successful_pass_records_three_sla_receipts(self):
        box = RegionRect(0.2, 0.2, 0.2, 0.2)
        coarse = _result([_finding("f1", Severity.WARNING, box)])
        analyzer = _HypothesisAwareAnalyzer(coarse, [RefinementResult()])
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            initial_response_sla_sec=0.02,
            first_refinement_sla_sec=0.04,
            total_analysis_sla_sec=0.08,
            finalization_reserve_sec=0.01,
            min_followup_budget_sec=0.001,
        )

        result = await interp.interpret("img", Modality.CXR, [])

        receipt = result.analysis_trace[-1]
        assert receipt["status"] == "completed"
        assert receipt["met"] == {
            "initial_response": True,
            "first_crop_refinement": True,
            "total": True,
        }
        assert receipt["budgets_sec"] == {
            "initial_response": 0.02,
            "first_crop_refinement": 0.04,
            "total": 0.08,
        }

    async def test_ekg_systematic_probes_can_discover_a_coarse_miss(self):
        coarse = _ekg_row_layout_result([])
        discovered = _finding(
            "st_probe",
            Severity.WARNING,
            RegionRect(0.2, 0.2, 0.1, 0.1),
            label="ST-T abnormality",
        )
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [
                RefinementResult(),
                RefinementResult(
                    (
                        RefinementDelta(
                            action=RefinementAction.ADD,
                            finding=discovered,
                            rationale="visible in the precordial safety crop",
                        ),
                    )
                ),
            ],
        )
        cropper = _RecordingCropper()
        interpreter = MultiPassInterpreter(
            analyzer=analyzer,
            cropper=cropper,
            max_zoom_targets=3,
            zoom_padding=0.0,
        )

        result = await interpreter.interpret(
            "coarse-image",
            Modality.EKG,
            ["lead_I", "lead_V1"],
            source_image_base64="original-image",
            source_size_px=(1000, 720),
        )

        assert len(analyzer.refine_calls) == 2
        assert all(call["hypothesis"] is None for call in analyzer.refine_calls)
        assert cropper.images == ["original-image", "original-image"]
        assert [region.y for region in cropper.regions] == pytest.approx([0.5, 0.0])
        assert [call["probe_id"] for call in analyzer.refine_calls] == [
            "ekg_systematic_precordial_leads",
            "ekg_systematic_limb_leads",
        ]
        assert list(analyzer.refine_calls[0]["crop_lead_regions"]) == [
            "lead_V1",
            "lead_V2",
            "lead_V3",
            "lead_V4",
            "lead_V5",
            "lead_V6",
        ]
        assert list(analyzer.refine_calls[1]["crop_lead_regions"]) == [
            "lead_I",
            "lead_II",
            "lead_III",
            "lead_aVR",
            "lead_aVL",
            "lead_aVF",
        ]
        assert result.severity is Severity.WARNING
        assert [finding.id for finding in result.findings] == ["st_probe"]
        assert any(
            event.get("stage") == "systematic_assist" for event in result.analysis_trace
        )

    async def test_row_layout_is_normalized_before_systematic_crop(self):
        coarse = _ekg_row_layout_result([])
        coarse.layout["format"] = "12lead_3x4"
        for lead in coarse.layout["leads"]:
            lead["bbox"][2] = 0.25
        analyzer = _HypothesisAwareAnalyzer(coarse, [RefinementResult()])
        cropper = _RecordingCropper()
        interpreter = MultiPassInterpreter(
            analyzer,
            cropper.__call__,
            max_zoom_targets=1,
            max_ekg_systematic_probes=1,
            zoom_padding=0.0,
        )

        result = await interpreter.interpret("img", Modality.EKG, [])

        assert result.layout["format"] == "12lead_12x1"
        assert cropper.regions == [RegionRect(0.0, 0.5, 1.0, 0.5)]
        projected_v1 = analyzer.refine_calls[0]["crop_lead_regions"]["lead_V1"]
        assert (projected_v1.x, projected_v1.y, projected_v1.w) == (0.0, 0.0, 1.0)
        assert projected_v1.h == pytest.approx(1 / 6)
        assert any(
            event.get("status") == "repaired_before_refinement"
            for event in result.analysis_trace
        )

    async def test_image_evidence_repairs_eight_lead_inventory_and_stale_warning(self):
        coarse = _ekg_row_layout_result([])
        coarse.layout["format"] = "12lead_3x4"
        coarse.layout["leads"] = coarse.layout["leads"][:8]
        for index, lead in enumerate(coarse.layout["leads"]):
            lead["bbox"] = [0.0, index / 8, 1.0, 1 / 8]
        coarse.validation_warnings = [
            "EKG layout is missing visible leads: lead_V3, lead_V4, lead_V5, lead_V6"
        ]
        analyzer = _HypothesisAwareAnalyzer(coarse, [RefinementResult()])
        cropper = _RowEvidenceCropper()
        interpreter = MultiPassInterpreter(
            analyzer,
            cropper.__call__,
            ekg_row_strip_detector=cropper.ekg_row_strip_evidence,
            max_zoom_targets=1,
            max_ekg_systematic_probes=1,
            zoom_padding=0.0,
        )

        result = await interpreter.interpret("img", Modality.EKG, [])

        assert result.layout["format"] == "12lead_12x1"
        assert len(result.layout["leads"]) == 12
        assert result.validation_warnings == []
        assert any(
            event.get("stage") == "layout_signal_check"
            and event.get("status") == "confirmed_12_row_strip"
            for event in result.analysis_trace
        )

    async def test_waveform_rhythm_attention_preempts_generic_probe_budget(self):
        coarse = _ekg_row_layout_result([])
        coarse.analysis_trace = [
            {
                "tool_audit": [
                    {
                        "tool": "ecg_founder_analyze_waveform",
                        "status": "ok",
                        "predictions": [{"label": "ATRIAL FIBRILLATION"}],
                    }
                ]
            }
        ]
        analyzer = _HypothesisAwareAnalyzer(coarse, [RefinementResult()])
        cropper = _RecordingCropper()
        interpreter = MultiPassInterpreter(
            analyzer,
            cropper,
            max_zoom_targets=1,
            max_ekg_systematic_probes=1,
            zoom_padding=0.0,
        )

        await interpreter.interpret("img", Modality.EKG, [])

        assert (
            analyzer.refine_calls[0]["probe_id"]
            == "ekg_systematic_waveform_rhythm_lead_II"
        )
        assert cropper.regions == [RegionRect(0.0, 1 / 12, 1.0, 1 / 12)]

    async def test_overlapping_waveform_probe_falls_back_to_other_lead_group(self):
        coarse = _ekg_row_layout_result(
            [
                _finding(
                    "rhythm",
                    Severity.WARNING,
                    RegionRect(0.1, 0.1, 0.2, 0.04),
                    label="Premature ventricular complex",
                )
            ]
        )
        coarse.analysis_trace = [
            {
                "tool_audit": [
                    {
                        "tool": "ecg_founder_analyze_waveform",
                        "status": "ok",
                        "predictions": [{"label": "PREMATURE VENTRICULAR COMPLEX"}],
                    }
                ]
            }
        ]
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [RefinementResult(), RefinementResult()],
        )
        interpreter = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            max_zoom_targets=2,
            max_ekg_systematic_probes=1,
            zoom_padding=0.0,
        )

        await interpreter.interpret("img", Modality.EKG, [])

        assert analyzer.refine_calls[0]["hypothesis"].id == "rhythm"
        assert analyzer.refine_calls[1]["probe_id"] == "ekg_systematic_precordial_leads"

    async def test_identical_waveform_and_generic_probe_regions_are_deduplicated(self):
        coarse = _ekg_row_layout_result([])
        coarse.analysis_trace = [
            {
                "tool_audit": [
                    {
                        "tool": "ecg_founder_analyze_waveform",
                        "status": "ok",
                        "predictions": [{"label": "RIGHT BUNDLE BRANCH BLOCK"}],
                    }
                ]
            }
        ]
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [RefinementResult(), RefinementResult()],
        )
        interpreter = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            max_zoom_targets=2,
            max_ekg_systematic_probes=2,
            zoom_padding=0.0,
        )

        await interpreter.interpret("img", Modality.EKG, [])

        assert [call["probe_id"] for call in analyzer.refine_calls] == [
            "ekg_systematic_waveform_attention_precordial_leads",
            "ekg_systematic_limb_leads",
        ]

    async def test_critical_triage_verifies_each_critical_before_support(self):
        rhythm = _finding(
            "rhythm",
            Severity.CRITICAL,
            None,
            label="irregular tachycardia",
            detail="irregular rapid rhythm",
        )
        rhythm = dataclasses.replace(rhythm, regions=["rhythm_strip"])
        st_t = _finding(
            "st-t",
            Severity.CRITICAL,
            None,
            label="anterior ST-T abnormality",
            detail="precordial ST-T change",
        )
        st_t = dataclasses.replace(
            st_t,
            regions=["lead_V2", "lead_V3", "lead_V4", "lead_V5"],
        )
        coarse = _ekg_row_layout_result([rhythm, st_t])
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [RefinementResult(), RefinementResult()],
        )
        interpreter = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            max_zoom_targets=2,
            max_ekg_systematic_probes=1,
            zoom_padding=0.0,
        )

        result = await interpreter.interpret("img", Modality.EKG, [])

        assert analyzer.refine_calls[0]["hypothesis"].id == "rhythm"
        assert analyzer.refine_calls[1]["hypothesis"].id == "st-t"
        triage = next(
            event
            for event in result.analysis_trace
            if event.get("stage") == "critical_triage"
        )
        assert triage["selected_critical_ids"] == ["rhythm", "st-t"]
        assert triage["support_probe_id"] == ""
        assert triage["overflow_critical_ids"] == []
        assert triage["planned_turn_count"] == 2
        assert result.incomplete is True
        assert result.review_required is True

    async def test_cxr_critical_triage_skips_unrelated_lower_priority_crops(self):
        box = RegionRect(0.1, 0.1, 0.2, 0.2)
        critical = _finding(
            "tension",
            Severity.CRITICAL,
            box,
            label="Tension pneumothorax",
            detail="Pleural line and contralateral mediastinal shift.",
        )
        warning = _finding(
            "effusion",
            Severity.WARNING,
            RegionRect(0.6, 0.6, 0.2, 0.2),
            label="Small pleural effusion",
            detail="Blunted costophrenic angle.",
        )
        info = _finding(
            "scar",
            Severity.INFO,
            RegionRect(0.4, 0.4, 0.1, 0.1),
            label="Apical scar",
            detail="Thin linear opacity.",
        )
        coarse = _result([warning, info, critical])
        analyzer = _HypothesisAwareAnalyzer(coarse, [RefinementResult()])
        interpreter = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            max_zoom_targets=3,
            zoom_padding=0.0,
        )

        result = await interpreter.interpret("img", Modality.CXR, [])

        assert [call["hypothesis"].id for call in analyzer.refine_calls] == ["tension"]
        triage = next(
            event
            for event in result.analysis_trace
            if event.get("stage") == "critical_triage"
        )
        assert triage["skipped_lower_priority_ids"] == ["effusion", "scar"]
        assert triage["skipped_lower_priority_categories"] == {
            "warning": 1,
            "info": 1,
        }
        assert triage["support_probe_id"] == ""

    async def test_critical_stemi_uses_one_reciprocal_support_crop(self):
        critical = dataclasses.replace(
            _finding(
                "stemi",
                Severity.CRITICAL,
                RegionRect(0.1, 0.55, 0.2, 0.08),
                label="Anterior STEMI pattern",
                detail="Contiguous ST elevation in V2-V4.",
            ),
            regions=["lead_V2", "lead_V3", "lead_V4"],
        )
        warning = _finding(
            "axis",
            Severity.WARNING,
            RegionRect(0.1, 0.1, 0.2, 0.08),
            label="Left axis deviation",
            detail="Predominantly negative inferior leads.",
        )
        coarse = _ekg_row_layout_result([warning, critical])
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [RefinementResult(), RefinementResult()],
        )
        interpreter = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            max_zoom_targets=2,
            max_ekg_systematic_probes=1,
            zoom_padding=0.0,
        )

        result = await interpreter.interpret("img", Modality.EKG, [])

        assert analyzer.refine_calls[0]["hypothesis"].id == "stemi"
        assert analyzer.refine_calls[1]["hypothesis"] is None
        assert analyzer.refine_calls[1]["probe_id"] == (
            "ekg_systematic_critical_support_limb_leads"
        )
        triage = next(
            event
            for event in result.analysis_trace
            if event.get("stage") == "critical_triage"
        )
        assert triage["support_reason"] == (
            "critical_territorial_reciprocal_crosscheck"
        )
        assert triage["skipped_lower_priority_ids"] == ["axis"]
        assert triage["planned_turn_count"] == 2

    async def test_critical_overflow_is_explicit_and_never_adds_extra_turn(self):
        findings = [
            _finding(
                f"critical-{index}",
                Severity.CRITICAL,
                RegionRect(0.1 * index, 0.1, 0.08, 0.08),
                label=f"Acute process {index}",
                detail=f"Localized time-critical morphology {index}.",
            )
            for index in range(1, 4)
        ]
        coarse = _result(findings)
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [RefinementResult(), RefinementResult()],
        )
        interpreter = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            max_zoom_targets=2,
            zoom_padding=0.0,
        )

        result = await interpreter.interpret("img", Modality.CXR, [])

        assert [call["hypothesis"].id for call in analyzer.refine_calls] == [
            "critical-1",
            "critical-2",
        ]
        triage = next(
            event
            for event in result.analysis_trace
            if event.get("stage") == "critical_triage"
        )
        assert triage["overflow_critical_ids"] == ["critical-3"]
        assert triage["planned_turn_count"] == 2
        assert triage["extra_turns_beyond_configured_budget"] == 0

    async def test_top_level_critical_alone_does_not_activate_triage(self):
        warning = _finding(
            "warning",
            Severity.WARNING,
            RegionRect(0.2, 0.2, 0.2, 0.2),
            label="Focal opacity",
            detail="Needs review.",
        )
        coarse = _result([warning])
        coarse.severity = Severity.CRITICAL
        analyzer = _HypothesisAwareAnalyzer(coarse, [RefinementResult()])
        interpreter = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            max_zoom_targets=1,
            zoom_padding=0.0,
        )

        result = await interpreter.interpret("img", Modality.CXR, [])

        assert analyzer.refine_calls[0]["hypothesis"].id == "warning"
        assert not any(
            event.get("stage") == "critical_triage" for event in result.analysis_trace
        )

    async def test_systematic_probe_survives_complete_crop_overlap(self):
        coarse = _ekg_row_layout_result(
            [
                _finding(
                    "global",
                    Severity.WARNING,
                    RegionRect(0.0, 0.0, 1.0, 1.0),
                    label="Unresolved global waveform pattern",
                )
            ]
        )
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [RefinementResult(), RefinementResult()],
        )
        interpreter = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            max_zoom_targets=2,
            max_ekg_systematic_probes=1,
            zoom_padding=0.0,
        )

        result = await interpreter.interpret("img", Modality.EKG, [])

        assert len(analyzer.refine_calls) == 2
        assert analyzer.refine_calls[1]["probe_id"].startswith("ekg_systematic_")
        assert any(
            event.get("stage") == "systematic_assist"
            and event.get("status") == "planned_overlap_fallback"
            for event in result.analysis_trace
        )

    async def test_precordial_hypothesis_crop_enables_balanced_lead_probe(self):
        finding = dataclasses.replace(
            _finding(
                "f1",
                Severity.INFO,
                RegionRect(0.2, 0.59, 0.12, 0.04),
                label="Unresolved anterior waveform change",
            ),
            regions=["lead_V2"],
        )
        coarse = _ekg_row_layout_result([finding])
        analyzer = _HypothesisAwareAnalyzer(coarse, [RefinementResult()])
        interpreter = MultiPassInterpreter(
            analyzer=analyzer,
            cropper=_RecordingCropper(),
            max_zoom_targets=1,
            max_ekg_systematic_probes=0,
            zoom_padding=0.0,
        )

        await interpreter.interpret("image", Modality.EKG, [])

        call = analyzer.refine_calls[0]
        assert call["hypothesis"].id == "f1"
        assert call["probe_id"] == "f1_precordial_leads"
        assert set(call["crop_lead_regions"]) == {"lead_V2"}

    async def test_limb_hypothesis_crop_keeps_original_probe_identity(self):
        finding = dataclasses.replace(
            _finding(
                "f1",
                Severity.INFO,
                RegionRect(0.2, 0.10, 0.12, 0.04),
                label="Unresolved limb-lead waveform change",
            ),
            regions=["lead_II"],
        )
        coarse = _ekg_row_layout_result([finding])
        analyzer = _HypothesisAwareAnalyzer(coarse, [RefinementResult()])
        interpreter = MultiPassInterpreter(
            analyzer=analyzer,
            cropper=_RecordingCropper(),
            max_zoom_targets=1,
            max_ekg_systematic_probes=0,
            zoom_padding=0.0,
        )

        await interpreter.interpret("image", Modality.EKG, [])

        call = analyzer.refine_calls[0]
        assert call["hypothesis"].id == "f1"
        assert call["probe_id"] == "f1"
        assert set(call["crop_lead_regions"]) == {"lead_II"}


def test_unlocalized_ekg_grounding_guard_preserves_study_triage() -> None:
    unlocalized = _finding(
        "urgent",
        Severity.CRITICAL,
        None,
        label="possible acute ST-T pattern",
    )
    localized = _finding(
        "localized",
        Severity.WARNING,
        RegionRect(0.2, 0.2, 0.1, 0.08),
    )
    result = _ekg_row_layout_result([unlocalized, localized])
    result.severity = Severity.CRITICAL

    guarded = apply_unlocalized_ekg_grounding_guard(result)

    assert guarded.severity is Severity.CRITICAL
    assert guarded.findings[0].severity is Severity.INFO
    assert guarded.findings[0].confidence == "low"
    assert "native ECG" in guarded.findings[0].question
    assert guarded.findings[1] == localized
    assert guarded.incomplete is True
    assert guarded.review_required is True
    assert guarded.analysis_trace[-1]["status"] == (
        "downgraded_unlocalized_actionable_finding"
    )


def test_boxed_info_guard_adds_uncertainty_without_moving_validated_box() -> None:
    box = RegionRect(0.21, 0.33, 0.14, 0.07)
    finding = _finding(
        "uncertain",
        Severity.INFO,
        box,
        label="Possible repolarization change",
    )
    result = _ekg_row_layout_result([finding])

    guarded = qualify_boxed_info_findings(result)

    assert guarded.findings[0].confidence == "low"
    assert "highlighted source-image region" in guarded.findings[0].question
    assert guarded.findings[0].bboxes == [box]
    assert guarded.review_required is True
    assert guarded.analysis_trace[-1]["status"] == "qualified_boxed_info_finding"
    assert guarded.analysis_trace[-1]["coordinates_moved"] is False


def test_boxed_info_guard_leaves_qualified_finding_unchanged() -> None:
    finding = dataclasses.replace(
        _finding("qualified", Severity.INFO, RegionRect(0.1, 0.2, 0.1, 0.1)),
        confidence="moderate",
    )
    result = _ekg_row_layout_result([finding])

    assert qualify_boxed_info_findings(result) is result


def test_ekg_overlay_guard_removes_broad_box_before_grounding_downgrade() -> None:
    broad = RegionRect(0.0, 0.08, 1.0, 0.08)
    result = _ekg_row_layout_result(
        [_finding("broad", Severity.WARNING, broad, label="Possible ST change")]
    )

    narrowed = apply_ekg_overlay_bbox_guard(result)
    guarded = apply_unlocalized_ekg_grounding_guard(narrowed)

    assert guarded.findings[0].bboxes == []
    assert guarded.findings[0].severity is Severity.INFO
    assert guarded.severity is Severity.WARNING
    assert narrowed.analysis_trace[-1]["removed_count"] == 1
    assert narrowed.analysis_trace[-1]["coordinates_moved"] is False


def test_ekg_study_level_duplicate_guard_prefers_lead_ii_without_moving_box() -> None:
    lead_ii_box = RegionRect(0.1, 0.09, 0.2, 0.04)
    precordial_box = RegionRect(0.2, 0.7, 0.2, 0.08)
    lead_ii = dataclasses.replace(
        _finding("coarse-rate", Severity.INFO, lead_ii_box, label="sinus bradycardia"),
        regions=["lead_II"],
        confidence="high",
    )
    redundant = dataclasses.replace(
        _finding(
            "crop-rate",
            Severity.INFO,
            precordial_box,
            label="Sinus Bradycardia",
        ),
        regions=["lead_V4"],
        confidence="moderate",
    )
    result = _ekg_row_layout_result([lead_ii, redundant])

    deduplicated = deduplicate_ekg_study_level_findings(result)

    assert [finding.id for finding in deduplicated.findings] == ["coarse-rate"]
    assert deduplicated.findings[0].bboxes == [lead_ii_box]
    event = deduplicated.analysis_trace[-1]
    assert event["status"] == "retracted_exact_study_level_duplicate"
    assert event["retained_finding_id"] == "coarse-rate"
    assert event["retracted_finding_id"] == "crop-rate"
    assert event["coordinates_moved"] is False


def test_ekg_duplicate_guard_does_not_collapse_local_morphology_findings() -> None:
    first = _finding(
        "v2-change",
        Severity.WARNING,
        RegionRect(0.1, 0.5, 0.1, 0.05),
        label="ST elevation",
    )
    second = _finding(
        "v5-change",
        Severity.WARNING,
        RegionRect(0.6, 0.8, 0.1, 0.05),
        label="ST elevation",
    )
    result = _ekg_row_layout_result([first, second])

    assert deduplicate_ekg_study_level_findings(result) is result


def test_unavailable_rhythm_strip_region_uses_bbox_center_lead_without_moving_box() -> (
    None
):
    box = RegionRect(0.1, 0.09, 0.2, 0.04)
    finding = _finding(
        "rhythm",
        Severity.CRITICAL,
        box,
        label="Irregular tachyarrhythmia",
    )
    finding = dataclasses.replace(finding, regions=["rhythm_strip"])
    result = _ekg_row_layout_result([finding])
    result.layout["rhythm_strip_bbox"] = None

    reconciled = reconcile_unavailable_ekg_rhythm_regions(result)

    assert reconciled.findings[0].regions == ["lead_II"]
    assert reconciled.findings[0].bboxes == [box]
    assert reconciled.findings[0].confidence == "low"
    assert reconciled.review_required is True
    assert reconciled.analysis_trace[-1]["status"] == (
        "replaced_unavailable_rhythm_strip"
    )
    assert reconciled.analysis_trace[-1]["coordinates_moved"] is False


def test_unavailable_unlocalized_rhythm_strip_region_is_removed() -> None:
    finding = dataclasses.replace(
        _finding("rhythm", Severity.INFO, None, label="Possible rhythm change"),
        regions=["rhythm_strip"],
    )
    result = _ekg_row_layout_result([finding])
    result.layout["rhythm_strip_bbox"] = None

    reconciled = reconcile_unavailable_ekg_rhythm_regions(result)

    assert reconciled.findings[0].regions == []
    assert reconciled.findings[0].bboxes == []
    assert reconciled.analysis_trace[-1]["status"] == (
        "removed_unlocalized_rhythm_strip"
    )


class TestTwoTurnGroupCoverage:
    @pytest.mark.parametrize("enabled", [False, True])
    async def test_opt_in_covers_both_groups_and_pairs_existing_hypotheses(
        self, enabled
    ):
        coarse = _ekg_row_layout_result(
            [
                _finding("limb", Severity.WARNING, RegionRect(0.2, 0.1, 0.15, 0.04)),
                _finding("chest", Severity.WARNING, RegionRect(0.2, 0.85, 0.15, 0.04)),
            ]
        )
        analyzer = _HypothesisAwareAnalyzer(
            coarse, [RefinementResult(), RefinementResult()]
        )
        cropper = _RecordingCropper()
        interpreter = MultiPassInterpreter(
            analyzer,
            cropper,
            max_zoom_targets=2,
            zoom_padding=0,
            prefer_ekg_group_coverage=enabled,
        )
        result = await interpreter.interpret(
            "source", Modality.EKG, [], source_size_px=(1000, 1200)
        )
        assert len(analyzer.refine_calls) == 2
        assert cropper.images == ["source", "source"]
        planned = [
            e
            for e in result.analysis_trace
            if e.get("status") == "full_lead_group_coverage"
        ]
        assert bool(planned) is enabled
        if enabled:
            assert cropper.regions == [
                RegionRect(0, 0.5, 1, 0.5),
                RegionRect(0, 0, 1, 0.5),
            ]
            assert [c["hypothesis"].id for c in analyzer.refine_calls] == [
                "chest",
                "limb",
            ]
            assert all(
                c["probe_id"].startswith("ekg_systematic_")
                for c in analyzer.refine_calls
            )
            assert [len(c["crop_lead_regions"]) for c in analyzer.refine_calls] == [
                6,
                6,
            ]
            assert planned[0]["diagnostic_completeness_asserted"] is False
        else:
            assert (
                cropper.regions[0].w < 1
            )  # The legacy narrow-first route is unchanged.

    @pytest.mark.parametrize(
        "case",
        [
            "missing",
            "unknown",
            "malformed",
            "critical",
            "one_turn",
            "three_turns",
            "probes_disabled",
            "local_attention",
            "other_modality",
            "duplicate_label",
            "waveform_attention",
        ],
    )
    async def test_coverage_policy_does_not_override_ineligible_or_priority_routes(
        self, case, monkeypatch
    ):
        coarse = _ekg_row_layout_result(
            [
                _finding(
                    "candidate", Severity.WARNING, RegionRect(0.2, 0.7, 0.15, 0.04)
                ),
            ]
        )
        options = {"max_zoom_targets": 2, "max_ekg_systematic_probes": 2}
        local = None
        if case == "missing":
            coarse.layout["leads"].pop()
        elif case == "unknown":
            coarse.layout["leads"][0]["label_visible"] = False
        elif case == "malformed":
            coarse.layout["leads"][0]["bbox"] = [0, 0, 2, 1]
        elif case == "critical":
            coarse.findings[0] = dataclasses.replace(
                coarse.findings[0],
                severity=Severity.CRITICAL,
                label="Possible acute ST-elevation ischemic pattern",
                detail="Contiguous ST elevation with reciprocal depression is visible.",
            )
            coarse.severity = Severity.CRITICAL
        elif case == "one_turn":
            options["max_zoom_targets"] = 1
        elif case == "three_turns":
            options["max_zoom_targets"] = 3
        elif case == "probes_disabled":
            options["max_ekg_systematic_probes"] = 0
        elif case == "local_attention":
            local = [RegionRect(0.5, 0.8, 0.1, 0.1)]
        elif case == "other_modality":
            coarse.modality = Modality.CXR
        elif case == "duplicate_label":
            coarse.layout["leads"].append(dict(coarse.layout["leads"][0]))
        elif case == "waveform_attention":
            monkeypatch.setattr(
                "medical_image_harness.multipass.select_ekg_waveform_attention_probe_regions",
                lambda _result: [("waveform_rhythm", RegionRect(0, 0.08, 1, 0.09))],
            )
        analyzer = _HypothesisAwareAnalyzer(coarse, [RefinementResult()] * 3)
        result = await MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            prefer_ekg_group_coverage=True,
            **options,
        ).interpret(
            "source",
            coarse.modality,
            [],
            source_size_px=(1000, 1200),
            local_candidate_regions=local,
        )
        assert not any(
            e.get("status") == "full_lead_group_coverage" for e in result.analysis_trace
        )
        assert len(analyzer.refine_calls) <= options["max_zoom_targets"]
        if case == "critical":
            assert analyzer.refine_calls[0]["hypothesis"].id == "candidate"
            assert any(
                e.get("stage") == "critical_triage" and e.get("status") == "activated"
                for e in result.analysis_trace
            )

    async def test_group_discovery_adds_source_remapped_finding_without_extra_turn(
        self,
    ):
        coarse = _ekg_row_layout_result([])
        addition = RefinementDelta(
            action=RefinementAction.ADD,
            finding=_finding("new", Severity.WARNING, RegionRect(0.2, 0.2, 0.2, 0.1)),
            rationale="Synthetic visible candidate in this group.",
        )
        analyzer = _HypothesisAwareAnalyzer(
            coarse, [RefinementResult((addition,)), RefinementResult()]
        )
        result = await MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            max_zoom_targets=2,
            zoom_padding=0,
            prefer_ekg_group_coverage=True,
        ).interpret("source", Modality.EKG, [], source_size_px=(1000, 1200))
        assert len(analyzer.refine_calls) == 2
        assert all(call["hypothesis"] is None for call in analyzer.refine_calls)
        finding = next(f for f in result.findings if f.id == "new")
        assert finding.bboxes[0].y == pytest.approx(0.6)
        assert finding.bboxes[0].h == pytest.approx(0.05)


class TestMultiPassRefinementSafety:
    async def test_ekg_budget_keeps_one_hypothesis_and_two_discovery_probes(self):
        coarse = _ekg_row_layout_result(
            [
                _finding(
                    "coarse_finding",
                    Severity.WARNING,
                    RegionRect(0.1, 0.6, 0.1, 0.08),
                )
            ]
        )
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [RefinementResult(), RefinementResult(), RefinementResult()],
        )
        interpreter = MultiPassInterpreter(
            analyzer=analyzer,
            cropper=_RecordingCropper(),
            max_zoom_targets=3,
            zoom_padding=0.0,
        )

        await interpreter.interpret(
            "image",
            Modality.EKG,
            [],
            source_size_px=(1000, 720),
        )

        assert len(analyzer.refine_calls) == 3
        assert analyzer.refine_calls[0]["hypothesis"].id == "coarse_finding"
        assert analyzer.refine_calls[1]["hypothesis"] is None
        assert analyzer.refine_calls[2]["hypothesis"] is None

    async def test_invalid_final_addition_falls_back_to_grounded_draft(self):
        coarse_box = RegionRect(0.2, 0.2, 0.2, 0.2)
        refined_local_box = RegionRect(0.25, 0.25, 0.2, 0.2)
        coarse = _result([_finding("f1", Severity.WARNING, coarse_box)])
        coarse.layout = {"format": "partial"}
        refinement = RefinementResult(
            (
                RefinementDelta(
                    action=RefinementAction.CONFIRM,
                    target_id="f1",
                    finding=_finding(
                        "f1",
                        Severity.WARNING,
                        refined_local_box,
                        detail="confirmed on crop",
                    ),
                    rationale="visible on source crop",
                ),
            )
        )
        final = _result(
            [
                _finding(
                    "invented",
                    Severity.CRITICAL,
                    RegionRect(0.8, 0.8, 0.1, 0.1),
                )
            ]
        )
        final.summary = "Reconciled final narrative."
        final.severity = Severity.CRITICAL
        final.checklist = {
            "x": ChecklistItem(value="reconciled", status=Severity.WARNING)
        }
        analyzer = _FinalizingAnalyzer(coarse, [refinement], final)
        interpreter = MultiPassInterpreter(
            analyzer=analyzer,
            cropper=_RecordingCropper(),
            max_zoom_targets=1,
        )

        result = await interpreter.interpret(
            "coarse-image",
            Modality.CXR,
            [],
            source_image_base64="original-image",
            source_size_px=(1000, 1000),
        )

        assert result.summary == "test"
        assert result.severity is Severity.WARNING
        assert [finding.id for finding in result.findings] == ["f1"]
        assert result.findings[0].detail == "confirmed on crop"
        assert result.findings[0].bboxes != [RegionRect(0.8, 0.8, 0.1, 0.1)]
        assert result.layout == {"format": "partial"}
        assert result.review_required is True
        assert analyzer.finalize_calls[0]["image"] == "original-image"
        assert analyzer.finalize_calls[0]["refinement_trace"]
        finalize_event = next(
            event for event in result.analysis_trace if event.get("stage") == "finalize"
        )
        assert finalize_event["status"] == "failed"
        assert finalize_event["error_type"] == "ValueError"
        assert "cannot add findings" in finalize_event["error"]
        assert result.analysis_trace[-1]["stage"] == "analysis_sla"

    async def test_empty_refinement_still_runs_final_report_turn(self):
        coarse = _result(
            [
                _finding(
                    "f1",
                    Severity.WARNING,
                    RegionRect(0.2, 0.2, 0.2, 0.2),
                )
            ]
        )
        final = _result([])
        final.summary = "Regional review found no additional finding."
        analyzer = _FinalizingAnalyzer(coarse, [RefinementResult()], final)
        interpreter = MultiPassInterpreter(
            analyzer=analyzer,
            cropper=_RecordingCropper(),
            max_zoom_targets=1,
        )

        result = await interpreter.interpret(
            "coarse-image",
            Modality.CXR,
            [],
            source_image_base64="original-image",
            source_size_px=(1000, 1000),
        )

        assert len(analyzer.finalize_calls) == 1
        assert result.summary == "Regional review found no additional finding."
        assert result.severity is Severity.NORMAL
        assert result.findings == []
        assert any(event.get("stage") == "refine" for event in result.analysis_trace)
        disposition = next(
            event
            for event in result.analysis_trace
            if event.get("stage") == "final_disposition"
        )
        assert disposition["status"] == "retracted"
        finalize_event = next(
            event for event in result.analysis_trace if event.get("stage") == "finalize"
        )
        assert finalize_event["status"] == "completed"
        assert finalize_event["retracted_count"] == 1
        assert result.analysis_trace[-1]["stage"] == "analysis_sla"

    async def test_final_retraction_cannot_erase_critical_triage_limitation(self):
        critical = _finding(
            "critical",
            Severity.CRITICAL,
            RegionRect(0.2, 0.2, 0.2, 0.2),
            label="Tension pneumothorax",
            detail="Possible pleural line and mediastinal shift.",
        )
        coarse = _result([critical])
        coarse.severity = Severity.CRITICAL
        final = _result([])
        final.summary = "Critical candidate was not reproduced on the original image."
        analyzer = _FinalizingAnalyzer(coarse, [RefinementResult()], final)
        interpreter = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            max_zoom_targets=1,
            zoom_padding=0.0,
        )

        result = await interpreter.interpret("img", Modality.CXR, [])

        assert result.findings == []
        assert result.incomplete is True
        assert result.review_required is True
        assert any(
            "Critical-first triage intentionally deferred" in reason
            for reason in result.incomplete_reasons
        )
        assert analyzer.finalize_calls[0]["draft"].incomplete is True
        assert any(
            event.get("stage") == "final_disposition"
            and event.get("status") == "retracted"
            for event in result.analysis_trace
        )

    async def test_retracted_ekg_critical_resumes_deferred_axes_in_final_turn(self):
        critical = _finding(
            "vt",
            Severity.CRITICAL,
            RegionRect(0.2, 0.1, 0.1, 0.05),
            label="Ventricular tachycardia",
            detail="Possible synchronized wide-complex run.",
        )
        coarse = _ekg_row_layout_result([critical])
        coarse.severity = Severity.CRITICAL
        refinement = RefinementResult(
            (
                RefinementDelta(
                    action=RefinementAction.RETRACT,
                    target_id="vt",
                    rationale="Not reproduced on the original-study crop.",
                ),
            )
        )
        final = _ekg_row_layout_result([])
        final.severity = Severity.INFO
        final.checklist = _complete_ekg_checklist()
        analyzer = _FinalizingAnalyzer(coarse, [refinement], final)
        interpreter = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            max_zoom_targets=1,
            max_ekg_systematic_probes=0,
            zoom_padding=0.0,
        )

        result = await interpreter.interpret("img", Modality.EKG, [])

        assert result.findings == []
        assert result.severity is Severity.INFO
        assert all(
            item.value.startswith("explicitly assessed")
            for item in result.checklist.values()
        )
        resume = next(
            event
            for event in result.analysis_trace
            if event.get("stage") == "critical_triage_resume_guard"
        )
        assert resume["status"] == "deferred_axes_resumed_on_original_study"
        assert resume["unresolved_axes"] == []
        assert result.incomplete is False
        assert result.incomplete_reasons == []
        assert result.review_required is False
        assert result.review_reasons == []

    async def test_finalizer_downgrade_keeps_unresolved_axes_explicit(self):
        box = RegionRect(0.2, 0.1, 0.1, 0.05)
        critical = _finding(
            "vt",
            Severity.CRITICAL,
            box,
            label="Ventricular tachycardia",
            detail="Possible synchronized wide-complex run.",
        )
        coarse = _ekg_row_layout_result([critical])
        coarse.severity = Severity.CRITICAL
        final_finding = _finding(
            "vt",
            Severity.INFO,
            box,
            label="Artifact-limited rhythm concern",
            detail="No confirmed ventricular run on original-study review.",
        )
        final = _ekg_row_layout_result([final_finding])
        final.severity = Severity.INFO
        final.checklist = {
            key: ChecklistItem(
                value="not_assessed_due_to_critical_triage",
                status=Severity.INFO,
            )
            for key in get_active_registry().resolve(Modality.EKG.value).checklist_keys
        }
        final.checklist["axis"] = ChecklistItem(
            value="Not assessed because the bounded final turn expired.",
            status=Severity.INFO,
        )
        analyzer = _FinalizingAnalyzer(coarse, [RefinementResult()], final)
        interpreter = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            max_zoom_targets=1,
            max_ekg_systematic_probes=0,
            zoom_padding=0.0,
        )

        result = await interpreter.interpret("img", Modality.EKG, [])

        assert result.findings[0].severity is Severity.INFO
        assert result.severity is Severity.INFO
        assert result.incomplete is True
        assert result.review_required is True
        resume = next(
            event
            for event in result.analysis_trace
            if event.get("stage") == "critical_triage_resume_guard"
        )
        assert resume["status"] == "deferred_axes_incomplete_fail_safe"
        assert resume["retained_critical_ids"] == []
        assert resume["unresolved_axes"]
        assert "axis" in resume["unresolved_axes"]
        assert resume["diagnosis_forced"] is False

    async def test_critical_finalization_timeout_fails_safe_without_false_severity(
        self,
    ):
        critical = _finding(
            "vt",
            Severity.CRITICAL,
            RegionRect(0.2, 0.1, 0.1, 0.05),
            label="Ventricular tachycardia",
            detail="Possible synchronized wide-complex run.",
        )
        coarse = _ekg_row_layout_result([critical])
        coarse.severity = Severity.CRITICAL
        refinement = RefinementResult(
            (
                RefinementDelta(
                    action=RefinementAction.RETRACT,
                    target_id="vt",
                    rationale="Not reproduced on the bounded crop.",
                ),
            )
        )
        analyzer = _FailingFinalizingAnalyzer(coarse, [refinement])
        interpreter = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            max_zoom_targets=1,
            max_ekg_systematic_probes=0,
            zoom_padding=0.0,
        )

        result = await interpreter.interpret("img", Modality.EKG, [])

        assert result.findings == []
        # A workflow timeout does not manufacture a clinical abnormality.
        # Incomplete/review flags below carry the unresolved safety state.
        assert result.severity is Severity.NORMAL
        assert result.incomplete is True
        assert result.review_required is True
        assert any(
            event.get("stage") == "finalize"
            and event.get("status") == "failed"
            and event.get("error_type") == "TimeoutError"
            for event in result.analysis_trace
        )
        resume = next(
            event
            for event in result.analysis_trace
            if event.get("stage") == "critical_triage_resume_guard"
        )
        assert resume["status"] == "deferred_axes_incomplete_fail_safe"
        assert resume["unresolved_axes"]
        assert resume["clinical_status_inferred"] is False

    async def test_no_crop_target_still_runs_final_report_turn(self):
        coarse = _result([])
        final = _result([])
        final.summary = "Normal study after complete review."
        final.checklist = {"x": ChecklistItem(value="normal", status=Severity.NORMAL)}
        analyzer = _FinalizingAnalyzer(coarse, [], final)
        interpreter = MultiPassInterpreter(
            analyzer=analyzer,
            cropper=_RecordingCropper(),
            max_zoom_targets=0,
        )

        result = await interpreter.interpret("image", Modality.CXR, [])

        assert len(analyzer.finalize_calls) == 1
        assert analyzer.finalize_calls[0]["refinement_trace"] == []
        assert result.summary == "Normal study after complete review."
        assert result.checklist["x"].value == "normal"
        sla = result.analysis_trace[-1]
        assert sla["stage"] == "analysis_sla"
        assert sla["first_crop_applicable"] is False

    async def test_failed_final_turn_has_retry_budget_and_explicit_unknown_checklist(
        self,
    ):
        coarse = _ekg_row_layout_result([])
        analyzer = _FailingFinalizingAnalyzer(coarse, [])
        interpreter = MultiPassInterpreter(
            analyzer=analyzer,
            cropper=_RecordingCropper(),
            max_zoom_targets=0,
            max_ekg_systematic_probes=0,
        )

        result = await interpreter.interpret("image", Modality.EKG, [])

        assert len(result.checklist) == 16
        assert result.checklist["rhythm"].value == "sinus"
        assert result.checklist["stemi_pattern"].status is Severity.INFO
        assert "Not assessed" in result.checklist["stemi_pattern"].value
        assert result.incomplete is True
        assert result.review_required is True
        finalize_event = next(
            event for event in result.analysis_trace if event.get("stage") == "finalize"
        )
        assert finalize_event["status"] == "failed"
        assert finalize_event["turn_budget_ms"] >= 75_000
        fallback_event = next(
            event
            for event in result.analysis_trace
            if event.get("stage") == "checklist_fallback"
        )
        assert fallback_event["clinical_status_inferred"] is False
        assert fallback_event["diagnosis_forced"] is False

    async def test_refinement_crop_uses_original_source_not_coarse_downscale(self):
        box = RegionRect(x=0.2, y=0.2, w=0.4, h=0.4)
        coarse = _result([_finding("f1", Severity.WARNING, box)])
        refinement = RefinementResult(
            (
                RefinementDelta(
                    action=RefinementAction.CONFIRM,
                    target_id="f1",
                    rationale="visible on source crop",
                ),
            )
        )
        analyzer = _HypothesisAwareAnalyzer(coarse, [refinement])
        cropper = _RecordingCropper()
        interpreter = MultiPassInterpreter(
            analyzer,
            cropper,
            zoom_padding=0.0,
            min_zoom_source_edge_px=0,
        )

        result = await interpreter.interpret(
            "coarse-downscale",
            Modality.CXR,
            [],
            source_image_base64="original-roi",
            source_size_px=(3000, 2000),
        )

        assert analyzer.images == ["coarse-downscale"]
        assert cropper.images == ["original-roi"]
        refine_event = next(
            event
            for event in result.analysis_trace
            if event.get("stage") == "refine" and event.get("status") == "completed"
        )
        assert refine_event["crop_source"] == "original_roi"

    async def test_no_abnormal_findings_skips_zoom(self):
        coarse = _result([])
        analyzer = _FakeAnalyzer([coarse])
        cropper = _RecordingCropper()
        interp = MultiPassInterpreter(analyzer, cropper)

        out = await interp.interpret("img", Modality.CXR, [])

        assert out is coarse
        assert cropper.regions == []
        assert len(analyzer.images) == 1  # only the coarse pass

    async def test_refines_abnormal_finding_bbox(self):
        coarse_box = RegionRect(x=0.5, y=0.5, w=0.2, h=0.2)
        coarse = _result(
            [_finding("f1", Severity.CRITICAL, coarse_box, detail="coarse")]
        )
        # Zoom returns a tighter bbox relative to the crop, plus new detail.
        zoom_box = RegionRect(x=0.4, y=0.4, w=0.1, h=0.1)
        zoom = _result([_finding("z", Severity.CRITICAL, zoom_box, detail="sharper")])
        analyzer = _FakeAnalyzer([coarse, zoom])
        cropper = _RecordingCropper()
        interp = MultiPassInterpreter(analyzer, cropper, zoom_padding=0.0)

        out = await interp.interpret("img", Modality.CXR, [])

        # Two analyze calls: coarse + one zoom.
        assert len(analyzer.images) == 2
        # The zoom received the cropped image, not the original.
        assert analyzer.images[1].startswith("crop::")
        # Coarse finding kept its id but got refined detail + remapped bbox.
        refined = out.findings[0]
        assert refined.id == "f1"
        assert refined.detail == "sharper"
        # Remapped global center: crop = padded coarse box (pad 0) = coarse box.
        # child center 0.45 within crop [0.5..0.7] -> 0.5 + 0.45*0.2 = 0.59
        b = refined.bboxes[0]
        assert (b.x + b.w / 2) == pytest.approx(0.59)

    async def test_refine_turn_receives_hypothesis_and_crop_context(self):
        box = RegionRect(x=0.2, y=0.3, w=0.4, h=0.3)
        coarse_finding = _finding(
            "f1",
            Severity.WARNING,
            box,
            label="opacity",
            detail="coarse",
        )
        coarse = _result([coarse_finding])
        confirmation = _finding(
            "different-id",
            Severity.CRITICAL,
            RegionRect(x=0.25, y=0.25, w=0.5, h=0.5),
            label="different label",
            detail="confirmed on targeted crop",
        )
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [
                RefinementResult(
                    (
                        RefinementDelta(
                            action=RefinementAction.CONFIRM,
                            target_id="f1",
                            finding=confirmation,
                            rationale="targeted second turn",
                        ),
                    )
                )
            ],
        )
        cropper = _RecordingCropper()
        interp = MultiPassInterpreter(analyzer, cropper, zoom_padding=0.0)

        out = await interp.interpret("img", Modality.CXR, ["left_lung"])

        assert analyzer.images == ["img"]
        assert len(analyzer.refine_calls) == 1
        call = analyzer.refine_calls[0]
        assert call["image"].startswith("crop::")
        assert call["hypothesis"] == coarse_finding
        assert call["crop_region"] == box
        confirmed = out.findings[0]
        assert confirmed.label == "opacity"
        assert confirmed.severity is Severity.WARNING
        assert confirmed.detail == "confirmed on targeted crop"
        assert (
            "[Crop-only evidence; ROI x=0.2000 y=0.3000 w=0.4000 h=0.3000] "
            "targeted second turn"
        ) in confirmed.notes

    async def test_confirm_preserves_coarse_result_severity_floor(self):
        box = RegionRect(x=0.2, y=0.3, w=0.4, h=0.3)
        coarse_finding = _finding("f1", Severity.WARNING, box)
        coarse = _result([coarse_finding])
        coarse.severity = Severity.CRITICAL
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [
                RefinementResult(
                    (
                        RefinementDelta(
                            RefinementAction.CONFIRM,
                            target_id="f1",
                        ),
                    )
                )
            ],
        )
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            zoom_padding=0.0,
        )

        out = await interp.interpret("img", Modality.CXR, [])

        assert out.severity is Severity.CRITICAL

    async def test_explicit_revise_is_not_coupled_to_first_delta(self):
        box = RegionRect(x=0.4, y=0.4, w=0.2, h=0.2)
        coarse = _result(
            [
                _finding(
                    "f1",
                    Severity.WARNING,
                    box,
                    label="possible opacity",
                    detail="coarse",
                )
            ]
        )
        addition = _finding(
            "new",
            Severity.WARNING,
            RegionRect(0.05, 0.05, 0.2, 0.2),
            label="adjacent finding",
        )
        revision = _finding(
            "ignored-payload-id",
            Severity.CRITICAL,
            RegionRect(0.25, 0.25, 0.5, 0.5),
            label="confirmed consolidation",
            detail="revised after crop",
        )
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [
                RefinementResult(
                    (
                        RefinementDelta(RefinementAction.ADD, finding=addition),
                        RefinementDelta(
                            RefinementAction.REVISE,
                            target_id="f1",
                            finding=revision,
                        ),
                    )
                )
            ],
        )
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            zoom_padding=0.0,
        )

        out = await interp.interpret("img", Modality.CXR, [])

        revised = next(finding for finding in out.findings if finding.id == "f1")
        assert revised.label == "confirmed consolidation"
        assert revised.severity is Severity.CRITICAL
        assert revised.detail == "revised after crop"
        assert revised.bboxes[0] == RegionRect(0.45, 0.45, 0.1, 0.1)
        assert any(finding.id == "new" for finding in out.findings)
        assert out.severity is Severity.CRITICAL

    async def test_explicit_retract_removes_coarse_and_updates_severity(self):
        box = RegionRect(x=0.3, y=0.3, w=0.3, h=0.3)
        coarse = _result([_finding("f1", Severity.CRITICAL, box)])
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [
                RefinementResult(
                    (
                        RefinementDelta(
                            RefinementAction.RETRACT,
                            target_id="f1",
                            rationale="targeted crop is normal",
                        ),
                    )
                )
            ],
        )
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            zoom_padding=0.0,
        )

        out = await interp.interpret("img", Modality.CXR, [])

        assert out.findings == []
        assert out.severity is Severity.NORMAL

    async def test_partial_crop_cannot_retract_disjoint_coarse_evidence(self):
        finding = Finding(
            id="multi-site",
            regions=[],
            label="Two-site signal anomaly",
            detail="Separate visible candidates require independent review",
            severity=Severity.WARNING,
            bboxes=[
                RegionRect(0.05, 0.05, 0.20, 0.10),
                RegionRect(0.70, 0.78, 0.10, 0.08),
            ],
        )
        coarse = _ekg_row_layout_result([finding])
        final = _ekg_row_layout_result([finding])
        analyzer = _FinalizingAnalyzer(
            coarse,
            [
                RefinementResult(
                    (
                        RefinementDelta(
                            RefinementAction.RETRACT,
                            target_id="multi-site",
                            rationale="the selected local crop is normal",
                        ),
                    )
                )
            ],
            final,
        )
        cropper = _RecordingCropper()
        interp = MultiPassInterpreter(
            analyzer,
            cropper,
            zoom_padding=0.0,
            max_zoom_targets=1,
            max_ekg_systematic_probes=0,
        )

        out = await interp.interpret("img", Modality.EKG, [])

        assert [item.id for item in out.findings] == ["multi-site"]
        assert cropper.regions == [finding.bboxes[0]]
        guard = next(
            event
            for event in out.analysis_trace
            if event.get("status") == "partial_crop_retraction_blocked"
        )
        assert guard["tool"] == "crop_coverage_guard"
        assert guard["uncovered_bbox_count"] == 1
        assert guard["blocked_action"] == "retract"
        assert guard["decision_applied"] is False
        assert guard["evidence_interpretation"] == (
            "inconclusive_partial_crop_coverage"
        )
        final_trace = analyzer.finalize_calls[0]["refinement_trace"]
        carried = next(
            event
            for event in final_trace
            if event.get("status") == "partial_crop_retraction_blocked"
        )
        assert carried["target_id"] == "multi-site"
        assert carried["decision_applied"] is False

    @pytest.mark.parametrize(
        ("action", "expected_status"),
        [
            (RefinementAction.REVISE, "partial_crop_revision_blocked"),
            (RefinementAction.CONFIRM, "partial_crop_confirmation_blocked"),
        ],
    )
    async def test_partial_crop_cannot_rebind_disjoint_coarse_evidence(
        self,
        action,
        expected_status,
    ):
        finding = Finding(
            id="multi-site",
            regions=["lead_I", "lead_V6"],
            label="Two-site signal anomaly",
            detail="Separate visible candidates require independent review",
            severity=Severity.WARNING,
            bboxes=[
                RegionRect(0.05, 0.05, 0.20, 0.10),
                RegionRect(0.70, 0.78, 0.10, 0.08),
            ],
        )
        proposed = Finding(
            id="ignored-payload-id",
            regions=["lead_I"],
            label="Rebound local diagnosis",
            detail="The selected crop alone changes the whole hypothesis.",
            severity=Severity.INFO,
            bboxes=[RegionRect(0.1, 0.1, 0.4, 0.4)],
        )
        coarse = _ekg_row_layout_result([finding])
        # The top-level coarse urgency may reflect image-wide context outside
        # the selected local crop. A blocked local mutation cannot lower it.
        coarse.severity = Severity.CRITICAL
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [
                RefinementResult(
                    (
                        RefinementDelta(
                            action,
                            target_id="multi-site",
                            finding=proposed,
                            rationale="Only the selected local crop was reviewed.",
                        ),
                    )
                )
            ],
        )
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            zoom_padding=0.0,
            max_zoom_targets=1,
            max_ekg_systematic_probes=0,
        )

        out = await interp.interpret("img", Modality.EKG, [])

        assert out.findings == [finding]
        assert out.severity is Severity.CRITICAL
        guard = next(
            event
            for event in out.analysis_trace
            if event.get("status") == expected_status
        )
        assert guard["blocked_action"] == action.value
        assert guard["uncovered_bbox_count"] == 1
        assert guard["decision_applied"] is False
        assert guard["severity_downgrade_authorized"] is False

    async def test_local_candidate_cannot_retract_bboxless_coarse_hypothesis(self):
        finding = _finding(
            "unlocalized",
            Severity.WARNING,
            None,
            label="Unlocalized opacity concern",
            detail="The coarse pass could not localize the concern.",
        )
        coarse = _result([finding])
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [
                RefinementResult(
                    (
                        RefinementDelta(
                            RefinementAction.RETRACT,
                            target_id="unlocalized",
                            rationale="The local candidate crop was normal.",
                        ),
                    )
                )
            ],
        )
        candidate = RegionRect(0.2, 0.2, 0.3, 0.3)
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            zoom_padding=0.0,
            max_zoom_targets=1,
        )

        out = await interp.interpret(
            "img",
            Modality.CXR,
            [],
            local_candidate_regions=[candidate],
        )

        assert out.findings == [finding]
        assert out.severity is Severity.WARNING
        guard = next(
            event
            for event in out.analysis_trace
            if event.get("status") == "partial_crop_retraction_blocked"
        )
        assert guard["source_bbox_count"] == 0
        assert guard["source_was_unlocalized"] is True
        assert guard["uncovered_bbox_count"] == 0
        assert guard["decision_applied"] is False

    @pytest.mark.parametrize(
        "action",
        [RefinementAction.REVISE, RefinementAction.CONFIRM],
    )
    async def test_semantic_only_targeted_delta_preserves_coarse_geometry_contract(
        self,
        action,
    ):
        box = RegionRect(0.2, 0.2, 0.3, 0.3)
        coarse_finding = _finding(
            "f1",
            Severity.WARNING,
            box,
            label="Coarse opacity concern",
            detail="Coarse semantics remain bound to this box.",
        )
        proposed = _finding(
            "ignored-payload-id",
            Severity.INFO,
            None,
            label="Different semantic conclusion",
            detail="No verified coordinates accompanied this mutation.",
        )
        coarse = _result([coarse_finding])
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [
                RefinementResult(
                    (
                        RefinementDelta(
                            action,
                            target_id="f1",
                            finding=proposed,
                            rationale="Semantic-only refinement.",
                        ),
                    ),
                    unlocalized_missing_receipt_delta_indexes=(0,),
                )
            ],
        )
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            zoom_padding=0.0,
            max_zoom_targets=1,
        )

        out = await interp.interpret("img", Modality.CXR, [])

        assert out.findings == [coarse_finding]
        assert out.severity is Severity.WARNING
        assert out.incomplete is True
        assert out.review_required is True
        guard = next(
            event
            for event in out.analysis_trace
            if event.get("stage") == "bbox_receipt_guardrail"
        )
        assert guard["action"] == action.value
        assert guard["decision_applied"] is False
        assert guard["coarse_finding_preserved"] is True
        assert guard["overlay_coordinates_asserted"] is False

    async def test_delta_cannot_revise_a_different_coarse_finding(self):
        box = RegionRect(x=0.3, y=0.3, w=0.3, h=0.3)
        coarse_finding = _finding("f1", Severity.WARNING, box, detail="coarse")
        coarse = _result([coarse_finding])
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [
                RefinementResult(
                    (
                        RefinementDelta(
                            RefinementAction.REVISE,
                            target_id="another-finding",
                            finding=_finding(
                                "another-finding",
                                Severity.CRITICAL,
                                RegionRect(0.1, 0.1, 0.2, 0.2),
                            ),
                        ),
                    )
                )
            ],
        )
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            zoom_padding=0.0,
        )

        out = await interp.interpret("img", Modality.CXR, [])

        assert out.findings == [coarse_finding]

    async def test_crop_region_is_subset_of_roi(self):
        # Privacy invariant: a zoom crop must never widen beyond the ROI.
        box = RegionRect(x=0.3, y=0.3, w=0.2, h=0.2)
        coarse = _result([_finding("f1", Severity.WARNING, box)])
        zoom = _result([_finding("z", Severity.WARNING, box)])
        analyzer = _FakeAnalyzer([coarse, zoom])
        cropper = _RecordingCropper()
        interp = MultiPassInterpreter(analyzer, cropper, zoom_padding=0.2)

        await interp.interpret("img", Modality.CXR, [])

        region = cropper.regions[0]
        assert region.x >= 0.0
        assert region.y >= 0.0
        assert region.x + region.w <= 1.0 + 1e-9
        assert region.y + region.h <= 1.0 + 1e-9

    async def test_failed_zoom_keeps_coarse_finding(self):
        box = RegionRect(x=0.3, y=0.3, w=0.2, h=0.2)
        coarse = _result([_finding("f1", Severity.WARNING, box, detail="coarse")])
        analyzer = _FakeAnalyzer([coarse])  # no zoom result -> pop raises

        def _boom(image_base64, region):
            raise RuntimeError("crop failed")

        interp = MultiPassInterpreter(analyzer, _boom)
        out = await interp.interpret("img", Modality.CXR, [])

        # Coarse finding survives unchanged.
        assert out.findings[0].detail == "coarse"
        assert out.findings[0].bboxes[0] == box

    async def test_transient_zoom_analysis_failure_is_retried(self):
        box = RegionRect(x=0.3, y=0.3, w=0.2, h=0.2)
        coarse = _result([_finding("f1", Severity.WARNING, box, detail="coarse")])
        zoom = _result([_finding("z1", Severity.WARNING, box, detail="retried")])
        analyzer = _FlakyZoomAnalyzer(coarse, zoom, failures_before_success=1)
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            zoom_padding=0.0,
            zoom_retry_attempts=1,
        )

        out = await interp.interpret("img", Modality.CXR, [])

        assert analyzer.zoom_calls == 2
        assert out.findings[0].detail == "retried"

    async def test_zero_zoom_retries_keeps_coarse_after_transient_failure(self):
        box = RegionRect(x=0.3, y=0.3, w=0.2, h=0.2)
        coarse = _result([_finding("f1", Severity.WARNING, box, detail="coarse")])
        zoom = _result([_finding("z1", Severity.WARNING, box, detail="never used")])
        analyzer = _FlakyZoomAnalyzer(coarse, zoom, failures_before_success=1)
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            zoom_padding=0.0,
            zoom_retry_attempts=0,
        )

        out = await interp.interpret("img", Modality.CXR, [])

        assert analyzer.zoom_calls == 1
        assert out.findings[0].detail == "coarse"

    async def test_local_candidate_cannot_confirm_unlocalized_whole_roi_finding(self):
        coarse = _result(
            [
                _finding(
                    "f1",
                    Severity.WARNING,
                    None,
                    label="possible opacity",
                    detail="coarse finding without coordinates",
                )
            ]
        )
        candidate = RegionRect(x=0.2, y=0.2, w=0.4, h=0.4)
        zoom = _result(
            [
                _finding(
                    "z",
                    Severity.WARNING,
                    RegionRect(x=0.25, y=0.25, w=0.5, h=0.5),
                    label="possible opacity",
                    detail="candidate crop confirms opacity",
                )
            ]
        )
        analyzer = _FakeAnalyzer([coarse, zoom])
        cropper = _RecordingCropper()
        interp = MultiPassInterpreter(analyzer, cropper, zoom_padding=0.0)

        out = await interp.interpret(
            "img",
            Modality.CXR,
            [],
            local_candidate_regions=[candidate],
        )

        assert len(analyzer.images) == 2
        assert cropper.regions == [candidate]
        refined = out.findings[0]
        assert refined.id == "f1"
        assert refined == coarse.findings[0]
        guard = next(
            e
            for e in out.analysis_trace
            if e.get("status") == "partial_crop_confirmation_blocked"
        )
        assert guard["source_was_unlocalized"] is True
        assert guard["decision_applied"] is False

    async def test_normal_safety_probe_can_be_disabled(self):
        coarse = _result([])
        analyzer = _FakeAnalyzer([coarse])
        cropper = _RecordingCropper()
        interp = MultiPassInterpreter(
            analyzer,
            cropper,
            max_normal_safety_probes=0,
        )

        out = await interp.interpret(
            "img",
            Modality.CXR,
            [],
            local_candidate_regions=[RegionRect(x=0.2, y=0.2, w=0.4, h=0.4)],
        )

        assert out is coarse
        assert len(analyzer.images) == 1
        assert cropper.regions == []

    async def test_normal_safety_probe_is_bounded_and_can_add_finding(self):
        coarse = _result([])
        discovered = _finding(
            "probe-finding",
            Severity.WARNING,
            RegionRect(0.25, 0.25, 0.5, 0.5),
            label="probe discovery",
        )
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [
                RefinementResult(
                    (
                        RefinementDelta(
                            RefinementAction.ADD,
                            finding=discovered,
                            rationale="bounded normal safety probe",
                        ),
                    )
                )
            ],
        )
        cropper = _RecordingCropper()
        interp = MultiPassInterpreter(
            analyzer,
            cropper,
            zoom_padding=0.0,
            max_normal_safety_probes=1,
        )
        first = RegionRect(x=0.1, y=0.1, w=0.3, h=0.3)
        second = RegionRect(x=0.6, y=0.6, w=0.2, h=0.2)

        out = await interp.interpret(
            "img",
            Modality.CXR,
            [],
            local_candidate_regions=[RegionRect(0.0, 0.0, 1.0, 1.0), first, second],
        )

        assert cropper.regions == [first]
        assert len(analyzer.refine_calls) == 1
        assert analyzer.refine_calls[0]["hypothesis"] is None
        assert out.severity is Severity.WARNING
        assert out.findings[0].bboxes[0] == RegionRect(0.175, 0.175, 0.15, 0.15)

    async def test_unresolved_discovery_is_kept_for_review_without_warning(self):
        coarse = _result([])
        discovered = Finding(
            id="possible-lvh",
            regions=["lead_V1"],
            label="Possible LVH voltage pattern",
            detail="Voltage appears high but secondary change is not confirmed.",
            severity=Severity.WARNING,
            confidence="moderate",
            question="Can a reviewer confirm calibrated voltage criteria?",
            bboxes=[RegionRect(0.25, 0.25, 0.5, 0.5)],
        )
        analyzer = _HypothesisAwareAnalyzer(
            coarse,
            [
                RefinementResult(
                    (
                        RefinementDelta(
                            RefinementAction.ADD,
                            finding=discovered,
                            rationale="unresolved discovery-only crop",
                        ),
                    )
                )
            ],
        )
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            zoom_padding=0.0,
            max_normal_safety_probes=1,
        )

        out = await interp.interpret(
            "img",
            Modality.CXR,
            ["lead_V1"],
            local_candidate_regions=[RegionRect(0.1, 0.1, 0.3, 0.3)],
        )

        assert out.severity is Severity.INFO
        assert out.findings[0].severity is Severity.INFO
        assert out.findings[0].confidence == "low"
        assert any(
            event.get("stage") == "refinement_guardrail"
            and event.get("status") == "downgraded_unresolved_discovery"
            for event in out.analysis_trace
        )

    async def test_full_frame_local_candidate_is_not_a_safety_probe(self):
        coarse = _result([])
        analyzer = _FakeAnalyzer([coarse])
        cropper = _RecordingCropper()
        interp = MultiPassInterpreter(analyzer, cropper)

        out = await interp.interpret(
            "img",
            Modality.CXR,
            [],
            local_candidate_regions=[RegionRect(0.0, 0.0, 1.0, 1.0)],
        )

        assert out is coarse
        assert analyzer.images == ["img"]
        assert cropper.regions == []

    async def test_extra_zoom_findings_appended(self):
        box = RegionRect(x=0.4, y=0.4, w=0.2, h=0.2)
        coarse = _result([_finding("f1", Severity.CRITICAL, box)])
        zbox = RegionRect(x=0.1, y=0.1, w=0.1, h=0.1)
        zoom = _result(
            [
                _finding("z1", Severity.CRITICAL, zbox, detail="primary"),
                _finding("z2", Severity.WARNING, zbox, detail="extra"),
            ]
        )
        analyzer = _FakeAnalyzer([coarse, zoom])
        interp = MultiPassInterpreter(analyzer, _RecordingCropper(), zoom_padding=0.0)

        out = await interp.interpret("img", Modality.CXR, [])

        ids = [f.id for f in out.findings]
        assert ids[0] == "f1"  # refined in place
        assert "f1_z2" in ids  # extra finding appended, linked to parent

    async def test_legacy_fallback_matches_second_finding_by_hypothesis(self):
        box = RegionRect(x=0.4, y=0.4, w=0.2, h=0.2)
        coarse = _result(
            [
                _finding(
                    "f1",
                    Severity.WARNING,
                    box,
                    label="target opacity",
                    detail="coarse",
                )
            ]
        )
        zoom = _result(
            [
                _finding(
                    "unrelated",
                    Severity.WARNING,
                    RegionRect(0.1, 0.1, 0.2, 0.2),
                    label="pleural fluid",
                    detail="first but unrelated",
                ),
                _finding(
                    "matched",
                    Severity.CRITICAL,
                    RegionRect(0.4, 0.4, 0.2, 0.2),
                    label="target opacity",
                    detail="second and matched",
                ),
            ]
        )
        analyzer = _FakeAnalyzer([coarse, zoom])
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            zoom_padding=0.0,
        )

        out = await interp.interpret("img", Modality.CXR, [])

        target = next(finding for finding in out.findings if finding.id == "f1")
        assert target.detail == "second and matched"
        assert target.severity is Severity.CRITICAL
        assert any(finding.id == "f1_unrelated" for finding in out.findings)

    async def test_legacy_normal_crop_retracts_target(self):
        box = RegionRect(x=0.4, y=0.4, w=0.2, h=0.2)
        coarse = _result([_finding("f1", Severity.WARNING, box)])
        analyzer = _FakeAnalyzer([coarse, _result([])])
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            zoom_padding=0.0,
        )

        out = await interp.interpret("img", Modality.CXR, [])

        assert out.findings == []
        assert out.severity is Severity.NORMAL

    async def test_legacy_non_normal_result_without_findings_keeps_coarse(self):
        box = RegionRect(x=0.4, y=0.4, w=0.2, h=0.2)
        coarse_finding = _finding("f1", Severity.WARNING, box)
        coarse = _result([coarse_finding])
        malformed_zoom = _result([])
        malformed_zoom.severity = Severity.WARNING
        analyzer = _FakeAnalyzer([coarse, malformed_zoom])
        interp = MultiPassInterpreter(
            analyzer,
            _RecordingCropper(),
            zoom_padding=0.0,
        )

        out = await interp.interpret("img", Modality.CXR, [])

        assert out is coarse
        assert out.findings == [coarse_finding]

    async def test_negative_max_zoom_rejected(self):
        with pytest.raises(ValueError):
            MultiPassInterpreter(
                _FakeAnalyzer([]), _RecordingCropper(), max_zoom_targets=-1
            )

    async def test_negative_zoom_retries_rejected(self):
        with pytest.raises(ValueError):
            MultiPassInterpreter(
                _FakeAnalyzer([]), _RecordingCropper(), zoom_retry_attempts=-1
            )


# ── resolution-aware zoom (screenshot 4K cap) ────────────────────────


class TestRegionSourceEdgePx:
    def test_short_edge_in_source_pixels(self):
        # 4K capture; region 10% wide x 5% tall -> short edge = 0.05*2160 = 108
        region = RegionRect(x=0.1, y=0.1, w=0.1, h=0.05)
        assert region_source_edge_px(region, (3840, 2160)) == 108

    def test_uses_min_of_width_and_height(self):
        region = RegionRect(x=0.0, y=0.0, w=0.5, h=0.02)
        # width 1920px, height 43px -> short edge 43
        assert region_source_edge_px(region, (3840, 2160)) == 43


class TestNeedsManualZoom:
    def test_small_region_needs_manual_zoom(self):
        # 2% of a 4K short edge = 43px < 64 -> source remains limited.
        region = RegionRect(x=0.1, y=0.1, w=0.02, h=0.02)
        assert needs_manual_zoom(region, (3840, 2160)) is True

    def test_large_region_digitally_zoomable(self):
        # 20% of 4K = 432px >= 64 -> no manual-zoom warning.
        region = RegionRect(x=0.1, y=0.1, w=0.2, h=0.2)
        assert needs_manual_zoom(region, (3840, 2160)) is False

    def test_threshold_is_configurable(self):
        region = RegionRect(x=0.1, y=0.1, w=0.2, h=0.2)  # 432px
        assert needs_manual_zoom(region, (3840, 2160), min_source_edge_px=500) is True


class TestExpandCropToMinSourceEdge:
    def test_expands_tight_bbox_around_center(self):
        region = RegionRect(x=0.4, y=0.4, w=0.02, h=0.04)

        expanded = expand_crop_to_min_source_edge(region, (1000, 800))

        assert expanded.w == pytest.approx(0.256)
        assert expanded.h == pytest.approx(0.32)
        assert expanded.x + expanded.w / 2 == pytest.approx(0.41)
        assert expanded.y + expanded.h / 2 == pytest.approx(0.42)

    def test_clamps_expansion_at_source_edge(self):
        region = RegionRect(x=0.98, y=0.98, w=0.02, h=0.02)

        expanded = expand_crop_to_min_source_edge(region, (1000, 1000))

        assert expanded.x == pytest.approx(0.744)
        assert expanded.y == pytest.approx(0.744)
        assert expanded.x + expanded.w == pytest.approx(1.0)
        assert expanded.y + expanded.h == pytest.approx(1.0)


class TestBuildManualZoomMessage:
    def test_includes_label_and_pixels(self):
        msg = build_manual_zoom_message("Lung nodule", 80)
        assert "Lung nodule" in msg
        assert "80px" in msg

    def test_blank_label_falls_back(self):
        msg = build_manual_zoom_message("   ", 50)
        assert "此區域" in msg


@pytest.mark.asyncio
class TestMultiPassResolutionAware:
    async def test_small_region_emits_hint_and_contextual_refinement_crop(self):
        # A tiny critical lesion in a 4K capture: 2% short edge = 43px < 64.
        box = RegionRect(x=0.4, y=0.4, w=0.02, h=0.02)
        coarse = _result([_finding("f1", Severity.CRITICAL, box, label="Micro-nodule")])
        analyzer = _HypothesisAwareAnalyzer(coarse, [RefinementResult()])
        cropper = _RecordingCropper()
        interp = MultiPassInterpreter(analyzer, cropper)

        out = await interp.interpret(
            "img", Modality.CXR, [], source_size_px=(3840, 2160)
        )

        # The user sees the source-resolution warning, while OpenClaw still
        # receives real neighboring pixels for a second visual turn.
        assert len(cropper.regions) == 1
        assert cropper.regions[0].w >= 256 / 3840
        assert cropper.regions[0].h >= 256 / 2160
        assert len(analyzer.images) == 1
        assert len(analyzer.refine_calls) == 1
        assert len(out.zoom_hints) == 1
        assert "Micro-nodule" in out.zoom_hints[0]
        # An empty refinement delta preserves the coarse finding.
        assert out.findings[0].bboxes[0] == box

    async def test_large_region_still_digitally_zoomed(self):
        box = RegionRect(x=0.3, y=0.3, w=0.3, h=0.3)  # 648px short edge
        coarse = _result([_finding("f1", Severity.CRITICAL, box)])
        zoom = _result(
            [_finding("z", Severity.CRITICAL, RegionRect(0.4, 0.4, 0.1, 0.1))]
        )
        analyzer = _FakeAnalyzer([coarse, zoom])
        cropper = _RecordingCropper()
        interp = MultiPassInterpreter(analyzer, cropper, zoom_padding=0.0)

        out = await interp.interpret(
            "img", Modality.CXR, [], source_size_px=(3840, 2160)
        )

        assert len(analyzer.images) == 2  # digital zoom happened
        assert cropper.regions  # crop was taken
        assert out.zoom_hints == []  # no manual hint needed

    async def test_unknown_source_size_zooms_as_before(self):
        # Without source_size_px the orchestrator can't reason about pixels, so
        # it digitally zooms every target (backward compatible).
        box = RegionRect(x=0.4, y=0.4, w=0.03, h=0.03)
        coarse = _result([_finding("f1", Severity.CRITICAL, box)])
        zoom = _result(
            [_finding("z", Severity.CRITICAL, RegionRect(0.4, 0.4, 0.1, 0.1))]
        )
        analyzer = _FakeAnalyzer([coarse, zoom])
        interp = MultiPassInterpreter(analyzer, _RecordingCropper(), zoom_padding=0.0)

        out = await interp.interpret("img", Modality.CXR, [])

        assert len(analyzer.images) == 2  # digital zoom still happens
        assert out.zoom_hints == []

    async def test_mixed_targets_split_between_crop_and_hint(self):
        big = RegionRect(x=0.1, y=0.1, w=0.3, h=0.3)  # digital
        small = RegionRect(x=0.6, y=0.6, w=0.02, h=0.02)  # limited + refined
        coarse = _result(
            [
                _finding("big", Severity.CRITICAL, big, label="Mass"),
                _finding("small", Severity.CRITICAL, small, label="Spot"),
            ]
        )
        zoom = _result(
            [_finding("z", Severity.CRITICAL, RegionRect(0.4, 0.4, 0.1, 0.1))]
        )
        analyzer = _FakeAnalyzer([coarse, zoom, zoom])
        cropper = _RecordingCropper()
        interp = MultiPassInterpreter(analyzer, cropper, zoom_padding=0.0)

        out = await interp.interpret(
            "img", Modality.CXR, [], source_size_px=(3840, 2160)
        )

        # Both targets are refined; the tiny one also receives a manual hint.
        assert len(cropper.regions) == 2
        assert len(out.zoom_hints) == 1
        assert "Spot" in out.zoom_hints[0]


# ── MultiPassAnalyzer drop-in adapter ──────────────────────────


@pytest.mark.asyncio
class TestMultiPassAnalyzer:
    """The adapter must be a drop-in VisionAnalyzerService for OverlayAgent."""

    async def test_analyze_routes_through_interpreter(self):
        coarse_box = RegionRect(x=0.5, y=0.5, w=0.2, h=0.2)
        coarse = _result([_finding("a", Severity.WARNING, coarse_box)])
        refined = _result(
            [_finding("a", Severity.WARNING, RegionRect(0.0, 0.0, 1.0, 1.0))]
        )
        inner = _FakeAnalyzer([coarse, refined])
        cropper = _RecordingCropper()
        interp = MultiPassInterpreter(inner, cropper)
        adapter = MultiPassAnalyzer(inner=inner, interpreter=interp)

        out = await adapter.analyze("img", Modality.CXR, [])

        # Multi-pass ran: a coarse + a refine call happened (2 analyzer images).
        assert len(inner.images) == 2
        assert cropper.regions  # a digital crop was taken
        assert out.findings

    async def test_analyze_with_source_size_routes_resolution_context(self):
        tiny_box = RegionRect(x=0.4, y=0.4, w=0.03, h=0.03)
        coarse = _result(
            [_finding("tiny", Severity.WARNING, tiny_box, label="Tiny target")]
        )
        refined = _result(
            [
                _finding(
                    "tiny",
                    Severity.WARNING,
                    RegionRect(0.4, 0.4, 0.2, 0.2),
                    label="Tiny target",
                )
            ]
        )
        inner = _FakeAnalyzer([coarse, refined])
        cropper = _RecordingCropper()
        interp = MultiPassInterpreter(inner, cropper)
        adapter = MultiPassAnalyzer(inner=inner, interpreter=interp)

        out = await adapter.analyze_with_source_size(
            "img",
            Modality.CXR,
            [],
            source_size_px=(100, 100),
        )

        assert len(inner.images) == 2
        assert len(cropper.regions) == 1
        assert cropper.regions[0] == RegionRect(0.0, 0.0, 1.0, 1.0)
        assert out.zoom_hints

    async def test_non_analyze_methods_delegate_to_inner(self):
        inner = _FakeAnalyzer([_result([])])
        interp = MultiPassInterpreter(inner, _RecordingCropper())
        adapter = MultiPassAnalyzer(inner=inner, interpreter=interp)

        assert adapter.is_connected() is True
        assert await adapter.chat("hi") == ""
        await adapter.connect()
        await adapter.disconnect()

    async def test_is_a_vision_analyzer_service(self):
        from medical_image_harness.protocols import VisionAnalyzerService

        inner = _FakeAnalyzer([_result([])])
        interp = MultiPassInterpreter(inner, _RecordingCropper())
        adapter = MultiPassAnalyzer(inner=inner, interpreter=interp)
        assert isinstance(adapter, VisionAnalyzerService)
