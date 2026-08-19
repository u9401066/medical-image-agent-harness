"""Unit tests for the multi-pass interpretation orchestrator.

Covers the pure coordinate maths (clamp / pad / remap), zoom-target selection,
and the coarse -> crop -> refine orchestration including the privacy invariant
that a zoom crop only ever shrinks the captured region.
"""

from __future__ import annotations

from dataclasses import replace

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
from medical_image_harness.multipass import (
    MultiPassAnalyzer,
    MultiPassInterpreter,
    RefinementAction,
    RefinementDelta,
    RefinementResult,
    apply_refinement_delta,
    build_manual_zoom_message,
    clamp_unit,
    covering_region,
    expand_crop_to_min_source_edge,
    needs_manual_zoom,
    pad_region,
    reconcile_final_report,
    region_source_edge_px,
    remap_bbox,
    select_ekg_systematic_probe_regions,
    select_zoom_targets,
)
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


def test_final_reconciliation_preserves_host_bound_scientific_ledger() -> None:
    box = RegionRect(0.1, 0.2, 0.3, 0.1)
    draft = _result([_finding("f1", Severity.WARNING, box)])
    draft.assessment_scope = "single_image_observation"
    draft.summary_observation_ids = ["obs-1"]
    draft.observations = [
        Observation(
            id="obs-1",
            anatomy="right lower lung",
            finding="focal opacity",
            polarity=Polarity.PRESENT,
            status=VerificationStatus.SUPPORTED,
            assessable=True,
            evidence_ids=["ev-1"],
        )
    ]
    draft.evidence = [
        Evidence(
            id="ev-1",
            kind="source_image",
            source_image_sha256="a" * 64,
            description="visible focal opacity",
            bboxes=[box],
        )
    ]
    draft.input_provenance = {"source_image_sha256": "a" * 64}
    draft.study_manifest = {"id": "study-1"}
    draft.workflow_events = [{"stage": "intake", "status": "completed"}]
    final = _result([])
    final.summary = "Reconciled narrative"

    result = reconcile_final_report(draft, final)

    assert result.summary == "Reconciled narrative"
    assert result.assessment_scope == "single_image_observation"
    assert result.summary_observation_ids == ["obs-1"]
    assert result.observations == draft.observations
    assert result.evidence == draft.evidence
    assert result.input_provenance == draft.input_provenance
    assert result.study_manifest == draft.study_manifest
    assert result.workflow_events == draft.workflow_events


# ── clamp_unit ───────────────────────────────────────────────────────


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


class TestEkgSystematicProbeRegions:
    def test_uses_declared_lead_layout_to_cover_limb_and_precordial_groups(self):
        result = _ekg_row_layout_result([])

        probes = select_ekg_systematic_probe_regions(result)

        assert [key for key, _region in probes] == [
            "limb_leads",
            "precordial_leads",
        ]
        assert probes[0][1].y == pytest.approx(0.0)
        assert probes[0][1].h == pytest.approx(0.5)
        assert probes[1][1].y == pytest.approx(0.5)
        assert probes[1][1].h == pytest.approx(0.5)

    def test_accepts_unprefixed_real_model_lead_names(self):
        result = _ekg_row_layout_result([])
        for lead in result.layout["leads"]:
            lead["name"] = lead["name"].removeprefix("lead_")

        probes = select_ekg_systematic_probe_regions(result)

        assert [key for key, _region in probes] == [
            "limb_leads",
            "precordial_leads",
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
            "limb_leads",
            "precordial_leads",
        ]

    def test_rejects_sparse_or_non_ekg_layout(self):
        result = _ekg_row_layout_result([])
        result.layout = {
            "leads": [
                {"name": "lead_I", "label_visible": True, "bbox": [0, 0, 1, 0.1]}
            ]
        }
        assert select_ekg_systematic_probe_regions(result) == []
        assert select_ekg_systematic_probe_regions(_result([])) == []


# ── select_zoom_targets ──────────────────────────────────────────────


class TestSelectZoomTargets:
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

    def test_refinement_changes_clear_unreconciled_canonical_links(self):
        crop = RegionRect(0.1, 0.1, 0.5, 0.5)
        linked = replace(
            _finding("f1", Severity.WARNING, RegionRect(0.2, 0.2, 0.2, 0.2)),
            evidence=["legacy evidence"],
            evidence_ids=["ev-1"],
            observation_ids=["obs-1"],
        )
        revised = _finding(
            "f1",
            Severity.CRITICAL,
            RegionRect(0.3, 0.3, 0.1, 0.1),
        )
        added = replace(
            _finding("f2", Severity.WARNING, RegionRect(0.1, 0.1, 0.1, 0.1)),
            evidence_ids=["invented-ev"],
            observation_ids=["invented-obs"],
        )

        after_revise = apply_refinement_delta(
            [linked],
            RefinementDelta(
                RefinementAction.REVISE,
                target_id="f1",
                finding=revised,
            ),
            crop_region=crop,
            expected_target_id="f1",
        )
        after_add = apply_refinement_delta(
            after_revise,
            RefinementDelta(RefinementAction.ADD, finding=added),
            crop_region=crop,
            expected_target_id=None,
        )

        assert after_revise[0].evidence == []
        assert after_revise[0].evidence_ids == []
        assert after_revise[0].observation_ids == []
        assert after_add[1].evidence_ids == []
        assert after_add[1].observation_ids == []


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
    ):
        self.refine_calls.append(
            {
                "image": image_base64,
                "modality": modality,
                "valid_regions": valid_regions,
                "hypothesis": hypothesis,
                "crop_region": crop_region,
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


class _RecordingCropper:
    """Fake cropper: records crop regions, returns a marker string."""

    def __init__(self) -> None:
        self.regions: list[RegionRect] = []
        self.images: list[str] = []

    def __call__(self, image_base64: str, region: RegionRect) -> str:
        self.images.append(image_base64)
        self.regions.append(region)
        return f"crop::{region.x:.3f},{region.y:.3f}"


@pytest.mark.asyncio
class TestMultiPassInterpreter:
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
        assert [region.y for region in cropper.regions] == pytest.approx([0.0, 0.5])
        assert result.severity is Severity.WARNING
        assert [finding.id for finding in result.findings] == ["st_probe"]
        assert any(
            event.get("stage") == "systematic_assist"
            for event in result.analysis_trace
        )

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

    async def test_final_report_turn_uses_source_image_and_cannot_replace_boxes(self):
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

        assert result.summary == "Reconciled final narrative."
        assert result.severity is Severity.WARNING
        assert [finding.id for finding in result.findings] == ["f1"]
        assert result.findings[0].detail == "confirmed on crop"
        assert result.findings[0].bboxes != [RegionRect(0.8, 0.8, 0.1, 0.1)]
        assert result.layout == {"format": "partial"}
        assert analyzer.finalize_calls[0]["image"] == "original-image"
        assert analyzer.finalize_calls[0]["refinement_trace"]
        assert result.analysis_trace[-1]["stage"] == "finalize"
        assert result.analysis_trace[-1]["status"] == "completed"

    async def test_negative_refinement_still_runs_final_report_reconciliation(self):
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
        assert analyzer.finalize_calls[0]["refinement_trace"]
        assert result.summary == "Regional review found no additional finding."
        assert [finding.id for finding in result.findings] == ["f1"]
        assert result.analysis_trace[-1]["stage"] == "finalize"

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
        assert result.analysis_trace[-1]["crop_source"] == "original_roi"

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
        assert "targeted second turn" in confirmed.notes

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

    async def test_local_candidate_refines_abnormal_finding_without_bbox(self):
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
        assert refined.detail == "candidate crop confirms opacity"
        assert refined.bboxes
        bbox = refined.bboxes[0]
        assert bbox.x == pytest.approx(0.3)
        assert bbox.y == pytest.approx(0.3)
        assert bbox.w == pytest.approx(0.2)
        assert bbox.h == pytest.approx(0.2)

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

        # The user sees the source-resolution warning, while the analyzer still
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
    """The adapter must be a drop-in VisionAnalyzerService."""

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
