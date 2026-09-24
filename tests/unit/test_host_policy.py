"""Host policy injection and scientific-binding boundaries of the shared engine."""

from dataclasses import replace

import pytest

from medical_image_harness.models import (
    AnalysisResult,
    ChecklistItem,
    ClaimType,
    Finding,
    Modality,
    RegionRect,
    Severity,
)
from medical_image_harness.multipass import (
    MultiPassInterpreter,
    RefinementAction,
    RefinementResult,
    apply_critical_triage_guard,
    complete_unassessed_checklist_fallback,
    reconcile_final_report,
)
from medical_image_harness.protocols import PENDING_MULTIPASS_REASON, StageTools


def test_refinement_action_is_a_direct_public_definition():
    assert RefinementAction.__name__ == "RefinementAction"
    assert RefinementAction.__module__ == "medical_image_harness.multipass"


def draft():
    return AnalysisResult(
        modality=Modality.CXR, summary="Synthetic observation",
        severity=Severity.WARNING,
        findings=[Finding(id="f1", regions=[], label="Synthetic opacity",
            detail="Visible synthetic marker", severity=Severity.WARNING,
            bboxes=[RegionRect(0.2, 0.2, 0.2, 0.2)],
            evidence_ids=["e1"], observation_ids=["o1"])],
        checklist={"host_axis": ChecklistItem(value="assessed", status=Severity.NORMAL)},
    )


@pytest.mark.parametrize("keys", [frozenset(), frozenset({"host_axis", "missing_axis"})])
def test_fallback_honors_exact_host_axes_including_empty(keys):
    source = draft()
    result = complete_unassessed_checklist_fallback(
        source, reason="Synthetic timeout", required_checklist_keys=keys,
    )
    assert set(result.checklist) == {"host_axis"} | keys
    if "missing_axis" in keys:
        assert result.incomplete and result.review_required
        assert result.checklist["missing_axis"].status is Severity.INFO


def test_default_public_axes_are_not_replaced_by_host_policy():
    source = draft()
    result = complete_unassessed_checklist_fallback(source, reason="Synthetic timeout")
    assert "projection_quality" in result.checklist
    assert "projection_quality" not in source.checklist


@pytest.mark.parametrize("keys,resolved", [
    (frozenset({"host_axis"}), True),
    (frozenset({"host_axis", "missing_axis"}), False),
    (frozenset(), False),
])
def test_pending_resolution_uses_host_axes(keys, resolved):
    source = draft()
    source.incomplete = True
    source.incomplete_reasons = [PENDING_MULTIPASS_REASON]
    result = reconcile_final_report(source, draft(), required_checklist_keys=keys)
    assert result.incomplete is not resolved
    assert (PENDING_MULTIPASS_REASON in result.incomplete_reasons) is not resolved


def test_critical_triage_honors_host_axes_without_public_profile_drift():
    source = replace(draft(), modality=Modality.EKG)
    critical = replace(source.findings[0], label="Ventricular tachycardia",
                       severity=Severity.CRITICAL)
    result = apply_critical_triage_guard(
        source, [critical], phase="coarse", required_checklist_keys=frozenset({"host_axis"}),
    )
    assert set(result.checklist) == {"host_axis"}
    assert result.checklist["host_axis"].status is Severity.INFO


@pytest.mark.parametrize("change", [
    {"label": "Different observation"},
    {"detail": "Different morphology"},
    {"severity": Severity.INFO},
    {"confidence": "low"},
    {"question": "Verify morphology?"},
    {"claim_type": ClaimType.DIAGNOSTIC_HYPOTHESIS},
])
def test_final_semantic_changes_invalidate_old_claim_bindings(change):
    source = draft()
    proposed = replace(source, findings=[replace(source.findings[0], **change)])
    result = reconcile_final_report(source, proposed)
    assert result.findings[0].evidence_ids == []
    assert result.findings[0].observation_ids == []
    assert result.findings[0].bboxes == source.findings[0].bboxes
    assert source.findings[0].evidence_ids == ["e1"]


def test_unchanged_final_claim_cannot_replace_trusted_references():
    source = draft()
    proposed = replace(source, findings=[replace(source.findings[0],
        evidence_ids=["forged"], observation_ids=["forged"])])
    result = reconcile_final_report(source, proposed)
    assert result.findings[0].evidence_ids == ["e1"]
    assert result.findings[0].observation_ids == ["o1"]


def test_finalizer_cannot_clear_host_bound_human_review():
    source = draft()
    source.input_provenance = {"source_image_sha256": "a" * 64}
    source.review_required = True
    source.review_reasons = ["Authorized human review"]
    source.summary_observation_ids = ["o1"]
    proposed = replace(source, input_provenance={"source_image_sha256": "b" * 64},
        review_required=False, review_reasons=[], summary_observation_ids=["forged"])
    result = reconcile_final_report(source, proposed)
    assert result.review_required
    assert result.review_reasons == source.review_reasons
    assert result.input_provenance == source.input_provenance
    assert result.summary_observation_ids == ["o1"]


@pytest.mark.parametrize("fail_final", [False, True])
async def test_interpreter_injects_stage_identities_and_failure_checklist(fail_final):
    class Analyzer:
        async def analyze(self, *_):
            return draft()

        async def refine(self, *_, **__):
            return RefinementResult()

        async def finalize(self, *_, draft, **__):
            if fail_final:
                raise TimeoutError("Synthetic failure")
            return draft

    tools = StageTools(coarse="host_read", refinement="host_crop", finalize="host_final")
    result = await MultiPassInterpreter(
        Analyzer(), lambda *_: "synthetic-crop", max_zoom_targets=1,
        stage_tools=tools,
        checklist_keys_for=lambda _: frozenset({"host_axis", "host_missing"}),
    ).interpret("synthetic-source", Modality.CXR, [])
    events = result.analysis_trace
    for stage, tool in [("coarse", tools.coarse), ("refine", "crop_region_base64"),
                        ("finalize", tools.finalize)]:
        assert any(e.get("stage") == stage and e.get("tool") == tool for e in events)
    if fail_final:
        assert set(result.checklist) == {"host_axis", "host_missing"}
        assert result.incomplete and result.review_required
    else:
        assert any(e.get("stage") == "final_disposition"
                   and e["tool"] == tools.finalize for e in events)
