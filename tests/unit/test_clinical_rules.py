"""Unit tests for the data-driven clinical consistency engine.

Covers the pure domain engine (condition matching, escalate-only severity, the
flag-for-review behavior, citations), the YAML rule-pack loader (override by id,
add-by-new-id, malformed-entry resilience), and the post-analyze hook.
"""

from __future__ import annotations

import pytest

from medical_image_harness.clinical_consistency import (
    ClinicalConsistencyHook,
)
from medical_image_harness.clinical_rules import (
    ClinicalConsistencyEngine,
    ClinicalRule,
    ConditionError,
    RuleCondition,
    builtin_rules,
    default_engine,
    group_by_modality,
)
from medical_image_harness.hooks import AnalyzeRequest
from medical_image_harness.models import (
    AnalysisResult,
    ChecklistItem,
    Finding,
    Modality,
    Severity,
)
from medical_image_harness.rule_loader import (
    build_clinical_engine,
    load_rule_pack_dir,
    merge_rules,
)


def _result(
    *,
    modality: Modality = Modality.EKG,
    severity: Severity = Severity.NORMAL,
    summary: str = "Normal study",
    checklist: dict[str, ChecklistItem] | None = None,
    findings: list[Finding] | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        modality=modality,
        summary=summary,
        severity=severity,
        findings=findings or [],
        checklist=checklist or {},
    )


def _item(value: str, status: Severity = Severity.NORMAL) -> ChecklistItem:
    return ChecklistItem(value=value, status=status)


# ── RuleCondition matching ───────────────────────────────────────────


class TestRuleCondition:
    def test_contains_any_on_checklist_value(self):
        cond = RuleCondition(
            field="checklist.st_segment", op="contains_any", values=("elevat",)
        )
        res = _result(checklist={"st_segment": _item("ST elevation V1-V4")})
        assert cond.matches(res) is True

    def test_contains_any_missing_key_is_false(self):
        cond = RuleCondition(
            field="checklist.st_segment", op="contains_any", values=("elevat",)
        )
        assert cond.matches(_result()) is False

    @pytest.mark.parametrize(
        ("summary", "expected"),
        [
            ("Definite anterior STEMI", True),
            ("No ischemia, but definite anterior STEMI", True),
            ("No STEMI pattern", False),
            ("STEMI cannot be excluded", False),
            ("Possible STEMI", False),
            ("NSTEMI", False),
        ],
    )
    def test_contains_any_asserted_respects_assertion_scope(
        self, summary: str, expected: bool
    ):
        cond = RuleCondition(
            field="summary", op="contains_any_asserted", values=("stemi",)
        )
        assert cond.matches(_result(summary=summary)) is expected

    @pytest.mark.parametrize(
        ("summary", "expected"),
        [
            ("Possible hyperacute ischemia", True),
            ("Hyperacute ischemia cannot be excluded", True),
            ("No hyperacute ischemia", False),
            ("No ischemia, but possible hyperacute ischemia", True),
        ],
    )
    def test_contains_any_non_negated_preserves_uncertain_triage(
        self, summary: str, expected: bool
    ):
        cond = RuleCondition(
            field="summary",
            op="contains_any_non_negated",
            values=("hyperacute ischemia",),
        )
        assert cond.matches(_result(summary=summary)) is expected

    def test_checklist_status_condition(self):
        cond = RuleCondition(
            field="checklist.st_segment.status",
            op="severity_at_least",
            value="info",
        )
        assert cond.matches(
            _result(checklist={"st_segment": _item("elevation", Severity.INFO)})
        )
        assert not cond.matches(_result(checklist={"st_segment": _item("isoelectric")}))
        assert not cond.matches(_result())

    def test_not_contains_any(self):
        cond = RuleCondition(field="summary", op="not_contains_any", values=("stemi",))
        assert cond.matches(_result(summary="benign")) is True
        assert cond.matches(_result(summary="STEMI suspected")) is False

    def test_all_text_spans_findings_and_checklist(self):
        cond = RuleCondition(
            field="all_text", op="contains_any", values=("pneumothorax",)
        )
        res = _result(
            modality=Modality.CXR,
            findings=[
                Finding(
                    id="f1",
                    regions=["right_apex"],
                    label="apical lucency",
                    detail="possible pneumothorax",
                    severity=Severity.WARNING,
                )
            ],
        )
        assert cond.matches(res) is True

    def test_severity_at_most(self):
        cond = RuleCondition(field="severity", op="severity_at_most", value="info")
        assert cond.matches(_result(severity=Severity.NORMAL)) is True
        assert cond.matches(_result(severity=Severity.INFO)) is True
        assert cond.matches(_result(severity=Severity.WARNING)) is False

    def test_severity_at_least(self):
        cond = RuleCondition(field="severity", op="severity_at_least", value="warning")
        assert cond.matches(_result(severity=Severity.CRITICAL)) is True
        assert cond.matches(_result(severity=Severity.NORMAL)) is False

    def test_invalid_operator_for_text_field_raises(self):
        with pytest.raises(ConditionError):
            RuleCondition(field="summary", op="severity_at_most", value="info")

    def test_invalid_operator_for_severity_field_raises(self):
        with pytest.raises(ConditionError):
            RuleCondition(field="severity", op="contains_any", values=("x",))

    def test_unknown_field_raises_on_match(self):
        cond = RuleCondition(field="bogus", op="contains_any", values=("x",))
        with pytest.raises(ConditionError):
            cond.matches(_result())


# ── ClinicalRule + engine ────────────────────────────────────────────


def _stemi_rule() -> ClinicalRule:
    return ClinicalRule(
        id="r-stemi",
        modality="EKG",
        description="ST elevation not flagged",
        conditions=(
            RuleCondition(
                field="checklist.st_segment", op="contains_any", values=("elevat",)
            ),
            RuleCondition(field="severity", op="severity_at_most", value="info"),
        ),
        message="ST 抬高卻評為正常 — 需排除 STEMI",
        guideline="Universal MI",
        guideline_version="2018",
        effective_date="2018-08-25",
        escalate_to=Severity.CRITICAL,
    )


class TestEngine:
    def test_rule_fires_only_when_all_conditions_hold(self):
        rule = _stemi_rule()
        fires = _result(
            severity=Severity.NORMAL,
            checklist={"st_segment": _item("marked ST elevation")},
        )
        no_fire = _result(
            severity=Severity.NORMAL,
            checklist={"st_segment": _item("isoelectric")},
        )
        assert rule.fires(fires) is True
        assert rule.fires(no_fire) is False

    def test_apply_escalates_severity_and_flags_review(self):
        engine = ClinicalConsistencyEngine(group_by_modality([_stemi_rule()]))
        res = _result(
            severity=Severity.NORMAL,
            checklist={"st_segment": _item("ST elevation")},
        )
        violations = engine.apply(res)

        assert len(violations) == 1
        assert res.severity is Severity.CRITICAL  # escalated up
        assert res.review_required is True
        assert res.review_reasons
        assert "STEMI" in res.review_reasons[0]
        assert "2018" in res.review_reasons[0]  # citation attached

    def test_apply_never_downgrades_severity(self):
        # Rule floor is CRITICAL but if result were already CRITICAL it stays;
        # and a WARNING floor must never pull a CRITICAL result down.
        rule = ClinicalRule(
            id="r-warn",
            modality="EKG",
            description="peaked T",
            conditions=(
                RuleCondition(
                    field="checklist.t_wave", op="contains_any", values=("peaked",)
                ),
            ),
            message="x",
            escalate_to=Severity.WARNING,
        )
        engine = ClinicalConsistencyEngine(group_by_modality([rule]))
        res = _result(
            severity=Severity.CRITICAL,
            checklist={"t_wave": _item("peaked")},
        )
        engine.apply(res)
        assert res.severity is Severity.CRITICAL  # not downgraded to WARNING

    def test_clean_result_untouched(self):
        engine = ClinicalConsistencyEngine(group_by_modality([_stemi_rule()]))
        res = _result(
            severity=Severity.NORMAL,
            checklist={"st_segment": _item("isoelectric, no acute changes")},
        )
        violations = engine.apply(res)
        assert violations == []
        assert res.severity is Severity.NORMAL
        assert res.review_required is False
        assert res.review_reasons == []

    def test_reasons_deduplicated(self):
        engine = ClinicalConsistencyEngine(
            group_by_modality([_stemi_rule(), _stemi_rule()])
        )
        res = _result(
            severity=Severity.NORMAL,
            checklist={"st_segment": _item("ST elevation")},
        )
        engine.apply(res)
        assert len(res.review_reasons) == 1

    def test_rules_scoped_by_modality(self):
        engine = ClinicalConsistencyEngine(group_by_modality([_stemi_rule()]))
        # A CXR result must not be matched by an EKG rule.
        res = _result(
            modality=Modality.CXR,
            severity=Severity.NORMAL,
            checklist={"st_segment": _item("ST elevation")},
        )
        assert engine.apply(res) == []

    def test_citation_format(self):
        assert "2018" in _stemi_rule().citation()
        assert "Universal MI" in _stemi_rule().citation()


# ── Built-in rules behavior ──────────────────────────────────────────


class TestBuiltinRules:
    def test_builtin_stemi_undercall_escalates(self):
        engine = default_engine()
        res = _result(
            modality=Modality.EKG,
            severity=Severity.NORMAL,
            summary="Definite anterior STEMI",
            checklist={
                "st_segment": _item("ST elevation in anterior leads", Severity.CRITICAL)
            },
        )
        engine.apply(res)
        assert res.severity is Severity.CRITICAL
        assert res.review_required is True

    def test_ambiguous_st_elevation_is_review_only(self):
        engine = default_engine()
        res = _result(
            modality=Modality.EKG,
            severity=Severity.INFO,
            summary="Mild concave elevation, likely benign early repolarization",
            checklist={
                "st_segment": _item(
                    "mild concave ST elevation; possible early repolarization",
                    Severity.INFO,
                )
            },
        )
        violations = engine.apply(res)
        assert [item.rule.id for item in violations] == ["ekg-st-elevation-not-flagged"]
        assert res.severity is Severity.INFO
        assert res.review_required is True

    @pytest.mark.parametrize(
        "summary",
        ["No STEMI pattern", "Cannot exclude STEMI", "Possible anterior STEMI"],
    )
    def test_non_asserted_stemi_does_not_force_critical(self, summary: str):
        res = _result(
            modality=Modality.EKG,
            severity=Severity.INFO,
            summary=summary,
        )
        default_engine().apply(res)
        assert res.severity is Severity.INFO
        assert res.review_required is False

    def test_possible_hyperacute_ischemia_escalates_triage_not_diagnosis(self):
        res = _result(
            modality=Modality.EKG,
            severity=Severity.WARNING,
            summary=(
                "Tall T waves may reflect hyperacute ischemia or benign "
                "repolarization; no definite STEMI is established."
            ),
        )

        violations = default_engine().apply(res)

        assert [item.rule.id for item in violations] == [
            "ekg-possible-hyperacute-ischemia-triage"
        ]
        assert res.severity is Severity.CRITICAL
        assert res.review_required is True
        assert "STEMI" not in res.summary.replace("no definite STEMI", "")

    def test_negated_hyperacute_ischemia_does_not_escalate(self):
        res = _result(
            modality=Modality.EKG,
            severity=Severity.WARNING,
            summary="Tall T waves are present, with no hyperacute ischemia.",
        )

        default_engine().apply(res)

        assert res.severity is Severity.WARNING
        assert res.review_required is False

    def test_uncertain_acute_injury_with_st_elevation_escalates_triage(self):
        res = _result(
            modality=Modality.EKG,
            severity=Severity.WARNING,
            summary="Anterior precordial ST-T abnormality",
            findings=[
                Finding(
                    id="anterior-stt",
                    regions=["lead_V2", "lead_V3", "lead_V4"],
                    label="Anterior precordial ST-T abnormality",
                    detail=(
                        "Mild concave ST elevation in V2-V4; early repolarization "
                        "versus acute anterior injury cannot be resolved."
                    ),
                    severity=Severity.WARNING,
                    confidence="low",
                    question="Can acute injury be excluded on the source tracing?",
                )
            ],
            checklist={
                "st_segment": _item("ST elevation in V2-V4", Severity.WARNING),
                "stemi_pattern": _item("absent", Severity.NORMAL),
            },
        )

        violations = default_engine().apply(res)

        assert [item.rule.id for item in violations] == [
            "ekg-uncertain-acute-injury-with-st-elevation-triage"
        ]
        assert res.severity is Severity.CRITICAL
        assert res.review_required is True
        assert "cannot be resolved" in res.findings[0].detail

    def test_benign_st_elevation_without_acute_injury_does_not_escalate(self):
        res = _result(
            modality=Modality.EKG,
            severity=Severity.WARNING,
            summary="Mild concave elevation consistent with early repolarization.",
            checklist={
                "st_segment": _item("ST elevation in V2-V4", Severity.WARNING)
            },
        )

        default_engine().apply(res)

        assert res.severity is Severity.WARNING
        assert res.review_required is False

    def test_negated_acute_injury_with_st_elevation_does_not_escalate(self):
        res = _result(
            modality=Modality.EKG,
            severity=Severity.WARNING,
            summary="ST elevation is present with no acute myocardial injury.",
            checklist={
                "st_segment": _item("ST elevation in V2-V4", Severity.WARNING)
            },
        )

        default_engine().apply(res)

        assert res.severity is Severity.WARNING
        assert res.review_required is False

    def test_builtin_pneumothorax_undercall_escalates(self):
        engine = default_engine()
        res = _result(
            modality=Modality.CXR,
            severity=Severity.NORMAL,
            summary="no acute findings; small apical pneumothorax not significant",
        )
        engine.apply(res)
        assert res.severity is Severity.CRITICAL
        assert res.review_required is True

    def test_negated_pneumothorax_does_not_escalate(self):
        res = _result(
            modality=Modality.CXR,
            severity=Severity.NORMAL,
            summary="No focal airspace disease, pleural effusion, or pneumothorax.",
        )
        default_engine().apply(res)
        assert res.severity is Severity.NORMAL
        assert res.review_required is False

    def test_builtin_does_not_flag_clean_ekg(self):
        engine = default_engine()
        res = _result(
            modality=Modality.EKG,
            severity=Severity.NORMAL,
            summary="Normal sinus rhythm",
            checklist={"st_segment": _item("isoelectric")},
        )
        engine.apply(res)
        assert res.review_required is False

    def test_builtins_have_citations(self):
        for rule in builtin_rules():
            assert rule.guideline, f"{rule.id} missing guideline citation"
            assert rule.message


# ── Rule-pack loader (modular update) ────────────────────────────────


_PACK = """
rules:
  - id: r-override
    modality: EKG
    description: overridden
    conditions:
      - field: checklist.st_segment
        op: contains_any
        values: ["elevat"]
    message: "overridden message"
    guideline: "Test Guideline"
    guideline_version: "9.9"
    escalate_to: warning
  - id: r-new
    modality: CXR
    description: new pleural-effusion under-call rule
    conditions:
      - field: all_text
        op: contains_any
        values: ["effusion"]
    message: "new rule"
    escalate_to: warning
"""


class TestRulePackLoader:
    def test_loads_only_rules_yaml(self, tmp_path):
        (tmp_path / "x.rules.yaml").write_text(_PACK, encoding="utf-8")
        (tmp_path / "ignored.yaml").write_text("rules: []", encoding="utf-8")
        (tmp_path / "z.rules.yaml.example").write_text(_PACK, encoding="utf-8")
        rules = load_rule_pack_dir(tmp_path)
        ids = {r.id for r in rules}
        assert ids == {"r-override", "r-new"}

    def test_missing_dir_returns_empty(self, tmp_path):
        assert load_rule_pack_dir(tmp_path / "nope") == []

    def test_malformed_rule_skipped_not_raised(self, tmp_path):
        bad = """
rules:
  - id: ok
    modality: EKG
    description: well-formed and documented
    conditions:
      - field: summary
        op: contains_any
        values: ["x"]
    message: "ok"
  - id: broken
    modality: EKG
    description: has no conditions
    conditions: []
    message: "no conditions -> skipped"
  - id: bad-op
    modality: EKG
    description: wrong operator for a text field
    conditions:
      - field: summary
        op: severity_at_most
        value: info
    message: "wrong op for text field -> skipped"
  - id: undocumented
    modality: EKG
    conditions:
      - field: summary
        op: contains_any
        values: ["y"]
    message: "no description -> skipped"
"""
        (tmp_path / "p.rules.yaml").write_text(bad, encoding="utf-8")
        rules = load_rule_pack_dir(tmp_path)
        assert {r.id for r in rules} == {"ok"}

    def test_merge_overrides_by_id(self):
        builtins = list(builtin_rules())
        overrides = load_rule_pack_dir_from_text(_PACK)
        # Sanity: override targets an id that may or may not be builtin; add r-new.
        merged = merge_rules(builtins, overrides)
        merged_ids = [r.id for r in merged]
        assert "r-new" in merged_ids
        # The override entry must appear exactly once (no duplicate id).
        assert merged_ids.count("r-override") == 1

    def test_build_engine_applies_overrides(self, tmp_path):
        (tmp_path / "p.rules.yaml").write_text(_PACK, encoding="utf-8")
        engine = build_clinical_engine(tmp_path)
        res = _result(
            modality=Modality.CXR,
            severity=Severity.NORMAL,
            summary="trace pleural effusion",
        )
        engine.apply(res)
        # The new CXR rule fired and escalated/flagged.
        assert res.review_required is True
        assert res.severity is Severity.WARNING

    def test_build_engine_without_dir_is_builtins(self):
        engine = build_clinical_engine(None)
        res = _result(
            modality=Modality.EKG,
            severity=Severity.NORMAL,
            checklist={"st_segment": _item("ST elevation", Severity.INFO)},
        )
        engine.apply(res)
        assert res.review_required is True


def load_rule_pack_dir_from_text(text: str) -> list[ClinicalRule]:
    """Helper: parse a pack from an in-memory string via a temp-like flow."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "mem.rules.yaml"
        p.write_text(text, encoding="utf-8")
        return load_rule_pack_dir(Path(d))


# ── Hook integration ─────────────────────────────────────────────────


class TestClinicalConsistencyHook:
    def _request(self) -> AnalyzeRequest:
        return AnalyzeRequest(
            image_base64="ZmFrZQ==",
            modality=Modality.EKG,
            valid_regions=[],
        )

    def test_hook_escalates_and_flags(self):
        hook = ClinicalConsistencyHook()
        res = _result(
            modality=Modality.EKG,
            severity=Severity.NORMAL,
            summary="Definite STEMI",
            checklist={"st_segment": _item("ST elevation", Severity.CRITICAL)},
        )
        out = hook.post_analyze(self._request(), res)
        assert out.severity is Severity.CRITICAL
        assert out.review_required is True

    def test_hook_passthrough_on_clean_result(self):
        hook = ClinicalConsistencyHook()
        res = _result(
            modality=Modality.EKG,
            severity=Severity.NORMAL,
            checklist={"st_segment": _item("isoelectric")},
        )
        out = hook.post_analyze(self._request(), res)
        assert out.review_required is False
        assert out.severity is Severity.NORMAL

    def test_hook_pre_analyze_is_passthrough(self):
        hook = ClinicalConsistencyHook()
        req = self._request()
        assert hook.pre_analyze(req) is req

    def test_hook_never_raises(self):
        # Even with an empty/odd result, the advisory hook must not raise.
        hook = ClinicalConsistencyHook()
        res = _result(modality=Modality.CT_BRAIN, severity=Severity.NORMAL)
        out = hook.post_analyze(self._request(), res)
        assert out is res


# ── Human-auditable text (catalogue / evidence / enforced description) ──


class TestAuditability:
    def test_condition_explain_is_human_readable(self):
        cond = RuleCondition(
            field="checklist.st_segment", op="contains_any", values=("elevat", "抬高")
        )
        text = cond.explain()
        assert "st_segment" in text
        assert "包含任一" in text
        assert "elevat" in text and "抬高" in text

    def test_condition_explain_severity(self):
        cond = RuleCondition(field="severity", op="severity_at_most", value="info")
        text = cond.explain()
        assert "整體嚴重度" in text
        assert "info" in text

    def test_rule_evidence_reports_matched_terms(self):
        res = _result(
            severity=Severity.NORMAL,
            checklist={"st_segment": _item("marked ST elevation")},
        )
        evidence = _stemi_rule().evidence(res)
        assert "elevat" in evidence

    def test_rule_evidence_empty_when_not_fired(self):
        res = _result(
            severity=Severity.NORMAL,
            checklist={"st_segment": _item("isoelectric")},
        )
        assert _stemi_rule().evidence(res) == ()

    def test_violation_audit_line_includes_evidence(self):
        engine = ClinicalConsistencyEngine(group_by_modality([_stemi_rule()]))
        res = _result(
            severity=Severity.NORMAL,
            checklist={"st_segment": _item("ST elevation")},
        )
        violations = engine.evaluate(res)
        line = violations[0].audit_line()
        assert "命中關鍵字" in line
        assert "elevat" in line

    def test_catalogue_entry_contains_all_audit_fields(self):
        entry = _stemi_rule().catalogue_entry()
        assert "r-stemi" in entry
        assert "ST elevation not flagged" in entry  # description
        assert "觸發條件" in entry
        assert "醫學依據" in entry
        assert "升級至 critical" in entry
        assert "標記人工複核" in entry

    def test_engine_catalogue_groups_by_modality(self):
        text = default_engine().catalogue()
        assert "=== CXR" in text
        assert "=== EKG" in text
        # every built-in rule id should appear in the catalogue
        for rule in builtin_rules():
            assert rule.id in text

    def test_empty_engine_catalogue_message(self):
        assert "沒有生效" in ClinicalConsistencyEngine({}).catalogue()

    def test_loader_rejects_rule_without_description(self, tmp_path):
        pack = """
rules:
  - id: undocumented
    modality: EKG
    conditions:
      - field: summary
        op: contains_any
        values: ["x"]
    message: "no description"
"""
        (tmp_path / "p.rules.yaml").write_text(pack, encoding="utf-8")
        assert load_rule_pack_dir(tmp_path) == []
