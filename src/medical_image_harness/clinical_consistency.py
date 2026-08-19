"""Clinical consistency hook — runs the guideline-grounded safety net.

A post-analyze guardrail that applies the data-driven
:class:`~medical_image_harness.clinical_rules.ClinicalConsistencyEngine` to the
AI's structured output. It may enforce an explicit severity floor and/or flag
for human review (never downgrades, never rewrites findings, never raises), so
it is an auditable advisory layer on top of an agent's read. Order it **after**
``OutputValidator`` so it operates on a schema-validated result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from medical_image_harness.clinical_rules import (
    ClinicalConsistencyEngine,
    default_engine,
)
from medical_image_harness.hooks import AnalyzeHook, AnalyzeRequest

if TYPE_CHECKING:
    from medical_image_harness.models import AnalysisResult

logger = structlog.get_logger(__name__)


class ClinicalConsistencyHook(AnalyzeHook):
    """Escalates/flags results that contradict their own structured read."""

    def __init__(self, engine: ClinicalConsistencyEngine | None = None) -> None:
        self._engine = engine or default_engine()

    def pre_analyze(self, request: AnalyzeRequest) -> AnalyzeRequest:
        return request  # advisory only operates on output

    def post_analyze(
        self,
        request: AnalyzeRequest,
        result: AnalysisResult,
    ) -> AnalysisResult:
        severity_before = result.severity
        violations = self._engine.apply(result)
        for violation in violations:
            floor = violation.rule.escalate_to
            if floor is None:
                action = "review_only"
            elif result.severity is severity_before:
                action = "severity_floor_already_met"
            else:
                action = "severity_escalated"
            logger.warning(
                "clinical_consistency_flag",
                rule_id=violation.rule.id,
                modality=result.modality.value,
                action=action,
                severity_before=severity_before.value,
                severity_after=result.severity.value,
                severity_floor=floor.value if floor is not None else None,
                guideline=violation.rule.guideline,
                evidence=list(violation.evidence),
                audit=violation.audit_line(),
            )
        return result
