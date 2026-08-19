"""Pure request and hook contracts for deterministic guardrails."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from medical_image_harness.models import AnalysisResult, Modality


class HookError(Exception):
    """Raised when a deterministic guardrail rejects an operation."""


@dataclass
class AnalyzeRequest:
    """Snapshot of one image-analysis request."""

    image_base64: str
    modality: Modality
    valid_regions: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class AnalyzeHook(ABC):
    """Provider-neutral middleware around an image-analysis call."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def pre_analyze(self, request: AnalyzeRequest) -> AnalyzeRequest:
        """Validate or transform a request before model inference."""

    @abstractmethod
    def post_analyze(
        self,
        request: AnalyzeRequest,
        result: AnalysisResult,
    ) -> AnalysisResult:
        """Validate or transform a structured result after inference."""
