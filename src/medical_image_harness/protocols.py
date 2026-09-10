"""Backend-neutral capabilities consumed by the scientific harness."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from medical_image_harness.models import AnalysisResult, Modality

PENDING_MULTIPASS_REASON = "pending_multipass_review"


@dataclass(frozen=True)
class StageTools:
    """Host-supplied auditable identities; never executable tool implementations."""

    coarse: str = "vision_analysis"
    refinement: str = "crop_region_base64+vision_analysis"
    finalize: str = "report_reconciliation"


class AnalyzerPort(Protocol):
    """Minimal provider-neutral capability required by the scientific loop."""

    async def analyze(
        self,
        image_base64: str,
        modality: Modality,
        valid_regions: list[str],
    ) -> AnalysisResult: ...


class VisionAnalyzerService(ABC):
    """Connected analyzer lifecycle implemented by host-owned adapters."""

    @abstractmethod
    async def analyze(
        self,
        image_base64: str,
        modality: Modality,
        valid_regions: list[str],
    ) -> AnalysisResult:
        """Analyze one source image and return a structured result."""

    @abstractmethod
    async def chat(self, message: str) -> str:
        """Ask a follow-up question about the current image session."""

    @abstractmethod
    async def connect(self) -> None:
        """Open the backend connection or session."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the backend connection or session."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return whether the adapter can accept work."""
