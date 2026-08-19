"""Input validation guard -- rejects malformed analyze requests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from medical_image_harness.hooks import AnalyzeHook, AnalyzeRequest, HookError
from medical_image_harness.profiles import (
    ModalityRegistry,
    get_active_registry,
)

if TYPE_CHECKING:
    from medical_image_harness.models import AnalysisResult

logger = structlog.get_logger(__name__)

# Limits (base64 size ~ 4/3 x raw bytes)
_MIN_B64_LEN = 128  # ~96 bytes raw -- too small to be a real image
_MAX_B64_LEN = 15_000_000  # ~11 MB raw -- prevent OOM


class InputGuard(AnalyzeHook):
    """Pre-analyze guardrail: validates image, modality, and regions."""

    def __init__(self, *, registry: ModalityRegistry | None = None) -> None:
        self._registry = registry or get_active_registry()

    def pre_analyze(self, request: AnalyzeRequest) -> AnalyzeRequest:
        # 1. Image must not be empty
        if not request.image_base64:
            raise HookError("影像資料為空 (empty image)")

        # 2. Image size bounds
        b64_len = len(request.image_base64)
        if b64_len < _MIN_B64_LEN:
            raise HookError(
                f"影像太小 ({b64_len} chars), 可能不是有效圖片"
            )
        if b64_len > _MAX_B64_LEN:
            raise HookError(
                f"影像過大 ({b64_len // 1_000_000} MB), 超過上限"
            )

        # 3. Modality must be supported (not AUTO at this stage)
        profile = self._registry.get(request.modality.value)
        if profile is None or not profile.supported:
            supported = ", ".join(self._registry.supported_keys())
            raise HookError(
                f"不支援的影像模態: {request.modality.value}, "
                f"支援: {supported}"
            )

        # 4. Valid regions must not be empty
        if not request.valid_regions:
            raise HookError("缺少有效區域定義 (valid_regions is empty)")

        logger.debug(
            "InputGuard passed",
            modality=request.modality.value,
            image_size=b64_len,
            regions=len(request.valid_regions),
        )
        return request

    def post_analyze(
        self, _request: AnalyzeRequest, result: AnalysisResult
    ) -> AnalysisResult:
        return result  # Input guard only validates input
