"""Scientific harness for auditable agent-assisted medical image co-reading."""

from medical_image_harness.models import (
    AnalysisResult,
    ChecklistItem,
    ClaimType,
    Evidence,
    Finding,
    Modality,
    Observation,
    Polarity,
    RegionRect,
    Severity,
    UserRegionAnnotation,
    VerificationStatus,
)
from medical_image_harness.profiles import (
    ModalityProfile,
    ModalityRegistry,
    default_registry,
)

__all__ = [
    "AnalysisResult",
    "ChecklistItem",
    "ClaimType",
    "Evidence",
    "Finding",
    "Modality",
    "ModalityProfile",
    "ModalityRegistry",
    "Observation",
    "Polarity",
    "RegionRect",
    "Severity",
    "UserRegionAnnotation",
    "VerificationStatus",
    "default_registry",
]

__version__ = "0.1.0"
