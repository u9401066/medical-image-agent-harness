"""Deterministic, PHI-free provenance records for reproducible evaluations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


@dataclass(frozen=True)
class TransformationRecord:
    """One deterministic transformation from an immutable source image."""

    operation: str
    output_sha256: str
    parameters: dict[str, object] = field(default_factory=dict)
    parent_sha256: str = ""


@dataclass(frozen=True)
class InputProvenance:
    """Non-identifying facts that bind an analysis to its exact input."""

    source_image_sha256: str
    source_kind: str
    deidentified: bool
    transformations: tuple[TransformationRecord, ...] = ()
    dataset_id: str = ""
    case_id: str = ""

    def __post_init__(self) -> None:
        if len(self.source_image_sha256) != 64:
            raise ValueError("source_image_sha256 must be a 64-character digest")
        int(self.source_image_sha256, 16)
        if not self.deidentified:
            raise ValueError("the public harness accepts only de-identified inputs")


@dataclass(frozen=True)
class RunProvenance:
    """Auditable run metadata without hidden chain-of-thought."""

    protocol_version: str
    skill_sha256: str
    schema_sha256: str
    model_id: str
    agent_surface: str
    input: InputProvenance
    started_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    tool_receipts: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return canonical_json_sha256(self.to_dict())
