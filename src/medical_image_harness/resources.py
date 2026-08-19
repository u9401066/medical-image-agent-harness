"""Load the canonical agent skill and modality references as package resources."""

from __future__ import annotations

import hashlib
from importlib.resources import files
from pathlib import Path
from typing import Any

_MODALITY_REFERENCES = {
    "EKG": "ekg.md",
    "ECG": "ekg.md",
    "DICOM-EKG-ANALYSIS": "ekg.md",
    "CXR": "cxr.md",
    "CHEST_XRAY": "cxr.md",
    "DICOM-CXR-ANALYSIS": "cxr.md",
    "CT_BRAIN": "ct-brain.md",
    "CTBRAIN": "ct-brain.md",
    "DICOM-CT-BRAIN-ANALYSIS": "ct-brain.md",
}


def _source_skill_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / ".agents"
        / "skills"
        / "medical-image-reading"
    )


def skill_root() -> Any:
    """Return the installed skill root, falling back to the source checkout."""

    installed = files("medical_image_harness").joinpath(
        "skills", "medical-image-reading"
    )
    if installed.is_dir():
        return installed
    source = _source_skill_root()
    if source.is_dir():
        return source
    raise FileNotFoundError("medical-image-reading skill resources are unavailable")


def load_skill() -> str:
    """Return the canonical cross-agent SKILL.md."""

    return skill_root().joinpath("SKILL.md").read_text(encoding="utf-8")


def load_reference(name: str) -> str:
    """Load one named skill reference without accepting path traversal."""

    filename = Path(name).name
    if filename != name or not filename.endswith(".md"):
        raise ValueError("reference name must be one Markdown filename")
    return skill_root().joinpath("references", filename).read_text(encoding="utf-8")


def load_modality_prompt(modality_or_skill: str) -> str:
    """Compose the shared protocol with the matching modality/output contract."""

    key = modality_or_skill.strip().upper().replace("-", "_")
    reference = _MODALITY_REFERENCES.get(key)
    if reference is None:
        reference = _MODALITY_REFERENCES.get(
            modality_or_skill.strip().upper(),
        )
    if reference is None:
        raise KeyError(f"unsupported modality or skill: {modality_or_skill}")
    sections = (
        load_skill(),
        load_reference("core-protocol.md"),
        load_reference(reference),
        load_reference("output-contract.md"),
    )
    return "\n\n".join(section.strip() for section in sections if section.strip())


def skill_sha256() -> str:
    """Hash all canonical instruction files in stable relative-path order."""

    root = skill_root()
    entries = [root.joinpath("SKILL.md")]
    references = root.joinpath("references")
    entries.extend(
        sorted(
            (entry for entry in references.iterdir() if entry.name.endswith(".md")),
            key=lambda entry: entry.name,
        )
    )
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
