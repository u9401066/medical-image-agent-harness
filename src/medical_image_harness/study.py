"""Provider-neutral study inventory and quality-gate contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImageAsset:
    """One de-identified image/frame identified without patient attributes."""

    id: str
    sha256: str
    modality: str
    source_kind: str
    view: str = ""
    laterality: str = ""
    orientation: str = ""
    series_ref: str = ""
    frame_number: int | None = None
    window: str = ""
    deidentified: bool = True

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("image asset id is required")
        if len(self.sha256) != 64:
            raise ValueError("image asset sha256 must be 64 characters")
        int(self.sha256, 16)
        if not self.deidentified:
            raise ValueError("study manifests accept only de-identified assets")
        if self.frame_number is not None and self.frame_number < 1:
            raise ValueError("frame_number is one-based")


@dataclass(frozen=True)
class StudyManifest:
    """Minimal inventory used to distinguish one image from a complete study."""

    id: str
    modality: str
    assets: tuple[ImageAsset, ...]
    complete: bool
    expected_views: tuple[str, ...] = ()
    expected_series: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def quality_issues(self) -> list[str]:
        issues = list(self.limitations)
        if not self.assets:
            issues.append("study has no image assets")
        duplicate_ids = _duplicates(asset.id for asset in self.assets)
        if duplicate_ids:
            issues.append("duplicate asset ids: " + ", ".join(duplicate_ids))
        modalities = {asset.modality.upper() for asset in self.assets}
        if modalities and modalities != {self.modality.upper()}:
            issues.append("asset modality disagrees with study modality")
        present_views = {asset.view.casefold() for asset in self.assets if asset.view}
        missing_views = [
            view for view in self.expected_views if view.casefold() not in present_views
        ]
        if missing_views:
            issues.append("missing expected views: " + ", ".join(missing_views))
        present_series = {asset.series_ref for asset in self.assets if asset.series_ref}
        missing_series = [
            series for series in self.expected_series if series not in present_series
        ]
        if missing_series:
            issues.append("missing expected series: " + ", ".join(missing_series))
        if not self.complete:
            issues.append("host marked study incomplete")
        return list(dict.fromkeys(issues))


def _duplicates(values) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates
