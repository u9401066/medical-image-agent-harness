import pytest

from medical_image_harness.study import ImageAsset, StudyManifest


def _asset(**overrides) -> ImageAsset:
    values = {
        "id": "image-1",
        "sha256": "a" * 64,
        "modality": "CXR",
        "source_kind": "rendered_image",
        "view": "PA",
    }
    values.update(overrides)
    return ImageAsset(**values)


def test_manifest_reports_missing_view_and_incomplete_host_state() -> None:
    manifest = StudyManifest(
        id="study-1",
        modality="CXR",
        assets=(_asset(),),
        complete=False,
        expected_views=("PA", "lateral"),
    )
    assert manifest.quality_issues() == [
        "missing expected views: lateral",
        "host marked study incomplete",
    ]


def test_manifest_rejects_non_deidentified_asset() -> None:
    with pytest.raises(ValueError, match="de-identified"):
        _asset(deidentified=False)
