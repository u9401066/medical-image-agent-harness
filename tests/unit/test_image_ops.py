import base64
import io

import pytest
from PIL import Image

from medical_image_harness.image_ops import (
    ImageOperationError,
    PillowCropper,
    crop_source_image,
    inspect_image,
)
from medical_image_harness.models import RegionRect


def _image() -> str:
    image = Image.new("RGB", (100, 80), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_crop_returns_source_bound_receipt() -> None:
    source = _image()
    result = crop_source_image(source, RegionRect(0.1, 0.25, 0.5, 0.5))
    assert result.pixel_box == (10, 20, 60, 60)
    assert result.source_sha256 == inspect_image(source).sha256
    assert inspect_image(result.image_base64).width == 50
    assert PillowCropper()(source, result.source_region) == result.image_base64


def test_out_of_bounds_or_empty_crop_fails_closed() -> None:
    source = _image()
    with pytest.raises(ImageOperationError, match="beyond"):
        crop_source_image(source, RegionRect(0.9, 0.9, 0.2, 0.2))
    with pytest.raises(ImageOperationError, match="positive area"):
        crop_source_image(source, RegionRect(0.2, 0.2, 0.0, 0.1))


def test_invalid_image_never_falls_back_to_input() -> None:
    invalid = base64.b64encode(b"not an image").decode("ascii")
    with pytest.raises(ImageOperationError, match="decoded safely"):
        crop_source_image(invalid, RegionRect(0.0, 0.0, 1.0, 1.0))
