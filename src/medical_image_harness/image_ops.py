"""Fail-closed image inspection and source-bound crop operations."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from medical_image_harness.models import RegionRect
from medical_image_harness.provenance import sha256_bytes

MAX_PIXELS = 100_000_000


class ImageOperationError(ValueError):
    """An image operation cannot preserve the declared safety invariant."""


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    format: str
    sha256: str


@dataclass(frozen=True)
class CropResult:
    image_base64: str
    source_region: RegionRect
    source_sha256: str
    crop_sha256: str
    pixel_box: tuple[int, int, int, int]


def decode_image(image_base64: str) -> tuple[bytes, Image.Image]:
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ImageOperationError("image is not valid base64") from exc
    if not raw:
        raise ImageOperationError("image is empty")
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageOperationError("image cannot be decoded safely") from exc
    if image.width <= 0 or image.height <= 0:
        raise ImageOperationError("image has invalid dimensions")
    if image.width * image.height > MAX_PIXELS:
        raise ImageOperationError("image exceeds the pixel safety limit")
    return raw, image


def inspect_image(image_base64: str) -> ImageInfo:
    raw, image = decode_image(image_base64)
    return ImageInfo(
        width=image.width,
        height=image.height,
        format=(image.format or "unknown").upper(),
        sha256=sha256_bytes(raw),
    )


def crop_source_image(image_base64: str, region: RegionRect) -> CropResult:
    """Crop only a strict, positive-area subset of the immutable input image."""

    if region.w <= 0.0 or region.h <= 0.0:
        raise ImageOperationError("crop region must have positive area")
    if region.x + region.w > 1.0 + 1e-9 or region.y + region.h > 1.0 + 1e-9:
        raise ImageOperationError("crop region extends beyond the source image")
    raw, image = decode_image(image_base64)
    left = max(0, round(region.x * image.width))
    top = max(0, round(region.y * image.height))
    right = min(image.width, round((region.x + region.w) * image.width))
    bottom = min(image.height, round((region.y + region.h) * image.height))
    if right <= left or bottom <= top:
        raise ImageOperationError("crop rounds to an empty pixel rectangle")
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG", optimize=True)
    crop_bytes = buffer.getvalue()
    return CropResult(
        image_base64=base64.b64encode(crop_bytes).decode("ascii"),
        source_region=region,
        source_sha256=sha256_bytes(raw),
        crop_sha256=sha256_bytes(crop_bytes),
        pixel_box=(left, top, right, bottom),
    )


class PillowCropper:
    """Callable adapter for :class:`medical_image_harness.multipass.ImageCropper`."""

    def __call__(self, image_base64: str, region: RegionRect) -> str:
        return crop_source_image(image_base64, region).image_base64
