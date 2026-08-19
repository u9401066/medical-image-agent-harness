import pytest

from medical_image_harness.provenance import InputProvenance, canonical_json_sha256


def test_canonical_json_hash_is_key_order_independent() -> None:
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256(
        {"a": 1, "b": 2}
    )


def test_input_provenance_fails_closed_without_deidentification() -> None:
    with pytest.raises(ValueError, match="de-identified"):
        InputProvenance(
            source_image_sha256="a" * 64,
            source_kind="rendered_image",
            deidentified=False,
        )


def test_canonical_json_hash_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_sha256({"x": float("nan")})
