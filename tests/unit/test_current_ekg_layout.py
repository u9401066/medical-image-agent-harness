"""Typed EKG lead-inventory parsing."""

import pytest

from medical_image_harness.ekg_layout import (
    STANDARD_EKG_LEADS,
    normalize_ekg_row_strip_layout,
    parse_ekg_lead_inventory,
)


def _layout(names: list[str]) -> dict[str, object]:
    return {
        "format": "12lead_rows",
        "leads": [
            {
                "name": name,
                "label_visible": True,
                "bbox": [0.0, index / len(names), 1.0, 1 / len(names)],
            }
            for index, name in enumerate(names)
        ],
    }


def test_inventory_normalizes_common_names_and_is_complete() -> None:
    names = [
        "lead_I",
        "LEAD-II",
        "lead iii",
        "aVR",
        "aVL",
        "aVF",
        "v 1",
        "V2",
        "lead_V3",
        "V-4",
        "v_5",
        "V6",
    ]

    inventory = parse_ekg_lead_inventory(_layout(names))

    assert inventory.complete is True
    assert tuple(inventory.by_name()) == tuple(
        f"lead_{name}" for name in STANDARD_EKG_LEADS
    )
    assert inventory.validation_warnings() == []


def test_inventory_reports_missing_duplicate_hidden_and_malformed_entries() -> None:
    layout = _layout(["I", "I", "II", "not-a-lead"])
    layout["leads"][2]["label_visible"] = False

    inventory = parse_ekg_lead_inventory(layout)

    assert inventory.complete is False
    assert inventory.duplicate_names == ("lead_I",)
    assert inventory.malformed_entries == 2
    assert "lead_II" in inventory.missing_names
    warnings = inventory.validation_warnings()
    assert any("duplicate leads" in warning for warning in warnings)
    assert any("missing visible leads" in warning for warning in warnings)


def test_inventory_rejects_out_of_bounds_bbox() -> None:
    layout = _layout(list(STANDARD_EKG_LEADS))
    layout["leads"][0]["bbox"] = [0.9, 0.0, 0.2, 0.1]

    inventory = parse_ekg_lead_inventory(layout)

    assert inventory.complete is False
    assert inventory.malformed_entries == 1
    assert "lead_I" in inventory.missing_names


@pytest.mark.parametrize("visibility", [None, False, 0, 1, "true", "false", [], {}])
def test_inventory_requires_explicit_boolean_visible_label(visibility: object) -> None:
    layout = _layout(["I"])
    if visibility is None:
        layout["leads"][0].pop("label_visible")
    else:
        layout["leads"][0]["label_visible"] = visibility
    inventory = parse_ekg_lead_inventory(layout)
    assert inventory.leads == ()
    assert inventory.malformed_entries == 1
    assert "lead_I" in inventory.missing_names


@pytest.mark.parametrize("pixel_evidence", [None, {
    "method": "local_black_ink_row_periodicity_v2", "status": "ok",
    "is_12_row_strip": True, "detected_row_count": 12, "consistent_gap_count": 11,
}])
def test_row_normalizer_does_not_invent_missing_label_visibility(pixel_evidence) -> None:
    layout = _layout(list(STANDARD_EKG_LEADS))
    for lead in layout["leads"]:
        lead.pop("label_visible")
    normalized, repaired = normalize_ekg_row_strip_layout(layout, image_evidence=pixel_evidence)
    assert not repaired
    assert normalized == layout
    assert not parse_ekg_lead_inventory(normalized).leads


def test_inventory_accepts_object_shaped_normalized_bboxes() -> None:
    layout = _layout(list(STANDARD_EKG_LEADS))
    for lead in layout["leads"]:
        x, y, w, h = lead["bbox"]
        lead["bbox"] = {"x": x, "y": y, "w": w, "h": h}

    inventory = parse_ekg_lead_inventory(layout)

    assert inventory.complete is True


def test_inventory_marks_empty_layout_as_missing() -> None:
    inventory = parse_ekg_lead_inventory({})

    assert inventory.complete is False
    assert inventory.source_present is False
    assert inventory.validation_warnings() == [
        "EKG layout is missing a lead inventory"
    ]


def test_row_strip_normalizer_repairs_clipped_3x4_geometry() -> None:
    layout = {
        "format": "12lead_3x4",
        "leads": [
            {
                "name": name,
                "label_visible": True,
                "bbox": [0.0, min(1.0, index * 0.11), 0.25, 0.11 if index < 10 else 0.0],
            }
            for index, name in enumerate(STANDARD_EKG_LEADS)
        ],
    }

    normalized, repaired = normalize_ekg_row_strip_layout(layout)
    inventory = parse_ekg_lead_inventory(normalized)

    assert repaired is True
    assert normalized["format"] == "12lead_12x1"
    assert inventory.complete is True
    assert inventory.by_name()["lead_V6"].y == 11 / 12
    assert all(region.w == 1.0 for region in inventory.by_name().values())


def test_row_strip_normalizer_repairs_complete_but_misdeclared_geometry() -> None:
    layout = _layout(list(STANDARD_EKG_LEADS))
    layout["format"] = "12lead_3x4"

    normalized, repaired = normalize_ekg_row_strip_layout(layout)

    assert repaired is True
    assert normalized["format"] == "12lead_12x1"


def test_row_strip_normalizer_recovers_clipped_v5_v6_tail() -> None:
    layout = {
        "format": "12lead_3x4",
        "leads": [
            {
                "name": name,
                "label_visible": True,
                "bbox": [0.0, index / 10, 1.0, 0.1 if index < 10 else 0.0],
            }
            for index, name in enumerate(STANDARD_EKG_LEADS[:11])
        ],
    }

    normalized, repaired = normalize_ekg_row_strip_layout(layout)
    inventory = parse_ekg_lead_inventory(normalized)

    assert repaired is True
    assert inventory.complete is True
    assert tuple(inventory.by_name()) == tuple(
        f"lead_{name}" for name in STANDARD_EKG_LEADS
    )
    assert "11/12 model lead declarations recovered" in normalized["notes"]


def test_row_strip_normalizer_leaves_true_3x4_layout_unchanged() -> None:
    layout = {
        "format": "12lead_3x4",
        "leads": [
            {
                "name": name,
                "label_visible": True,
                "bbox": [(index % 4) / 4, (index // 4) / 3, 0.25, 1 / 3],
            }
            for index, name in enumerate(STANDARD_EKG_LEADS)
        ],
    }

    normalized, repaired = normalize_ekg_row_strip_layout(layout)

    assert repaired is False
    assert normalized == layout


def test_row_strip_normalizer_recovers_eight_leads_only_with_image_evidence() -> None:
    layout = _layout(list(STANDARD_EKG_LEADS[:8]))
    layout["format"] = "12lead_3x4"
    evidence = {
        "method": "local_black_ink_row_periodicity_v2",
        "status": "ok",
        "is_12_row_strip": True,
        "detected_row_count": 12,
        "consistent_gap_count": 10,
    }

    unchanged, repaired_without_evidence = normalize_ekg_row_strip_layout(layout)
    normalized, repaired = normalize_ekg_row_strip_layout(
        layout,
        image_evidence=evidence,
    )

    assert repaired_without_evidence is False
    assert unchanged == layout
    assert repaired is True
    assert normalized["format"] == "12lead_12x1"
    assert parse_ekg_lead_inventory(normalized).complete is True


def test_row_strip_normalizer_accepts_format_separators_and_hidden_tail() -> None:
    layout = _layout(list(STANDARD_EKG_LEADS))
    layout["format"] = "12-lead standard"
    for index, lead in enumerate(layout["leads"]):
        lead["bbox"] = [0.0, index * 0.12, 1.0, 0.12]
    for lead in layout["leads"][-3:]:
        lead["label_visible"] = False
        lead["bbox"] = [0.0, 0.0, 0.0, 0.0]
    evidence = {
        "method": "local_black_ink_row_periodicity_v2",
        "status": "ok",
        "is_12_row_strip": True,
        "detected_row_count": 12,
        "consistent_gap_count": 11,
    }

    normalized, repaired = normalize_ekg_row_strip_layout(
        layout,
        image_evidence=evidence,
    )

    assert repaired is True
    assert normalized["format"] == "12lead_12x1"
    assert parse_ekg_lead_inventory(normalized).complete is True


def test_row_strip_normalizer_repairs_degenerate_full_frame_geometry_with_evidence() -> None:
    layout = {
        "format": "standard_12_lead",
        "leads": [
            {
                "name": name,
                "label_visible": index < 9,
                "bbox": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
            }
            for index, name in enumerate(STANDARD_EKG_LEADS)
        ],
    }
    evidence = {
        "method": "local_black_ink_row_periodicity_v2",
        "status": "ok",
        "is_12_row_strip": True,
        "detected_row_count": 12,
        "consistent_gap_count": 11,
    }

    normalized, repaired = normalize_ekg_row_strip_layout(
        layout,
        image_evidence=evidence,
    )

    assert repaired is True
    assert normalized["format"] == "12lead_12x1"
    assert parse_ekg_lead_inventory(normalized).complete is True
    assert normalized["leads"][1]["bbox"] == [0.0, 1 / 12, 1.0, 1 / 12]


def test_row_strip_normalizer_builds_geometry_from_compact_order_and_evidence() -> None:
    layout = {
        "format": "12lead_12x1",
        "lead_order": list(STANDARD_EKG_LEADS),
        "rhythm_strip_leads": [],
        "rhythm_strip_bbox": None,
    }
    evidence = {
        "method": "local_black_ink_row_periodicity_v2",
        "status": "ok",
        "is_12_row_strip": True,
        "detected_row_count": 12,
        "consistent_gap_count": 11,
    }

    unchanged, repaired_without_evidence = normalize_ekg_row_strip_layout(layout)
    normalized, repaired = normalize_ekg_row_strip_layout(
        layout,
        image_evidence=evidence,
    )

    assert repaired_without_evidence is False
    assert unchanged == layout
    assert repaired is True
    assert parse_ekg_lead_inventory(normalized).complete is True


@pytest.mark.parametrize(
    ("detected_row_count", "consistent_gap_count"),
    [(9, 6), (11, 9), (13, 9)],
)
def test_compact_row_layout_tolerates_bounded_detector_peak_error(
    detected_row_count: int,
    consistent_gap_count: int,
) -> None:
    layout = {
        "format": "12lead_12x1",
        "lead_order": list(STANDARD_EKG_LEADS),
        "rhythm_strip_bbox": None,
    }
    evidence = {
        "method": "local_black_ink_row_periodicity_v2",
        "status": "ok",
        "is_12_row_strip": False,
        "detected_row_count": detected_row_count,
        "consistent_gap_count": consistent_gap_count,
        "median_row_period_normalized": 1 / 12,
        "vertical_span_ratio": 0.95,
    }

    normalized, repaired = normalize_ekg_row_strip_layout(
        layout,
        image_evidence=evidence,
    )

    assert repaired is True
    assert parse_ekg_lead_inventory(normalized).complete is True


def test_compact_row_layout_rejects_nonperiodic_near_miss_evidence() -> None:
    layout = {
        "format": "12lead_12x1",
        "lead_order": list(STANDARD_EKG_LEADS),
    }
    evidence = {
        "method": "local_black_ink_row_periodicity_v2",
        "status": "ok",
        "is_12_row_strip": False,
        "detected_row_count": 11,
        "consistent_gap_count": 9,
        "median_row_period_normalized": 0.14,
        "vertical_span_ratio": 0.95,
    }

    normalized, repaired = normalize_ekg_row_strip_layout(
        layout,
        image_evidence=evidence,
    )

    assert repaired is False
    assert normalized == layout

