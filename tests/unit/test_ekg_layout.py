"""Typed EKG lead-inventory parsing."""

from medical_image_harness.ekg_layout import (
    STANDARD_EKG_LEADS,
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


def test_inventory_marks_empty_layout_as_missing() -> None:
    inventory = parse_ekg_lead_inventory({})

    assert inventory.complete is False
    assert inventory.source_present is False
    assert inventory.validation_warnings() == [
        "EKG layout is missing a lead inventory"
    ]
