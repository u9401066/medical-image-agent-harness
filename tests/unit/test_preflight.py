from copy import deepcopy

import pytest
from test_schema import _payload

from medical_image_harness.schema import (
    load_schema,
    preflight_validation_errors,
    validation_errors,
)


def prefix():
    payload = _payload()
    payload["analysis_trace"] = [
        item
        for item in payload["analysis_trace"]
        if item["stage"] not in {"contract_validation", "human_handoff"}
    ]
    return payload


def test_preflight_is_not_final_acceptance_and_does_not_rewrite_history():
    payload = prefix()
    before = deepcopy(payload)
    schema = load_schema()
    assert preflight_validation_errors(payload) == []
    assert validation_errors(payload)
    assert payload == before and load_schema() == schema
    assert schema["properties"]["analysis_trace"]["minItems"] == 6


@pytest.mark.parametrize(
    "change",
    [
        "missing_blind",
        "order",
        "duplicate",
        "failed",
        "future_validation",
        "future_handoff",
        "bad_box",
        "unbound_box",
        "unknown_observation",
        "missing_axis",
        "not_reviewed",
        "wrong_scope",
        "complete",
        "quality",
    ],
)
def test_preflight_retains_content_and_executed_prefix_guards(change):
    payload = prefix()
    trace = payload["analysis_trace"]
    if change == "missing_blind":
        trace[:] = [item for item in trace if item["stage"] != "blind_pass"]
    elif change == "order":
        trace.reverse()
    elif change == "duplicate":
        trace.append(deepcopy(trace[-1]))
    elif change == "failed":
        trace[-1]["status"] = "failed"
    elif change.startswith("future"):
        trace.append(
            {
                "stage": "human_handoff"
                if change.endswith("handoff")
                else "contract_validation",
                "status": "completed",
                "detail": "Not yet executed.",
            }
        )
    elif change == "bad_box":
        payload["findings"][0]["bboxes"][0]["w"] = 2
    elif change == "unbound_box":
        payload["findings"][0]["bboxes"][0]["source_image_sha256"] = "b" * 64
    elif change == "unknown_observation":
        payload["summary_observation_ids"] = ["missing"]
    elif change == "missing_axis":
        del payload["checklist"]["lungs"]
    elif change == "not_reviewed":
        payload["review_required"] = False
    elif change == "wrong_scope":
        payload["assessment_scope"] = "complete_study"
    elif change == "complete":
        payload["incomplete"] = False
    else:
        payload["image_quality"]["adequacy"] = "diagnostic"
    assert preflight_validation_errors(payload)


def test_final_contract_still_requires_real_validation_and_handoff_events():
    payload = _payload()
    assert validation_errors(payload) == []
    assert preflight_validation_errors(payload)
