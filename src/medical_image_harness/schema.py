"""JSON Schema loading and result validation."""

from __future__ import annotations

import math
from importlib.resources import files
from pathlib import Path
from typing import Any

import jsonschema

from medical_image_harness.profiles import default_registry

_TRACE_ORDER = {
    stage: index
    for index, stage in enumerate(
        (
            "intake",
            "quality_gate",
            "blind_pass",
            "independent_evidence",
            "reconcile",
            "targeted_second_look",
            "contract_validation",
            "human_handoff",
        )
    )
}
_REQUIRED_TRACE_STAGES = {
    "intake",
    "quality_gate",
    "blind_pass",
    "reconcile",
    "contract_validation",
    "human_handoff",
}


def _source_schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas" / "analysis-result.schema.json"


def schema_text() -> str:
    installed = files("medical_image_harness").joinpath(
        "schemas", "analysis-result.schema.json"
    )
    if installed.is_file():
        return installed.read_text(encoding="utf-8")
    return _source_schema_path().read_text(encoding="utf-8")


def load_schema() -> dict[str, Any]:
    import json

    payload = json.loads(schema_text())
    jsonschema.Draft202012Validator.check_schema(payload)
    return payload


def validation_errors(payload: object) -> list[str]:
    validator = jsonschema.Draft202012Validator(load_schema())
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    rendered: list[str] = []
    for error in errors:
        path = "/".join(str(part) for part in error.absolute_path) or "$"
        rendered.append(f"{path}: {error.message}")
    rendered.extend(_semantic_errors(payload))
    return rendered


def _semantic_errors(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    errors: list[str] = _non_finite_errors(payload)

    def records(key: str) -> list[dict[str, Any]]:
        value = payload.get(key)
        if not isinstance(value, list):
            return []
        return [entry for entry in value if isinstance(entry, dict)]

    def list_value(value: object) -> list[Any]:
        return value if isinstance(value, list) else []

    def unique_ids(key: str) -> set[str]:
        seen: set[str] = set()
        for entry in records(key):
            identifier = entry.get("id")
            if not isinstance(identifier, str) or not identifier:
                continue
            if identifier in seen:
                errors.append(f"{key}: duplicate id {identifier!r}")
            seen.add(identifier)
        return seen

    observation_ids = unique_ids("observations")
    evidence_ids = unique_ids("evidence")
    unique_ids("findings")
    observations = {entry.get("id"): entry for entry in records("observations")}
    evidence_by_id = {entry.get("id"): entry for entry in records("evidence")}

    for reference in list_value(payload.get("summary_observation_ids")):
        if not isinstance(reference, str):
            continue
        if reference not in observation_ids:
            errors.append(f"summary_observation_ids: unresolved id {reference!r}")
            continue
        observation = observations[reference]
        if not observation.get("assessable") or observation.get("status") not in {
            "supported",
            "possible",
        }:
            errors.append(
                "summary_observation_ids: impression references an unverified "
                f"observation {reference!r}"
            )

    modality = payload.get("modality")
    if isinstance(modality, str):
        profile = default_registry().get(modality)
        if profile is not None:
            checklist = payload.get("checklist")
            keys = set(checklist) if isinstance(checklist, dict) else set()
            missing = sorted(profile.checklist_keys - keys)
            if missing:
                errors.append("checklist: missing required axes: " + ", ".join(missing))
    checklist = payload.get("checklist")
    if isinstance(checklist, dict):
        for key, item in checklist.items():
            if not isinstance(item, dict) or not item.get("assessable"):
                continue
            reference = item.get("evidence")
            if reference not in observation_ids:
                errors.append(
                    f"checklist/{key}: assessable item lacks a resolved observation"
                )

    for finding in records("findings"):
        finding_id = finding.get("id", "?")
        finding_observation_ids = {
            reference
            for reference in list_value(finding.get("observation_ids"))
            if isinstance(reference, str)
        }
        finding_evidence_ids = {
            reference
            for reference in list_value(finding.get("evidence_ids"))
            if isinstance(reference, str)
        }
        observation_evidence_ids: set[str] = set()
        for reference in finding_observation_ids:
            if reference not in observation_ids:
                errors.append(
                    f"findings/{finding_id}: unresolved observation {reference!r}"
                )
                continue
            observation = observations[reference]
            observation_evidence_ids.update(
                reference
                for reference in list_value(observation.get("evidence_ids"))
                if isinstance(reference, str)
            )
            if observation.get("polarity") == "absent" or observation.get("status") in {
                "contradicted",
                "unevaluable",
            }:
                errors.append(
                    f"findings/{finding_id}: references a non-retainable observation "
                    f"{reference!r}"
                )
        for reference in finding_evidence_ids:
            if reference not in evidence_ids:
                errors.append(f"findings/{finding_id}: unresolved evidence {reference!r}")
        if not finding_evidence_ids.issubset(observation_evidence_ids):
            errors.append(
                f"findings/{finding_id}: evidence is not linked through its observations"
            )

        linked_box_signatures = {
            _box_signature(box)
            for reference in finding_evidence_ids
            for box in list_value(
                evidence_by_id.get(reference, {}).get("bboxes")
            )
            if isinstance(box, dict)
        }
        for index, box in enumerate(list_value(finding.get("bboxes"))):
            if not isinstance(box, dict):
                continue
            path = f"findings/{finding_id}/bboxes/{index}"
            errors.extend(_box_errors(box, path))
            if box.get("verified") is not True:
                errors.append(f"{path}: a retained finding box must be verified")
            if _box_signature(box) not in linked_box_signatures:
                errors.append(f"{path}: box is not present in linked evidence")

    for observation in records("observations"):
        references = observation.get("evidence_ids", [])
        for reference in references if isinstance(references, list) else []:
            if isinstance(reference, str) and reference not in evidence_ids:
                errors.append(
                    f"observations/{observation.get('id', '?')}: unresolved evidence {reference!r}"
                )
        if observation.get("assessable") and not references:
            errors.append(
                f"observations/{observation.get('id', '?')}: assessable claim has no evidence"
            )

    provenance = payload.get("input_provenance")
    source_hash = (
        provenance.get("source_image_sha256")
        if isinstance(provenance, dict)
        else None
    )
    manifest = payload.get("study_manifest")
    assets = list_value(manifest.get("assets")) if isinstance(manifest, dict) else []
    asset_by_id = {
        asset.get("id"): asset
        for asset in assets
        if isinstance(asset, dict) and isinstance(asset.get("id"), str)
    }
    asset_hashes = {
        asset.get("sha256")
        for asset in asset_by_id.values()
        if isinstance(asset.get("sha256"), str)
    }
    if len(asset_by_id) != len(assets):
        errors.append("study_manifest/assets: asset ids must be present and unique")
    if isinstance(source_hash, str) and source_hash not in asset_hashes:
        errors.append("input_provenance: source hash is not a study-manifest asset")
    allowed_hashes = set(asset_hashes)
    if isinstance(provenance, dict):
        transformations = provenance.get("transformations", [])
        if isinstance(transformations, list):
            for index, transform in enumerate(transformations):
                if not isinstance(transform, dict):
                    continue
                parent_hash = transform.get("parent_sha256")
                if parent_hash not in allowed_hashes:
                    errors.append(
                        f"input_provenance/transformations/{index}: unknown parent hash"
                    )
                output_hash = transform.get("output_sha256")
                if isinstance(output_hash, str):
                    allowed_hashes.add(output_hash)

    if isinstance(manifest, dict):
        if manifest.get("modality") != modality:
            errors.append("study_manifest: modality does not match result")
        present_views = {
            asset.get("view", "").casefold()
            for asset in asset_by_id.values()
            if isinstance(asset.get("view"), str) and asset.get("view")
        }
        present_series = {
            asset.get("series_ref")
            for asset in asset_by_id.values()
            if isinstance(asset.get("series_ref"), str) and asset.get("series_ref")
        }
        missing_views = {
            view
            for view in list_value(manifest.get("expected_views"))
            if isinstance(view, str) and view.casefold() not in present_views
        }
        missing_series = {
            series
            for series in list_value(manifest.get("expected_series"))
            if isinstance(series, str) and series not in present_series
        }
        if manifest.get("complete") and (missing_views or missing_series):
            errors.append("study_manifest: complete study is missing expected inputs")

    primary_assets = [
        asset for asset in asset_by_id.values() if asset.get("sha256") == source_hash
    ]
    if (
        isinstance(provenance, dict)
        and primary_assets
        and primary_assets[0].get("source_kind") != provenance.get("source_kind")
    ):
        errors.append("input_provenance: source kind disagrees with primary asset")

    _check_study_scope(payload, manifest, provenance, errors)

    for evidence in records("evidence"):
        evidence_hash = evidence.get("source_image_sha256")
        if evidence_hash not in allowed_hashes:
            errors.append(
                f"evidence/{evidence.get('id', '?')}: source hash is not in provenance"
            )
        if evidence.get("kind") == "tool_output" and (
            not evidence.get("tool_name") or not evidence.get("tool_version")
        ):
            errors.append(
                f"evidence/{evidence.get('id', '?')}: tool output lacks name/version"
            )
        if evidence.get("kind") != "tool_output" and evidence.get(
            "source_ref"
        ) not in asset_by_id:
            errors.append(
                f"evidence/{evidence.get('id', '?')}: source_ref is not a manifest asset"
            )
        for index, box in enumerate(list_value(evidence.get("bboxes"))):
            if not isinstance(box, dict):
                continue
            path = f"evidence/{evidence.get('id', '?')}/bboxes/{index}"
            errors.extend(_box_errors(box, path))
            bound_hash = box.get("source_image_sha256")
            if bound_hash != evidence_hash:
                errors.append(f"{path}: source hash mismatch")

    _check_trace(payload, errors)
    return errors


def _box_signature(box: dict[str, Any]) -> tuple[object, ...]:
    return tuple(
        box.get(key)
        for key in (
            "x",
            "y",
            "w",
            "h",
            "coordinate_space",
            "verified",
            "source_image_sha256",
        )
    )


def _non_finite_errors(value: object, path: str = "$") -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{path}: non-finite numbers are forbidden"]
    if isinstance(value, dict):
        return [
            error
            for key, item in value.items()
            for error in _non_finite_errors(item, f"{path}/{key}")
        ]
    if isinstance(value, list):
        return [
            error
            for index, item in enumerate(value)
            for error in _non_finite_errors(item, f"{path}/{index}")
        ]
    return []


def _box_errors(box: dict[str, Any], path: str) -> list[str]:
    values = tuple(box.get(key) for key in ("x", "y", "w", "h"))
    if not all(isinstance(value, int | float) for value in values):
        return []
    x, y, width, height = values
    if not all(math.isfinite(value) for value in values):
        return [f"{path}: box coordinates must be finite"]
    if x + width > 1.0 + 1e-9 or y + height > 1.0 + 1e-9:
        return [f"{path}: box exceeds source bounds"]
    return []


def _check_study_scope(
    payload: dict[str, Any],
    manifest: object,
    provenance: object,
    errors: list[str],
) -> None:
    if not isinstance(manifest, dict) or not isinstance(provenance, dict):
        return
    modality = payload.get("modality")
    source_kind = provenance.get("source_kind")
    screenshot_ct = modality == "CT_BRAIN" and source_kind == "screenshot"
    incomplete_study = manifest.get("complete") is not True or screenshot_ct
    scope = payload.get("assessment_scope")
    if screenshot_ct and scope != "single_image_observation":
        errors.append("assessment_scope: CT screenshot must be a single-image observation")
    elif incomplete_study and scope == "complete_study":
        errors.append("assessment_scope: incomplete inputs cannot represent a complete study")
    elif not incomplete_study and scope != "complete_study":
        errors.append("assessment_scope: a complete manifest must use complete_study")
    if screenshot_ct and manifest.get("complete") is True:
        errors.append("study_manifest: a CT screenshot cannot be a complete study")
    if screenshot_ct:
        for collection in ("observations", "findings"):
            entries = payload.get(collection)
            if not isinstance(entries, list):
                continue
            for index, entry in enumerate(entries):
                if (
                    isinstance(entry, dict)
                    and entry.get("claim_type") != "descriptive_observation"
                ):
                    errors.append(
                        f"{collection}/{index}: CT screenshot permits only "
                        "descriptive observations"
                    )
        findings = payload.get("findings")
        if isinstance(findings, list):
            for index, finding in enumerate(findings):
                if isinstance(finding, dict) and finding.get("confidence") == "high":
                    errors.append(
                        f"findings/{index}: CT screenshot cannot carry high "
                        "diagnostic confidence"
                    )
    if not incomplete_study:
        return
    image_quality = payload.get("image_quality")
    adequacy = image_quality.get("adequacy") if isinstance(image_quality, dict) else None
    issues = image_quality.get("issues") if isinstance(image_quality, dict) else None
    if adequacy == "diagnostic":
        errors.append("image_quality: incomplete study cannot be diagnostic")
    if not issues:
        errors.append("image_quality: incomplete study must state quality issues")
    if payload.get("incomplete") is not True or not payload.get("incomplete_reasons"):
        errors.append("incomplete: incomplete study must fail closed with reasons")
    if not manifest.get("limitations"):
        errors.append("study_manifest: incomplete study must state limitations")


def _check_trace(payload: dict[str, Any], errors: list[str]) -> None:
    trace = payload.get("analysis_trace")
    if not isinstance(trace, list):
        return
    stages = [
        event.get("stage")
        for event in trace
        if isinstance(event, dict) and isinstance(event.get("stage"), str)
    ]
    if len(stages) != len(set(stages)):
        errors.append("analysis_trace: workflow stages must be unique")
    missing = sorted(_REQUIRED_TRACE_STAGES - set(stages))
    if missing:
        errors.append("analysis_trace: missing required stages: " + ", ".join(missing))
    order = [_TRACE_ORDER[stage] for stage in stages if stage in _TRACE_ORDER]
    if order != sorted(order):
        errors.append("analysis_trace: workflow stages are out of order")
    events = {
        event.get("stage"): event
        for event in trace
        if isinstance(event, dict) and isinstance(event.get("stage"), str)
    }
    for stage in {"intake", "quality_gate", "contract_validation", "human_handoff"}:
        if events.get(stage, {}).get("status") != "completed":
            errors.append(f"analysis_trace: {stage} must be completed")
    diagnostic = (
        isinstance(payload.get("image_quality"), dict)
        and payload["image_quality"].get("adequacy") != "non_diagnostic"
    )
    if diagnostic:
        for stage in {"blind_pass", "reconcile"}:
            if events.get(stage, {}).get("status") != "completed":
                errors.append(f"analysis_trace: {stage} must be completed")
    if events.get("independent_evidence", {}).get("status") == "completed" and events.get(
        "blind_pass", {}
    ).get("status") != "completed":
        errors.append("analysis_trace: tools cannot precede a completed blind pass")


def validate_payload(payload: object) -> None:
    errors = validation_errors(payload)
    if errors:
        raise ValueError("invalid analysis result: " + "; ".join(errors))
