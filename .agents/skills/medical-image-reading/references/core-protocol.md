# Core scientific protocol

Protocol version: `0.1.0`

Machine-audited invariant IDs:

- `DEIDENTIFIED_INPUT_ONLY`
- `QUALITY_GATE_BEFORE_INFERENCE`
- `BLIND_PASS_BEFORE_OPTIONAL_TOOLS`
- `ATOMIC_OBSERVATION_LEDGER`
- `SOURCE_BOUND_LOCALIZATION`
- `EXPLICIT_HYPOTHESIS_CHALLENGE`
- `APPROPRIATE_ABSTENTION`
- `AUTHORIZED_HUMAN_REVIEW`

## 1. Intake and trust boundary

Create a PHI-free run record before interpretation:

- hash the exact immutable source bytes;
- record `source_kind` (`dicom`, rendered image, screenshot, waveform render);
- record the trusted de-identification assertion and every transform/crop hash;
- identify available views, frames, series, orientation, and prior comparison;
- state which agent surface and model/tool versions will touch the data.

Do not treat a successful metadata scrub as proof that pixels contain no burned-in
identifiers. Pixel/graphics/structured-content cleaning needs its own review.

## 2. Quality and completeness gate

Assess modality agreement, anatomy coverage, laterality/orientation labels, view or
lead identity, resolution, exposure/contrast/window, motion, truncation, artifacts,
and study completeness. Use `diagnostic`, `limited`, or `non_diagnostic`.

- `non_diagnostic`: stop pathology inference; record limitations and review steps.
- `limited`: answer only the claims supported by the available data; mark every
  unsupported axis `not_assessable`.
- A single screenshot never implies a complete CT or multi-view radiographic study.

## 3. Blind systematic pass

Inspect the pixels without optional model/tool/prior-report suggestions. Record
observations by anatomy and search axis, including pertinent negatives only where
the anatomy is actually assessable. Keep description separate from impression.

Each atomic observation records:

- polarity: `present`, `absent`, or `uncertain`;
- verification status: `supported`, `possible`, `contradicted`, or `unevaluable`;
- anatomy/laterality/temporal qualifier;
- evidence IDs bound to exact image/frame coordinates;
- limitations and the next discriminating question.

## 4. Independent evidence pass

Optional classifiers, detectors, segmenters, waveform models, OCR, or priors run
after the blind pass. Preserve their input hash, model revision, calibration ID,
and raw output. Do not silently convert a classifier label into a report finding or
a waveform label into an image bbox.

## 5. Reconcile and challenge

Build four lists: agreements, conflicts, unsupported claims, and uninspected/high-
risk regions. Revisit conflicts from the immutable source at an appropriate scale,
window, frame, or view. A refinement turn must emit an explicit `confirm`, `revise`,
`retract`, `add`, or `unevaluable` decision with a short visible-evidence rationale.

For a crop rectangle `(px, py, pw, ph)` and a crop-local box `(cx, cy, cw, ch)`,
the source box is `(px + cx*pw, py + cy*ph, cw*pw, ch*ph)`. Clamp and validate the
result; retain both source hash and transform receipt.

## 6. Reconcile the report

The impression may reference only verified observation IDs. Recheck polarity,
laterality, temporal comparison, measurement basis, severity, certainty, and
checklist consistency. A high-risk possible claim can mandate urgent human review
without being promoted to a confirmed diagnosis.

## 7. Human handoff

Return a structured draft, evidence map, explicit limitations, disagreement list,
and concrete review questions. The harness never signs or writes back a report.
