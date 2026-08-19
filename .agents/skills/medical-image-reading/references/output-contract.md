# Output contract

Canonical schema: `schemas/analysis-result.schema.json` at the repository root.
Protocol version: `0.1.0`; schema version: `1.0.0`.

The result separates observations, evidence, and impressions. At minimum it records:

- exact input/protocol provenance and technical image quality;
- an explicit assessment scope (`single_image_observation`, `partial_study`, or
  `complete_study`) and a PHI-free study manifest;
- one modality-specific checklist item for every required search axis, including
  whether it was assessable;
- atomic observations with polarity and verification status;
- a claim type distinguishing pixel-level description from a study-level diagnostic
  hypothesis; CT screenshots permit descriptive observations only;
- abnormal or unresolved `findings` linked to observation/evidence IDs;
- normalized source-image boxes with source hash and verification state;
- summary, triage severity, limitations, next steps, and human-review reasons;
- only auditable workflow events, never hidden reasoning.

The analyzer produces the observation/evidence draft. The trusted host adds exact
hashes, study inventory, protocol versions, assessment scope, and workflow events;
the model must never invent these bindings. Only the assembled payload is passed to
`medical-image-harness validate` or `AnalysisResult.to_contract_payload()`.

Contract rules that require semantic validation in addition to JSON Schema:

1. IDs are unique and all references resolve.
2. Every evidence object is bound to the run's source or a recorded transform.
3. `x + w <= 1` and `y + h <= 1`; width and height are positive.
4. Every retained finding box is verified, source-hash-bound, inside source bounds,
   and exactly present in evidence linked through the finding's observations.
5. Normal/absent observations have no retained finding boxes.
6. A low-certainty candidate has a reviewer question.
7. Every result is a `research_draft` with `review_required=true`; incomplete inputs
   additionally require explicit limitations and non-diagnostic/limited quality.
8. CT screenshots are single-image observations, never complete-study reads.
   Their observations/findings are descriptive only and cannot carry high diagnostic
   confidence.
9. Required intake, QC, blind-pass, reconciliation, contract-validation, and human-
   handoff events are unique and ordered; optional tools cannot precede blind read.
10. Impression claims cite verified observation IDs; unsupported tool output cannot
   enter the impression.

Use `medical-image-harness validate RESULT.json` for deterministic validation.
Adapters may add transport envelopes but must not weaken this payload contract.
