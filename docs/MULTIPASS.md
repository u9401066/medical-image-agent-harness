# Shared bounded multi-pass engine

`medical_image_harness.multipass` owns the provider-neutral scientific loop:
coarse interpretation, immutable-source crop planning, hypothesis challenges,
source-coordinate remapping, critical-first routing, bounded stage deadlines,
explicit final dispositions, and review/incompleteness guards. It has no capture,
GUI, network, provider SDK, or product imports. The host supplies the analyzer,
cropper, and optional signal calibration/layout detection capabilities.

This revision extracts the current product engine from source checkpoint
`e1ef13d87d90890bdba5d78918d2509ee01bc410`, not the earlier public snapshot.
No compatibility forwarding module is required in the consuming application.

## Host policy is explicit

`MultiPassInterpreter(..., checklist_keys_for=callback, stage_tools=StageTools(...))`
allows a host to retain its existing checklist policy and auditable provider stage
identities without putting provider code in this package. The default uses the
public modality profiles. Public CXR includes projection quality and public CT has
a research checklist; a host must not silently adopt different axes during a
structural migration. An explicitly empty set remains empty, not a default lookup.

The analyzer's optional refinement turn receives the hypothesis, exact crop,
probe identity, and only lead regions declared visible in that source crop. A
missing/unreadable lead label is not inferred from a layout template. A failed crop
does not fall back to transmitting the full source. Partial crops cannot retract,
revise, or confirm an unlocalized whole-image hypothesis or uncovered source boxes.

## Drafts are not validated evidence contracts

The engine returns typed drafts. The canonical schema and semantic validator are
still mandatory after trusted host assembly. In particular:

- Final reconciliation retains host provenance, study scope, observation/evidence
  ledger and workflow events; the model cannot replace those bindings.
- Changed clinical semantics (including claim type) invalidate previous claim
  references. Changed summary text is not automatically linked to old observations.
- Crop-derived replacements cannot reuse stale full-source evidence references.
- A host-bound human-review requirement cannot be cleared by final model output.
- Missing axes remain unassessed, not negative. Workflow failure does not itself
  manufacture an abnormal finding or raise clinical severity.

This is not a completed host assembler, calibrated diagnostic system, or proof of
clinical accuracy. Public CI uses deterministic synthetic fixtures only.

## Regression evidence and collection correction

Current-engine and strict partial/hidden-label layout tests are included alongside
the earlier public regressions. Inspection found 38 parametrized edge cases hidden
inside another test function in the source suite; they are now collected. A static
test rejects nested or shadowed test definitions to prevent recurrence.

Two stale assertions were corrected against the existing engine behavior: a local
candidate cannot confirm an unlocalized source hypothesis, and a timeout carries
incomplete/review flags without inventing clinical severity. The imported engine
was not changed to satisfy those stale assertions. Provider injection and canonical
binding protections have separate new regression tests.
