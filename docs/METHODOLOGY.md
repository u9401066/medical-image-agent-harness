# Methodology

The harness treats image reading as a versioned measurement workflow rather than a
single model response.

## Claim graph

```text
impression claim
  → observation ID(s)
  → evidence ID(s)
  → source image / series / frame
  → coordinate space + transform hash
  → agent/tool/model version + calibration record
```

This graph prevents a report sentence from silently outrunning its pixels or tool
evidence. Tool output stays independent until reconciliation, and every crop remains
bound to the immutable source.

The model-facing analyzer output is a draft. A trusted host, not the model, binds
source hashes and the de-identified study manifest, records ordered workflow events,
sets the assessment scope, and validates the final `research_draft` envelope. A CT
screenshot cannot validate as a complete or diagnostic study result.

## Bias controls

- The agent records a blind systematic pass before optional expert tools or priors.
- Refinement is hypothesis-aware and may confirm, revise, retract, add, or abstain.
- Missing inputs become `not_assessable`, not normal.
- Severity expresses review urgency; certainty expresses evidentiary support.
- The final summary may reference only verified observation IDs.

## Evaluation hierarchy

Contract and provenance checks precede clinical metrics. Subsequent layers cover
study/QC, claim precision and recall, localization, calibration/coverage-risk,
urgent misses, unsupported assertions, robustness, subgroup performance, and human-
AI team studies. Every metric keeps its own denominator and confidence interval.

Public CI proves software invariants on synthetic fixtures. It does not prove
clinical performance. Clinical claims require appropriately governed data, patient-
level splits, blinded adjudication, external validation, and human-factors evidence.
