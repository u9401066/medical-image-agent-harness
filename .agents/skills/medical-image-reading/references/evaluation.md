# Evaluation protocol

Evaluation is versioned research evidence, not a product claim.

## Dataset and split integrity

- Use patient-level separation and record dataset/version/license/DUA.
- Keep private or credentialed datasets outside git, release artifacts, and public CI.
- Report site, device, view, quality, and demographic subgroup results where lawful.
- Public CI uses synthetic or clearly redistributable de-identified fixtures only.

## Metric layers

1. Contract: schema/reference validity, source hashes, transform receipts, fail-closed
   tool errors, and rerun comparability.
2. Study/QC: modality/view/lead/series completeness, orientation/laterality, window,
   de-identification, and appropriate abstention.
3. Claims: reviewer-defined concept precision/recall with explicit aliases; preserve
   negation, uncertainty, anatomy, laterality, and temporal qualifiers.
4. Localization: coordinate round-trip plus IoU/FROC for boxes, Dice/HD95 for masks,
   and wrong-frame/wrong-coordinate-space failures.
5. Calibration: Brier/ECE and coverage-risk only when probabilities have a versioned
   calibration set; qualitative self-confidence is reported separately.
6. Safety: urgent miss rate, unsupported assertion rate, laterality/negation error,
   evidence coverage, disagreement handling, and abstention appropriateness.
7. Human factors: blinded multi-reader adjudication and human-only versus human+AI;
   this is required before any clinical-effectiveness claim.

Report denominators and confidence intervals for every metric. Do not collapse
non-gradable cases into normal, compare natural-language strings verbatim, or use a
single aggregate score to hide an urgent miss.

For cross-agent testing, run the same golden contract cases in Codex and Copilot,
then compare normalized structured fields and mandatory safety-rule IDs. Keep live
model runs out of untrusted fork jobs and never expose credentials to repository code.
