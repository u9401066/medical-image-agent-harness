# Medical Image Agent Harness

This repository contains a provider-neutral research harness for auditable medical
image co-reading. It does not contain a viewer, screen capture, PACS writeback,
commercial model integration, or autonomous diagnostic workflow.

## Working agreements

- Use `uv` for Python environments and run `uv run pytest` plus
  `uv run python scripts/check_compatibility.py` after meaningful changes.
- Keep the canonical agent method in
  `.agents/skills/medical-image-reading/`; do not duplicate it under another skill
  discovery directory.
- For medical-image reading or evaluation tasks, load the `medical-image-reading`
  skill and only the modality references needed for the task.
- Preserve immutable raw predictions and source-image provenance. Validators return
  normalized copies; they do not silently rewrite source evidence.
- All public fixtures must be deterministic synthetic data or have documented
  redistribution rights and de-identification provenance.
- Never add PHI, credentials, PACS/RIS/EHR writes, proprietary model weights, UI
  automation, OpenClaw/plugin code, or private product adapters.
- Report research metrics with denominators and uncertainty. Never describe a mock
  or public-CI smoke test as clinical validation.

## Code review rules

- Reject any crop path that can fall back to the uncropped input after a crop error.
- Reject source boxes with zero area, bounds overflow, unbound hashes, or crop-local
  coordinates mislabeled as source coordinates.
- Reject conclusions that convert missing views/leads/series or absent tool labels
  into normal findings.
- Reject changes that weaken human-review, abstention, provenance, or licensing
  gates without an explicit documented rationale and regression test.
