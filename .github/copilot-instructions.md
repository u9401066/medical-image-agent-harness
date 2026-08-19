# Copilot repository instructions

For any request to inspect, interpret, ground, compare, or evaluate a medical image
or ECG render, use the canonical agent skill at
`.agents/skills/medical-image-reading/SKILL.md` and read only its relevant modality
references. Do not reproduce the protocol in this file.

This repository is research/evaluation software, not an autonomous diagnosis or
clinical writeback system. Keep inputs de-identified, preserve source provenance,
fail closed on non-diagnostic or incomplete data, and require authorized human
review. Use `uv` and run the repository compatibility and test suites after changes.
