# Medical Image Agent Harness

A provider-neutral, testable scientific harness for agent-assisted medical image
co-reading. It gives Codex and GitHub Copilot the same systematic method, structured
evidence contract, multi-pass verification loop, safety gates, and evaluation API.

> Research and evaluation use only. This project is not a medical device, does not
> provide autonomous diagnosis, and does not write to clinical systems. A qualified
> human retains responsibility for every interpretation and action.

## Why this exists

`MultiPassInterpreter(..., prefer_ekg_group_coverage=True)` optionally reserves a
two-crop budget for both observed limb and precordial groups. It requires a complete,
valid, explicitly labeled 12-lead inventory **before** any layout normalization,
two bounded group crops, and at least two enabled systematic probes. Existing
hypotheses can be verified inside those groups; discovery may add source-remapped
findings. Critical-first triage, waveform/local attention, partial/unknown inputs,
other modalities and other budgets keep their existing routes. The default is
off. Coverage is a crop-planning fact, not proof of a complete or correct diagnosis;
deadlines can still prevent a planned crop from executing. No model call is added.

Vision-capable agents can inspect an image, but a reliable research workflow needs
more than a prompt. This repository makes the method inspectable and testable:

```text
de-identified source/study manifest
  → integrity + quality/completeness gate
  → blind systematic observation pass
  → optional independent tools/models
  → disagreement-led second look
  → source-coordinate evidence verification
  → observation-ledger reconciliation
  → schema + safety validation
  → authorized human review
```

The public harness intentionally excludes viewer/UI code, screen capture, PACS/EHR
writeback, product plugins, credentials, and proprietary model weights.

## Codex and Copilot

The canonical cross-agent workflow is
[`.agents/skills/medical-image-reading/SKILL.md`](.agents/skills/medical-image-reading/SKILL.md).
Both Codex and GitHub Copilot discover project skills from `.agents/skills`.

- `AGENTS.md` bootstraps Codex and repository maintenance rules.
- `.github/copilot-instructions.md` is a thin Copilot fallback.
- `.github/agents/medical-image-reader.agent.md` exposes a named Copilot agent.
- `.codex/agents/medical-image-reviewer.toml` exposes a named Codex subagent.

The platform adapters route to the shared skill; they do not fork the medical method.

## Python API

The package contains provider-neutral models and ports, deterministic input/output
guards, cited clinical-consistency rules, source-safe image operations, multi-pass
and rhythm-strip refinement, provenance fingerprints, and transparent evaluation
metrics.

Inline Python annotations are exposed through the packaged `py.typed` marker.
Consumers can type-check direct model imports without adding untyped-import
suppression. This is typing metadata, not a whole-codebase type-check or clinical
validation claim.

```bash
uv sync --extra dev
uv run python scripts/check_compatibility.py
uv run pytest
uv run medical-image-harness fingerprint
uv run medical-image-harness validate result.json
```

Minimal adapter contract:

```python
from medical_image_harness.protocols import AnalyzerPort


class MyAgentAdapter(AnalyzerPort):
    async def analyze(self, image_base64, modality, valid_regions):
        ...
```

`AnalyzerPort` returns a typed model draft. `OutputValidator` performs backward-
compatible draft normalization only; it is not the release gate. The trusted host
must attach the de-identified study manifest, exact provenance, observation/evidence
ledger, ordered workflow events, assessment scope, and human-review disposition,
then call `AnalysisResult.to_contract_payload()`. That method invokes the canonical
schema and semantic validator and fails closed when any binding is absent.

Hosts that validate content before making it available for review can use the
separate `schema.preflight_validation_errors(payload)` API. It checks the same
content against an executed prefix and forbids future validation/handoff events.
It is not canonical acceptance: the normal serializer/CLI still require real
completed validation and handoff records. See the
[preflight boundary](.agents/skills/medical-image-reading/references/output-contract.md#preflight-before-actual-human-handoff).

See [Methodology](docs/METHODOLOGY.md), [integration boundary](docs/INTEGRATION.md),
the [bounded multi-pass engine and host policy](docs/MULTIPASS.md),
and [prior-art/license research](docs/PRIOR_ART.md).

## Supported protocol profiles

- 12-lead EKG/ECG renders, with label/layout and calibration gating
- chest radiographs, with projection/quality gating and systematic review
- CT brain, with a strict distinction between a screenshot and a complete study

Profiles are research protocol baselines, not validated diagnostic claims. New
modalities should add a versioned checklist, schema fixtures, quality rules, and
evaluation plan before being advertised as supported.

## Submodule use

A private product can pin this repository as a Git submodule and implement the small
adapter port. Keep dependency direction one-way: private product → public harness.
Do not copy the canonical skill into multiple discovery paths; materialize only a
thin parent-repository adapter when an agent starts above the submodule.

## License and data

Code and original protocol text are Apache-2.0. No model weights, clinical datasets,
or patient images are distributed. Upstream code licenses do not automatically
cover downloaded weights or datasets; audit each layer independently.
