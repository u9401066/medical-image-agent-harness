# Public/private integration boundary

## Public harness owns

- models, schemas, modality protocols, and the canonical cross-agent skill;
- input/QC/output guardrails and source-safe image transforms;
- provider-neutral analyzer/refinement/finalizer ports;
- multi-pass, rhythm-strip, localization, provenance, and evaluation logic;
- synthetic/redistributable fixtures and deterministic compatibility tests.

## Private product owns

- desktop/window/screen capture, ROI UI, viewer and overlay rendering;
- OpenClaw or other gateway/plugin transport, MCP integration, credentials;
- PACS/RIS/EHR and clinical writeback, packaging, installers, telemetry;
- proprietary model selection, weights, site configuration, and governed datasets.

Dependency direction is always private product → public harness. The public package
must never import the product namespace or transport SDK.

## Recommended parent layout

```text
private-product/
  third_party/medical-image-agent-harness/   # pinned Git submodule
  .agents/skills/medical-image-reading/      # generated thin adapter only
  src/private_product/adapters/               # private AnalyzerPort implementation
```

Codex scans ancestor `.agents/skills` paths, not arbitrary descendant submodules.
When agents start from the private root, generate a tiny parent skill that tells the
agent to load the pinned canonical skill. Do not duplicate the method. Run a drift
check in CI and update the submodule through a reviewable PR.

For packaging, depend on the Python package from the pinned submodule. Runtime skill
formats for a private gateway should be generated from the canonical references plus
private tool augmentation; never hand-edit two copies.
