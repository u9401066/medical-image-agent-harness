# Contributing

Use `uv sync --extra dev`, make focused changes, and run:

```bash
uv run python scripts/check_compatibility.py
uv run ruff check .
uv run pytest --cov=medical_image_harness --cov-report=term-missing
uv build
uv run python scripts/check_distribution.py
```

New modality protocols need a technical-quality gate, systematic checklist,
structured-output fixtures, failure/abstention cases, provenance rules, evaluation
plan, and supporting references. Do not submit patient images, credentialed dataset
content, proprietary weights, copied report phrase banks, or private product code.

Contributions are accepted under Apache-2.0. Clearly identify any third-party source
and its code/content/data/model license.
