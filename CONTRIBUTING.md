# Contributing

Changes should preserve the forecast-origin contract and the distinction between model selection, calibration, and final assessment.

1. Create an isolated environment and install `.[dev]`.
2. Add or update a test for every methodological change, especially feature timing.
3. Run `ruff check .`, `ruff format --check .`, and `pytest`.
4. Regenerate artifacts only when the protocol or data changes.
5. State whether a change was conceived before or after inspecting holdout results.

For new datasets, add provenance, license, timezone, weather-vintage, revision, and missing-data documentation. Never commit confidential utility or customer data.

For new models, compare under the same information set, folds, holdout, random-seed policy, and compute budget. Report negative and failed results; do not remove a strong baseline because it wins.
