# Contributing

Use the repository-local environment described in `ENVIRONMENT.md`. Before a
change is submitted, run:

```powershell
conda run -n alloymodelforge-dev python -m pytest
conda run -n alloymodelforge-dev python scripts\scientific_validation.py
```

Scientific changes must include a test with a physical invariant or an
independent reference result. Tests that only assert that a file exists or a
function returns are not sufficient for model-building changes. Do not change
scientific tolerances without documenting the physical or numerical reason.

External backends must record their name and version and must never silently
fall back to an AlloyModelForge implementation.
