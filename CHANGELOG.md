# Changelog

## 0.2.0 - 2026-08-27

- Renamed the public software brand from AlloyForge to AlloyModelForge to make
  its role as an atomistic alloy model generator explicit.
- Added the formal `alloy-model-forge` and `alloy-model-forge-web` commands,
  while preserving the legacy short commands and Python import path.
- Renamed the reproducible Conda environment to `alloymodelforge-dev` and the
  default local Web data directory to `.alloymodelforge/jobs`.
- Updated Web, documentation, citation metadata, validation reports and
  publication terminology for the new public name.

## Unreleased

- Upgraded native SQS to finite-size fixed-composition pair targets, region
  feasibility diagnostics, internal/touching pair scopes, composition-directed
  swaps, adaptive acceptance-rate control and reproducible early stopping.
- Added dimensionless objective decomposition, independent multi-start
  statistics, MILP proof of joint regional-composition feasibility, crossing
  interface bonds, non-random pair propensity targets and exact small-cell
  optimality benchmarks.
- Added Native SQS v4 directional source-to-target Warren-Cowley objectives
  with frozen phase reference compositions and restricted finite-composition
  angular triplet motifs with local incremental updates.
- Added serialized, non-blocking OVITO render jobs with unique outputs and status API.
- Added Tachyon figure legends, counts, fractions, DPI metadata and publication controls.
- Added grain equivalent-diameter, centroid and empty-grain validation.
- Added inclusion/core-shell composition, connectivity, center-offset and radial phase profiles.
- Added CNA/PTM interior-versus-interface statistics.
- Added model resource limits, disk-space checks and safe completed-job deletion.
- Added publication workflow, interpretation limits and reproducibility checklist.

## 0.1.0 - 2026-08-25

- Initial FCC, BCC, HCP, polycrystal, multiphase and inclusion builders.
- Native SQS, sqsgenerator and icet adapters with SRO reporting.
- OVITO CNA/PTM analysis, CLI, Python API and optional browser interface.
- Scientific validation, provenance and deterministic structure fingerprints.
