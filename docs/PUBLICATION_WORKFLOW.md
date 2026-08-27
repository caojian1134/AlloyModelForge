# AlloyModelForge publication and reproducibility workflow

## Scientific scope

AlloyModelForge constructs and validates initial atomistic models. Energy minimization,
molecular dynamics and density-functional calculations are deliberately outside
its scope. CNA/PTM labels describe the unrelaxed starting geometry and must not be
interpreted as a prediction of thermodynamic phase stability.

## Reproducible model generation

1. Archive the exact YAML specification and its `random_seed`.
2. Build in the `alloymodelforge-dev` environment from `environment.yml`.
3. Archive `metadata.json`, `validation.json`, `model.extxyz` and the intended
   downstream format.
4. Report the structure SHA-256, AlloyModelForge version, dependency versions, box,
   periodicity, phase compositions, grain count and chemical placement method.
5. Keep the Extended XYZ file as the authoritative model because formats such as
   POSCAR and CIF cannot retain all grain, phase, interface, CNA and PTM arrays.

## Publication rendering

The Web application's interactive view uses 3Dmol for rapid inspection. The
`OVITO high-quality render` action submits the complete structure to a serialized
background queue and uses OVITO Tachyon. Each render receives a unique filename.

CLI example:

```powershell
conda run -n alloymodelforge-dev alloy-model-forge render-ovito output/model.extxyz `
  --output output/cna-section.png --color-by cna --filter-by half-x `
  --projection orthographic --width 2400 --height 1800 --dpi 300
```

Generated figures include a color key, atom counts and fractions. For a core-shell
model use an X/Y/Z half-section and color by `inclusion` or `phase`; an opaque full
model naturally hides its core.

## Required validation evidence

- atom count, composition error and minimum interatomic distance;
- output round-trip integrity and SHA-256 fingerprint;
- actual phase and grain populations;
- equivalent grain diameter distribution, empty-grain check and boundary fraction;
- orientation matrices, Euler angles and symmetry-reduced adjacent-grain
  misorientation distribution;
- inclusion atom counts, composition, center offset, maximum radial extent and
  radial phase profile;
- periodic seam, local-density and interface/triple-junction statistics;
- CNA/PTM statistics split conceptually between crystalline interiors and
  interfaces when interpreting the result.

## Known limitations

- Initial grain and phase interfaces are geometrically cleaned but not relaxed.
- CNA/PTM may classify interface, free-surface and strongly strained atoms as
  `Other`; this is expected and is not by itself a construction failure.
- Atom-fraction-derived grain volumes are estimates for multiphase systems with
  different atomic densities.
- Inclusion connectivity is inferred from generated labels and radial profiles;
  downstream physical stability requires an external MD/DFT workflow.
- The local Web service is designed for one workstation. OVITO work is serialized
  to protect its global visualization scene.

## Acceptance checklist

A result is ready to support a manuscript only when tests pass in the declared
Conda environment, validation has no unexplained warnings, all requested grains
are populated, composition tolerances are met, output round trips are recorded,
and every figure can be regenerated from an archived model plus command line.
