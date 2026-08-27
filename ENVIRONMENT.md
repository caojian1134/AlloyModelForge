# AlloyModelForge development environment

## Environments

- Full scientific environment: Conda environment `alloymodelforge-dev`
- Python: 3.13 (currently resolved and tested as 3.13.15, 64-bit Windows)
- SciPy: 1.18.1 (pinned because 1.18.0 changes the fixed-seed rotation result)
- Definition: `environment.yml`
- Includes: icet 3.0, sqsgenerator 0.5.6, OVITO 3.16.0, Web and tests
- Bootstrap/update command: `powershell -ExecutionPolicy Bypass -File scripts\bootstrap-dev.ps1`

Do not install or run AlloyModelForge from the Conda `base` environment. All project
commands, tests, Web services, and scientific validation use `alloymodelforge-dev`.
The environment file uses `nodefaults`, and the bootstrap script enforces strict
channel priority, so a machine-level `.condarc` cannot silently add the Anaconda
defaults channel. The script also persists `PYTHONNOUSERSITE=1` in this environment
to prevent unrelated packages from `%APPDATA%\\Python` entering validation runs.

## Verified capabilities

The Conda environment includes ASE, NumPy, SciPy, FastAPI, icet,
sqsgenerator, and the official OVITO distribution. `pip check` passes and all
81 tests pass in the current release, including the environment-contract test,
the real icet backend, Web
workflow coverage, scientific invariants, and OVITO CNA/PTM tests.

## Reproducibility baseline

Run:

```powershell
conda run -n alloymodelforge-dev python scripts\reproducibility_check.py
```

Expected structure fingerprint:

```text
96b3b5a0860682320679c70617c7134e7179aea5f7fbee6dbbadcf18c6f49753
```

Expected atom count and composition: 459 atoms, Co=230, Ni=229. The minimum
distance is 1.554171787371 angstrom. The expected validation status is
`WARNING`, because this deliberately small random-orientation polycrystal has
non-commensurate periodic seams and a high boundary fraction. The script
returns a non-zero exit code if any part of this baseline changes.

## Conda environment with icet

The authoritative full scientific environment is created with:

```powershell
conda env create -f environment.yml
conda activate alloymodelforge-dev
```

It installs the conda-forge binary build of `icet 3.0` and the official OVITO
Conda build from `https://conda.ovito.org`, avoiding local C++ compilation and
the Qt DLL conflict caused by mixing the PyPI OVITO wheel with Conda. PyPI
`icet 3.2` remains source-only on this platform;
the package extra accepts compatible icet releases from 3.0 through 3.2, while
the reproducible Conda environment deliberately pins the tested 3.0 build.
SciPy is installed from the pip section at exactly 1.18.1 after Conda resolves
the compiled scientific stack; this is required for the documented fixed-seed
fingerprint.
