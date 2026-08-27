$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentFile = Join-Path $projectRoot "environment.yml"
$environmentName = "alloymodelforge-dev"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONNOUSERSITE = "1"
$previousChannelPriority = $env:CONDA_CHANNEL_PRIORITY
$env:CONDA_CHANNEL_PRIORITY = "strict"

Push-Location $projectRoot
try {
    $existing = conda env list --json | ConvertFrom-Json
    $found = $existing.envs | Where-Object { (Split-Path $_ -Leaf) -eq $environmentName }
    if ($found) {
        conda env update -n $environmentName -f $environmentFile --prune
    } else {
        conda env create -f $environmentFile
    }
    conda env config vars set -n $environmentName `
        PYTHONNOUSERSITE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
    conda run -n $environmentName python -m pip check
    conda run -n $environmentName python -m pytest -q
    conda run -n $environmentName python scripts\reproducibility_check.py
} finally {
    if ($null -eq $previousChannelPriority) {
        Remove-Item Env:CONDA_CHANNEL_PRIORITY -ErrorAction SilentlyContinue
    } else {
        $env:CONDA_CHANNEL_PRIORITY = $previousChannelPriority
    }
    Pop-Location
}
