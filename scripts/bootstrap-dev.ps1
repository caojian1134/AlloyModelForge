$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentFile = Join-Path $projectRoot "environment.yml"
$environmentName = "alloymodelforge-dev"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONNOUSERSITE = "1"
$previousChannelPriority = $env:CONDA_CHANNEL_PRIORITY
$env:CONDA_CHANNEL_PRIORITY = "strict"

function Assert-CondaSucceeded {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "Conda failed during $Step (exit code $LASTEXITCODE)."
    }
}

Push-Location $projectRoot
try {
    $environmentList = conda env list --json
    Assert-CondaSucceeded "environment discovery"
    $existing = $environmentList | ConvertFrom-Json
    $found = $existing.envs | Where-Object { (Split-Path $_ -Leaf) -eq $environmentName }
    if ($found) {
        conda env update -n $environmentName -f $environmentFile --prune
        Assert-CondaSucceeded "environment update"
    } else {
        conda env create -f $environmentFile
        Assert-CondaSucceeded "environment creation"
    }
    conda env config vars set -n $environmentName `
        PYTHONNOUSERSITE=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
    Assert-CondaSucceeded "environment configuration"
    conda run -n $environmentName python -m pip check
    Assert-CondaSucceeded "dependency validation"
    conda run -n $environmentName python -m pytest -q
    Assert-CondaSucceeded "regression tests"
    conda run -n $environmentName python scripts\reproducibility_check.py
    Assert-CondaSucceeded "reproducibility validation"
} finally {
    if ($null -eq $previousChannelPriority) {
        Remove-Item Env:CONDA_CHANNEL_PRIORITY -ErrorAction SilentlyContinue
    } else {
        $env:CONDA_CHANNEL_PRIORITY = $previousChannelPriority
    }
    Pop-Location
}
