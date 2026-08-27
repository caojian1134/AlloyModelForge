$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Push-Location $projectRoot
try {
    conda run -n alloymodelforge-dev alloy-model-forge-web --host 127.0.0.1 --port 8765 --no-browser
} finally {
    Pop-Location
}
