"""The setup command must stop before later checks when Conda fails."""

from pathlib import Path
import shutil
import subprocess

import pytest


SHELLS = [name for name in ("powershell", "pwsh") if shutil.which(name)]


@pytest.mark.parametrize("shell", SHELLS or [None])
@pytest.mark.parametrize("existing_environment", [False, True])
@pytest.mark.parametrize("failure_index", [0, 1, 2, 3, 4, 5, None])
def test_bootstrap_stops_at_failed_conda_command(
    tmp_path, shell, existing_environment, failure_index
):
    if shell is None:
        pytest.skip("PowerShell is required to exercise the Windows bootstrap")

    root = Path(__file__).resolve().parents[1]
    bootstrap = str(root / "scripts" / "bootstrap-dev.ps1").replace("'", "''")
    log = tmp_path / "calls.txt"
    escaped_log = str(log).replace("'", "''")
    environments = '["C:/envs/alloymodelforge-dev"]' if existing_environment else "[]"
    harness = tmp_path / "bootstrap-harness.ps1"
    harness.write_text(
        f"""$ErrorActionPreference = 'Stop'
$global:callIndex = 0
function conda {{
    Add-Content -LiteralPath '{escaped_log}' -Value ($args -join ' ')
    $global:LASTEXITCODE = 0
    if ($global:callIndex -eq {failure_index if failure_index is not None else -1}) {{
        $global:LASTEXITCODE = 23
    }}
    if ($global:callIndex -eq 0) {{ '{{"envs":{environments}}}' }}
    $global:callIndex += 1
}}
& '{bootstrap}'
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    calls = log.read_text(encoding="utf-8-sig").splitlines()
    if failure_index is None:
        assert result.returncode == 0, result.stderr
        assert len(calls) == 6
        assert "reproducibility_check.py" in calls[-1]
    else:
        assert result.returncode != 0
        assert len(calls) == failure_index + 1
        assert "exit code 23" in result.stderr
