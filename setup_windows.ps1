param()

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$PythonLauncher = "py"
if (-not (Get-Command $PythonLauncher -ErrorAction SilentlyContinue)) {
    $PythonLauncher = "python"
}

if (-not (Test-Path ".venv")) {
    & $PythonLauncher -m venv .venv
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt
& $Python .\dml_cluster\torch_install.py --install

Write-Host "[setup] ready. Run .\run_single_device_benchmarks.ps1 next."

