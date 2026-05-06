param(
    [string]$Model = "all",
    [string]$Dataset = "all",
    [int]$BatchSize = 16,
    [int]$Batches = 20,
    [int]$WarmupBatches = 3,
    [int]$LogEvery = 100,
    [double]$PowerSampleInterval = 2.0,
    [int]$MaxSamples = 0,
    [int]$NumWorkers = 0,
    [string]$Device = "auto",
    [switch]$Download,
    [switch]$Amp
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$ArgsList = @(
    "benchmarks\single_device_benchmark.py",
    "--model", $Model,
    "--dataset", $Dataset,
    "--batch-size", "$BatchSize",
    "--batches", "$Batches",
    "--warmup-batches", "$WarmupBatches",
    "--log-every", "$LogEvery",
    "--power-sample-interval", "$PowerSampleInterval",
    "--max-samples", "$MaxSamples",
    "--num-workers", "$NumWorkers",
    "--device", $Device
)
if ($Download) {
    $ArgsList += "--download"
}
if ($Amp) {
    $ArgsList += "--amp"
}

& $Python @ArgsList
