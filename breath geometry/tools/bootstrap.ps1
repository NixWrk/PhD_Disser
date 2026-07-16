param(
    [string]$Python = "python",
    [switch]$WithGeometry,
    [int]$PipTimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $Python -m venv $Venv
}

$PipNetworkArgs = @(
    "--disable-pip-version-check",
    "--timeout", $PipTimeoutSeconds,
    "--retries", "2"
)

& $VenvPython -m pip install @PipNetworkArgs --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Unable to upgrade the local packaging toolchain"
}

if ($WithGeometry) {
    & $VenvPython -m pip install @PipNetworkArgs -e ($RepoRoot + "[geometry,dev]")
} else {
    & $VenvPython -m pip install @PipNetworkArgs -e ($RepoRoot + "[dev]")
}
if ($LASTEXITCODE -ne 0) {
    throw "Unable to install the project dependencies into .venv"
}

& $VenvPython -m pip freeze --all |
    Set-Content -Encoding utf8 -LiteralPath (Join-Path $RepoRoot "requirements.lock.txt")

& $VenvPython -m pytest $RepoRoot
