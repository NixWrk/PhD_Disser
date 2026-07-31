param(
    [string]$Python = "python",
    [switch]$WithGeometry,
    [switch]$SkipPackagingUpgrade,
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

if (-not $SkipPackagingUpgrade) {
    & $VenvPython -m pip install @PipNetworkArgs --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to upgrade the local packaging toolchain"
    }
}

$ProjectInstallArgs = @()
if ($SkipPackagingUpgrade) {
    $ProjectInstallArgs = @("--no-build-isolation", "--no-index")
}

if ($WithGeometry) {
    & $VenvPython -m pip install @PipNetworkArgs @ProjectInstallArgs `
        -e ($RepoRoot + "[geometry,notebook,dev]")
} else {
    & $VenvPython -m pip install @PipNetworkArgs @ProjectInstallArgs `
        -e ($RepoRoot + "[notebook,dev]")
}
if ($LASTEXITCODE -ne 0) {
    throw "Unable to install the project dependencies into .venv"
}

& $VenvPython -m ipykernel install --sys-prefix --name breathgeom `
    --display-name "Breath Geometry (.venv)"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to register the local Jupyter kernel"
}

& $VenvPython -m pip freeze --all --exclude-editable |
    Set-Content -Encoding utf8 -LiteralPath (Join-Path $RepoRoot "requirements.lock.txt")

& $VenvPython -m pytest $RepoRoot
