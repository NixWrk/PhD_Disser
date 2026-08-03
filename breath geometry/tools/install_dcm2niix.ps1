param(
    [string]$Version = "v1.0.20260416",
    [string]$ExpectedSha256 = "969bca4fc41d5f82658acef9d0ed9cbfbd4114ec8e8668906910241fcbb2c048"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$CacheDir = Join-Path $PSScriptRoot "cache"
$BinDir = Join-Path $PSScriptRoot "bin"
$Archive = Join-Path $CacheDir "dcm2niix_win_$Version.zip"
$Url = "https://github.com/rordenlab/dcm2niix/releases/download/$Version/dcm2niix_win.zip"

New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

if (-not (Test-Path -LiteralPath $Archive)) {
    curl.exe -L --fail --show-error --retry 2 --retry-all-errors --connect-timeout 20 --output $Archive $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to download $Url"
    }
}

$ActualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
if ($ActualSha256 -ne $ExpectedSha256.ToLowerInvariant()) {
    throw "SHA-256 mismatch for $Archive"
}

$ExtractDir = Join-Path $CacheDir "dcm2niix_$Version"
if (-not (Test-Path -LiteralPath $ExtractDir)) {
    Expand-Archive -LiteralPath $Archive -DestinationPath $ExtractDir
}

$Executable = Get-ChildItem -Recurse -File -Filter "dcm2niix.exe" -LiteralPath $ExtractDir |
    Select-Object -First 1
if (-not $Executable) {
    throw "dcm2niix.exe not found after extraction"
}

Copy-Item -Force -LiteralPath $Executable.FullName -Destination (Join-Path $BinDir "dcm2niix.exe")
& (Join-Path $BinDir "dcm2niix.exe") --version
