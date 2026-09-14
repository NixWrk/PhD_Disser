param(
    [Parameter(Mandatory = $true)]
    [string]$Baseline,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [Parameter(Mandatory = $true)]
    [string]$Dependencies,

    [Parameter(Mandatory = $true)]
    [string]$Python,

    [Parameter(Mandatory = $true)]
    [string]$Matlab,

    [double]$TargetEdgeMm = 0.5,
    [double]$ContactRadiusMm = 8.0
)

$ErrorActionPreference = "Stop"
$modelRoot = Split-Path -Parent $PSScriptRoot
$baselinePath = (Resolve-Path -LiteralPath $Baseline).Path
$dependenciesPath = (Resolve-Path -LiteralPath $Dependencies).Path
$outputPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Output))
$statusPath = Join-Path $outputPath "spatial_orchestrator_status.json"
$eidorsStartup = (Resolve-Path -LiteralPath $env:EIDORS_STARTUP).Path
$currentStage = "initialization"

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

function Write-Status {
    param(
        [string]$Status,
        [string]$Stage,
        [string]$Message
    )

    $value = [ordered]@{
        status = $Status
        stage = $Stage
        message = $Message
        pid = $PID
        updated_utc = [DateTime]::UtcNow.ToString("o")
        baseline = $baselinePath
        output = $outputPath
        target_edge_mm = $TargetEdgeMm
        contact_neighbourhood_radius_mm = $ContactRadiusMm
        scope = "contact_neighbourhood_only_not_global_volume_or_interface_convergence"
    }
    $temporary = "$statusPath.tmp"
    $value | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

function Invoke-External {
    param(
        [string]$Stage,
        [string]$Executable,
        [string[]]$Arguments
    )

    $script:currentStage = $Stage
    Write-Status -Status "running" -Stage $Stage -Message "Spatial-convergence workflow is running"
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stage '$Stage' failed with exit code $LASTEXITCODE"
    }
}

function Convert-ToMatlabPath {
    param([string]$Path)
    return $Path.Replace("\", "/").Replace("'", "''")
}

try {
    $tool = Join-Path $PSScriptRoot "tepc_spatial_convergence.py"
    $preparer = Join-Path $PSScriptRoot "prepare_surface_sensitivity.py"
    $runner = Join-Path $PSScriptRoot "run_surface_sensitivity.py"

    Invoke-External -Stage "refine_contact_neighbourhood" -Executable $Python -Arguments @(
        $tool,
        "refine",
        "--baseline", $baselinePath,
        "--output", $outputPath,
        "--deps", $dependenciesPath,
        "--target-edge-mm", $TargetEdgeMm.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--radius-mm", $ContactRadiusMm.ToString([Globalization.CultureInfo]::InvariantCulture)
    )

    $modelMatlab = Convert-ToMatlabPath $modelRoot
    $srcMatlab = Convert-ToMatlabPath (Join-Path $modelRoot "src")
    $vendorMatlab = Convert-ToMatlabPath (Join-Path $modelRoot "vendor_stl_eidors")
    $eidorsMatlab = Convert-ToMatlabPath $eidorsStartup
    $legacyManifest = Convert-ToMatlabPath (Join-Path $baselinePath "manifest_before_refinement.json")
    $fineMesh = Convert-ToMatlabPath (Join-Path $outputPath "refinement/refined_mesh.mat")
    $exportDirectory = Convert-ToMatlabPath (Join-Path $outputPath "refinement/export")
    $exportCommand = "run('$eidorsMatlab'); addpath('$modelMatlab'); addpath('$srcMatlab'); addpath('$vendorMatlab'); trkg4_export_refined_surface_model('$legacyManifest','$fineMesh','$exportDirectory');"
    Invoke-External -Stage "export_fine_fem_model" -Executable $Matlab -Arguments @("-batch", $exportCommand)

    Invoke-External -Stage "configure_fine_studies" -Executable $Python -Arguments @(
        $tool,
        "configure",
        "--baseline", $baselinePath,
        "--output", $outputPath,
        "--deps", $dependenciesPath
    )

    $fineManifest = Convert-ToMatlabPath (Join-Path $outputPath "manifest.json")
    $contactCommand = "run('$eidorsMatlab'); addpath('$modelMatlab'); addpath('$srcMatlab'); addpath('$vendorMatlab'); r=trkg4_prepare_surface_sensitivity('$fineManifest'); assert(strcmp(r.status,'passed'));"
    Invoke-External -Stage "prepare_contacts" -Executable $Matlab -Arguments @("-batch", $contactCommand)

    Invoke-External -Stage "seal_pilot" -Executable $Python -Arguments @(
        $preparer,
        "seal",
        "--base", $outputPath
    )

    Invoke-External -Stage "run_pilot_78_states" -Executable $Python -Arguments @(
        $runner,
        "--base", $outputPath,
        "--deps", $dependenciesPath,
        "--python", $Python,
        "--matlab", $Matlab,
        "--pilot"
    )

    Invoke-External -Stage "compare_1mm_to_0p5mm" -Executable $Python -Arguments @(
        $tool,
        "analyse",
        "--baseline", $baselinePath,
        "--output", $outputPath,
        "--deps", $dependenciesPath
    )

    Write-Status -Status "completed" -Stage "analysis" -Message "Contact-local two-level spatial-convergence screen completed"
}
catch {
    Write-Status -Status "failed" -Stage $currentStage -Message $_.Exception.Message
    throw
}
