param(
    [Parameter(Mandatory = $true)][string]$Baseline,
    [Parameter(Mandatory = $true)][string]$Output,
    [Parameter(Mandatory = $true)][string]$Dependencies,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$Matlab,
    [Parameter(Mandatory = $true)][string]$EidorsStartup,
    [string]$Surface = "",
    [string]$PreparationReport = "",
    [int]$GmshThreads = 16,
    [int]$SolverThreads = 4
)

$ErrorActionPreference = "Stop"
$modelRoot = Split-Path -Parent $PSScriptRoot
$baselinePath = (Resolve-Path -LiteralPath $Baseline).Path
$dependenciesPath = (Resolve-Path -LiteralPath $Dependencies).Path
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$matlabPath = (Resolve-Path -LiteralPath $Matlab).Path
$eidorsPath = (Resolve-Path -LiteralPath $EidorsStartup).Path
$outputPath = if ([System.IO.Path]::IsPathRooted($Output)) { [System.IO.Path]::GetFullPath($Output) } else { [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Output)) }
if (-not $Surface) { $Surface = Join-Path $modelRoot "output\nik_body_arm_parameter_surface_1mm_candidate.stl" }
if (-not $PreparationReport) { $PreparationReport = Join-Path $modelRoot "output\nik_stl_preparation_report.json" }
$surfacePath = (Resolve-Path -LiteralPath $Surface).Path
$preparationPath = (Resolve-Path -LiteralPath $PreparationReport).Path
$selectionPath = Join-Path $outputPath "state_selection.json"
$statusPath = Join-Path $outputPath "orchestrator_status.json"
$tool = Join-Path $PSScriptRoot "tepc_volume_convergence.py"
$gmshTool = Join-Path $PSScriptRoot "gmsh_arm_parameter_mesh.py"
$solver = Join-Path $PSScriptRoot "run_electrode_sensitivity.py"
$currentStage = "initialization"

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

function Write-Status {
    param([string]$Status, [string]$Stage, [string]$Message)
    $value = [ordered]@{
        status = $Status
        stage = $Stage
        message = $Message
        pid = $PID
        updated_utc = [DateTime]::UtcNow.ToString("o")
        baseline = $baselinePath
        output = $outputPath
        selection = $selectionPath
        spatial_scope = "independent_volume_mesh_levels_L12_L09_L06"
        physical_validation = $false
    }
    $temporary = "$statusPath.tmp"
    $value | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

function Quote-ProcessArgument {
    param([string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }
    $builder = [System.Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') { $slashes++ }
        elseif ($character -eq '"') {
            [void]$builder.Append((('\' * (2 * $slashes + 1)) -join ''))
            [void]$builder.Append('"')
            $slashes = 0
        }
        else {
            if ($slashes) {
                [void]$builder.Append((('\' * $slashes) -join ''))
                $slashes = 0
            }
            [void]$builder.Append($character)
        }
    }
    if ($slashes) { [void]$builder.Append((('\' * (2 * $slashes)) -join '')) }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Invoke-Logged {
    param([string]$Stage, [string]$Executable, [string[]]$Arguments, [string]$LogBase)
    $script:currentStage = $Stage
    Write-Status -Status "running" -Stage $Stage -Message "Numerical spatial validation is running"
    $stdout = "$LogBase.stdout.log"
    $stderr = "$LogBase.stderr.log"
    $argumentText = (($Arguments | ForEach-Object { Quote-ProcessArgument $_ }) -join " ")
    $process = Start-Process -FilePath $Executable -ArgumentList $argumentText -WorkingDirectory $modelRoot -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if ($process.ExitCode -ne 0) { throw "Stage '$Stage' failed with exit code $($process.ExitCode). See $stderr" }
}

function Convert-ToMatlabPath {
    param([string]$Path)
    return $Path.Replace("\", "/").Replace("'", "''")
}

function Wait-Or-BuildMesh {
    param([string]$LevelId, [string]$LevelPath, [double]$TrunkSizeMm)
    $meshDirectory = Join-Path $LevelPath "mesh"
    $mesh = Join-Path $meshDirectory ("body_trunk{0}mm.msh" -f [int]$TrunkSizeMm)
    $report = Join-Path $meshDirectory "gmsh_build.json"
    $externalStatus = Join-Path $LevelPath "stage_status.json"
    if ((Test-Path -LiteralPath $mesh) -and (Test-Path -LiteralPath $report)) { return $mesh }

    if (Test-Path -LiteralPath $externalStatus) {
        $external = Get-Content -LiteralPath $externalStatus -Raw | ConvertFrom-Json
        if ($external.status -eq "running" -and (Get-Process -Id $external.pid -ErrorAction SilentlyContinue)) {
            $script:currentStage = "wait_existing_gmsh_$LevelId"
            while (Get-Process -Id $external.pid -ErrorAction SilentlyContinue) {
                Write-Status -Status "running" -Stage $currentStage -Message "Waiting for the already running Gmsh process $($external.pid)"
                Start-Sleep -Seconds 20
            }
            if ((Test-Path -LiteralPath $mesh) -and (Test-Path -LiteralPath $report)) { return $mesh }
            throw "Existing Gmsh process for $LevelId ended without a complete mesh/report"
        }
    }

    if (Test-Path -LiteralPath $meshDirectory) {
        $entries = Get-ChildItem -LiteralPath $meshDirectory -Force
        if ($entries.Count -gt 0) { throw "Partial mesh directory for $LevelId is preserved; use a new output or inspect it" }
    }
    else { New-Item -ItemType Directory -Path $meshDirectory | Out-Null }
    Invoke-Logged -Stage "gmsh_$LevelId" -Executable $pythonPath -LogBase (Join-Path $meshDirectory "gmsh") -Arguments @(
        $gmshTool, $surfacePath, $preparationPath, $mesh, $report,
        "--arm-size-mm", "1",
        "--trunk-size-mm", $TrunkSizeMm.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--transition-mm", "5",
        "--threads", "$GmshThreads",
        "--algorithm-3d", "1"
    )
    if (-not ((Test-Path -LiteralPath $mesh) -and (Test-Path -LiteralPath $report))) { throw "Gmsh did not produce both mesh and report for $LevelId" }
    return $mesh
}

function Complete-Level {
    param([string]$LevelId, [string]$FolderName, [double]$TrunkSizeMm)
    $levelPath = Join-Path $outputPath $FolderName
    New-Item -ItemType Directory -Path $levelPath -Force | Out-Null
    $logs = Join-Path $levelPath "logs"
    New-Item -ItemType Directory -Path $logs -Force | Out-Null
    $mesh = Wait-Or-BuildMesh -LevelId $LevelId -LevelPath $levelPath -TrunkSizeMm $TrunkSizeMm

    $classification = Join-Path $levelPath "classification\classification.json"
    if (-not (Test-Path -LiteralPath $classification)) {
        $command = "run('$(Convert-ToMatlabPath $eidorsPath)'); addpath('$(Convert-ToMatlabPath $modelRoot)'); addpath('$(Convert-ToMatlabPath (Join-Path $modelRoot "src"))'); addpath('$(Convert-ToMatlabPath (Join-Path $modelRoot "vendor_stl_eidors"))'); trkg4_classify_volume_convergence_mesh('$(Convert-ToMatlabPath $mesh)','$(Convert-ToMatlabPath (Join-Path $levelPath "classification"))','$LevelId',$($TrunkSizeMm.ToString([Globalization.CultureInfo]::InvariantCulture)));"
        Invoke-Logged -Stage "classify_$LevelId" -Executable $matlabPath -Arguments @("-batch", $command) -LogBase (Join-Path $logs "classification")
    }

    if (-not (Test-Path -LiteralPath (Join-Path $levelPath "refinement\refinement.json"))) {
        Invoke-Logged -Stage "contact_refinement_$LevelId" -Executable $pythonPath -LogBase (Join-Path $logs "refinement") -Arguments @(
            $tool, "refine-contacts", "--baseline", $baselinePath, "--level", $levelPath, "--deps", $dependenciesPath,
            "--target-edge-mm", "1", "--radius-mm", "8"
        )
    }

    if (-not (Test-Path -LiteralPath (Join-Path $levelPath "export\volume_export.json"))) {
        $command = "run('$(Convert-ToMatlabPath $eidorsPath)'); addpath('$(Convert-ToMatlabPath $modelRoot)'); addpath('$(Convert-ToMatlabPath (Join-Path $modelRoot "src"))'); addpath('$(Convert-ToMatlabPath (Join-Path $modelRoot "vendor_stl_eidors"))'); trkg4_export_volume_convergence_level('$(Convert-ToMatlabPath (Join-Path $baselinePath "manifest.json"))','$(Convert-ToMatlabPath (Join-Path $levelPath "refinement\refined_mesh.mat"))','$(Convert-ToMatlabPath (Join-Path $levelPath "refinement\refinement.json"))','$(Convert-ToMatlabPath (Join-Path $levelPath "export"))','$LevelId');"
        Invoke-Logged -Stage "export_$LevelId" -Executable $matlabPath -Arguments @("-batch", $command) -LogBase (Join-Path $logs "export")
    }

    if (-not (Test-Path -LiteralPath (Join-Path $levelPath "manifest.json"))) {
        Invoke-Logged -Stage "configure_$LevelId" -Executable $pythonPath -LogBase (Join-Path $logs "configure") -Arguments @(
            $tool, "configure-level", "--baseline", $baselinePath, "--level", $levelPath, "--deps", $dependenciesPath
        )
    }

    if (-not (Test-Path -LiteralPath (Join-Path $levelPath "contacts\preparation.json"))) {
        $command = "run('$(Convert-ToMatlabPath $eidorsPath)'); addpath('$(Convert-ToMatlabPath $modelRoot)'); addpath('$(Convert-ToMatlabPath (Join-Path $modelRoot "src"))'); addpath('$(Convert-ToMatlabPath (Join-Path $modelRoot "vendor_stl_eidors"))'); r=trkg4_prepare_surface_sensitivity('$(Convert-ToMatlabPath (Join-Path $levelPath "manifest.json"))'); assert(strcmp(r.status,'passed'));"
        Invoke-Logged -Stage "contacts_$LevelId" -Executable $matlabPath -Arguments @("-batch", $command) -LogBase (Join-Path $logs "contacts")
    }

    if (-not (Test-Path -LiteralPath (Join-Path $levelPath "validation_plan.json"))) {
        Invoke-Logged -Stage "studies_$LevelId" -Executable $pythonPath -LogBase (Join-Path $logs "studies") -Arguments @(
            $tool, "make-studies", "--level", $levelPath, "--selection", $selectionPath, "--deps", $dependenciesPath, "--threads", "$SolverThreads"
        )
    }

    $plan = Get-Content -LiteralPath (Join-Path $levelPath "validation_plan.json") -Raw | ConvertFrom-Json
    foreach ($job in $plan.jobs) {
        $study = Join-Path $levelPath $job.study
        $result = Join-Path $levelPath $job.output
        if (-not (Test-Path -LiteralPath (Join-Path $result "completion.json"))) {
            if ((Test-Path -LiteralPath $result) -and (Get-ChildItem -LiteralPath $result -Force)) { throw "Partial solver output is preserved for $LevelId/$($job.montage)" }
            Invoke-Logged -Stage ("solve_" + $LevelId + "_" + $job.montage) -Executable $pythonPath -LogBase (Join-Path $logs ("solve_" + $job.montage)) -Arguments @(
                "-X", "utf8", $solver, "--study", $study, "--output", $result, "--deps", $dependenciesPath, "--threads", "$SolverThreads"
            )
        }
    }
    Invoke-Logged -Stage "verify_$LevelId" -Executable $pythonPath -LogBase (Join-Path $logs "verify") -Arguments @(
        $tool, "verify-level", "--level", $levelPath, "--deps", $dependenciesPath
    )
}

try {
    Invoke-Logged -Stage "select_states" -Executable $pythonPath -LogBase (Join-Path $outputPath "state_selection") -Arguments @(
        $tool, "select-states", "--baseline", $baselinePath, "--output", $selectionPath, "--deps", $dependenciesPath
    )
    Complete-Level -LevelId "L09" -FolderName "level09" -TrunkSizeMm 9
    Complete-Level -LevelId "L06" -FolderName "level06" -TrunkSizeMm 6

    $analysis = Join-Path $outputPath "analysis"
    if (-not (Test-Path -LiteralPath (Join-Path $analysis "summary.json"))) {
        Invoke-Logged -Stage "cross_level_analysis" -Executable $pythonPath -LogBase (Join-Path $outputPath "analysis") -Arguments @(
            $tool, "analyse", "--baseline", $baselinePath,
            "--level09", (Join-Path $outputPath "level09"), "--level06", (Join-Path $outputPath "level06"),
            "--selection", $selectionPath, "--output", $analysis, "--deps", $dependenciesPath
        )
    }
    Write-Status -Status "completed" -Stage "analysis" -Message "Independent-volume numerical convergence workflow completed"
}
catch {
    Write-Status -Status "failed" -Stage $currentStage -Message $_.Exception.Message
    throw
}