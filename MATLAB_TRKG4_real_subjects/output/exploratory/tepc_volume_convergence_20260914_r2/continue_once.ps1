$ErrorActionPreference = "Stop"

$project = "D:\Аспа\Kalmykov_PhD\MATLAB_TRKG4_real_subjects"
$output = Join-Path $project "output\exploratory\tepc_volume_convergence_20260914_r2"
$baseline = Join-Path $project "output\exploratory\tepc_preparation_20260911"
$deps = Join-Path $project "output\exploratory\arm_sigma_20260908\python_solver_deps"
$python = "C:\PC\Python\python.exe"
$matlab = "C:\PC\Matlab\2025b\bin\matlab.exe"
$eidors = "C:\PC\FEM\eidors-v3.12-ng\eidors\startup.m"
$surface = Join-Path $project "output\nik_body_arm_parameter_surface_1mm_candidate.stl"
$preparation = Join-Path $project "output\nik_stl_preparation_report.json"
$orchestrator = Join-Path $project "tools\run_tepc_volume_convergence.ps1"
$statusPath = Join-Path $output "detached_status.json"
$pipelineLog = Join-Path $output "continuation_pipeline.log"

$meshes = @(
    [pscustomobject]@{
        level = "L09"
        pid = 27308
        target = Join-Path $output "level09\mesh\body_trunk9mm.msh"
        report = Join-Path $output "level09\mesh\gmsh_build.json"
    },
    [pscustomobject]@{
        level = "L06"
        pid = 2632
        target = Join-Path $output "level06\mesh\body_trunk6mm.msh"
        report = Join-Path $output "level06\mesh\gmsh_build.json"
    }
)

function Write-RunStatus {
    param([string]$Status, [string]$Stage, [string]$Message, [object]$Details = $null)
    $value = [ordered]@{
        status = $Status
        stage = $Stage
        message = $Message
        updated = [DateTimeOffset]::Now.ToString("o")
        launcher_pid = $PID
        output = $output
        details = $Details
    }
    $temporary = "$statusPath.tmp"
    $value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

try {
    while ($true) {
        $details = @($meshes | ForEach-Object {
            $process = Get-Process -Id $_.pid -ErrorAction SilentlyContinue
            [ordered]@{
                level = $_.level
                pid = $_.pid
                is_running = $null -ne $process
                cpu_seconds = if ($null -eq $process) { $null } else { $process.TotalProcessorTime.TotalSeconds }
                private_memory_bytes = if ($null -eq $process) { $null } else { $process.PrivateMemorySize64 }
            }
        })
        $running = @($details | Where-Object { $_.is_running })
        if ($running.Count -eq 0) { break }
        Write-RunStatus -Status "running" -Stage "parallel_gmsh" -Message "Independent meshes are running; continuation monitor is active" -Details $details
        Start-Sleep -Seconds 30
    }

    $missing = @($meshes | ForEach-Object {
        if (-not (Test-Path -LiteralPath $_.target)) { $_.target }
        if (-not (Test-Path -LiteralPath $_.report)) { $_.report }
    })
    if ($missing.Count -gt 0) {
        throw "Gmsh processes ended without required output: $($missing -join '; ')"
    }

    foreach ($mesh in $meshes) {
        $build = Get-Content -LiteralPath $mesh.report -Raw | ConvertFrom-Json
        if ($build.status -ne "candidate_mesh_requires_external_quality_and_matlab_qc") {
            throw "$($mesh.level) build report status is '$($build.status)'"
        }
    }

    Write-RunStatus -Status "running" -Stage "eidors_pipeline" -Message "Meshes completed; running classification, QC, EIDORS solves and analysis"
    & $orchestrator -Baseline $baseline -Output $output -Dependencies $deps -Python $python -Matlab $matlab -EidorsStartup $eidors -Surface $surface -PreparationReport $preparation -GmshThreads 16 -SolverThreads 4 *>> $pipelineLog

    $summary = Join-Path $output "analysis\summary.json"
    if (-not (Test-Path -LiteralPath $summary)) {
        throw "The pipeline ended without analysis summary"
    }
    Write-RunStatus -Status "completed" -Stage "analysis" -Message "Numerical volume-convergence workflow completed" -Details @{ summary = $summary }
}
catch {
    Write-RunStatus -Status "failed" -Stage "continuation" -Message $_.Exception.Message
    $_ | Out-String | Add-Content -LiteralPath $pipelineLog -Encoding UTF8
    throw
}
