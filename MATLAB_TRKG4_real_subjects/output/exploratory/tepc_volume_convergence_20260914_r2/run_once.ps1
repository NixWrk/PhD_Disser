$ErrorActionPreference = "Stop"
$project = "D:\Аспа\Kalmykov_PhD\MATLAB_TRKG4_real_subjects"
$output = "D:\Аспа\Kalmykov_PhD\MATLAB_TRKG4_real_subjects\output\exploratory\tepc_volume_convergence_20260914_r2"
$baseline = Join-Path $project "output\exploratory\tepc_preparation_20260911"
$deps = Join-Path $project "output\exploratory\arm_sigma_20260908\python_solver_deps"
$python = "C:\PC\Python\python.exe"
$matlab = "C:\PC\Matlab\2025b\bin\matlab.exe"
$eidors = "C:\PC\FEM\eidors-v3.12-ng\eidors\startup.m"
$surface = Join-Path $project "output\nik_body_arm_parameter_surface_1mm_candidate.stl"
$preparation = Join-Path $project "output\nik_stl_preparation_report.json"
$gmshTool = Join-Path $project "tools\gmsh_arm_parameter_mesh.py"
$orchestrator = Join-Path $project "tools\run_tepc_volume_convergence.ps1"
$statusPath = Join-Path $output "detached_status.json"

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

function Quote-Argument {
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

function Start-Mesh {
    param([string]$Level, [int]$Trunk)
    $directory = Join-Path $output ("level{0}\mesh" -f $Level)
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $target = Join-Path $directory ("body_trunk{0}mm.msh" -f $Trunk)
    $report = Join-Path $directory "gmsh_build.json"
    if ((Test-Path -LiteralPath $target) -or (Test-Path -LiteralPath $report)) {
        throw "The output for L$Level is not new"
    }
    $arguments = @(
        "-X", "utf8", $gmshTool, $surface, $preparation, $target, $report,
        "--arm-size-mm", "1", "--trunk-size-mm", "$Trunk", "--transition-mm", "5",
        "--threads", "16", "--algorithm-3d", "1"
    )
    $argumentText = (($arguments | ForEach-Object { Quote-Argument $_ }) -join " ")
    $process = Start-Process -FilePath $python -ArgumentList $argumentText -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $directory "gmsh.stdout.log") -RedirectStandardError (Join-Path $directory "gmsh.stderr.log")
    return [pscustomobject]@{ level = "L$Level"; target = $target; report = $report; process = $process }
}

try {
    Write-RunStatus -Status "running" -Stage "starting_parallel_meshes" -Message "Starting independent L09 and L06 volume meshes"
    $runs = @((Start-Mesh -Level "09" -Trunk 9), (Start-Mesh -Level "06" -Trunk 6))
    while (($runs | Where-Object { -not $_.process.HasExited }).Count -gt 0) {
        foreach ($run in $runs) { $run.process.Refresh() }
        $details = @($runs | ForEach-Object {
            [ordered]@{
                level = $_.level
                pid = $_.process.Id
                has_exited = $_.process.HasExited
                cpu_seconds = if ($_.process.HasExited) { $null } else { $_.process.TotalProcessorTime.TotalSeconds }
                private_memory_bytes = if ($_.process.HasExited) { $null } else { $_.process.PrivateMemorySize64 }
            }
        })
        Write-RunStatus -Status "running" -Stage "parallel_gmsh" -Message "Independent meshes are running" -Details $details
        Start-Sleep -Seconds 30
    }
    foreach ($run in $runs) {
        $run.process.WaitForExit()
        if ($run.process.ExitCode -ne 0) { throw "$($run.level) Gmsh failed with exit code $($run.process.ExitCode)" }
        if (-not ((Test-Path -LiteralPath $run.target) -and (Test-Path -LiteralPath $run.report))) { throw "$($run.level) ended without mesh/report" }
    }
    Write-RunStatus -Status "running" -Stage "eidors_pipeline" -Message "Meshes completed; running classification, QC, EIDORS solves and analysis"
    & $orchestrator -Baseline $baseline -Output $output -Dependencies $deps -Python $python -Matlab $matlab -EidorsStartup $eidors -Surface $surface -PreparationReport $preparation -GmshThreads 16 -SolverThreads 4
    if ($LASTEXITCODE -ne 0) { throw "Volume-convergence orchestrator failed with exit code $LASTEXITCODE" }
    $summary = Join-Path $output "analysis\summary.json"
    if (-not (Test-Path -LiteralPath $summary)) { throw "The pipeline ended without analysis summary" }
    Write-RunStatus -Status "completed" -Stage "analysis" -Message "Numerical volume-convergence workflow completed" -Details @{summary=$summary}
}
catch {
    Write-RunStatus -Status "failed" -Stage "launcher" -Message $_.Exception.Message
    throw
}