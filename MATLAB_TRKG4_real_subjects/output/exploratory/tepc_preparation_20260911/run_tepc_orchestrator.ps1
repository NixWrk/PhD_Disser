$ErrorActionPreference = 'Stop'

$project = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$baseRelative = 'output/exploratory/tepc_preparation_20260911'
$depsRelative = 'output/exploratory/arm_sigma_20260908/python_solver_deps'
$python = 'C:\PC\Python\python.exe'
$matlab = 'C:/PC/Matlab/2025b/bin/matlab.exe'
$statusFile = Join-Path $PSScriptRoot 'orchestrator_status.json'
$startedUtc = [DateTime]::UtcNow.ToString('o')
$currentStage = 'initialization'
$commit = (& git -C $project rev-parse HEAD).Trim()

function Write-State {
    param(
        [string]$Status,
        [string]$Stage,
        [string]$Message
    )
    $record = [ordered]@{
        status = $Status
        stage = $Stage
        message = $Message
        pid = $PID
        commit = $commit
        started_utc = $startedUtc
        updated_utc = [DateTime]::UtcNow.ToString('o')
        base = $baseRelative
    }
    $temporary = "$statusFile.tmp"
    $json = $record | ConvertTo-Json -Depth 4
    [System.IO.File]::WriteAllText(
        $temporary,
        $json,
        [System.Text.UTF8Encoding]::new($false)
    )
    Move-Item -LiteralPath $temporary -Destination $statusFile -Force
}

function Invoke-Checked {
    param(
        [string]$Stage,
        [string]$Executable,
        [string[]]$CommandArguments
    )
    $script:currentStage = $Stage
    Write-State -Status 'running' -Stage $Stage -Message "Started $Stage"
    & $Executable @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE"
    }
    Write-State -Status 'running' -Stage $Stage -Message "Completed $Stage"
}

try {
    Set-Location -LiteralPath $project
    $common = @(
        'tools/run_surface_sensitivity.py',
        '--base', $baseRelative,
        '--deps', $depsRelative
    )
    Invoke-Checked -Stage 'pilot' -Executable $python -CommandArguments (
        $common + @('--matlab', $matlab, '--pilot')
    )
    Invoke-Checked -Stage 'full' -Executable $python -CommandArguments (
        $common + @('--matlab', $matlab, '--full')
    )
    Invoke-Checked -Stage 'analysis' -Executable $python -CommandArguments (
        $common + @('--analyze')
    )
    Invoke-Checked -Stage 'report' -Executable $python -CommandArguments @(
        'tools/build_surface_sensitivity_report.py',
        '--base', $baseRelative
    )
    Write-State -Status 'completed' -Stage 'report' -Message 'TEPC calculation, analysis, and report completed'
    exit 0
}
catch {
    Write-State -Status 'failed' -Stage $currentStage -Message $_.Exception.Message
    [Console]::Error.WriteLine($_.Exception.ToString())
    exit 1
}
