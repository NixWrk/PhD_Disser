[CmdletBinding()]
param(
    [ValidateRange(5, 1440)]
    [int]$IntervalMinutes = 15,
    [string]$TaskName = "Kalmykov research reporting",
    [ValidateRange(10, 300)]
    [int]$InitialRunTimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$reportScript = Join-Path $PSScriptRoot "research_report.py"
$python = Get-Command python -ErrorAction Stop
$configPath = Join-Path $projectRoot "research-report.json"
$config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json

if (-not (Test-Path -LiteralPath $config.archive_base)) {
    throw "Archive drive is unavailable: $($config.archive_base)"
}
if (-not (Test-Path -LiteralPath $reportScript)) {
    throw "Reporting script is missing: $reportScript"
}

$arguments = "`"$reportScript`" update"
$action = New-ScheduledTaskAction -Execute $python.Source -Argument $arguments -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Validate, build HTML/Markdown, and publish the research archive." `
    -Force | Out-Null

$receiptPath = Join-Path $projectRoot "research\generated\scheduled-run.json"
$previousRunId = $null
if (Test-Path -LiteralPath $receiptPath) {
    try {
        $previousReceipt = Get-Content -LiteralPath $receiptPath -Raw -Encoding UTF8 |
            ConvertFrom-Json
        $previousRunId = $previousReceipt.run_id
    }
    catch {
        $previousRunId = $null
    }
}

$startedAt = Get-Date
Start-ScheduledTask -TaskName $TaskName
$deadline = $startedAt.AddSeconds($InitialRunTimeoutSeconds)

do {
    Start-Sleep -Seconds 1
    $task = Get-ScheduledTask -TaskName $TaskName
    $taskInfo = $task | Get-ScheduledTaskInfo
    $freshRun = $taskInfo.LastRunTime -ge $startedAt.AddSeconds(-2)
} while ((-not $freshRun -or $task.State -eq "Running") -and (Get-Date) -lt $deadline)

if (-not $freshRun -or $task.State -eq "Running") {
    throw "Task '$TaskName' did not finish its initial run within $InitialRunTimeoutSeconds seconds."
}
if ($taskInfo.LastTaskResult -ne 0) {
    throw "Task '$TaskName' initial run failed with code $($taskInfo.LastTaskResult)."
}

if (-not (Test-Path -LiteralPath $receiptPath)) {
    throw "Task finished without a run receipt: $receiptPath"
}
$receipt = Get-Content -LiteralPath $receiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
$receiptStarted = [DateTimeOffset]::Parse($receipt.started_at)
$receiptCompleted = [DateTimeOffset]::Parse($receipt.completed_at)
$verifiedAt = Get-Date
if (
    [string]::IsNullOrWhiteSpace($receipt.run_id) -or
    $receipt.run_id -eq $previousRunId -or
    $receiptStarted.UtcDateTime -lt $startedAt.ToUniversalTime().AddSeconds(-2) -or
    $receiptStarted.UtcDateTime -gt $verifiedAt.ToUniversalTime().AddSeconds(2) -or
    $receiptCompleted.UtcDateTime -lt $receiptStarted.UtcDateTime -or
    $receiptCompleted.UtcDateTime -gt $verifiedAt.ToUniversalTime().AddSeconds(2)
) {
    throw "Task run receipt does not belong to the initiated run."
}

if ($receipt.outcome -eq "synced") {
    $manifestPath = Join-Path $config.archive_root ".research-mirror-manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        throw "Synchronized run has no archive manifest: $manifestPath"
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (
        $manifest.transaction_id -ne $receipt.archive_transaction_id -or
        $manifest.project_id -ne $config.project_id -or
        $manifest.source_git_branch -ne $receipt.git_branch -or
        $manifest.source_git_commit -ne $receipt.git_commit -or
        -not $receipt.git_clean -or
        -not $manifest.source_git_clean
    ) {
        throw "Run receipt does not match the committed archive manifest."
    }
    Write-Host (
        "Task '$TaskName' installed; archive publication verified. " +
        "Interval: $IntervalMinutes minutes; next run: $($taskInfo.NextRunTime)."
    )
}
elseif ($receipt.outcome -eq "deferred_dirty") {
    Write-Warning (
        "Task execution is verified, but archive publication is NOT verified: " +
        "the working tree contains uncommitted changes. A clean scheduled run will retry."
    )
    Write-Host (
        "Task '$TaskName' installed. Interval: $IntervalMinutes minutes; " +
        "next run: $($taskInfo.NextRunTime)."
    )
}
else {
    throw "Unexpected task run outcome: $($receipt.outcome)"
}
