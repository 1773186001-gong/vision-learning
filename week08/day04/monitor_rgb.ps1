param([Parameter(Mandatory=$true)][string]$JobDir)
$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$JobsRoot = Join-Path $PSScriptRoot 'jobs'
$resolvedJob = [IO.Path]::GetFullPath($JobDir)
if ($ProjectDir -ne 'D:\vision-learning-server' -or
    -not $resolvedJob.StartsWith($JobsRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Unexpected server/project/job directory.'
}
$statePath = Join-Path $resolvedJob 'state.json'
$state = [ordered]@{ status = 'VERIFYING_LAUNCH'; monitor_pid = $PID; python_pid = $null
    started_at = (Get-Date).ToString('o'); finished_at = $null; exit_code = $null; output = $null; error = $null }
function Write-State {
    $temp = Join-Path $resolvedJob ('state_' + [Guid]::NewGuid().ToString('N') + '.tmp')
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temp -Encoding UTF8
    Move-Item -LiteralPath $temp -Destination $statePath -Force
}
try {
    Write-State
    $verifiedPath = Join-Path $resolvedJob 'launch_verified.json'
    for ($attempt = 0; $attempt -lt 30 -and -not (Test-Path -LiteralPath $verifiedPath); $attempt++) {
        Start-Sleep -Seconds 1
    }
    if (-not (Test-Path -LiteralPath $verifiedPath)) { throw 'Unverified launcher; no training started.' }
    $verified = Get-Content -LiteralPath $verifiedPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($verified.monitor_pid -ne $PID) { throw 'Verified monitor PID mismatch.' }
    $settings = Get-Content -LiteralPath (Join-Path $resolvedJob 'parameters.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($settings.project_dir -ne $ProjectDir -or $settings.python -ne 'D:\cv-server-env\Scripts\python.exe' -or
        $settings.entry -ne (Join-Path $PSScriptRoot 'train_rgb.py') -or
        $settings.data_root -ne (Join-Path $ProjectDir 'data\LLVIP_1000') -or
        -not $settings.output.StartsWith((Join-Path $PSScriptRoot 'outputs') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Unexpected job paths; refusing execution.'
    }
    if (Test-Path -LiteralPath $settings.output) { throw 'Output appeared after launch; refuse to overwrite.' }
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $arguments = @('-u', '-B', ('"{0}"' -f $settings.entry), '--data-root', ('"{0}"' -f $settings.data_root),
                   '--output', ('"{0}"' -f $settings.output), '--device', 'cuda:0')
    if ($settings.keep_awake) { $arguments += '--keep-awake' }
    $process = Start-Process -FilePath $settings.python -ArgumentList $arguments -WorkingDirectory $ProjectDir `
        -RedirectStandardOutput (Join-Path $resolvedJob 'stdout.log') `
        -RedirectStandardError (Join-Path $resolvedJob 'stderr.log') -WindowStyle Hidden -PassThru
    $state.status = 'RUNNING'
    $state.python_pid = $process.Id
    $state.output = $settings.output
    Write-State
    $null = $process.Handle
    $process.WaitForExit()
    $state.exit_code = $process.ExitCode
    $state.finished_at = (Get-Date).ToString('o')
    if ($process.ExitCode -eq 0 -and (Test-Path -LiteralPath (Join-Path $settings.output 'completed.json'))) {
        $state.status = 'SUCCEEDED'
    } else { $state.status = 'FAILED' }
    Write-State
} catch {
    $state.status = 'FAILED'
    $state.finished_at = (Get-Date).ToString('o')
    $state.error = $_.Exception.Message
    Write-State
    exit 1
}
