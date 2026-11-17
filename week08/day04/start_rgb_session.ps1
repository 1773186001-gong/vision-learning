# Day04 session-only policy variant. Original launcher and monitor are unchanged.
# Invoke this file with powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File ...
param(
    [ValidateSet('start', 'status')][string]$Action = 'status',
    [string]$JobDir,
    [switch]$KeepAwake
)
$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ([IO.Path]::GetFullPath($ProjectDir) -ne 'D:\vision-learning-server') {
    throw 'This launcher is server-only: D:\vision-learning-server'
}
$JobsRoot = Join-Path $PSScriptRoot 'jobs'
$Python = 'D:\cv-server-env\Scripts\python.exe'
$Entry = Join-Path $PSScriptRoot 'train_rgb.py'
$Monitor = Join-Path $PSScriptRoot 'monitor_rgb.ps1'

if ($Action -eq 'status') {
    if (-not $JobDir) { throw 'Provide the exact JobDir printed by start.' }
    $resolvedJob = [IO.Path]::GetFullPath($JobDir)
    if (-not $resolvedJob.StartsWith($JobsRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'JobDir must be inside Day04/jobs.'
    }
    $state = Get-Content -LiteralPath (Join-Path $resolvedJob 'state.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $state | ConvertTo-Json -Depth 8
    foreach ($processIdValue in @($state.monitor_pid, $state.python_pid)) {
        if ($processIdValue) {
            Get-CimInstance Win32_Process -Filter "ProcessId = $processIdValue" |
                Select-Object ProcessId, ParentProcessId, Name, CommandLine
        }
    }
    foreach ($log in @('stdout.log', 'stderr.log')) {
        $path = Join-Path $resolvedJob $log
        if (Test-Path -LiteralPath $path) { Get-Content -LiteralPath $path -Encoding UTF8 -Tail 8 }
    }
    exit 0
}

foreach ($path in @($Python, $Entry, $Monitor)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing: $path" }
}
# Never terminate an existing job; refuse concurrent RGB training.
$running = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains('train_rgb.py') })
if ($running.Count) { throw 'RGB Python process already exists. Inspect it; do not launch another.' }

$Outputs = Join-Path $PSScriptRoot 'outputs'
$index = 1
do {
    $runName = 'rgb_baseline_run{0:D2}' -f $index
    $Output = Join-Path $Outputs $runName
    $index++
} while (Test-Path -LiteralPath $Output)

if (-not (Test-Path -LiteralPath $JobsRoot)) {
    New-Item -ItemType Directory -Path $JobsRoot | Out-Null
}
$newJobName = '{0}_{1}_{2}' -f $runName, (Get-Date -Format 'yyyyMMdd_HHmmss'), ([Guid]::NewGuid().ToString('N').Substring(0,8))
$newJob = Join-Path $JobsRoot $newJobName
New-Item -ItemType Directory -Path $newJob | Out-Null
$parameters = [ordered]@{
    project_dir = $ProjectDir; python = $Python; entry = $Entry
    data_root = (Join-Path $ProjectDir 'data\LLVIP_1000')
    output = $Output; keep_awake = [bool]$KeepAwake
    created_at = (Get-Date).ToString('o')
}
$parameters | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $newJob 'parameters.json') -Encoding UTF8
[ordered]@{ status = 'WAITING_FOR_VERIFIED_MONITOR'; monitor_pid = $null; python_pid = $null; output = $Output } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $newJob 'state.json') -Encoding UTF8

# WMI service creates the monitor outside the SSH process tree. No scheduled task,
# service, password storage, environment installation, or power-plan edits.
$PowerShellExe = Join-Path $PSHOME 'powershell.exe'
$commandLine = '"{0}" -NoProfile -ExecutionPolicy RemoteSigned -WindowStyle Hidden -File "{1}" -JobDir "{2}"' -f $PowerShellExe, $Monitor, $newJob
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $commandLine; CurrentDirectory = $ProjectDir; ProcessStartupInformation = $startup
}
$receipt = [ordered]@{ return_value = $created.ReturnValue; monitor_pid = $created.ProcessId; command_line = $commandLine }
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $newJob 'launch_receipt.json') -Encoding UTF8
if ($created.ReturnValue -ne 0) { throw "WMI launch refused ($($created.ReturnValue)); training not started; preserve $newJob" }
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $($created.ProcessId)"
if (-not $process) { throw "Monitor missing; no launch approval file written. Inspect $newJob" }
$parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($process.ParentProcessId)"
if (-not $parent -or $parent.Name -ne 'WmiPrvSE.exe') {
    throw 'Cannot verify monitor is outside SSH tree; monitor will time out without training.'
}
[ordered]@{ monitor_pid = $created.ProcessId; parent_pid = $parent.ProcessId; parent_name = $parent.Name; verified_at = (Get-Date).ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $newJob 'launch_verified.json') -Encoding UTF8
Write-Output "JobDir: $newJob"
Write-Output "Output: $Output"
Write-Output "Verified WMI monitor PID: $($created.ProcessId)"
Write-Output 'Launch accepted. Reconnect and inspect state/PIDs/log progress before claiming persistence or success.'

