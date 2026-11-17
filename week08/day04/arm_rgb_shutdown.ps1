# User-requested shutdown after successful RGB training AND verified local retrieval.
param()
$ErrorActionPreference = 'Stop'
if ($PSScriptRoot -ne 'D:\vision-learning-server\week08\day04') { throw 'Server-only script.' }
$job = Join-Path $PSScriptRoot 'jobs\rgb_baseline_run01_20261030_125726_0d9fec62'
$pythonExe = 'D:\cv-server-env\Scripts\python.exe'
$watcher = Join-Path $PSScriptRoot 'shutdown_after_rgb.py'
foreach ($path in @($pythonExe, $watcher, (Join-Path $job 'state.json'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing: $path" }
}
if (Test-Path -LiteralPath (Join-Path $job 'shutdown_watcher_status.json')) {
    throw 'Shutdown watcher already exists; inspect its status rather than starting another.'
}
$command = '"{0}" -u -B "{1}" --job "{2}"' -f $pythonExe, $watcher, $job
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $command; CurrentDirectory = 'D:\vision-learning-server'; ProcessStartupInformation = $startup
}
if ($created.ReturnValue -ne 0) { throw "Watcher launch failed: $($created.ReturnValue)" }
Write-Output "Shutdown watcher PID: $($created.ProcessId)"
Write-Output 'Waiting for training success AND verified download marker; shutdown is NOT scheduled yet.'
