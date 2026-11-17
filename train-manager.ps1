
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("start", "status", "log", "stop")]
    [string]$Action,

    [string]$Script = "server_train_demo.py",

    [int]$Tail = 40
)

$ErrorActionPreference = "Stop"

$ProjectDir = "D:\vision-learning-server"
$Python = "D:\cv-server-env\Scripts\python.exe"
$LogDir = Join-Path $ProjectDir "logs"
$StateDir = Join-Path $ProjectDir "run-state"
$StateFile = Join-Path $StateDir "current-job.json"
$MonitorScript = Join-Path $ProjectDir "train-monitor.ps1"

New-Item -ItemType Directory -Force -Path $LogDir, $StateDir |
    Out-Null

function Read-JobState {
    if (-not (Test-Path $StateFile)) {
        return $null
    }

    try {
        return Get-Content -Raw -Encoding UTF8 $StateFile |
            ConvertFrom-Json
    }
    catch {
        Write-Warning "Cannot read state file."
        return $null
    }
}

function Get-TrackedProcess($State) {
    if ($null -eq $State -or -not $State.MonitorPID) {
        return $null
    }

    Get-CimInstance Win32_Process `
        -Filter "ProcessId = $($State.MonitorPID)" `
        -ErrorAction SilentlyContinue
}

switch ($Action) {
    "start" {
        if (-not (Test-Path $Python)) {
            throw "Python environment not found: $Python"
        }

        if (-not (Test-Path $MonitorScript)) {
            throw "Monitor script not found: $MonitorScript"
        }

        if ([IO.Path]::GetFileName($Script) -ne $Script) {
            throw "Please provide a script filename only."
        }

        $ScriptPath = Join-Path $ProjectDir $Script

        if (-not (Test-Path $ScriptPath)) {
            throw "Training script not found: $ScriptPath"
        }

        $oldState = Read-JobState
        if ($null -ne $oldState) {
            $oldMonitor = Get-TrackedProcess $oldState
            if ($null -ne $oldMonitor) {
                throw "A managed training task is already running."
            }
        }

        $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
        $OutLog = Join-Path $LogDir "background_$RunId.log"
        $ErrLog = Join-Path $LogDir "background_$RunId.err"

        $state = [ordered]@{
            PID = $null
            MonitorPID = $null
            Script = $Script
            ScriptPath = $ScriptPath
            StartedAt = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
            FinishedAt = $null
            Status = "STARTING"
            ExitCode = $null
            OutputLog = $OutLog
            ErrorLog = $ErrLog
        }

        $state |
            ConvertTo-Json |
            Set-Content -Path $StateFile -Encoding UTF8

        $monitorArgs = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", $MonitorScript,
            "-Python", $Python,
            "-ScriptPath", $ScriptPath,
            "-ProjectDir", $ProjectDir,
            "-StateFile", $StateFile,
            "-OutputLog", $OutLog,
            "-ErrorLog", $ErrLog
        )

        $monitor = Start-Process `
            -FilePath "powershell.exe" `
            -ArgumentList $monitorArgs `
            -WorkingDirectory $ProjectDir `
            -WindowStyle Hidden `
            -PassThru

        $state.MonitorPID = $monitor.Id
        $state |
            ConvertTo-Json |
            Set-Content -Path $StateFile -Encoding UTF8

        Write-Output "Training monitor started."
        Write-Output "Monitor PID: $($monitor.Id)"
        Write-Output "Script: $ScriptPath"
        Write-Output "Output log: $OutLog"
        Write-Output "Error log: $ErrLog"
    }

    "status" {
        $state = Read-JobState

        if ($null -eq $state) {
            Write-Output "No tracked training job."
            break
        }

        $monitor = Get-TrackedProcess $state

        if ($null -ne $monitor -and
            $state.Status -notin @("SUCCEEDED", "FAILED", "STOPPED")) {
            Write-Output "Status: RUNNING"
        }
        else {
            Write-Output "Status: $($state.Status)"
        }

        Write-Output "Script: $($state.Script)"
        Write-Output "Monitor PID: $($state.MonitorPID)"
        Write-Output "Started: $($state.StartedAt)"
        Write-Output "Finished: $($state.FinishedAt)"
        Write-Output "Exit code: $($state.ExitCode)"
        Write-Output "Output log: $($state.OutputLog)"
        Write-Output "Error log: $($state.ErrorLog)"
    }

    "log" {
        $state = Read-JobState

        if ($null -eq $state) {
            Write-Output "No tracked training job."
            break
        }

        foreach ($file in @($state.OutputLog, $state.ErrorLog)) {
            Write-Output ""
            Write-Output "===== $file ====="

            if (Test-Path $file) {
                Get-Content -Encoding UTF8 $file -Tail $Tail
            }
            else {
                Write-Output "Log file not created yet."
            }
        }

        $appLogs = Get-ChildItem $LogDir `
            -Filter "train_demo_*.log" |
            Sort-Object LastWriteTime -Descending

        if ($appLogs.Count -gt 0) {
            Write-Output ""
            Write-Output "===== Latest application training log ====="
            Get-Content -Encoding UTF8 $appLogs[0].FullName -Tail $Tail
        }
    }

    "stop" {
        $state = Read-JobState

        if ($null -eq $state) {
            Write-Output "No tracked training job."
            break
        }

        $monitor = Get-TrackedProcess $state

        if ($null -eq $monitor) {
            Write-Output "No active monitor process found."
            Write-Output "Current status: $($state.Status)"
            break
        }

        # Stop the monitor and its Python child process.
        $pythonProc = Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -eq "python.exe" -and
                $_.CommandLine -and
                $_.CommandLine.Contains($state.ScriptPath)
            }

        foreach ($p in $pythonProc) {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        }

        Stop-Process -Id $state.MonitorPID -Force `
            -ErrorAction SilentlyContinue

        $state.Status = "STOPPED"
        $state.FinishedAt = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        $state |
            ConvertTo-Json |
            Set-Content -Path $StateFile -Encoding UTF8

        Write-Output "Stop requested for managed training task."
    }
}
