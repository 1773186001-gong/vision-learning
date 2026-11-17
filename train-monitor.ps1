
param(
    [Parameter(Mandatory = $true)]
    [string]$Python,

    [Parameter(Mandatory = $true)]
    [string]$ScriptPath,

    [Parameter(Mandatory = $true)]
    [string]$ProjectDir,

    [Parameter(Mandatory = $true)]
    [string]$StateFile,

    [Parameter(Mandatory = $true)]
    [string]$OutputLog,

    [Parameter(Mandatory = $true)]
    [string]$ErrorLog
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$state = Get-Content -Raw -Encoding UTF8 $StateFile |
    ConvertFrom-Json

try {
    $state.Status = "RUNNING"
    $state |
        ConvertTo-Json |
        Set-Content -Path $StateFile -Encoding UTF8

    $proc = Start-Process `
        -FilePath $Python `
        -ArgumentList @("-u", $ScriptPath) `
        -WorkingDirectory $ProjectDir `
        -RedirectStandardOutput $OutputLog `
        -RedirectStandardError $ErrorLog `
        -PassThru `
        -Wait

    $exitCode = $proc.ExitCode

    $state = Get-Content -Raw -Encoding UTF8 $StateFile |
        ConvertFrom-Json

    $state.ExitCode = $exitCode
    $state.FinishedAt = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")

    if ($exitCode -eq 0) {
        $state.Status = "SUCCEEDED"
    }
    else {
        $state.Status = "FAILED"
    }

    $state |
        ConvertTo-Json |
        Set-Content -Path $StateFile -Encoding UTF8
}
catch {
    try {
        $state = Get-Content -Raw -Encoding UTF8 $StateFile |
            ConvertFrom-Json

        $state.Status = "FAILED"
        $state.ExitCode = $null
        $state.FinishedAt = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        $state.Error = $_.Exception.Message

        $state |
            ConvertTo-Json |
            Set-Content -Path $StateFile -Encoding UTF8
    }
    catch {
        # Preserve the original failure if state writing also fails.
    }

    exit 1
}
