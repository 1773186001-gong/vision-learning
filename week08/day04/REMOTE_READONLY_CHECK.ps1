# Initial read-only inspection, to be executed ONLY after remote authorization.
$ErrorActionPreference = 'Stop'
$ProjectDir = 'D:\vision-learning-server'
$Python = 'D:\cv-server-env\Scripts\python.exe'
Write-Output 'HOSTNAME:'
hostname
Write-Output 'LOCAL IPV4 ADDRESSES:'
Get-NetIPAddress -AddressFamily IPv4 | Select-Object IPAddress, InterfaceAlias
Write-Output 'REQUIRED PATHS:'
foreach ($path in @($Python, $ProjectDir, (Join-Path $ProjectDir 'data\LLVIP_1000'),
                   (Join-Path $ProjectDir 'week08\day03\preflight_ir.py'),
                   (Join-Path $ProjectDir 'week08\day04'))) {
    [pscustomobject]@{ path = $path; exists = (Test-Path -LiteralPath $path) } | Format-List
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Server interpreter missing.' }
Write-Output 'EXISTING DAY03 ENVIRONMENT CHECK:'
& $Python -B (Join-Path $ProjectDir 'week08\day03\preflight_ir.py')
if ($LASTEXITCODE -ne 0) { throw 'Day03 environment check failed.' }
Write-Output 'DATA COUNTS:'
foreach ($split in @('train','val','test')) {
    foreach ($relative in @("images\$split\visible", "images\$split\infrared", "annotations_voc\$split")) {
        $folder = Join-Path (Join-Path $ProjectDir 'data\LLVIP_1000') $relative
        if (-not (Test-Path -LiteralPath $folder -PathType Container)) { throw "Missing: $folder" }
        [pscustomobject]@{ folder = $folder; files = @(Get-ChildItem -LiteralPath $folder -File).Count } | Format-List
    }
}
Get-FileHash -LiteralPath (Join-Path $ProjectDir 'data\LLVIP_1000\manifest.csv') -Algorithm SHA256 | Format-List
Write-Output 'EXISTING DAY04 FILES (DO NOT OVERWRITE):'
$day04 = Join-Path $ProjectDir 'week08\day04'
if (Test-Path -LiteralPath $day04) {
    Get-ChildItem -LiteralPath $day04 -File -Recurse | Select-Object FullName, Length
}
Write-Output 'RUNNING PYTHON PROCESSES (READ ONLY; DO NOT STOP):'
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Select-Object ProcessId, ParentProcessId, CommandLine
Write-Output 'POWER (READ ONLY):'
powercfg /getactivescheme
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE
powercfg /query SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE
powercfg /query SCHEME_CURRENT SUB_BUTTONS LIDACTION
Get-CimInstance Win32_Battery | Select-Object BatteryStatus, EstimatedChargeRemaining
Write-Output 'GPU UTILIZATION (IF TOOL AVAILABLE):'
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
}
