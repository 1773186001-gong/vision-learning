param([string]$Output='week10/day05/outputs/roi_nms_replay_run01')
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$outputPath=[IO.Path]::GetFullPath((Join-Path $projectRoot $Output))
if (-not $outputPath.StartsWith(([IO.Path]::GetFullPath($PSScriptRoot)+[IO.Path]::DirectorySeparatorChar),[StringComparison]::OrdinalIgnoreCase)) { throw 'Output outside day05' }
foreach ($p in @($outputPath,($outputPath+'_process.log'),($outputPath+'_process.json'))) { if (Test-Path -LiteralPath $p) { throw 'Refusing overwrite' } }
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($outputPath)) | Out-Null
$pythonPath='D:\python\.venv\Scripts\python.exe'
$scriptPath=Join-Path $PSScriptRoot 'replay_roi_nms.py'
$started=[DateTimeOffset]::Now.ToString('o')
Push-Location -LiteralPath $projectRoot
try { & $pythonPath -B -X utf8 $scriptPath --output $outputPath 2>&1 | Tee-Object -FilePath ($outputPath+'_process.log'); $processExit=$LASTEXITCODE } finally { Pop-Location }
@{command="& '$pythonPath' -B -X utf8 '$scriptPath' --output '$outputPath'";working_directory=$projectRoot;started_at=$started;finished_at=[DateTimeOffset]::Now.ToString('o');exit_code=$processExit} | ConvertTo-Json | Set-Content -LiteralPath ($outputPath+'_process.json') -Encoding utf8
Write-Output "DAY05_PROCESS_EXIT_CODE=$processExit"
exit $processExit
