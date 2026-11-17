param([string]$Output='week10/day06/outputs/success_case_trace_run01')
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$target=[IO.Path]::GetFullPath((Join-Path $projectRoot $Output))
if (-not $target.StartsWith(([IO.Path]::GetFullPath($PSScriptRoot)+[IO.Path]::DirectorySeparatorChar),[StringComparison]::OrdinalIgnoreCase)) { throw 'Output outside day06' }
foreach ($p in @($target,($target+'_process.log'),($target+'_process.json'))) { if (Test-Path -LiteralPath $p) { throw 'Refusing overwrite' } }
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) | Out-Null
$pythonPath='D:\python\.venv\Scripts\python.exe'
$scriptPath=Join-Path $PSScriptRoot 'trace_success.py'
$started=[DateTimeOffset]::Now.ToString('o')
Push-Location -LiteralPath $projectRoot
try { & $pythonPath -B -X utf8 $scriptPath --output $target 2>&1 | Tee-Object -FilePath ($target+'_process.log'); $processExit=$LASTEXITCODE } finally { Pop-Location }
@{command="& '$pythonPath' -B -X utf8 '$scriptPath' --output '$target'";working_directory=$projectRoot;started_at=$started;finished_at=[DateTimeOffset]::Now.ToString('o');exit_code=$processExit} | ConvertTo-Json | Set-Content -LiteralPath ($target+'_process.json') -Encoding utf8
Write-Output "DAY06_PROCESS_EXIT_CODE=$processExit"
exit $processExit
