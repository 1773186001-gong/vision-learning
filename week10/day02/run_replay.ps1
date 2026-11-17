param([string]$Output='week10/day02/outputs/upstream_replay_run01')
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$target=[IO.Path]::GetFullPath((Join-Path $root $Output))
if (-not $target.StartsWith(([IO.Path]::GetFullPath($PSScriptRoot)+[IO.Path]::DirectorySeparatorChar),[StringComparison]::OrdinalIgnoreCase)) { throw 'Output outside day02' }
foreach ($p in @($target,($target+'_process.log'),($target+'_process.json'))) { if (Test-Path -LiteralPath $p) { throw 'Refusing overwrite' } }
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) | Out-Null
$python='D:\python\.venv\Scripts\python.exe'
$script=Join-Path $PSScriptRoot 'replay_upstream.py'
$started=[DateTimeOffset]::Now.ToString('o')
Push-Location -LiteralPath $root
try { & $python -B -X utf8 $script --output $target 2>&1 | Tee-Object -FilePath ($target+'_process.log'); $code=$LASTEXITCODE } finally { Pop-Location }
@{command="& '$python' -B -X utf8 '$script' --output '$target'"; working_directory=$root; started_at=$started; finished_at=[DateTimeOffset]::Now.ToString('o');exit_code=$code} | ConvertTo-Json | Set-Content -LiteralPath ($target+'_process.json') -Encoding utf8
Write-Output "DAY02_PROCESS_EXIT_CODE=$code"
exit $code
