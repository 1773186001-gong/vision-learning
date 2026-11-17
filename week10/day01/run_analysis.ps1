param([string]$Output = 'week10/day01/outputs/candidate_review_run01')
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$outputPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $Output))
$dayRoot = [IO.Path]::GetFullPath($PSScriptRoot) + [IO.Path]::DirectorySeparatorChar
if (-not $outputPath.StartsWith($dayRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Output must remain inside week10/day01' }
$recordPath = $outputPath + '_process.json'
$processLog = $outputPath + '_process.log'
if ((Test-Path -LiteralPath $outputPath) -or (Test-Path -LiteralPath $recordPath) -or (Test-Path -LiteralPath $processLog)) { throw 'Refusing to overwrite existing output or process evidence' }
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($outputPath)) | Out-Null
$pythonPath = 'D:\python\.venv\Scripts\python.exe'
$scriptPath = Join-Path $PSScriptRoot 'analyze_candidates.py'
$commandText = "& '$pythonPath' -B -X utf8 '$scriptPath' --output '$outputPath'"
$startedAt = [DateTimeOffset]::Now.ToString('o')
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -B -X utf8 $scriptPath --output $outputPath 2>&1 | Tee-Object -FilePath $processLog
    $analysisExitCode = $LASTEXITCODE
} finally { Pop-Location }
@{ command = $commandText; working_directory = $projectRoot; started_at = $startedAt; finished_at = [DateTimeOffset]::Now.ToString('o'); exit_code = $analysisExitCode; process_log = $processLog } | ConvertTo-Json | Set-Content -LiteralPath $recordPath -Encoding utf8
Write-Output "DAY01_PROCESS_EXIT_CODE=$analysisExitCode"
exit $analysisExitCode
