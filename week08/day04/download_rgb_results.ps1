# Run on the LOCAL PC. Read-only remote retrieval; no training or shutdown.
param([string]$Server = '192.168.0.149')
$ErrorActionPreference = 'Stop'
$expected = 'D:\python\test\vision-learning\week08\day04'
if ($PSScriptRoot -ne $expected) { throw 'Run this script from the local Day04 project.' }
if ($Server -notmatch '^[a-zA-Z0-9.-]+$') { throw 'Invalid server hostname.' }
$scp = (Get-Command scp.exe -ErrorAction Stop).Source
$tag = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8)
$output = Join-Path $PSScriptRoot ('outputs\rgb_baseline_retrieved_' + $tag)
$job = Join-Path $PSScriptRoot ('checks\retrieval_' + $tag + '\job')
foreach ($destination in @($output, $job)) {
    if (Test-Path -LiteralPath $destination) { throw "Destination already exists: $destination" }
    New-Item -ItemType Directory -Path $destination | Out-Null
}
$remote = 'gxbb@' + $Server + ':D:/vision-learning-server/week08/day04/'
$run = $remote + 'outputs/rgb_baseline_run01/'
$sources = @('best.pth','last.pth','test_predictions.pth','*.json','*.csv','*.txt','*.md','*.png','prediction_examples') | ForEach-Object { $run + $_ }
$record = Join-Path (Split-Path -Parent $job) 'transfer.json'
$info = [ordered]@{ status='started'; output=$output; job=$job; server=$Server; remote_run='rgb_baseline_run01'; training_started=$false; shutdown_requested=$false }
$info | ConvertTo-Json | Set-Content -LiteralPath $record -Encoding UTF8
Write-Host 'Enter the server password only at the OpenSSH prompt, never in chat.'
Write-Host "Results destination: $output"
& $scp -r @sources $output
if ($LASTEXITCODE -ne 0) { throw 'Result transfer failed. Partial files are preserved; no automatic retry.' }
$remoteJob = $remote + 'jobs/rgb_baseline_run01_20261030_125726_0d9fec62/'
& $scp ($remoteJob + '*.json') ($remoteJob + '*.log') $job
if ($LASTEXITCODE -ne 0) { throw 'Job transfer failed. Partial files are preserved; no automatic retry.' }
$info.status = 'downloaded_pending_review'
$info | ConvertTo-Json | Set-Content -LiteralPath $record -Encoding UTF8
Write-Host 'DOWNLOAD_FINISHED - local verification is still required.'
Write-Host "Output: $output"
Write-Host "Job: $job"
Write-Host 'No model training, remote file changes, or shutdown were performed.'
