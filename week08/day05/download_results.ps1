# Local inbound transfer only. Does not rerun inference/training or change remote files.
param([string]$Server='192.168.0.149')
$ErrorActionPreference='Stop'
if ($PSScriptRoot -ne 'D:\python\test\vision-learning\week08\day05') { throw 'Local project only.' }
if ($Server -notmatch '^[a-zA-Z0-9.-]+$') { throw 'Invalid server.' }
$tag=(Get-Date -Format 'yyyyMMdd_HHmmss')+'_'+[guid]::NewGuid().ToString('N').Substring(0,8)
$output=Join-Path $PSScriptRoot 'outputs\fusion_concat_run01'
if (Test-Path -LiteralPath $output) { $output=Join-Path $PSScriptRoot ('outputs\fusion_concat_retrieved_'+$tag) }
if (Test-Path -LiteralPath $output) { throw 'Destination already exists.' }
New-Item -ItemType Directory -Path $output | Out-Null
$remote='gxbb@'+$Server+':D:/vision-learning-server/week08/day05/outputs/fusion_concat_run01/'
$sources=@('best.pth','last.pth','test_predictions.pth','*.json','*.csv','*.log','*.md','*.png','prediction_examples') | ForEach-Object { $remote+$_ }
Write-Host "Download destination: $output"
Write-Host 'Enter password only at SSH prompts. Per-epoch checkpoints stay on server.'
& scp -r @sources $output
if ($LASTEXITCODE -ne 0) { throw 'Transfer failed; partial files preserved. Do not claim success or automatically retry.' }
$record=Join-Path $PSScriptRoot ('checks\result_transfer_'+$tag+'.json')
[ordered]@{status='downloaded_pending_verification';output=$output;server=$Server;remote_run='fusion_concat_run01';training_repeated=$false;test_inference_repeated=$false} | ConvertTo-Json | Set-Content -LiteralPath $record -Encoding UTF8
Write-Host "DOWNLOAD_FINISHED: $output"
Write-Host 'Local checkpoint/cache/metric verification is still required.'
