# Local read-only retrieval of this exact authorized job's small status/log files.
param([string]$Server='192.168.0.149')
$ErrorActionPreference='Stop'
if ($PSScriptRoot -ne 'D:\python\test\vision-learning\week08\day05') { throw 'Local project only.' }
if ($Server -notmatch '^[a-zA-Z0-9.-]+$') { throw 'Invalid server.' }
$tag=(Get-Date -Format 'yyyyMMdd_HHmmss')+'_'+[guid]::NewGuid().ToString('N').Substring(0,8)
$folder=Join-Path $PSScriptRoot ('checks\remote_status_'+$tag)
if (Test-Path -LiteralPath $folder) { throw 'Destination already exists.' }
New-Item -ItemType Directory -Path $folder | Out-Null
$source='gxbb@'+$Server+':D:/vision-learning-server/week08/day05/jobs/fusion_concat_run01_20261102_142408_3a904cd8'
& scp -r $source $folder
if ($LASTEXITCODE -ne 0) { throw 'Status download failed; partial files preserved, no retry.' }
Write-Host "STATUS_DOWNLOADED: $folder"
Write-Host 'No training, remote changes, or model downloads performed.'
