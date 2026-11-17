# LOCAL interactive command-line handoff; existing SSH auth only, no saved password.
param([string]$Server='192.168.0.149')
$ErrorActionPreference='Stop'
if ($PSScriptRoot -ne 'D:\python\test\vision-learning\week08\day05') { throw 'Local project only.' }
if ($Server -notmatch '^[a-zA-Z0-9.-]+$') { throw 'Invalid server.' }
$destination='gxbb@'+$Server
$bundle=Join-Path $PSScriptRoot 'day05_package.zip'
if (-not (Test-Path -LiteralPath $bundle)) { throw 'Package missing.' }
$encodeCommand={param([string]$Text) [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Text))}
$check=@'
$ErrorActionPreference='Stop'
if ($env:COMPUTERNAME -ne 'DESKTOP-BFIO0RA') { throw 'Wrong server.' }
if (-not (Test-Path -LiteralPath 'D:\cv-server-env\Scripts\python.exe')) { throw 'Python missing.' }
if (-not (Test-Path -LiteralPath 'D:\vision-learning-server\data\LLVIP_1000')) { throw 'Data missing.' }
$running=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train_(ir|rgb|fusion)\.py' })
if ($running.Count) { $running | Select-Object ProcessId,CommandLine; throw 'Existing training; stop.' }
& D:\cv-server-env\Scripts\python.exe -B -c "import socket,torch,torchvision; from pathlib import Path; from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights as W; p=Path(torch.hub.get_dir())/'checkpoints'/W.DEFAULT.url.rsplit('/',1)[-1]; print(socket.gethostname(),torch.__version__,torchvision.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); assert p.is_file(); print('COCO cache:',p)"
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
powercfg /getactivescheme
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE
powercfg /query SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE
powercfg /query SCHEME_CURRENT SUB_BUTTONS LIDACTION
'@
Write-Host 'Enter server password only at the SSH prompt. Read-only checks first.'
& ssh $destination ('powershell.exe -NoProfile -EncodedCommand '+(& $encodeCommand $check))
if ($LASTEXITCODE -ne 0) { throw 'Remote read-only checks failed; no upload/start.' }
$remoteZip='D:/vision-learning-server/week08/day05_package_'+[Guid]::NewGuid().ToString('N')+'.zip'
& scp $bundle ($destination+':'+$remoteZip)
if ($LASTEXITCODE -ne 0) { throw 'Upload failed; no execution.' }
$installer=@'
import zipfile,hashlib,pathlib,sys
z=pathlib.Path(sys.argv[1]); root=pathlib.Path('D:/vision-learning-server/week08/day05')
with zipfile.ZipFile(z) as archive:
 items=[]
 for info in archive.infolist():
  target=(root/info.filename).resolve()
  if not target.is_relative_to(root.resolve()) or info.is_dir(): raise ValueError('Invalid archive member')
  data=archive.read(info)
  if target.exists() and (not target.is_file() or hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(data).digest()): raise FileExistsError('Different existing file: '+str(target))
  items.append((target,data))
 for target,data in items:
  if not target.exists():
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f: f.write(data)
print('Package verified; only new files installed:',root)
'@
$payload=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($installer))
$run=@"
`$ErrorActionPreference='Stop'
& D:\cv-server-env\Scripts\python.exe -B -c "import base64;exec(base64.b64decode('$payload'))" '$remoteZip'
if (`$LASTEXITCODE -ne 0) { throw 'Package conflict; no training started.' }
& D:\cv-server-env\Scripts\python.exe -B -c "import sys;sys.path.insert(0,'D:/vision-learning-server/week08/day05');from fusion_common import verify_package;verify_package();print('Runtime package hashes match')"
if (`$LASTEXITCODE -ne 0) { throw 'Package integrity failed.' }
& powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File D:\vision-learning-server\week08\day05\start_fusion.ps1 -Action start -KeepAwake
if (`$LASTEXITCODE -ne 0) { throw 'Background launch failed.' }
"@
& ssh $destination ('powershell.exe -NoProfile -EncodedCommand '+(& $encodeCommand $run))
if ($LASTEXITCODE -ne 0) { throw 'Remote launch failed. Preserve logs; no automatic retry.' }
Write-Host 'Launch returned. Inspect the exact JobDir state/log progression after disconnect before claiming persistence or success.'
