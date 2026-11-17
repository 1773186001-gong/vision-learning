import sys,json,platform,shutil,subprocess
from pathlib import Path
from fusion_common import *


def inspect(root,cuda=True):
    protected=protect()
    result=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,torchvision=ir.torchvision.__version__,
                cuda_build=torch.version.cuda,data=inspect_data(root,True),protected=protected,power=power_status())
    if cuda:
        device=ir.device_check('cuda:0')
        if torch.__version__!='2.14.1+cu126' or ir.torchvision.__version__!='0.29.1+cu126': raise RuntimeError('Unexpected server versions')
        if shutil.disk_usage(ROOT).free<5*1024**3: raise RuntimeError('Need 5GiB free disk; no cleanup')
        result['gpu']=torch.cuda.get_device_name(device)
        result['nvidia_smi']=subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=True).stdout
    return result


if __name__=='__main__': print(json.dumps(inspect(ROOT/'data/LLVIP_1000'),ensure_ascii=False,indent=2))
