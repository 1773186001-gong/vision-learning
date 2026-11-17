"""只读检查环境、子集、代码和预训练缓存；无下载/训练。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights as W

root=Path(__file__).resolve().parents[2]
weights=Path(torch.hub.get_dir())/'checkpoints'/W.DEFAULT.url.rsplit('/',1)[-1]
files=['week08/day01/llvip_dataset.py','week08/day02/paired_letterbox.py',
       'week08/day02/paired_horizontal_flip.py']
files += ['week08/day03/'+x for x in ('ir_common.py','train_ir.py','evaluate_ir.py','predict_ir.py')]
report={'python':sys.executable,'torch':torch.__version__,'torchvision':torchvision.__version__,
        'cuda':torch.cuda.is_available(),'cuda_build':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'weights_url':W.DEFAULT.url,'weights_path':str(weights),'weights_exists':weights.is_file(),
        'dependencies':{x:importlib.util.find_spec(x) is not None for x in ('matplotlib','PIL','torchmetrics','pycocotools')},
        'files':{f:{'exists':(root/f).is_file(),
                    'sha256':hashlib.sha256((root/f).read_bytes()).hexdigest() if (root/f).is_file() else None} for f in files},
        'manifest_exists':(root/'data/LLVIP_1000/manifest.csv').is_file()}
print(json.dumps(report,ensure_ascii=False,indent=2))
