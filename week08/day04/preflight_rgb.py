"""本机只读检查RGB实验前置条件；除显式--report外不写文件，不连接服务器。"""
import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image
from rgb_common import (MANIFEST_SHA, REFERENCE_RUN, ROOT, WEIGHTS_SHA,
                        ir, output_path, protected_hashes, read_reference, rgb_loader, runtime_hashes,
                        write_new_json)
from llvip_dataset import _parse_voc_annotation


def power_status():
    result={'platform':platform.system(), 'settings_changed':False,
            'limitation':'后台和保持唤醒请求均不能保证合盖、断电、重启或强制睡眠后继续运行'}
    if os.name=='nt':
        commands={'active_scheme':['powercfg','/getactivescheme'],
                  'sleep':['powercfg','/query','SCHEME_CURRENT','SUB_SLEEP','STANDBYIDLE'],
                  'hibernate':['powercfg','/query','SCHEME_CURRENT','SUB_SLEEP','HIBERNATEIDLE'],
                  'lid':['powercfg','/query','SCHEME_CURRENT','SUB_BUTTONS','LIDACTION']}
        for key,command in commands.items():
            try:
                completed=subprocess.run(command,capture_output=True,timeout=10)
                result[key]={'exit_code':completed.returncode,
                             'stdout':completed.stdout.decode('mbcs',errors='replace'),
                             'stderr':completed.stderr.decode('mbcs',errors='replace')}
            except (OSError,subprocess.TimeoutExpired) as error:
                result[key]={'error':str(error)}
    return result


def inspect_data(root, full_decode):
    root=Path(root).resolve()
    if ir.sha(root/'manifest.csv')!=MANIFEST_SHA:
        raise ValueError('LLVIP_1000清单SHA256不符')
    result={}
    all_scenes=[]
    for split in ('train','val','test'):
        ds=rgb_loader(root,split).dataset
        expected={r.stem for r in ds.records}
        for folder in (root/'images'/split/'visible',root/'images'/split/'infrared',root/'annotations_voc'/split):
            files=[p for p in folder.iterdir() if p.is_file()]
            if len(files)!=len(ds) or {p.stem for p in files}!=expected:
                raise ValueError(f'{folder}文件集合与manifest不一致')
        scenes=sorted({r.scene_id for r in ds.records})
        if any(set(scenes)&old for old in all_scenes):
            raise ValueError('场景前缀跨train/val/test重复')
        all_scenes.append(set(scenes))
        result[split]={'pairs':len(ds),'RGB':len(ds),'IR':len(ds),'XML':len(ds),
                       'scenes':scenes,'person_boxes':0,'negative_stems':[],
                       'decode_all_images':full_decode}
        for i,r in enumerate(ds.records):
            sizes=[]
            for path in (r.visible_path,r.infrared_path):
                if path.stem!=r.stem or not path.is_file(): raise ValueError(f'配对文件不符：{path}')
                with Image.open(path) as image:
                    sizes.append(image.size)
                    if full_decode: image.load()
            target,size=_parse_voc_annotation(r.annotation_path,i,r.stem)
            if sizes!=[size,size]: raise ValueError(f'{r.stem}图像/XML尺寸不一致')
            if size!=(1280,1024): raise ValueError(f'{r.stem}原图尺寸变化：{size}')
            if len(target['boxes'])!=r.person_boxes or (len(target['boxes'])==0)!=r.is_negative:
                raise ValueError(f'{r.stem}目标数量/空标注标记与manifest不符')
            if not all(ir.torch.isfinite(value).all() for value in target.values()):
                raise ValueError(f'{r.stem}标注含NaN/Inf')
            result[split]['person_boxes']+=len(target['boxes'])
            if r.is_negative: result[split]['negative_stems'].append(r.stem)
    return result


def inspect(data_root, require_cuda=True, full_decode=True):
    protected=protected_hashes()
    day04_hashes=runtime_hashes()
    reference=read_reference()
    weights=ir.weights_info()
    weights['sha256']=ir.sha(weights['path']) if weights['exists'] else None
    result={'python_executable':sys.executable,'python_version':platform.python_version(),
            'platform':platform.platform(),'torch':ir.torch.__version__,
            'torchvision':ir.torchvision.__version__,'cuda_build':ir.torch.version.cuda,
            'cuda_available':ir.torch.cuda.is_available(),'weights':weights,
            'protected_file_count':len(protected),'protected_files_match':True,
            'day04_runtime_hashes':day04_hashes,
            'data':inspect_data(data_root,full_decode),'power':power_status(),
            'disk_free_bytes':shutil.disk_usage(ROOT).free,
            'gpu':None,'cuda_training_checked':False}
    if require_cuda:
        device=ir.device_check('cuda:0')
        if ir.torch.__version__!=reference['config']['torch'] or ir.torchvision.__version__!=reference['config']['torchvision']:
            raise ValueError('服务器Torch/TorchVision与IR实测环境不一致')
        if weights['sha256']!=WEIGHTS_SHA: raise ValueError('预训练缓存缺失或完整SHA256不同；禁止自动下载')
        if importlib.util.find_spec('matplotlib') is None:
            raise RuntimeError('matplotlib缺失，无法完成约定的曲线输出；不擅自安装')
        # 仅CPU加载完整COCO预训练结构，确认缓存可使用；不做前向或更新。
        cached_model=ir.build_model(True)
        del cached_model
        properties=ir.torch.cuda.get_device_properties(device)
        if properties.name!='NVIDIA GeForce RTX 3060 Laptop GPU':
            raise RuntimeError(f'GPU与指定服务器不符：{properties.name}')
        if result['disk_free_bytes']<3*1024**3:
            raise RuntimeError('空闲磁盘少于3GiB，无法安全保留逐轮checkpoint；不清理旧文件')
        result.update(gpu=properties.name,gpu_total_memory_bytes=properties.total_memory,
                      gpu_preflight_passed=True)
    else:
        result['limitation']='本地准备检查，不能证明服务器CUDA训练通过'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT/'data/LLVIP_1000')
    p.add_argument('--output',type=Path)
    p.add_argument('--report',type=Path)
    p.add_argument('--local',action='store_true',help='不要求本机CUDA或预训练缓存')
    p.add_argument('--headers-only',action='store_true',help='不完整解码图片，必须如实标记')
    a=p.parse_args()
    if a.report and not a.report.resolve().is_relative_to((ROOT/'week08/day04').resolve()):
        raise ValueError('检查报告只允许保存在Day04中')
    if a.report and a.report.exists(): raise FileExistsError('报告已存在，不覆盖')
    if a.output:
        target=output_path(a.output)
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise FileExistsError('输出已存在且非空；请换新的RGB run目录')
    ir.torch.set_num_threads(2)
    report=inspect(a.data_root,not a.local,not a.headers_only)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        write_new_json(a.report,report)


if __name__=='__main__': main()
