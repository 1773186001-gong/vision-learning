import sys
from pathlib import Path
import torch
from fusion_model import ROOT, ir, build_model, restore
sys.path.insert(0,str(ROOT/'week08/day04'))
from preflight_rgb import inspect_data, power_status
from train_rgb import Transcript, temporary_awake
from train_ir import atomic_save, cuda_sync


def loader(root,split,shuffle=False):
    ds=ir.LLVIPDataset(root,split,modality='paired',infrared_channels=3,
                      transforms=ir.PairedLetterboxAndFlip(split))
    if len(ds)!={'train':700,'val':150,'test':150}[split]: raise ValueError('Dataset count mismatch')
    return ir.DataLoader(ds,batch_size=1,shuffle=shuffle,num_workers=0,
                        collate_fn=ir.detection_collate_fn,generator=torch.Generator().manual_seed(42))


def move(images,targets,device):
    return [{k:v.to(device) for k,v in im.items()} for im in images],[{k:v.to(device) for k,v in t.items()} for t in targets]


def update(model,opt,images,targets,device):
    images,targets=move(images,targets,device); opt.zero_grad(set_to_none=True)
    losses=model(images,targets)
    if set(losses)!=set(ir.LOSS_KEYS): raise ValueError('Four loss fields required')
    total=sum(losses.values())
    if not torch.isfinite(total) or not all(torch.isfinite(v) for v in losses.values()): raise FloatingPointError('Invalid loss')
    total.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None): raise FloatingPointError('Invalid gradient')
    opt.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError('Invalid parameter')
    return {'total_loss':total.item(),**{k:v.item() for k,v in losses.items()}}


@torch.inference_mode()
def infer(model,data,device):
    model.eval(); predictions=[]; truths=[]; stems=[]
    for i,(images,targets) in enumerate(data):
        moved,_=move(images,targets,device)
        for p,t in zip(model(moved),targets):
            p={k:v.detach().cpu() for k,v in p.items()}
            if not all(torch.isfinite(v).all() for v in p.values()): raise FloatingPointError('Invalid prediction')
            predictions.append(p); truths.append(t['boxes'].cpu())
        stems.append(data.dataset.get_record(i).stem)
    return predictions,truths,stems


def protect():
    spec=__import__('json').loads((Path(__file__).parent/'protected_inputs.json').read_text(encoding='utf-8'))
    checked={}; missing=[]
    server=ROOT==Path('D:/vision-learning-server')
    for name,digest in spec.items():
        remote=name.replace('rgb_baseline_retrieved_20261030_213029_fa237fce','rgb_baseline_run01') if server else name
        path=ROOT/remote
        if server and not path.is_file(): missing.append(name); continue
        if ir.sha(path)!=digest: raise ValueError('Historical file changed: '+name)
        checked[name]=digest
    required=['week08/day01/llvip_dataset.py','week08/day02/paired_letterbox.py','week08/day02/paired_horizontal_flip.py','week08/day03/ir_common.py']
    for mode,folder in [('IR','week08/day03/outputs/ir_baseline_run01'),('RGB','week08/day04/outputs/rgb_baseline_retrieved_20261030_213029_fa237fce')]:
        required.extend(folder+'/'+n for n in ('best.pth','last.pth','test_predictions.pth'))
    if any(n not in checked for n in required): raise ValueError('Required historical inputs missing')
    return dict(checked=checked,missing_historical_files_not_present_on_server=missing)


def save(path,obj):
    with Path(path).open('x',encoding='utf-8') as f: __import__('json').dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def verify_package():
    folder=Path(__file__).parent
    spec=__import__('json').loads((folder/'package_manifest.json').read_text(encoding='utf-8'))
    for name,digest in spec.items():
        if ir.sha(folder/name)!=digest: raise ValueError('Day05 package changed: '+name)
    return spec
