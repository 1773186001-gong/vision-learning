import argparse
from pathlib import Path
from fusion_common import *


def evaluate(model,root,device):
    preds,gts,stems=infer(model,loader(root,'val'),device)
    return ir.metrics(preds,gts)


def final_test(model,root,device,out):
    if not (out/'test_started.json').exists() or (out/'test_predictions.pth').exists(): raise RuntimeError('Test guard')
    preds,gts,stems=infer(model,loader(root,'test'),device)
    cache=dict(modality='fusion_concat',predictions=preds,truths=gts,stems=stems)
    with (out/'test_predictions.pth').open('xb') as f: torch.save(cache,f)
    scores=ir.metrics(preds,gts); save(out/'test_metrics.json',scores)
    return scores


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(Path(__file__).parent.resolve()/'outputs') or a.output.exists(): raise ValueError('New Day05 output required')
    m,_=restore(a.checkpoint,ir.device_check('cuda:0')); save(a.output,evaluate(m,a.data_root,torch.device('cuda:0')))
