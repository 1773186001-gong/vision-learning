"""取回后只读核对RGB产物，严格加载checkpoint和复算缓存，不进行训练或模型推理。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from PIL import Image
from rgb_common import (comparison, ir, output_path, protected_hashes, read_json,
                        read_reference, runtime_hashes, write_new_json)


def review(output, job):
    config=read_json(output/'config.json')
    assert config['modality']=='visible' and config['device']=='cuda:0'
    assert config['gpu']=='NVIDIA GeForce RTX 3060 Laptop GPU'
    assert config['day04_runtime_hashes']==runtime_hashes()
    assert config['protected_hashes']==protected_hashes()==read_json(output/'protected_after.json')
    with (output/'training_log.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    assert [r['epoch'] for r in rows]==list(range(1,11))
    assert all(math.isfinite(v) for r in rows for v in r.values())
    for row in rows:
        assert row['lr']==0.001
        assert abs(row['total_loss']-sum(row[k] for k in ir.LOSS_KEYS))<1e-6
        scores=read_json(output/f'val_epoch_{int(row["epoch"]):02d}.json')
        assert all(row[k]==scores[k] for k in ('precision','recall','AP50','mAP50_95'))
    best_epoch=int(max(rows,key=lambda row:row['mAP50_95'])['epoch'])
    checkpoints={}
    for name,epoch in (('best.pth',best_epoch),('last.pth',10)):
        model,ck=ir.checkpoint_model(output/name,ir.torch.device('cpu'))
        assert ck['epoch']==epoch and ck['config']==config
        assert len(ck['history'])==epoch
        assert all(all(actual[k]==value for k,value in wanted.items()) for actual,wanted in zip(ck['history'],rows))
        assert ck['optimizer']['state']
        assert all(k in ck for k in ('python_rng','torch_rng','cuda_rng','loader_rng'))
        checkpoints[name]={'epoch':epoch,'bytes':(output/name).stat().st_size,
                           'sha256':ir.sha(output/name),'strict_load':True,'history_matches_csv':True}
        del model,ck
    assert read_json(output/'test_started.json')['checkpoint_sha256']==checkpoints['best.pth']['sha256']
    cache=ir.torch.load(output/'test_predictions.pth',map_location='cpu',weights_only=False)
    assert cache['modality']=='visible'
    assert len(cache['stems'])==len(cache['predictions'])==len(cache['truths'])==150
    assert len(set(cache['stems']))==150
    for prediction in cache['predictions']:
        boxes=prediction['boxes']
        assert boxes.dtype==ir.torch.float32 and boxes.ndim==2 and boxes.shape[1]==4
        assert ir.torch.isfinite(boxes).all() and (boxes>=0).all() and (boxes<=640).all()
        assert (boxes[:,2:]>boxes[:,:2]).all()
        assert (prediction['labels']==1).all()
        assert ir.torch.isfinite(prediction['scores']).all()
    scores=ir.metrics(cache['predictions'],cache['truths'])
    completed=read_json(output/'completed.json')
    assert scores==read_json(output/'test_metrics.json')==completed['test']
    assert completed['best_epoch']==best_epoch and completed['epochs']==10
    compared=comparison(config,rows,best_epoch,scores,read_reference())
    assert compared==read_json(output/'rgb_vs_ir.json')
    state=read_json(job/'state.json')
    assert state['status']=='SUCCEEDED' and state['exit_code']==0
    exit_status=read_json(output/'exit_status.json')
    assert exit_status['exit_code']==0
    pngs=[output/'loss_curve.png',*sorted((output/'prediction_examples').glob('*.png'))]
    for path in pngs:
        with Image.open(path) as image: image.verify()
    assert not list(output.glob('failure_*.json'))
    result={'status':'passed','review_device':'cpu','training_or_model_inference_during_review':False,
            'checkpoints':checkpoints,'csv_epochs':10,'best_epoch':best_epoch,
            'cache_images':150,'GT_boxes':sum(len(t) for t in cache['truths']),
            'test':scores,'comparison_matches':True,'protected_files_unchanged':40,
            'epoch_seconds_sum':sum(r['epoch_seconds'] for r in rows),
            'peak_allocated_bytes':int(max(r['peak_allocated_bytes'] for r in rows)),
            'peak_reserved_bytes':int(max(r['peak_reserved_bytes'] for r in rows)),
            'decoded_PNG_count':len(pngs),'monitor_state':state,'exit_status':exit_status,
            'resume_after_interruption_tested':False,
            'note':'本地SHA用于保存后核对，不等于全部文件与服务器另做逐字节远程哈希对照'}
    write_new_json(output/'local_result_review.json',result)
    files=sorted(p for p in output.rglob('*') if p.is_file())
    write_new_json(output/'artifact_sha256.json',{
        str(p.relative_to(output)):{'bytes':p.stat().st_size,'sha256':ir.sha(p)} for p in files})
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--job',type=Path,required=True)
    a=p.parse_args()
    review(output_path(a.output),a.job)
