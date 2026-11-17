"""Local CPU checkpoint/cache audit; never model forward, training, or network."""
import argparse,csv,json,math,time
from pathlib import Path
from collections import Counter
from PIL import Image
from fusion_common import *
from error_trace import trace


def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))


def review(out,job):
    verify_package(); protected=protect()
    c=read(out/'config.json'); done=read(out/'completed.json')
    assert c['modality']=='fusion_concat' and c['device']=='cuda:0'
    assert done['epochs']==10 and c['gpu']=='NVIDIA GeForce RTX 3060 Laptop GPU'
    rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader((out/'training_log.csv').open(encoding='utf-8-sig'))]
    assert [r['epoch'] for r in rows]==list(range(1,11))
    for row in rows:
        assert all(math.isfinite(v) for v in row.values()) and row['lr']==.001
        assert abs(row['total_loss']-sum(row[k] for k in ir.LOSS_KEYS))<1e-6
        val=read(out/f'val_epoch_{int(row["epoch"]):02d}.json')
        assert all(row[k]==val[k] for k in ('precision','recall','AP50','mAP50_95'))
    best=int(max(rows,key=lambda r:r['mAP50_95'])['epoch']); assert best==done['best_epoch']
    checkpoints={}
    for name,epoch in [('best.pth',best),('last.pth',10)]:
        model,ck=restore(out/name,torch.device('cpu'))
        assert ck['epoch']==epoch and ck['config']==c and ck['history']==rows[:epoch]
        assert ck['optimizer']['state'] and all(k in ck for k in ('torch_rng','python_rng','cuda_rng','loader_rng'))
        for p in model.parameters(): assert torch.isfinite(p).all()
        checkpoints[name]=dict(epoch=epoch,bytes=(out/name).stat().st_size,sha256=ir.sha(out/name),strict_load=True)
        del model,ck
    assert read(out/'test_started.json')['checkpoint_sha256']==checkpoints['best.pth']['sha256']
    cache=torch.load(out/'test_predictions.pth',map_location='cpu',weights_only=False)
    assert cache['modality']=='fusion_concat' and len(set(cache['stems']))==len(cache['stems'])==150
    assert len(cache['truths'])==len(cache['predictions'])==150 and sum(map(len,cache['truths']))==348
    metric=ir.metrics(cache['predictions'],cache['truths'])
    assert metric==read(out/'test_metrics.json')==done['test']
    compared=read(out/'three_model_comparison.json'); recomputed={}; cases={}
    details=read(out/'fusion_error_details.json')
    assert details['stems']==cache['stems'] and details['truths']==[t.tolist() for t in cache['truths']]
    for mode in ('IR','RGB','Fusion'):
        saved=cache if mode=='Fusion' else torch.load(Path(__file__).parent/'references'/mode/'test_predictions.pth',map_location='cpu',weights_only=False)
        assert saved['stems']==cache['stems']
        assert all(torch.equal(a,b) for a,b in zip(saved['truths'],cache['truths']))
        metrics=ir.metrics(saved['predictions'],saved['truths']); assert metrics==compared[mode]['metrics']
        types=Counter(); per_image=[]
        for i,(p,g) in enumerate(zip(saved['predictions'],saved['truths'])):
            assert p['boxes'].dtype==torch.float32 and p['boxes'].shape[1:]==(4,)
            assert torch.isfinite(p['boxes']).all() and (p['boxes']>=0).all() and (p['boxes']<=640).all()
            assert (p['boxes'][:,2:]>p['boxes'][:,:2]).all() and (p['labels']==1).all()
            rows_p=trace(p,g); types.update(r['fp_type'] for r in rows_p if r['status']=='FP')
            if mode=='Fusion': assert rows_p==details['predictions'][i]
            counts=dict(TP=sum(r['status']=='TP' for r in rows_p),FP=sum(r['status']=='FP' for r in rows_p),FN=len(g)-sum(r['status']=='TP' for r in rows_p))
            per_image.append(counts)
            if saved['stems'][i] in ('190001','190040','190474','190435','190593'):
                cases.setdefault(saved['stems'][i],{})[mode]=dict(**counts,FP_types=dict(Counter(r['fp_type'] for r in rows_p if r['status']=='FP')))
        for cat,v in compared[mode]['fp_categories'].items(): assert v==dict(count=types[cat],fraction=types[cat]/metrics['FP'])
        assert sum(types.values())==metrics['FP']
        recomputed[mode]=dict(metrics=metrics,FP_types=dict(types))
        if mode=='Fusion':
            selections=read(out/'prediction_examples/selection.json')
            for cat,stems in selections['selected'].items():
                assert selections['shortfall'][cat]==3-len(stems)
                for stem in stems:
                    count=per_image[cache['stems'].index(stem)]
                    assert (count['TP']>0 and count['FP']==count['FN']==0) if cat=='success' else (count['FN']>0 if cat=='miss' else count['FP']>0)
    state=read(job/'state.json'); assert state['status']=='SUCCEEDED' and state['exit_code']==0
    exits=list(out.glob('exit_status_*.json')); assert len(exits)==1 and read(exits[0])['exit_code']==0
    assert not list(out.glob('failure_*.json'))
    smoke=read(out/'smoke_test.json'); assert smoke['all_state_restored_exactly'] and smoke['eval_forward_finite']
    pngs=list(out.rglob('*.png'))
    for p in pngs:
        with Image.open(p) as im: im.verify()
    old=read(out/'protected_after.json')
    for name,digest in old['checked'].items(): assert protected['checked'][name]==digest
    audit=dict(status='passed',review_device='cpu',training=False,model_forward=False,epochs=10,best_epoch=best,
               checkpoints=checkpoints,test_cache_images=150,GT_boxes=348,comparison=recomputed,cases=cases,
               decoded_pngs=len(pngs),smoke=smoke,epoch_seconds_sum=sum(r['epoch_seconds'] for r in rows),
               peak_allocated_bytes=int(max(r['peak_allocated_bytes'] for r in rows)),peak_reserved_bytes=int(max(r['peak_reserved_bytes'] for r in rows)),
               state=state,exit_status=read(exits[0]),historical_local_hashes_verified=len(protected['checked']),
               remote_historical_hashes_verified=len(old['checked']),resume_interrupted_training_tested=False,
               limitation='Local artifact hashes are not an independent full remote per-file SHA comparison')
    folder=Path(__file__).parent/'checks'/f'final_review_{time.time_ns()}'; folder.mkdir()
    save(folder/'review.json',audit)
    save(folder/'artifact_sha256.json',{str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=ir.sha(p)) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(audit,ensure_ascii=False,indent=2)); print('AUDIT_FOLDER',folder)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--job',type=Path,required=True)
    a=p.parse_args(); torch.set_num_threads(2); review(a.output.resolve(),a.job.resolve())
