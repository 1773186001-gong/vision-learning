"""Preflight, strict paired smoke, 10 epochs, val selection, single test, offline report."""
import argparse,contextlib,json,random,time,traceback,os,shutil
from pathlib import Path
from fusion_common import *
from preflight_fusion import inspect
from evaluate_fusion import evaluate,final_test


def optimizer(model):
    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001,momentum=.9,weight_decay=.0005)


def smoke(model,train,device):
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    opt=optimizer(model); rows=[]; torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for negative in (False,True):
        i=next(i for i,r in enumerate(train.dataset.records) if r.is_negative==negative)
        images,target=train.dataset[i]
        before=model.detector.roi_heads.box_predictor.cls_score.weight.detach().clone()
        row=update(model,opt,[images],[target],device)
        if torch.equal(before,model.detector.roi_heads.box_predictor.cls_score.weight): raise RuntimeError('No smoke update')
        rows.append(dict(stem=train.dataset.get_record(i).stem,**row))
    model.eval()
    with torch.no_grad():
        moved,_=move([images],[target],device); prediction=model(moved)
        if not all(torch.isfinite(v).all() for p in prediction for v in p.values()): raise FloatingPointError('Invalid eval smoke')
    cuda_sync(device)
    result=dict(samples=rows,eval_forward_finite=True,branch_shapes=model.detector.backbone.last_shapes,
                peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device))
    model.load_state_dict(initial,strict=True)
    assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items())
    result['all_state_restored_exactly']=True
    del opt,initial,prediction,moved,before
    model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    return result


def run(a,out):
    torch.set_num_threads(2); verify_package(); protected=protect()
    save(out/f'preflight_{time.time_ns()}.json',inspect(a.data_root))
    # Real COCO structure/data/feature check is mandatory before CUDA smoke/training.
    from check_day05 import check
    save(out/f'structure_{time.time_ns()}.json',check(a.data_root,Path(ir.weights_info()['path'])))
    ir.seed_all(42); device=ir.device_check('cuda:0'); train=loader(a.data_root,'train',True)
    config=dict(modality='fusion_concat',epochs=10,batch_size=1,num_workers=0,seed=42,lr=.001,momentum=.9,weight_decay=.0005,
                input_size=640,amp=False,scheduler=None,torch=torch.__version__,torchvision=ir.torchvision.__version__,
                python=sys.version,device=str(device),gpu=torch.cuda.get_device_name(device),manifest_sha256=ir.sha(a.data_root/'manifest.csv'),
                weights_sha256=ir.sha(ir.weights_info()['path']),architecture='independent RGB FPN + IR FPN -> concat/1x1 -> shared RPN/RoI',
                command=__import__('subprocess').list2cmdline([sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:]]))
    history=[]; best=-1.; start=1
    if a.resume:
        if a.resume.resolve()!=out/'last.pth' or (out/'test_started.json').exists(): raise ValueError('Unsafe resume')
        model,ck=restore(a.resume,device)
        for key in config:
            if key!='command' and config[key]!=ck['config'][key]: raise ValueError('Resume config mismatch: '+key)
        config=ck['config']; history=ck['history']; start=ck['epoch']+1; best=ck['best_score']
        if start>10: raise RuntimeError('Completed training cannot resume')
    else:
        model=build_model(True).to(device); save(out/'config.json',config)
        save(out/'model_metadata.json',dict(architecture=model.architecture,
            total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            parameters={n:dict(shape=list(p.shape),trainable=p.requires_grad,source='new average identity' if '.fusion.' in n else ('new person predictor' if '.box_predictor.' in n else 'official COCO')) for n,p in model.named_parameters()}))
        save(out/'smoke_test.json',smoke(model,train,device)); print('SMOKE PASSED',flush=True); ir.seed_all(42)
    opt=optimizer(model)
    if a.resume:
        opt.load_state_dict(ck['optimizer']); torch.set_rng_state(ck['torch_rng']); random.setstate(ck['python_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng']); train.generator.set_state(ck['loader_rng'])
        if (out/'training_log.csv').exists(): shutil.copy2(out/'training_log.csv',out/f'csv_before_resume_{time.time_ns()}.csv')
        del ck
    for ep in range(start,11):
        if (out/f'epoch_{ep:02d}.pth').exists() or list(out.glob('*.tmp')): raise FileExistsError('Ambiguous interrupted epoch; preserve files')
        cuda_sync(device); begin=time.perf_counter(); torch.cuda.reset_peak_memory_stats(device); model.train()
        sums={k:0. for k in ('total_loss',)+ir.LOSS_KEYS}
        for i,(images,targets) in enumerate(train,1):
            values=update(model,opt,images,targets,device)
            for k in sums: sums[k]+=values[k]
            if i%50==0:
                print(f'fusion epoch={ep} batch={i}/700 loss={values["total_loss"]:.6f}',flush=True)
                ir.save_json(out/'progress.json',dict(epoch=ep,batch=i,pid=os.getpid()))
        scores=evaluate(model,a.data_root,device); cuda_sync(device)
        row=dict(epoch=ep,**{k:v/len(train) for k,v in sums.items()},lr=opt.param_groups[0]['lr'],epoch_seconds=time.perf_counter()-begin,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                 **{k:scores[k] for k in ('precision','recall','AP50','mAP50_95')})
        improved=scores['mAP50_95']>best; best=max(best,scores['mAP50_95']); history.append(row)
        ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=ep,history=history,best_score=best,config=config,
                trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=train.generator.get_state())
        for path in [out/f'epoch_{ep:02d}.pth',out/'last.pth']+([out/'best.pth'] if improved else []): atomic_save(ck,path)
        save(out/f'val_epoch_{ep:02d}.json',scores); ir.write_history(history,out)
        # Separate validation-only curve, no new inference.
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(); ax.plot([r['epoch'] for r in history],[r['mAP50_95'] for r in history],label='val mAP50:95'); ax.legend(); fig.savefig(out/'mAP_curve.png',dpi=150); plt.close(fig)
        print(json.dumps(row),flush=True)
    del ck,opt,model; torch.cuda.empty_cache()
    model,ck=restore(out/'best.pth',device)
    save(out/'test_started.json',dict(best_epoch=ck['epoch'],checkpoint_sha256=ir.sha(out/'best.pth')))
    scores=final_test(model,a.data_root,device,out); best_epoch=ck['epoch']; del model,ck; torch.cuda.empty_cache()
    from predict_fusion import render_and_report
    render_and_report(a.data_root,out,history,best_epoch,scores)
    assert protect()==protected
    save(out/'protected_after.json',protected)
    save(out/'completed.json',dict(modality='fusion_concat',epochs=10,best_epoch=best_epoch,test=scores))
    print('COMPLETED FUSION',json.dumps(scores),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cuda:0'],default='cuda:0'); p.add_argument('--resume',type=Path); p.add_argument('--keep-awake',action='store_true'); a=p.parse_args()
    out=a.output.resolve(); allowed=Path(__file__).parent.resolve()/'outputs'
    if not out.is_relative_to(allowed) or out==allowed: raise ValueError('Day05 independent output required')
    if a.resume:
        if not out.is_dir() or (out/'completed.json').exists(): raise ValueError('Invalid resume output')
    else:
        original=out; i=2
        while out.exists(): out=original.with_name(original.name+f'_{i:02d}'); i+=1
    out.mkdir(parents=True,exist_ok=True); code=1; begin=time.perf_counter()
    with (out/f'console_{time.time_ns()}.log').open('x',encoding='utf-8') as log:
        with contextlib.redirect_stdout(Transcript(sys.stdout,log)),contextlib.redirect_stderr(Transcript(sys.stderr,log)):
            try:
                with temporary_awake(a.keep_awake): run(a,out)
                code=0
            except BaseException as e:
                traceback.print_exc(); save(out/f'failure_{time.time_ns()}.json',dict(error=repr(e),automatic_retry=False,settings_changed=False))
            finally: save(out/f'exit_status_{time.time_ns()}.json',dict(exit_code=code,seconds=time.perf_counter()-begin,pid=os.getpid()))
    return code


if __name__=='__main__': raise SystemExit(main())
