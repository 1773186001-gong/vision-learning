"""RGB完整基线：检查→共享冒烟→10轮训练→val选模→一次test→缓存出图与IR对照。"""
import argparse
import contextlib
import ctypes
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from rgb_common import (MANIFEST_SHA, REFERENCE_RUN, SETTINGS, WEIGHTS_SHA,
                        atomic_save, cuda_sync, ir, output_path, protected_hashes,
                        read_reference, rgb_loader, runtime_hashes, smoke, update, write_new_json)


class Transcript:
    def __init__(self, stream, file): self.stream,self.file=stream,file
    def write(self,text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)
    def flush(self): self.stream.flush(); self.file.flush()


@contextlib.contextmanager
def temporary_awake(enabled):
    # 仅明确传--keep-awake时使用线程级请求；不改电源计划，不阻止合盖/强制睡眠。
    if enabled:
        if os.name!='nt': raise RuntimeError('临时保持唤醒只支持Windows')
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise OSError('临时保持唤醒请求失败；停止，不修改电源配置')
    try:
        yield
    finally:
        if enabled:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def build_config(args, protected):
    device=ir.device_check(args.device)
    if device.type!='cuda': raise RuntimeError('正式RGB训练必须使用实际RTX 3060 CUDA')
    weights=ir.weights_info()
    weights['sha256']=ir.sha(weights['path']) if weights['exists'] else None
    if weights['sha256']!=WEIGHTS_SHA: raise ValueError('官方预训练缓存缺失或哈希不同，停止而不下载')
    return {**SETTINGS,'modality':'visible','model_input':'visible only; no IR features',
            'data_root':str(args.data_root.resolve()),'manifest_sha256':ir.sha(args.data_root/'manifest.csv'),
            'weights':weights,'torch':ir.torch.__version__,'torchvision':ir.torchvision.__version__,
            'python':platform.python_version(),'python_executable':sys.executable,
            'cuda_build':ir.torch.version.cuda,'device':str(device),'gpu':ir.torch.cuda.get_device_name(device),
            'cpu_threads':2,'train_flip_probability':0.5,'val_test_flip_probability':0.0,
            'score_floor':0.001,'max_detections':100,'pr_score_threshold':0.5,'pr_iou':0.5,
            'cudnn_benchmark':False,'cudnn_deterministic':True,
            'deterministic_algorithms_required':False,
            'tf32_matmul':ir.torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn':ir.torch.backends.cudnn.allow_tf32,
            'keep_awake_requested':args.keep_awake,'protected_hashes':protected,
            'day04_runtime_hashes':runtime_hashes(),
            'command':subprocess.list2cmdline([sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:]])}


def run(args, output, run_begin):
    from preflight_rgb import inspect
    from evaluate_rgb import final_test
    from predict_rgb import render_saved
    from report_rgb import write_report
    ir.torch.set_num_threads(2)
    # 所有检查先完成，之后才固定种子和构造正式初始模型，避免检查消耗RNG。
    preflight=inspect(args.data_root,require_cuda=True,full_decode=True)
    write_new_json(output/f'preflight_{time.time_ns()}.json' if args.resume else output/'preflight.json',preflight)
    protected=protected_hashes()
    reference=read_reference()
    ir.seed_all(42)
    config=build_config(args,protected)
    if config['manifest_sha256']!=MANIFEST_SHA: raise ValueError('数据清单改变')
    for key in (*SETTINGS,'torch','torchvision','manifest_sha256'):
        if config[key]!=reference['config'][key]: raise ValueError(f'与IR对照不同：{key}')
    device=ir.device_check(args.device)
    train=rgb_loader(args.data_root,'train',42,True)
    val=rgb_loader(args.data_root,'val')
    history,start,best=[],1,-1.0
    if args.resume:
        if args.resume.resolve()!=output/'last.pth': raise ValueError('仅允许同run的last.pth恢复')
        model,saved=ir.checkpoint_model(args.resume,device)
        # 启动命令允许变化；科学配置、来源和运行环境不可改变。
        for key,value in config.items():
            if key!='command' and saved['config'].get(key)!=value:
                raise ValueError(f'续训配置不一致：{key}')
        config=saved['config']
        history,start,best=saved['history'],saved['epoch']+1,saved['best_score']
        if saved['epoch']>=10 or (output/'test_started.json').exists():
            raise ValueError('已完成训练或启动过test，不自动重复评估')
    else:
        # 直接从官方COCO预训练构造，绝不读取IR best/last来初始化RGB。
        model=ir.build_model(True).to(device)
        write_new_json(output/'config.json',config)
        head_before=model.roi_heads.box_predictor.cls_score.weight.detach().cpu().clone()
        ir.torch.cuda.reset_peak_memory_stats(device)
        smoke_result=smoke(model,train,device)
        cuda_sync(device)
        ir.torch.testing.assert_close(model.roi_heads.box_predictor.cls_score.weight.detach().cpu(),
                                      head_before,rtol=0,atol=0)
        smoke_result['peak_reserved_bytes']=ir.torch.cuda.max_memory_reserved(device)
        smoke_result['modality']='visible'
        write_new_json(output/'smoke_test.json',smoke_result)
        print('SMOKE PASSED',json.dumps(smoke_result),flush=True)
        ir.seed_all(42)
        write_new_json(output/'initialization.json',{
            'source':'official COCO detector, fresh 2-class head; NOT IR checkpoint',
            'weights_sha256':config['weights']['sha256'],'smoke_weights_restored':True,
            'head_restored_exactly':True,'seed_reset_to':42,
            'head_first_values':head_before.flatten()[:8].tolist()})
        del head_before
    optimizer=ir.torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                 lr=0.001,momentum=0.9,weight_decay=0.0005)
    if args.resume:
        optimizer.load_state_dict(saved['optimizer'])
        ir.torch.set_rng_state(saved['torch_rng'])
        random.setstate(saved['python_rng'])
        train.generator.set_state(saved['loader_rng'])
        ir.torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if (output/'training_log.csv').exists():
            shutil.copy2(output/'training_log.csv',output/f'log_before_resume_{time.time_ns()}.csv')
    for epoch in range(start,11):
        if (output/f'epoch_{epoch:02d}.pth').exists(): raise FileExistsError('已有同轮快照，不覆盖')
        cuda_sync(device)
        begin=time.perf_counter()
        ir.torch.cuda.reset_peak_memory_stats(device)
        model.train()
        totals={k:0.0 for k in ('total_loss',)+ir.LOSS_KEYS}
        for batch,(images,targets) in enumerate(train,1):
            values=update(model,optimizer,images,targets,device)
            for key in totals: totals[key]+=values[key]
            if batch%100==0:
                print(f'RGB epoch={epoch} batch={batch}/700 loss={values["total_loss"]:.6f}',flush=True)
                ir.save_json(output/'progress.json',{'phase':'train','epoch':epoch,'batch':batch,'pid':os.getpid()})
        predictions,truths,_=ir.infer(model,val,device)
        scores=ir.metrics(predictions,truths)
        cuda_sync(device)
        row={'epoch':epoch,**{k:v/len(train) for k,v in totals.items()},
             'lr':optimizer.param_groups[0]['lr'],'epoch_seconds':time.perf_counter()-begin,
             'peak_allocated_bytes':ir.torch.cuda.max_memory_allocated(device),
             'peak_reserved_bytes':ir.torch.cuda.max_memory_reserved(device),
             **{k:scores[k] for k in ('precision','recall','AP50','mAP50_95')}}
        improved=scores['mAP50_95']>best
        if improved: best=scores['mAP50_95']
        history.append(row)
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                    'trainable_names':[n for n,p in model.named_parameters() if p.requires_grad],
                    'history':history,'best_score':best,'config':config,
                    'torch_rng':ir.torch.get_rng_state(),'python_rng':random.getstate(),
                    'cuda_rng':ir.torch.cuda.get_rng_state_all(),'loader_rng':train.generator.get_state()}
        paths=[output/f'epoch_{epoch:02d}.pth',output/'last.pth']
        if improved: paths.append(output/'best.pth')
        for path in paths:
            if path.with_suffix('.tmp').exists(): raise FileExistsError(f'存在未完成保存：{path.with_suffix(".tmp")}')
            atomic_save(checkpoint,path)
        ir.write_history(history,output)
        write_new_json(output/f'val_epoch_{epoch:02d}.json',scores)
        print(json.dumps(row),flush=True)
    del checkpoint,optimizer,model
    best_model,saved=ir.checkpoint_model(output/'best.pth',device)
    write_new_json(output/'test_started.json',{'checkpoint_sha256':ir.sha(output/'best.pth'),
                                             'best_epoch':saved['epoch'],'modality':'visible'})
    scores=final_test(best_model,args.data_root,device,output)
    del best_model
    selection=render_saved(args.data_root,output/'test_predictions.pth',output/'prediction_examples')
    # 只读已存缓存复算；不再运行模型或选择新的测试阈值。
    cache=ir.torch.load(output/'test_predictions.pth',map_location='cpu',weights_only=False)
    if scores!=ir.metrics(cache['predictions'],cache['truths']): raise ValueError('缓存指标不一致')
    write_new_json(output/'result_checks.json',{'cache_images':len(cache['stems']),
                   'GT_boxes':sum(len(t) for t in cache['truths']),'cached_metrics_match':True,
                   'test_inference_runs':1,'extra_model_inference_for_visualization':False})
    write_new_json(output/'protected_after.json',protected_hashes())
    timing={'epoch_seconds_sum':sum(r['epoch_seconds'] for r in history),
            'run_seconds_before_report':time.perf_counter()-run_begin,
            'max_epoch_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in history),
            'max_epoch_peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in history)}
    write_report(output,config,history,saved['epoch'],scores,selection,timing)
    write_new_json(output/'completed.json',{'modality':'visible','epochs':10,
                   'best_epoch':saved['epoch'],'test':scores,'timing':timing})
    print('COMPLETED RGB',json.dumps(scores),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cuda:0'],default='cuda:0')
    p.add_argument('--resume',type=Path)
    p.add_argument('--keep-awake',action='store_true',help='需用户授权；临时请求保持系统唤醒，不改变电源设置')
    args=p.parse_args()
    output=output_path(args.output)
    if args.resume:
        if not output.is_dir() or (output/'completed.json').exists(): raise ValueError('不能恢复已完成或不存在的run')
    elif output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('输出非空，禁止覆盖；请选择新的RGB run目录')
    output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    exit_code=1
    with (output/f'console_{time.time_ns()}.txt').open('x',encoding='utf-8') as log:
        with contextlib.redirect_stdout(Transcript(sys.stdout,log)), contextlib.redirect_stderr(Transcript(sys.stderr,log)):
            try:
                with temporary_awake(args.keep_awake): run(args,output,start)
                exit_code=0
            except BaseException as error:
                traceback.print_exc()
                write_new_json(output/f'failure_{time.time_ns()}.json',{
                    'type':type(error).__name__,'message':str(error),'automatic_retries':0,
                    'settings_changed_on_failure':False,
                    'recovery':'保留所有记录；仅在test未启动时经人工核对从最后完整epoch恢复；不得自动重试'})
            finally:
                status={'exit_code':exit_code,'pid':os.getpid(),'seconds':time.perf_counter()-start}
                # 恢复也保留原来的退出记录。
                name=f'exit_status_resume_{time.time_ns()}.json' if args.resume else 'exit_status.json'
                write_new_json(output/name,status)
                print('EXIT_STATUS',json.dumps(status),flush=True)
    return exit_code


if __name__=='__main__': raise SystemExit(main())
