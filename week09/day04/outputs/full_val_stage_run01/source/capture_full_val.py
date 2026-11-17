"""Authorized fixed 144-image capture plus 6-image reuse; no tuning/training/test."""
import argparse
from contextlib import ExitStack,redirect_stdout,redirect_stderr
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'week09/day03'))
import capture_single_val as original
import torch
import torchvision
from agnostic_capture import AgnosticCapture,LOCK,exact
from capture_ledger import Ledger,UnknownExecution,json_write,sha
from offline_full_gt import load_json,read_csv,save_json,write_csv,reuse_preflight,Tee,D1
from full_val_analysis import compare_item,merge_full

def remap(path):
    normalized=str(path).replace('\\','/')
    for anchor in ('week09/','week08/','data/'):
        if anchor in normalized:return ROOT/normalized[normalized.index(anchor):]
    raise ValueError('Cannot remap source path')

def preflight():
    baseline,_,rpn_path=original.preflight()  # checks trusted original environment, restore sources and best SHA
    _,day1,_,val,_,_=reuse_preflight()
    expected=load_json(HERE/'checks/full_val_expected_inputs_run01.json')
    for relative,h in expected['sha256'].items():
        if sha(ROOT/relative)!=h:raise RuntimeError('Input/source changed: '+relative)
    proof=load_json(HERE/'outputs/six_image_all_gt_run02/reuse_evidence.json')
    for c in proof['captures']:
        if sha(remap(c['raw_path']))!=c['raw_sha256']:raise RuntimeError('Reused capture differs')
        if c['original_config']!=day1['models'][c['model']]['rpn_config']:raise RuntimeError('Reuse config differs')
        for p,h in c['inputs_sha256'].items():
            if sha(remap(p))!=h:raise RuntimeError('Reuse input differs')
    reused={c['stem'] for c in proof['captures']}
    remaining=[(i,r) for i,r in enumerate(val) if r['stem'] not in reused]
    supplied=read_csv(HERE/'outputs/six_image_all_gt_run02/remaining_val_manifest.csv')
    if [(int(r['image_index']),r['stem']) for r in supplied]!=[(i,r['stem']) for i,r in remaining]:raise RuntimeError('Authorized remaining list differs')
    if len(remaining)!=144 or len(proof['captures'])!=12 or any(int(r['person_boxes'])==0 for _,r in remaining):raise RuntimeError('Unexpected authorized scope; no extra real empty-GT test')
    environment={'hostname':platform.node(),'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
        'torchvision':torchvision.__version__,'gpu':torch.cuda.get_device_name(0),'platform':platform.platform()}
    return baseline,day1,val,remaining,proof,expected,environment

def receipt_signature(model,mr,index,expected,day1):
    files={k:v for k,v in expected['sha256'].items() if '/images/val/' not in k and '/annotations_voc/val/' not in k}
    inputs={k:v for k,v in expected['sha256'].items() if any(k.endswith(mr[f]) for f in ('visible','infrared','annotation_voc'))}
    spec={'model':model,'stem':mr['stem'],'image_index':index,'inputs':inputs,'sources':files,
          'checkpoint_sha256':day1['models'][model]['checkpoint']['sha256_before'],'config':day1['models'][model]['rpn_config']}
    return hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()

@torch.inference_mode()
def capture_item(model,name,image,target,ledger,initial_state,expected_config,device,prefix):
    detector=model if name=='IR' else model.detector
    cfg=original.day1.rpn_configuration(detector);original.config_check(cfg,expected_config)
    state_before=original.day1.tensor_digest(model)
    if state_before!=initial_state:raise RuntimeError('Loaded persistent state changed before item')
    methods={k:(k in detector.rpn.__dict__,detector.rpn.__dict__.get(k)) for k in ('filter_proposals','_get_top_n_idx')}
    old_filter=detector.rpn.filter_proposals;old_top=detector.rpn._get_top_n_idx
    old_nms=original.core.rpn_module.box_ops.batched_nms
    hooks=original.hook_state([model,detector.transform,detector.rpn])
    observed={'model_forward_calls':0,'transform_calls':0,'rpn_calls':0};cap=None;error=None
    try:
        with ExitStack() as stack:
            def model_pre(module,inputs):
                observed['model_forward_calls']+=1
                if observed['model_forward_calls']!=1:raise RuntimeError('Repeated original forward forbidden')
            def transform_hook(module,inputs,output):
                images,targets=output;observed['transform_calls']+=1
                observed['transform_gt_all']=targets[0]['boxes'].detach().clone();observed['transform_sizes']=list(images.image_sizes)
                if observed['transform_calls']!=1 or images.image_sizes!=[(640,640)]:raise ValueError('Transform geometry/calls')
                exact(targets[0]['boxes'],target['boxes'].to(device),'same complete transformed GT')
            def rpn_pre(module,inputs):
                nonlocal cap
                images,features,targets=inputs
                if cap is not None or tuple(features)!=('0','1','pool'):raise ValueError('RPN calls/actual keys')
                exact(targets[0]['boxes'],observed['transform_gt_all'],'same-call full GT')
                if list(images.image_sizes)!=observed['transform_sizes']:raise ValueError('RPN image size')
                observed['layer_keys']=tuple(features);observed['feature_shapes']={k:list(v.shape) for k,v in features.items()}
                cap=stack.enter_context(AgnosticCapture(module,tuple(features),targets[0]['boxes'],prefix))
            def rpn_post(module,inputs,output):
                observed['rpn_calls']+=1;proposals,losses=output
                if losses or len(proposals)!=1 or cap is None or len(cap.records)!=1:raise ValueError('RPN output differs')
                observed['rpn_forward_boxes']=proposals[0].detach().clone()
                exact(proposals[0],cap.records[0]['final_boxes'],'actual RPN output')
            for module,kind,fn in ((model,'pre',model_pre),(detector.transform,'post',transform_hook),(detector.rpn,'pre',rpn_pre),(detector.rpn,'post',rpn_post)):
                handle=module.register_forward_pre_hook(fn) if kind=='pre' else module.register_forward_hook(fn)
                stack.callback(handle.remove)
            moved=[image['infrared'].to(device)] if name=='IR' else [{k:v.to(device) for k,v in image.items()}]
            moved_targets=[{k:v.to(device) for k,v in target.items()}]
            model(moved,moved_targets)
        if cap is None or len(cap.states)!=1 or any(observed[k]!=1 for k in ('model_forward_calls','transform_calls','rpn_calls')):raise ValueError('Incomplete original forward')
    except BaseException:error=traceback.format_exc()
    finally:
        state_after=original.day1.tensor_digest(model)
        restored=(original.hooks_restored(hooks) and not LOCK.locked() and detector.rpn.filter_proposals==old_filter
            and detector.rpn._get_top_n_idx==old_top and original.core.rpn_module.box_ops.batched_nms is old_nms
            and all((k in detector.rpn.__dict__,detector.rpn.__dict__.get(k))==v for k,v in methods.items())
            and (cap is None or not cap.active))
        info={'forward_calls':observed['model_forward_calls'],'state_sha256_before':state_before,'state_sha256_after':state_after,
            'state_equal_to_initial':state_after==initial_state,'config':cfg,'config_unchanged':original.day1.rpn_configuration(detector)==cfg,
            'restoration_verified':restored,'same_device_exact_replay':cap is not None and len(cap.states)==1,'forward_error':error}
        data={'schema':'day04_gt_independent_v1','signature':ledger.signature,'observed':original.cpu(observed),
            'records':original.cpu(cap.records) if cap else [],'stages':original.cpu(cap.states) if cap else [],
            'catalogues':cap.catalogues if cap else [],'verification':info}
        if cap is not None and cap.records:ledger.commit(data)
        else:json_write(ledger.path/'capture_failure.json',{'status':'execution_unknown_no_complete_capture','verification':info})
    if error or not restored or state_before!=state_after or not info['config_unchanged']:raise RuntimeError('Original forward/capture failed; preserve ledger: '+str(error))
    return data

def validate_new(data,ledger,mr,index,model,history):
    if data.get('signature')!=ledger.signature:raise RuntimeError('Capture signature differs')
    v=data['verification']
    if v['forward_error'] or v['forward_calls']!=1 or not all(v[k] for k in ('state_equal_to_initial','config_unchanged','restoration_verified','same_device_exact_replay')) or v['state_sha256_before']!=v['state_sha256_after']:raise RuntimeError('Raw capture invariants failed')
    _,image,comparisons=compare_item(data,model,mr,index,history)
    return {'status':'passed','gt_count':image['gt_count'],'new_forward_calls':1,'day01_comparisons':comparisons,
        'same_device_exact_replay':True,'restoration_verified':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(HERE):raise ValueError('Output outside Day04')
    out.mkdir(parents=True,exist_ok=args.resume)
    attempts=out/'attempts';attempts.mkdir(exist_ok=True);attempt=attempts/f"attempt_{len(list(attempts.iterdir()))+1:03d}";attempt.mkdir()
    meta={'status':'running','scope':'full_validation','command':subprocess.list2cmdline(sys.orig_argv),'new_forward_calls_this_attempt':0,
        'started_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'training':False,'test_inference':False,'thresholds_modified':False}
    before={};entries=[];code=1;models={}
    with (attempt/'run.log').open('x',encoding='utf-8') as log:
      with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
       try:
        print('COMMAND',meta['command'],flush=True)
        baseline,day1,val,remaining,proof,expected,environment=preflight();meta['environment']=environment
        original.day1.ir.seed_all(42);torch.set_num_threads(2)
        protected=[p for root in (ROOT/'week08',ROOT/'week09/day01',ROOT/'week09/day02',ROOT/'week09/day03',HERE) for p in root.rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts and not p.is_relative_to(out)]
        protected.extend(ROOT/p for p in expected['sha256'])
        before=original.snapshot_files(protected);meta['historical_sha256_before']=before
        meta['source_sha256']={str(p):sha(p) for p in HERE.glob('*.py')};meta['seed']=42;meta['threads']=2
        meta['cudnn_deterministic']=torch.backends.cudnn.deterministic;meta['cudnn_benchmark']=torch.backends.cudnn.benchmark
        print('PREFLIGHT_PASSED',json.dumps(environment),'protected_files',len(before),'remaining_images',len(remaining),flush=True)
        history={(r['model'],r['stem'],int(r['gt_index'])):r for r in read_csv(D1/'rpn_per_gt.csv')}
        for c in proof['captures']:
            folder=out/'reuse'/f"{c['model']}_{c['stem']}";folder.mkdir(parents=True,exist_ok=args.resume)
            raw=folder/'raw_capture.pth';source=remap(c['raw_path'])
            if raw.exists():
                if not args.resume or sha(raw)!=c['raw_sha256']:raise ValueError('Reused destination differs')
            else:shutil.copy2(source,raw)
            if sha(raw)!=c['raw_sha256']:raise ValueError('Reused transfer differs')
            data=torch.load(raw,weights_only=True,map_location='cpu');compare_item(data,c['model'],val[c['image_index']],c['image_index'],history)
            entries.append(dict(c,relative_raw=raw.relative_to(out).as_posix(),raw_sha256=sha(raw),new_forward_calls=0,
                source_raw=str(source),source_raw_sha256=c['raw_sha256'],origin='reused_day03'))
        dataset=original.day1.fusion.loader(ROOT/'data/LLVIP_1000','val',shuffle=False).dataset
        ir_dataset=original.day1.ir.loader(ROOT/'data/LLVIP_1000','val').dataset
        for model_name in ('IR','Fusion'):
            checkpoint=ROOT/('week08/day03/outputs/ir_baseline_run01/best.pth' if model_name=='IR' else 'week08/day05/outputs/fusion_concat_run01/best.pth')
            restore=original.day1.ir.checkpoint_model if model_name=='IR' else original.day1.fusion.restore
            model,ck=restore(checkpoint,'cuda:0');del ck;model.eval()
            detector=model if model_name=='IR' else model.detector
            original.config_check(original.day1.rpn_configuration(detector),day1['models'][model_name]['rpn_config'])
            models[model_name]=(model,original.day1.tensor_digest(model))
        for offset,(index,mr) in enumerate(remaining):
            image,target=dataset[index];ir_image,ir_target=ir_dataset[index]
            if dataset.get_record(index).stem!=mr['stem']:raise ValueError('Dataset identity differs')
            exact(image['infrared'],ir_image,'IR input equivalence')
            for key in target:exact(target[key],ir_target[key],'paired target equivalence')
            for name in ('IR','Fusion'):
                signature=receipt_signature(name,mr,index,expected,day1)
                path=out/'items'/f"block_{offset//10:03d}"/f"{name}_{mr['stem']}";ledger=Ledger(path,signature)
                action=ledger.action()
                if action=='new':
                    ledger.start({'model':name,'stem':mr['stem'],'image_index':index,'scope':'val','command':meta['command'],
                        'new_forward_calls_authorized':1,'started_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
                    data=capture_item(models[name][0],name,image,target,ledger,models[name][1],day1['models'][name]['rpn_config'],'cuda:0',f"{out.name}/{name}/{mr['stem']}")
                    meta['new_forward_calls_this_attempt']+=1
                else:
                    data=torch.load(path/'raw_capture.pth',weights_only=True,map_location='cpu')
                verification=validate_new(data,ledger,mr,index,name,history)
                if action!='skip':ledger.accept(verification)
                entries.append({'model':name,'stem':mr['stem'],'scene_id':mr['scene_id'],'image_index':index,
                    'relative_raw':(path/'raw_capture.pth').relative_to(out).as_posix(),'raw_sha256':sha(path/'raw_capture.pth'),
                    'new_forward_calls':1,'origin':'new_day04','signature':signature,'ledger_relative':path.relative_to(out).as_posix()})
                print('ITEM_VERIFIED',name,mr['stem'],'GT',verification['gt_count'],'action',action,
                    'max_day01_delta',max((abs(r['actual_difference']) for r in verification['day01_comparisons']),default=0),flush=True)
            if offset==0:print('FIRST_FORMAL_IMAGE_BOTH_MODELS_ACCEPTED',mr['stem'],'included_in_288',flush=True)
            if (offset+1)%10==0 or offset+1==len(remaining):print('BLOCK_PROGRESS',offset+1,'of',len(remaining),'images',flush=True)
        if sum(e['new_forward_calls'] for e in entries)!=288:raise ValueError('Expected 288 new original forwards in total ledger')
        after=original.snapshot_files(protected)
        if before!=after:raise ValueError('Protected files changed; attribution stopped')
        write_csv(out/'forward_ledger.csv',entries,list(dict.fromkeys(k for e in entries for k in e)))
        verification=merge_full(out,entries)
        meta.update(status='completed',new_forward_calls=288,reused_model_image_items=12,actual=verification['actual'])
        code=0;print('FULL_VAL_COMPLETED',json.dumps(verification['actual']),'new_forwards=288 reused_forwards=0',flush=True)
       except BaseException:
        meta['status']='failed';meta['failure']=traceback.format_exc();traceback.print_exc()
        save_json(attempt/'failure.json',{'status':'failed','error':meta['failure'],'attribution_allowed':False,'automatic_rerun':False})
       finally:
        if before:
            meta['historical_sha256_after']=original.snapshot_files(protected)
            meta['historical_files_unchanged']=before==meta['historical_sha256_after']
            if not meta['historical_files_unchanged']:meta['status']='failed';code=1
        meta['exit_code']=code;meta['finished_at_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        save_json(attempt/'run_metadata.json',meta);save_json(attempt/'execution_status.json',{'status':meta['status'],'exit_code':code})
        if code==0:
            save_json(out/'run_metadata.json',meta);save_json(out/'execution_status.json',{'status':'completed','exit_code':0})
        print('PROCESS_SCRIPT_EXIT',code,flush=True)
    return code
if __name__=='__main__':sys.exit(main())
