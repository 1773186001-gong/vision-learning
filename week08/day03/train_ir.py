"""IR完整基线：最小冒烟→10轮训练→验证选模→一次最终测试及代表性预测图。"""
import argparse
import contextlib
import copy
import importlib.util
import json
import random
import sys
import time
import traceback
from pathlib import Path

from ir_common import *


def cuda_sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def update(model, optimizer, images, targets, device):
    images, targets = move(images, targets, device)
    optimizer.zero_grad(set_to_none=True)
    losses = model(images, targets)
    if set(losses) != set(LOSS_KEYS):
        raise ValueError(f'Loss字段异常：{losses.keys()}')
    total = sum(losses.values())
    if not torch.isfinite(total) or not all(torch.isfinite(v).all() for v in losses.values()):
        raise FloatingPointError('Loss出现NaN/Inf')
    total.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise FloatingPointError('梯度出现NaN/Inf')
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('更新后参数出现NaN/Inf')
    return {'total_loss': total.item(), **{k: v.item() for k,v in losses.items()}}


def smoke(model, data, device):
    state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001)
    indices = [next(i for i,r in enumerate(data.dataset.records) if not r.is_negative),
               next(i for i,r in enumerate(data.dataset.records) if r.is_negative)]
    rows = []
    model.train()
    for index in indices:
        image, target = data.dataset[index]
        before = model.roi_heads.box_predictor.cls_score.weight.detach().clone()
        row = update(model, opt, [image], [target], device)
        if torch.equal(before, model.roi_heads.box_predictor.cls_score.weight):
            raise RuntimeError('冒烟：参数未更新')
        rows.append({'stem': data.dataset.get_record(index).stem, **row})
    image, target = data.dataset[indices[0]]
    image_list, transformed = model.transform([image.to(device)],
        [{k:v.to(device) for k,v in target.items()}])
    if image_list.image_sizes != [(640,640)]:
        raise ValueError('模型内部resize改变了640尺寸')
    torch.testing.assert_close(transformed[0]['boxes'].cpu(), target['boxes'], rtol=0, atol=0)
    model.load_state_dict(state)
    torch.set_rng_state(rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    del opt, image_list, transformed
    return {'samples': rows, 'resize_boxes_unchanged': True,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'weights_restored_before_training': True}


def atomic_save(checkpoint, path):
    temp = path.with_suffix('.tmp')
    torch.save(checkpoint, temp)
    temp.replace(path)


def final_report(output, config, history, checkpoint, evaluation):
    text = '# Day03 IR 单模态基线真实训练结果\n\n'
    text += '本报告仅在10轮训练和正式test全部成功后生成。\n\n'
    text += '## 配置\n\n```json\n'+json.dumps(config,ensure_ascii=False,indent=2)+'\n```\n\n'
    text += '## 逐轮真实指标\n\n|Epoch|total|classifier|box_reg|objectness|rpn_box|P|R|AP50|mAP50:95|seconds|peak bytes|\n|---|---|---|---|---|---|---|---|---|---|---|---|\n'
    for r in history:
        vals=[str(r['epoch'])]+[f'{r[k]:.6f}' for k in ('total_loss',)+LOSS_KEYS+('precision','recall','AP50','mAP50_95','epoch_seconds')]+[str(r['peak_allocated_bytes'])]
        text += '|'+ '|'.join(vals)+'|\n'
    text += f'\n最佳Epoch：{checkpoint["epoch"]}；模型 best.pth。last.pth 为第10轮。\n'
    text += '\n## 最终test\n\n```json\n'+json.dumps(evaluation,indent=2)+'\n```\n'
    text += '\nAP是本项目单类101点插值协议，不声称完全等价于COCO API。图片见prediction_examples，失败/中断见failure文件，冒烟见smoke_test.json。\n'
    text += '\n学习重点：四项Loss、验证选模与测试隔离、置信阈值和AP积分、IR三通道迁移的局限。\n'
    (output/'DAY03_IR_TRAINING_GPT_HANDOFF.md').write_text(text,encoding='utf-8')


def run(args):
    seed_all(42)
    torch.set_num_threads(2)
    device = device_check(args.device)
    output = args.output.resolve()
    config = {'model': 'fasterrcnn_mobilenet_v3_large_320_fpn', 'epochs':10,
              'batch_size':1, 'num_workers':0, 'seed':42, 'lr':0.001,
              'momentum':0.9, 'weight_decay':0.0005, 'scheduler':None, 'amp':False,
              'letterbox':640, 'min_size':640, 'max_size':640,
              'data_root':str(args.data_root.resolve()), 'manifest_sha256':sha(args.data_root/'manifest.csv'),
              'selection':'val mAP50_95', 'weights':weights_info(),
              'torch':torch.__version__, 'torchvision':torchvision.__version__,
              'device':str(device), 'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None}
    config['weights']['sha256'] = sha(config['weights']['path']) if config['weights']['exists'] else None
    train = loader(args.data_root, 'train',42,True)
    val = loader(args.data_root,'val')
    history, start, best = [],1,-1.0
    if args.resume:
        if args.resume.resolve() != (output/'last.pth'):
            raise ValueError('续训只允许从同一输出目录 last.pth 恢复')
        model, saved = checkpoint_model(args.resume,device)
        if saved['config'] != config:
            raise ValueError('续训配置/数据/权重缓存环境变化，停止；不能静默改参')
        history,start,best = saved['history'],saved['epoch']+1,saved['best_score']
        if saved['epoch'] >= 10:
            raise ValueError('10轮已完成；不允许重跑正式测试；用评估/推理工具检查已保存结果')
        if (output/'test_started.json').exists():
            raise ValueError('测试已启动过，停止避免再次访问test')
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('输出目录非空，禁止覆盖；请显式续训或选择新目录')
        model = build_model(True).to(device)
        output.mkdir(parents=True,exist_ok=True)
        save_json(output/'config.json',config)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
        lr=0.001,momentum=0.9,weight_decay=0.0005)
    if args.resume:
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        random.setstate(saved['python_rng'])
        train.generator.set_state(saved['loader_rng'])
        if device.type=='cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        # 从checkpoint历史重建派生CSV，先保留中断时原CSV而不是覆盖。
        if (output/'training_log.csv').exists():
            import shutil
            shutil.copy2(output/'training_log.csv',output/f'log_before_resume_{time.time_ns()}.csv')
    else:
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        smoke_result = smoke(model,train,device)
        save_json(output/'smoke_test.json',smoke_result)
        print('SMOKE PASSED',smoke_result,flush=True)
        seed_all(42)  # 冒烟不计入正式训练；恢复权重并重置固定种子。
    logpath=output/f'console_{time.time_ns()}.txt'
    class Tee:
        def __init__(self, stream, f): self.stream,self.f=stream,f
        def write(self,text):
            self.stream.write(text); self.f.write(text); self.f.flush(); return len(text)
        def flush(self): self.stream.flush(); self.f.flush()
    with logpath.open('w',encoding='utf-8') as f, contextlib.redirect_stdout(Tee(sys.stdout,f)), contextlib.redirect_stderr(Tee(sys.stderr,f)):
        try:
            for epoch in range(start,11):
                if (output/f'epoch_{epoch:02d}.pth').exists():
                    raise FileExistsError('已有同轮checkpoint，停止避免覆盖')
                cuda_sync(device)
                begin=time.perf_counter()
                if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
                model.train()
                totals={k:0.0 for k in ('total_loss',)+LOSS_KEYS}
                for batch,(images,targets) in enumerate(train,1):
                    values=update(model,optimizer,images,targets,device)
                    for k in totals: totals[k]+=values[k]
                    if batch%100==0: print(f'epoch={epoch} batch={batch}/700 loss={values["total_loss"]:.6f}',flush=True)
                preds,gts,_=infer(model,val,device)
                scores=metrics(preds,gts)
                cuda_sync(device)
                row={'epoch':epoch, **{k:v/len(train) for k,v in totals.items()},
                     'lr':optimizer.param_groups[0]['lr'],'epoch_seconds':time.perf_counter()-begin,
                     'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                     'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,
                     **{k:scores[k] for k in ('precision','recall','AP50','mAP50_95')}}
                improved=scores['mAP50_95']>best
                if improved: best=scores['mAP50_95']
                history.append(row)
                ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                    'trainable_names':[name for name,p in model.named_parameters() if p.requires_grad],
                    'history':history,'best_score':best,'config':config,
                    'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                    'loader_rng':train.generator.get_state()}
                atomic_save(ck,output/f'epoch_{epoch:02d}.pth')
                atomic_save(ck,output/'last.pth')
                if improved: atomic_save(ck,output/'best.pth')
                write_history(history,output)
                save_json(output/f'val_epoch_{epoch:02d}.json',scores)
                print(json.dumps(row),flush=True)
            # 只有十轮结束后才读取test；标记先落盘，失败后也不自动重试测试。
            best_model,checkpoint=checkpoint_model(output/'best.pth',device)
            save_json(output/'test_started.json',{'checkpoint_sha256':sha(output/'best.pth'),
                                                'best_epoch':checkpoint['epoch']})
            from evaluate_ir import evaluate_final
            evaluation=evaluate_final(best_model,args.data_root,device,output)
            from predict_ir import render_saved
            render_saved(args.data_root,output/'test_predictions.pth',output/'prediction_examples')
            save_json(output/'completed.json',{'best_epoch':checkpoint['epoch'],'test':evaluation})
            final_report(output,config,history,checkpoint,evaluation)
            print('COMPLETED',evaluation,flush=True)
        except BaseException as e:
            traceback.print_exc()
            save_json(output/f'failure_{time.time_ns()}.json',{'type':type(e).__name__,'message':str(e),
                      'recovery':'从最后完整epoch的last.pth恢复；本轮未完成部分不计入'})
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',type=Path)
    args=p.parse_args()
    # 外层日志覆盖权重检查和冒烟阶段；每次启动唯一名字，不覆盖旧记录。
    args.output.parent.mkdir(parents=True,exist_ok=True)
    class Transcript:
        def __init__(self, stream, file): self.stream,self.file=stream,file
        def write(self,text):
            self.stream.write(text); self.file.write(text); self.file.flush(); return len(text)
        def flush(self): self.stream.flush(); self.file.flush()
    launch_log=args.output.parent/f'{args.output.name}_launch_{time.time_ns()}.log'
    with launch_log.open('w',encoding='utf-8') as log:
        with contextlib.redirect_stdout(Transcript(sys.stdout,log)), contextlib.redirect_stderr(Transcript(sys.stderr,log)):
            try:
                run(args)
            except BaseException as error:
                traceback.print_exc()
                if args.output.is_dir():
                    save_json(args.output/f'entry_failure_{time.time_ns()}.json',
                              {'type':type(error).__name__, 'message':str(error)})
                raise SystemExit(1)
