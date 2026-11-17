"""独立入口仅评估val；正式test由完整训练在选模后调用一次。"""
import argparse
from pathlib import Path
from rgb_common import ir, output_path, read_json, rgb_loader, write_new_json


def final_test(model, data_root, device, output):
    # 只有train_rgb创建test_started标记并确认best后才可进入。
    if not (output/'test_started.json').is_file():
        raise RuntimeError('缺少正式测试启动记录')
    for name in ('test_predictions.pth','test_metrics.json'):
        if (output/name).exists():
            raise FileExistsError(f'已有{name}，不再次运行test')
    predictions,truths,stems = ir.infer(model,rgb_loader(data_root,'test'),device)
    scores = ir.metrics(predictions,truths)
    with (output/'test_predictions.pth').open('xb') as stream:
        ir.torch.save({'modality':'visible','predictions':predictions,
                       'truths':truths,'stems':stems},stream)
    write_new_json(output/'test_metrics.json',scores)
    return scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    a.output=output_path(a.output)
    if a.output.exists(): raise FileExistsError('验证输出已存在，不覆盖')
    device=ir.device_check(a.device)
    # 先查看配置，拒绝将IR checkpoint当RGB结果。
    if read_json(a.checkpoint.parent/'config.json').get('modality')!='visible':
        raise ValueError('不是RGB实验checkpoint')
    model,ck=ir.checkpoint_model(a.checkpoint,device)
    if ck['config'].get('modality')!='visible': raise ValueError('checkpoint模态不符')
    predictions,truths,_=ir.infer(model,rgb_loader(a.data_root,'val'),device)
    result={'split':'val','epoch':ck['epoch'],**ir.metrics(predictions,truths)}
    write_new_json(a.output,result)
    print(result)


if __name__=='__main__': main()
