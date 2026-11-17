"""验证集评估；正式test由训练流程只运行一次并缓存预测。"""
import argparse
from ir_common import *


def evaluate_final(model, root, device, output):
    test=loader(root,'test')
    predictions,truths,stems=infer(model,test,device)
    result=metrics(predictions,truths)
    torch.save({'predictions':predictions,'truths':truths,'stems':stems},output/'test_predictions.pth')
    save_json(output/'test_metrics.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('评估文件已存在，禁止覆盖')
    model,ck=checkpoint_model(a.checkpoint,device_check(a.device))
    predictions,truths,stems=infer(model,loader(a.data_root,'val'),torch.device(a.device))
    save_json(a.output,{'epoch':ck['epoch'],'split':'val',**metrics(predictions,truths)})
    print(a.output.read_text(encoding='utf-8'))
