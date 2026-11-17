"""本地工程检查，不下载权重、不训练；合成指标断言与真实IR/模型变换检查。"""
import ast
import json
from pathlib import Path
from ir_common import *


def main():
    output=Path(__file__).parent/'checks'
    output.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob('*.py'):
        ast.parse(source.read_text(encoding='utf-8'))
    gt=[torch.tensor([[1.,2.,11.,22.]])]
    perfect=[{'boxes':gt[0].clone(),'scores':torch.tensor([0.9])}]
    m=metrics(perfect,gt)
    assert m['AP50']==m['mAP50_95']==m['precision']==m['recall']==1
    missed=metrics([{'boxes':torch.empty((0,4)), 'scores':torch.empty(0)}],gt)
    assert missed['FN']==1 and missed['mAP50_95']==0
    duplicate=metrics([{'boxes':gt[0].repeat(2,1),'scores':torch.tensor([0.9,0.8])}],gt)
    assert duplicate['TP']==1 and duplicate['FP']==1 and duplicate['precision']==0.5
    empty=metrics([{'boxes':torch.empty((0,4)),'scores':torch.empty(0)}],[torch.empty((0,4))])
    assert empty['TP']==empty['FP']==empty['FN']==0
    # 仅构造未加载权重的模型以检查结构/变换，绝不以此作为基线训练。
    model=build_model(False)
    assert model.roi_heads.box_predictor.cls_score.out_features==2
    assert model.transform.min_size==(640,) and model.transform.max_size==640
    assert all(m.eps==1e-5 for m in model.modules() if isinstance(m,FrozenBatchNorm2d))
    data=loader(ROOT/'data/LLVIP_1000','val')
    image,target=data.dataset[0]
    transformed,targets=model.transform([image],[target])
    assert transformed.image_sizes==[(640,640)]
    torch.testing.assert_close(targets[0]['boxes'],target['boxes'],rtol=0,atol=0)
    negative=loader(ROOT/'data/LLVIP_1000','train').dataset
    idx=next(i for i,r in enumerate(negative.records) if r.is_negative)
    _,target_empty=negative[idx]
    assert target_empty['boxes'].shape==(0,4)
    result={'syntax':'passed','synthetic_metric_cases':'perfect/miss/duplicate/empty passed; NOT experiment metrics',
            'real_val_stem':data.dataset.get_record(0).stem,'ir_shape':list(image.shape),
            'model_transform_image_sizes':transformed.image_sizes,
            'boxes_unchanged':True,'negative_stem':negative.get_record(idx).stem,
            'weights':weights_info(),'training_run':False}
    save_json(output/'local_checks.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
