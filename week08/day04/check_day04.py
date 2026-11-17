"""本地CPU准备验证：真实RGB/空标注、共享代码身份、采样/预处理/保护；不训练。"""
import argparse
import ast
import json
import sys
import time
from pathlib import Path
import torch
from rgb_common import (REFERENCE_RUN, ROOT, ir, output_path, protected_hashes,
                        read_reference, rgb_loader, smoke, update, write_new_json)
from preflight_rgb import inspect_data
from predict_rgb import select_examples
import train_ir


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/LLVIP_1000')
    args=parser.parse_args()
    torch.set_num_threads(2)
    protected_before=protected_hashes()
    reference=read_reference()
    for file in Path(__file__).parent.glob('*.py'):
        ast.parse(file.read_text(encoding='utf-8'))
    assert update is train_ir.update and smoke is train_ir.smoke
    data_check=inspect_data(args.data_root,full_decode=True)
    results=[]
    for split in ('train','val','test'):
        rgb=rgb_loader(args.data_root,split)
        baseline=ir.loader(args.data_root,split)
        paired=ir.LLVIPDataset(args.data_root,split,modality='paired',
                              transforms=ir.PairedLetterboxAndFlip(split))
        assert rgb.dataset.modality=='visible'
        assert rgb.collate_fn is ir.detection_collate_fn
        assert rgb.batch_size==1 and rgb.num_workers==0
        assert rgb.dataset.transforms.horizontal_flip.p==(0.5 if split=='train' else 0.0)
        assert [r.stem for r in rgb.dataset.records]==[r.stem for r in baseline.dataset.records]
        shuffled=rgb_loader(args.data_root,split,42,True)
        ir_shuffled=ir.loader(args.data_root,split,42,True)
        assert list(shuffled.sampler)==list(ir_shuffled.sampler)
        indices=[0]
        if split=='train': indices.append(next(i for i,r in enumerate(rgb.dataset.records) if r.is_negative))
        for index in indices:
            torch.manual_seed(42)
            image,target=rgb.dataset[index]
            torch.manual_seed(42)
            pair,paired_target=paired[index]
            torch.testing.assert_close(image,pair['visible'],rtol=0,atol=0)
            assert image.dtype==torch.float32 and image.shape==(3,640,640)
            assert torch.isfinite(image).all() and image.min()>=0 and image.max()<=1
            for key in target:
                torch.testing.assert_close(target[key],paired_target[key],rtol=0,atol=0)
            assert target['boxes'].dtype==torch.float32 and target['boxes'].shape[1:]==(4,)
            expected_area=(target['boxes'][:,2]-target['boxes'][:,0])*(target['boxes'][:,3]-target['boxes'][:,1])
            torch.testing.assert_close(target['area'],expected_area,rtol=0,atol=0)
            assert target['labels'].dtype==torch.int64
            results.append({'split':split,'stem':rgb.dataset.get_record(index).stem,
                            'RGB_shape':list(image.shape),'boxes_shape':list(target['boxes'].shape),
                            'equals_paired_visible':True,'RGB_equals_IR':torch.equal(image,pair['infrared'])})
        images,targets=next(iter(rgb))
        assert len(images)==len(targets)==1 and isinstance(images[0],torch.Tensor)
    # 只构造结构并执行预处理，不加载随机权重进行模型前向/训练。
    model=ir.build_model(False)
    assert model.roi_heads.box_predictor.cls_score.out_features==2
    val=rgb_loader(args.data_root,'val')
    image,target=val.dataset[0]
    processed,processed_targets=model.transform([image],[target])
    assert processed.image_sizes==[(640,640)]
    torch.testing.assert_close(target['boxes'],processed_targets[0]['boxes'],rtol=0,atol=0)
    try: output_path(REFERENCE_RUN)
    except ValueError: pass
    else: raise AssertionError('没有拒绝写入IR目录')
    # 仅验证案例分类/优先排序的合成断言，不是RGB检测成绩。
    gt=torch.tensor([[1.,1.,11.,11.]])
    saved={'stems':['perfect','miss','duplicate'], 'truths':[gt,gt,gt],
           'predictions':[{'boxes':gt,'scores':torch.tensor([.9])},
                          {'boxes':torch.empty((0,4)),'scores':torch.empty(0)},
                          {'boxes':gt.repeat(2,1),'scores':torch.tensor([.9,.8])}]}
    selected,counts=select_examples(saved,['duplicate','miss','perfect'])
    assert selected=={'success':[0],'miss':[1],'false_positive':[2]}
    assert counts['duplicate']=={'TP':1,'FP':1,'FN':0}
    assert protected_hashes()==protected_before
    report={'status':'passed','device':'cpu','python':sys.executable,
            'torch':torch.__version__,'syntax':'passed',
            'shared_update_and_smoke_identity':True,'metric_function':'original ir_common.metrics',
            'data_integrity':data_check,'real_samples':results,
            'train_val_test_sampling_and_flip_configuration':'passed',
            'model_transform_640_boxes_unchanged':True,
            'IR_output_path_rejected':True,'protected_files_unchanged':len(protected_before),
            'synthetic_case_selection':'passed; NOT RGB experiment metrics',
            'IR_reference_best_epoch':reference['best_epoch'],
            'training_or_model_forward_run':False,'GPU_training_status':'not run; awaiting authorization'}
    folder=Path(__file__).parent/'checks'
    folder.mkdir(exist_ok=True)
    path=folder/f'local_checks_{time.time_ns()}.json'
    write_new_json(path,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('SAVED',path)


if __name__=='__main__': main()
