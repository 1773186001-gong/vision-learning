"""Real paired data + real COCO loading + branch/feature checks, no detection inference/training."""
import argparse,json,time,sys
from pathlib import Path
from fusion_common import *


def check(data_root,weights_path):
    torch.set_num_threads(2); ir.seed_all(42); before=protect()
    model=build_model(True,weights_path).eval(); b=model.detector.backbone
    assert all(torch.equal(v,b.infrared.state_dict()[k]) for k,v in b.rgb.state_dict().items())
    assert next(b.rgb.parameters()).data_ptr()!=next(b.infrared.parameters()).data_ptr()
    assert [p.requires_grad for p in b.rgb.parameters()]==[p.requires_grad for p in b.infrared.parameters()]
    assert model.detector.roi_heads.box_predictor.cls_score.out_features==2
    samples=[]
    for split in ('train','val','test'):
        data=loader(data_root,split)
        assert data.dataset.transforms.horizontal_flip.p==(.5 if split=='train' else 0)
        indices=[0]
        if split=='train': indices.append(next(i for i,r in enumerate(data.dataset.records) if r.is_negative))
        for i in indices:
            pair,target=data.dataset[i]
            torch.testing.assert_close(pair['infrared'][0],pair['infrared'][2],rtol=0,atol=0)
            assert target['boxes'].dtype==torch.float32 and target['boxes'].shape[1:]==(4,)
            samples.append(dict(split=split,stem=data.dataset.get_record(i).stem,boxes=target['boxes'].tolist(),
                                image_shapes={k:list(v.shape) for k,v in pair.items()}))
    pair,target=loader(data_root,'val').dataset[0]
    packed=torch.cat([pair['visible'],pair['infrared']]); original=packed.clone()
    with torch.no_grad():
        transformed,ts=model.detector.transform([packed],[target])
        assert transformed.image_sizes==[(640,640)]
        torch.testing.assert_close(ts[0]['boxes'],target['boxes'],rtol=0,atol=0)
        expected=(packed-torch.tensor(model.detector.transform.image_mean)[:,None,None])/torch.tensor(model.detector.transform.image_std)[:,None,None]
        torch.testing.assert_close(transformed.tensors[0],expected,rtol=0,atol=0)
        a=b.rgb(transformed.tensors[:,:3]); c=b.infrared(transformed.tensors[:,3:]); fused=b(transformed.tensors)
        errors={}
        for k in a:
            expected=(a[k]+c[k])/2
            torch.testing.assert_close(fused[k],expected,rtol=1e-5,atol=1e-5)
            errors[k]=float((fused[k]-expected).abs().max())
        # Route sensitivity: changing IR leaves independently extracted RGB features unchanged.
        changed=transformed.tensors.clone(); changed[:,3:]=0
        c2=b.infrared(changed[:,3:]); assert any(not torch.equal(c[k],c2[k]) for k in c)
    assert torch.equal(packed,original)
    batch=next(iter(loader(data_root,'train'))); assert isinstance(batch[0][0],dict)
    assert protect()==before
    return dict(status='passed',python=sys.executable,torch=torch.__version__,torchvision=ir.torchvision.__version__,
                weights_sha256=ir.sha(weights_path),architecture=model.architecture,branch_shapes=b.last_shapes,
                fusion_max_abs_errors=errors,real_samples=samples,independent_branches=True,
                branch_frozen_names=[n for n,p in b.rgb.named_parameters() if not p.requires_grad],
                parameter_count=sum(p.numel() for p in model.parameters()),trainable_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                normalization_and_boxes_checked=True,model_detection_forward=False,backbone_forward=True,
                GPU_smoke='pending',training='not started',protected_files=len(before['checked']))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=ROOT/'data/LLVIP_1000')
    p.add_argument('--weights-path',type=Path,default=Path(ir.weights_info()['path']))
    a=p.parse_args(); r=check(a.data_root,a.weights_path)
    folder=Path(__file__).parent/'checks'; folder.mkdir(exist_ok=True)
    save(folder/f'local_{time.time_ns()}.json',r); print(json.dumps(r,ensure_ascii=False,indent=2))
