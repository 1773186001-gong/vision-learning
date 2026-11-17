"""Two independent COCO MobileNet+FPN branches; shared detection heads."""
import copy
import sys
from pathlib import Path
from collections import OrderedDict
from unittest.mock import patch
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'week08/day03'))
import ir_common as ir
SHA='907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5'


class DualFPN(nn.Module):
    def __init__(self, backbone, keys):
        super().__init__()
        self.rgb=backbone
        self.infrared=copy.deepcopy(backbone)
        self.keys=tuple(keys)
        self.out_channels=backbone.out_channels
        c=self.out_channels
        self.fusion=nn.ModuleDict({k:nn.Conv2d(2*c,c,1) for k in keys})
        with torch.no_grad():
            for conv in self.fusion.values():
                conv.weight.zero_(); conv.bias.zero_()
                for i in range(c): conv.weight[i,i,0,0]=conv.weight[i,i+c,0,0]=.5
        self.last_shapes={}

    def forward(self, packed):
        if packed.ndim!=4 or packed.shape[1]!=6: raise ValueError('Expected internally packed paired N6HW')
        a,b=self.rgb(packed[:,:3]),self.infrared(packed[:,3:])
        if tuple(a)!=self.keys or tuple(b)!=self.keys: raise ValueError('FPN keys changed')
        result=OrderedDict()
        for k in self.keys:
            if a[k].shape!=b[k].shape: raise ValueError('Unaligned branch feature shapes')
            result[k]=self.fusion[k](torch.cat([a[k],b[k]],1))
            self.last_shapes[k]={'rgb':list(a[k].shape),'infrared':list(b[k].shape),'fused':list(result[k].shape)}
        return result


class PairedDetector(nn.Module):
    def __init__(self, detector):
        super().__init__(); self.detector=detector

    def forward(self, images, targets=None):
        packed=[]
        for image in images:
            if set(image)!={'visible','infrared'}: raise ValueError('Explicit paired modality dict required')
            a,b=image['visible'],image['infrared']
            if a.shape!=b.shape or a.shape!=(3,640,640): raise ValueError('RGB/IR must share 3x640x640 geometry')
            if a.dtype!=torch.float32 or b.dtype!=torch.float32: raise ValueError('float32 inputs required')
            if not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise ValueError('Nonfinite image')
            packed.append(torch.cat([a,b],0))
        # Pack only for one shared geometry transform, split BEFORE separate feature extraction.
        # No 6-channel convolution and no prediction-level fusion.
        return self.detector(packed,targets)


def build_model(pretrained=True, weights_path=None):
    if pretrained:
        path=Path(weights_path or ir.weights_info()['path']).resolve()
        if not path.is_file() or ir.sha(path)!=SHA: raise ValueError('Verified official COCO cache required; no auto download')
        if path.parent.name!='checkpoints': raise ValueError('Weight must be inside a checkpoints cache directory')
        with patch('torch.hub.get_dir',return_value=str(path.parent.parent)):
            detector=ir.build_model(True)
    else:
        detector=ir.build_model(False)  # For immediate strict checkpoint restoration only.
    detector.backbone.eval()
    with torch.no_grad(): features=detector.backbone(torch.zeros(1,3,640,640))
    keys=tuple(features)
    anchors=detector.rpn.anchor_generator
    if len(keys)!=len(anchors.sizes) or len(keys)!=len(anchors.aspect_ratios): raise ValueError('Anchor/FPN count mismatch')
    pool=detector.roi_heads.box_roi_pool
    pool.featmap_names=[k for k in pool.featmap_names if k in keys]
    if not pool.featmap_names: raise ValueError('No compatible RoI feature keys')
    detector.backbone=DualFPN(detector.backbone,keys)
    detector.transform.image_mean=list(detector.transform.image_mean)*2
    detector.transform.image_std=list(detector.transform.image_std)*2
    model=PairedDetector(detector)
    model.architecture={'feature_shapes':{k:list(v.shape) for k,v in features.items()},
        'roi_keys':pool.featmap_names,'anchor_sizes':anchors.sizes,'anchor_ratios':anchors.aspect_ratios,
        'out_channels':detector.backbone.out_channels,'fusion_initialization':'[0.5 I, 0.5 I], zero bias',
        'pretrained_loaded':pretrained,'source':'official COCO only' if pretrained else 'strict restore skeleton'}
    return model


def restore(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=build_model(False)
    model.load_state_dict(ck['model'],strict=True)
    for name,p in model.named_parameters(): p.requires_grad_(name in ck['trainable_names'])
    return model.to(device),ck
