"""从正式测试的缓存预测中选成功/漏检/误检案例出图，不再次跑测试。"""
import argparse
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image
from ir_common import *


def render_saved(root, cache, output):
    if output.exists(): raise FileExistsError('预测目录已存在，禁止覆盖')
    saved=torch.load(cache,map_location='cpu',weights_only=False)
    ds=loader(root,'test').dataset
    candidates={'success':[], 'miss':[], 'false_positive':[]}
    for i,(pred,gt) in enumerate(zip(saved['predictions'],saved['truths'])):
        s=metrics([pred],[gt])
        if s['FN']: candidates['miss'].append(i)
        if s['FP']: candidates['false_positive'].append(i)
        if s['TP'] and not s['FN'] and not s['FP']: candidates['success'].append(i)
    output.mkdir(parents=True)
    selection={}
    for category,indices in candidates.items():
        selection[category]=[saved['stems'][i] for i in indices[:3]]
        for i in indices[:3]:
            image,t=ds[i]
            if ds.get_record(i).stem != saved['stems'][i]: raise ValueError('缓存样本顺序不符')
            picture=to_pil_image(image)
            draw=ImageDraw.Draw(picture)
            for box in t['boxes'].tolist(): draw.rectangle(box,outline='lime',width=2)
            pred=saved['predictions'][i]
            for box,score in zip(pred['boxes'].tolist(),pred['scores'].tolist()):
                if score>=0.5:
                    draw.rectangle(box,outline='red',width=2)
                    draw.text((box[0],max(0,box[1]-12)),f'person {score:.2f}',fill='red')
            draw.text((5,5),f'{category}: GT green; prediction red; score>=0.5',fill='white')
            picture.save(output/f'{category}_{saved["stems"][i]}.png')
    # 某类案例不存在时如实记录空列表，不制造误检/漏检。
    save_json(output/'selection.json',selection)
    return selection


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(render_saved(a.data_root,a.predictions,a.output))
