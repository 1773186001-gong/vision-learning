"""仅从一次正式test的缓存出RGB预测图；优先IR已展示的编号，按真实匹配分类。"""
import argparse
from pathlib import Path
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image
from rgb_common import REFERENCE_RUN, ir, output_path, read_json, rgb_loader, write_new_json


def select_examples(saved, preferred):
    candidates={'success':[], 'miss':[], 'false_positive':[]}
    counts={}
    for i,(prediction,gt) in enumerate(zip(saved['predictions'],saved['truths'])):
        m=ir.metrics([prediction],[gt])
        counts[saved['stems'][i]]={k:m[k] for k in ('TP','FP','FN')}
        if m['FN']: candidates['miss'].append(i)
        if m['FP']: candidates['false_positive'].append(i)
        if m['TP'] and not m['FN'] and not m['FP']: candidates['success'].append(i)
    rank={stem:i for i,stem in enumerate(preferred)}
    selected={key:sorted(indices,key=lambda i:(rank.get(saved['stems'][i],len(rank)),i))[:3]
              for key,indices in candidates.items()}
    return selected,counts


def render_saved(root, cache, output, reference=REFERENCE_RUN):
    output=output_path(output)
    if output.exists(): raise FileExistsError('RGB预测图目录已存在，不覆盖')
    saved=ir.torch.load(cache,map_location='cpu',weights_only=False)
    if saved.get('modality')!='visible': raise ValueError('预测缓存不是RGB')
    ds=rgb_loader(root,'test').dataset
    if len(saved['stems'])!=len(ds) or saved['stems']!=[r.stem for r in ds.records]:
        raise ValueError('缓存样本顺序与清单不一致')
    preferred=[]
    for values in read_json(reference/'prediction_examples/selection.json').values():
        for stem in values:
            if stem not in preferred: preferred.append(stem)
    selected,counts=select_examples(saved,preferred)
    output.mkdir(parents=True)
    selection={key:[saved['stems'][i] for i in indices] for key,indices in selected.items()}
    for category,indices in selected.items():
        for i in indices:
            image,target=ds[i]
            ir.torch.testing.assert_close(target['boxes'],saved['truths'][i],rtol=0,atol=0)
            picture=to_pil_image(image)
            draw=ImageDraw.Draw(picture)
            for box in target['boxes'].tolist(): draw.rectangle(box,outline='lime',width=2)
            prediction=saved['predictions'][i]
            for box,score in zip(prediction['boxes'].tolist(),prediction['scores'].tolist()):
                if score>=0.5:
                    draw.rectangle(box,outline='red',width=2)
                    draw.text((box[0],max(0,box[1]-12)),f'person {score:.2f}',fill='red')
            stem=saved['stems'][i]
            draw.text((5,5),f'RGB {category} {stem}: GT green; prediction red; score>=0.5',fill='white')
            picture.save(output/f'{category}_{stem}.png')
    write_new_json(output/'selection.json',selection)
    write_new_json(output/'case_details.json',{'preferred_IR_stems':preferred,
                   'counts':{stem:counts[stem] for stems in selection.values() for stem in stems},
                   'shortfall':{key:3-len(values) for key,values in selection.items()},
                   'note':'类别可重叠；不足3张时如实记录，不制造失败案例'})
    return selection


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(render_saved(a.data_root,a.predictions,a.output))
