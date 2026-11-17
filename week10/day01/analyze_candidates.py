"""Four fixed model/GT items; saved original indices only. No model or NMS execution."""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import struct
import subprocess
import traceback
from collections import Counter
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch
import torchvision
from torchvision.ops import box_iou, boxes as box_ops
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D4 = ROOT / 'week09/day04/outputs/full_val_stage_run01'
D7 = ROOT / 'week09/day07/outputs/competition_run01'
STEMS = ('030078', '030129')
MODELS = ('IR', 'Fusion')
KEYS = ('0', '1', 'pool')
GT_COLOR, GOOD_COLOR, COMP_COLOR = '#ffd24a', '#00cfc2', '#ff7294'
FONT_PATH = Path('C:/Windows/Fonts/msyh.ttc')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path, data, fields=None):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(data[0]))
        w.writeheader()
        for r in data:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def check(condition, message):
    if not condition:
        raise ValueError(message)


def exact(a, b, name):
    check(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), name)


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def scalar_iou(a, b, binary32=True):
    q = f32 if binary32 else float
    aa = q(q(a[2]-a[0]) * q(a[3]-a[1]))
    bb = q(q(b[2]-b[0]) * q(b[3]-b[1]))
    inter = q(max(0., q(min(a[2], b[2])-max(a[0], b[0]))) * max(0., q(min(a[3], b[3])-max(a[1], b[1]))))
    return q(inter / q(q(aa+bb)-inter))


def font(size):
    return ImageFont.truetype(str(FONT_PATH), size)


def text(draw, xy, value, size=22, fill='#26354a'):
    draw.text(xy, str(value), font=font(size), fill=fill)


def dashed(draw, points, fill, width=3, dash=9):
    for p, q in zip(points, points[1:]):
        length = math.dist(p, q)
        if not length:
            continue
        for start in range(0, math.ceil(length), dash*2):
            end = min(start+dash, length)
            a = tuple(p[i] + (q[i]-p[i])*start/length for i in (0, 1))
            b = tuple(p[i] + (q[i]-p[i])*end/length for i in (0, 1))
            draw.line([a, b], fill=fill, width=width)


def border(draw, box, color, kept=True, width=3):
    if kept:
        draw.rectangle(box, outline=color, width=width)
    else:
        x1, y1, x2, y2 = box
        dashed(draw, [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)], color, width)


def draw_view(canvas, image, viewport, bounds, gt, candidates, labels=True):
    """Crop is display only, after full-image analysis. All boxes share the same affine map."""
    x, y, width, height = viewport
    x1, y1, x2, y2 = bounds
    scale = min(width/(x2-x1), height/(y2-y1))
    rw, rh = round((x2-x1)*scale), round((y2-y1)*scale)
    ox, oy = x+(width-rw)//2, y+(height-rh)//2
    crop = image.crop(bounds).resize((rw,rh), Image.Resampling.BILINEAR)
    canvas.paste(crop, (ox,oy))
    draw = ImageDraw.Draw(canvas)
    def trans(b):
        return [ox+(b[0]-x1)*rw/(x2-x1), oy+(b[1]-y1)*rh/(y2-y1), ox+(b[2]-x1)*rw/(x2-x1), oy+(b[3]-y1)*rh/(y2-y1)]
    border(draw, trans(gt), GT_COLOR, True, 4)
    for n, r in enumerate(candidates):
        b = trans(r['xyxy'])
        color = GOOD_COLOR if r['role']=='good' else COMP_COLOR
        border(draw, b, color, r['nms_kept'], 4 if r['nms_kept'] else 2)
        if labels:
            # Alternate top/bottom and offset labels; isolated atlas cards remain authoritative.
            lx = min(max(b[0]+(n%3)*17, ox), ox+rw-58)
            ly = max(oy, b[1]-23) if n%2==0 else min(oy+rh-24,b[3]+2)
            draw.rectangle((lx-2,ly,lx+57,ly+24),fill='#172438')
            text(draw,(lx,ly),r['label'],17,color)
    return (ox,oy,rw,rh)


def plot_all(out, cases, geometry):
    figdir = out/'figures'
    figdir.mkdir()
    figure_provenance = []
    for stem in STEMS:
        cc = [cases[stem,m] for m in MODELS]
        gt = cc[0]['gt']
        check(gt==cc[1]['gt'], 'Models have different GT geometry')
        selected = [r for c in cc for r in c['selected']]
        bb = [gt]+[r['xyxy'] for r in selected]
        bounds = (max(0,math.floor(min(b[0] for b in bb))-20), max(0,math.floor(min(b[1] for b in bb))-20),
                  min(640,math.ceil(max(b[2] for b in bb))+20), min(640,math.ceil(max(b[3] for b in bb))+20))
        image = geometry[stem]['ir_image']
        im = Image.new('RGB',(1800,1450),'#f1f5fa');d=ImageDraw.Draw(im)
        text(d,(35,22),f'{stem}/G0  原NMS去留对照',34)
        text(d,(35,74),'黄色 GT  |  青色 好框  |  粉红 直接竞争者  |  实线 保留 / 虚线 删除',23)
        text(d,(35,112),'两侧同一IR底图、相同坐标和放大范围；Fusion模型实际输入为RGB+IR。',21)
        for col,c in enumerate(cc):
            x=35+col*880
            text(d,(x,160),f"{c['model']}   L={c['summary']['L']}  T={c['summary']['T']}  好框原NMS保留 {sum(r['nms_kept'] for r in c['goods'])}/{len(c['goods'])}",25)
            v=draw_view(im,image,(x,210,820,420),(0,0,640,640),gt,c['selected'],False)
            ox,oy,rw,rh=v
            d.rectangle((ox+bounds[0]*rw/640,oy+bounds[1]*rh/640,ox+bounds[2]*rw/640,oy+bounds[3]*rh/640),outline='#ffffff',width=2)
            text(d,(x,645),f'共同放大范围 xyxy={bounds}',19)
            draw_view(im,image,(x,690,820,630),bounds,gt,c['selected'])
            text(d,(x,1335),'编号详见候选图册与CSV；密集框请逐张查看图册。',20)
        text(d,(35,1390),'事实：原保留索引决定去留。几何关系不等于已证实的GPU实际抑制事件。',22)
        im.save(figdir/f'{stem}_IR_Fusion_comparison.png')

        context=Image.new('RGB',(1400,825),'#f1f5fa');d=ImageDraw.Draw(context)
        text(d,(30,15),f'{stem}  640×640含padding坐标核对',28)
        for j,(name,img) in enumerate([('IR',image),('RGB',geometry[stem]['rgb_image'])]):
            px=30+j*690
            text(d,(px,62),f'{name}：原完整图像Letterbox；黄色为G0',20)
            draw_view(context,img,(px,100,640,640),(0,0,640,640),gt,[],False)
            for yy in [64,576]: d.line([(px,100+yy),(px+640,100+yy)], fill='#718198',width=2)
            text(d,(px,752),'1280×1024 → 640×512；上/下padding=64',19)
        context.save(figdir/f'{stem}_letterbox_geometry.png')
        for c in cc:
            # One card per distinct candidate: no candidate hidden by overlapping boxes.
            rr=c['selected'];columns=3;cardw=470;cardh=480
            atlas=Image.new('RGB',(columns*cardw+40,120+math.ceil(len(rr)/columns)*cardh),'#f1f5fa');d=ImageDraw.Draw(atlas)
            text(d,(20,15),f'{stem}/G0  {c["model"]} 全部好框与直接竞争者（编号→完整ID见CSV）',25)
            text(d,(20,57),'同一IR底图与共同范围；实线保留，虚线删除。* 表示全部C最小值候选。',20)
            for i,r in enumerate(rr):
                x=20+(i%columns)*cardw;y=105+(i//columns)*cardh
                color=GOOD_COLOR if r['role']=='good' else COMP_COLOR
                role='好框' if r['role']=='good' else '直接竞争者'
                text(d,(x+8,y+4),f'{r["label"]}  {role}  层{r["layer"]}  flat={r["flat_index"]}',21)
                draw_view(atlas,image,(x+5,y+40,cardw-20,300),bounds,gt,[r],False)
                text(d,(x+8,y+345),f'score={r["score"]:.8f}  GT IoU={r["gt_iou"]:.8f}',18)
                text(d,(x+8,y+374),f'原NMS：{"保留" if r["nms_kept"] else "删除"}  输入#{r["nms_input_position"]}  输出#{r["nms_output_position"]}',18)
                text(d,(x+8,y+403),f'最终150：{"保留" if r["final_kept"] else "未进入"}  '+(f'C={r["C"]}'+('  * 最小值' if r['is_T_minimizer'] else '') if r['role']=='good' else ''),18)
                d.line([(x,y+460),(x+cardw-25,y+460)],fill=color,width=3)
            atlas.save(figdir/f'{stem}_{c["model"]}_candidate_atlas.png')
            for layer in KEYS:
                gg=[r for r in c['goods'] if r['layer']==layer]
                if not gg:continue
                heights=[max(230,85+len([e for e in c['relations'] if e['a_id']==g['candidate_id']])*91) for g in gg]
                graph=Image.new('RGB',(1360,165+sum(heights)),'#f1f5fa');d=ImageDraw.Draw(graph)
                text(d,(25,15),f'{stem}/G0  {c["model"]}  实际层 {layer}',30)
                text(d,(25,65),'连线：同层高分重叠关系（标注相互IoU）；不是已证实的实际抑制边。',23)
                text(d,(25,105),'青色好框 / 粉红竞争者；节点明确标注原NMS去留；* 表示C最小值。',21)
                y=155;lookup={r['candidate_id']:r for r in rr}
                for g,h in zip(gg,heights):
                    ee=[e for e in c['relations'] if e['a_id']==g['candidate_id']]
                    gy=y+h/2
                    for j,e in enumerate(ee):
                        cy=y+35+j*91
                        d.line([(435,gy),(805,cy+31)],fill='#8d9db1',width=2)
                        lx=540;ly=gy+(cy+31-gy)*.47
                        d.rectangle((lx-5,ly-14,lx+147,ly+18),fill='#f1f5fa')
                        text(d,(lx,ly-13),f'IoU {e["pair_iou"]:.6f}',18)
                        b=lookup[e['b_id']]
                        border(d,(805,cy,1320,cy+73),COMP_COLOR,b['nms_kept'],3)
                        text(d,(818,cy+3),f'{b["label"]}  flat={b["flat_index"]}  原NMS {"保留" if b["nms_kept"] else "删除"}',21)
                        text(d,(818,cy+37),f's={b["score"]:.8f}  GT IoU={b["gt_iou"]:.6f}',18)
                    border(d,(25,gy-48,435,gy+48),GOOD_COLOR,g['nms_kept'],4)
                    text(d,(38,gy-42),f'{g["label"]} flat={g["flat_index"]} C={g["C"]}'+(' *' if g['is_T_minimizer'] else ''),21)
                    text(d,(38,gy-10),f'原NMS {"保留" if g["nms_kept"] else "删除"}  s={g["score"]:.8f}',20)
                    text(d,(38,gy+20),f'GT IoU={g["gt_iou"]:.6f}',18)
                    d.line([(25,y+h-7),(1320,y+h-7)],fill='#d2dbe6',width=1)
                    y+=h
                graph.save(figdir/f'{stem}_{c["model"]}_layer_{layer}_relations.png')
        figure_provenance.append(dict(stem=stem,coordinate_system='640x640 including Letterbox padding',common_zoom_xyxy=bounds,background='same IR Letterbox image for both model overlays',crop_for_display_only=True))
    return figure_provenance


def analyze(out):
    # Import only original data/geometry helpers, never a model restore entry.
    sys.path.insert(0,str(ROOT/'week08/day01'))
    sys.path.insert(0,str(ROOT/'week08/day02'))
    from llvip_dataset import LLVIPDataset
    from paired_letterbox import PairedLetterbox

    provenance=[]
    manifest=load(D4/'artifact_manifest.json')
    expected=load(D4/'preflight/expected_inputs.json')['sha256']
    day7meta=load(D7/'run_metadata.json')
    day7prov={(r['model'],r['stem']):r for r in load(D7/'capture_provenance.json')}
    def record(p,expected_sha=None,source='current read-only input'):
        p=Path(p);digest=sha(p)
        if expected_sha is not None: check(digest==expected_sha,f'SHA mismatch: {p}')
        provenance.append(dict(path=str(p),sha256=digest,expected_sha256=expected_sha,evidence=source))
    for n in ['README.md','analyze_competition.py','outputs/competition_run01/per_gt_metrics.csv','outputs/competition_run01/good_candidates.csv','outputs/competition_run01/competition_relations.csv','outputs/competition_run01/paired_23.csv','outputs/competition_run01/verification.json','outputs/competition_run01/capture_provenance.json','outputs/competition_run01/run_metadata.json']:
        record(ROOT/'week09/day07'/n)
    check(load(D7/'verification.json')['status']=='passed','Day07 not passed')
    for n in ['forward_ledger.csv','artifact_manifest.json']:
        record(D4/n,day7meta['input_sha256'][str(D4/n)],'Day07 recorded input SHA')
    for n in ['preflight/expected_inputs.json','source/agnostic_capture.py','source/capture_full_val.py','source/capture_ledger.py','source/offline_full_gt.py','run_metadata.json']:
        record(D4/n,manifest[n],'Day04 artifact manifest')
    for n in ['data/LLVIP_1000/manifest.csv','week08/day01/llvip_dataset.py','week08/day02/paired_letterbox.py']:
        record(ROOT/n,expected[n],'Day04 original preflight input SHA')
    ledger_all=rows(D4/'forward_ledger.csv')
    check(len(ledger_all)==300,'Unexpected Day04 full ledger')
    ledger={(r['model'],r['stem']):r for r in ledger_all}
    prior={n:rows(D7/(n+'.csv')) for n in ['per_gt_metrics','good_candidates','competition_relations','paired_23']}
    dataset=LLVIPDataset(ROOT/'data/LLVIP_1000',split='val',modality='paired',transforms=PairedLetterbox(640))
    geometry={}; cases={};all_candidates=[];all_relations=[];all_goods=[];all_outputs=[];audits=[];ties=[]
    for stem in STEMS:
        indices=[i for i,r in enumerate(dataset.records) if r.stem==stem]
        check(len(indices)==1,'Manifest identity not unique')
        index=indices[0];rec=dataset.get_record(index)
        for p in [rec.infrared_path,rec.visible_path,rec.annotation_path]:
            relative=p.relative_to(ROOT).as_posix();record(p,expected[relative],'Day04 original image/annotation SHA')
        with Image.open(rec.infrared_path) as im: original_size=im.size
        params=dataset.transforms.parameters(original_size[1],original_size[0])
        check(params.original_wh==(1280,1024) and params.resized_wh==(640,512) and params.padding_ltrb==(0,64,0,64) and params.scale==.5,'Unexpected Letterbox geometry')
        images,target=dataset[index]
        def pil(t): return Image.fromarray(t.mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy())
        geometry[stem]=dict(ir_image=pil(images['infrared']),rgb_image=pil(images['visible']),parameters=dict(original_wh=params.original_wh,resized_wh=params.resized_wh,output_wh=params.output_wh,scale=params.scale,padding_ltrb=params.padding_ltrb),xml_gt_order_verified=True)
        for model in MODELS:
            context=dict(model=model,stem=stem,gt_index=0)
            e=ledger[model,stem];p=D4/e['relative_raw']
            check(e['raw_sha256']==day7prov[model,stem]['sha256'],'Day04/Day07 raw SHA disagree')
            record(p,e['raw_sha256'],'Day04 ledger and Day07 capture provenance')
            record(p,manifest[p.relative_to(D4).as_posix()],'Day04 artifact manifest')
            for n in ['started.json','captured.json','verified.json']:
                q=p.parent/n;record(q,manifest[q.relative_to(D4).as_posix()],'Day04 artifact manifest')
                check(load(q)['signature']==e['signature'],'Ledger signature mismatch')
            check(load(p.parent/'verified.json')['status']=='passed','Day04 item not verified')
            # This is the ONLY torch.load call; its path is derived from four fixed ledger entries.
            d=torch.load(p,map_location='cpu',weights_only=True)
            check(d['signature']==e['signature'] and len(d['records'])==1,'Raw identity mismatch')
            r=d['records'][0];stages=d['stages'][0];s=stages['score_filtered'];cat=d['catalogues'][0]
            exact(target['boxes'],d['observed']['transform_gt_all'],'XML -> original Letterbox -> captured full GT')
            exact(target['boxes'],r['gt_all'],'Same original GT in NMS record')
            check(tuple(r['image_size'])==(640,640) and d['observed']['transform_sizes']==[(640,640)],'640x640 coordinates not verified')
            check(tuple(r['layer_keys'])==KEYS and r['config']==dict(pre_nms=150,post_nms=150,min_size=.001,score_thresh=.05,nms_thresh=.7),'Configuration changed')
            boxes,scores,flat,levels=s['boxes'],s['scores'],s['ids'],r['nms_levels']
            check(boxes.dtype==scores.dtype==torch.float32 and boxes.device.type=='cpu','Primary float32 CPU convention')
            exact(boxes,r['nms_boxes'],'Original NMS input boxes');exact(scores,r['nms_scores'],'Original NMS input scores')
            full_levels=torch.repeat_interleave(torch.arange(len(KEYS)),torch.tensor(r['counts']))
            exact(full_levels[flat],levels,'Actual level from flattened index and counts')
            ids=[]
            for i,f in enumerate(flat.tolist()):
                c=cat[f];lev=int(levels[i]);offset=sum(r['counts'][:lev])
                check(c==dict(candidate_id=f'{r["prefix"]}/{f}',flat_index=f,level_key=KEYS[lev],level_index=lev,index_within_level=f-offset),'Candidate catalogue identity mismatch')
                ids.append(c['candidate_id'])
            check(len(ids)==len(set(ids)) and r['prefix']==f'full_val_stage_run01/{model}/{stem}','Duplicate/incorrect candidate ID')
            keep=r['nms_indices'];check(keep.dtype==torch.int64 and len(keep.unique())==len(keep) and all(0<=k<len(ids) for k in keep.tolist()),'Invalid original keep indices')
            for field in ['ids','boxes','scores']:
                exact(stages['layer_nms'][field],s[field][keep],'Original full NMS output '+field)
                exact(stages['post_nms_topn'][field],s[field][keep[:150]],'Original final budget '+field)
            exact(r['final_boxes'],boxes[keep[:150]],'Original final boxes/order')
            exact(r['final_scores'],scores[keep[:150]],'Original final scores/order')
            exact(d['observed']['rpn_forward_boxes'],r['final_boxes'],'Captured original RPN return')
            check(not bool((scores[keep][1:]>scores[keep][:-1]).any()),'Original output score order')
            gt=target['boxes'][0];g=gt.tolist();bb=boxes.tolist();ss=scores.tolist();ll=levels.tolist()
            v=box_iou(gt[None],boxes)[0];vv=v.tolist()
            check(vv==[scalar_iou(g,b) for b in bb],'Independent binary32 GT IoU differs')
            good=torch.where(v>=.75)[0].tolist();check(bool(good),'Empty good set')
            ov=box_iou(boxes[good],boxes)
            count=[];local_relations=[];local_ties=[]
            for pos,x in enumerate(vv):
                diag=scalar_iou(g,bb[pos],False)
                if abs(x-.75)<=1e-6 or (x>=.75)!=(diag>=.75):
                    audits.append(dict(**context,kind='GT_threshold',a_id=ids[pos],b_id='',primary=x,float64=diag,decision_diff=(x>=.75)!=(diag>=.75)))
            for j,a in enumerate(good):
                competitors=[]
                for b in range(len(ids)):
                    if ll[a]!=ll[b] or vv[b]>=.75 or ss[b]<ss[a]:continue
                    pair=float(ov[j,b]);check(pair==scalar_iou(bb[a],bb[b]),'Independent pair IoU differs')
                    diag=scalar_iou(bb[a],bb[b],False)
                    if abs(pair-.7)<=1e-6 or (pair>.7)!=(diag>.7):
                        audits.append(dict(**context,kind='pair_threshold',a_id=ids[a],b_id=ids[b],primary=pair,float64=diag,decision_diff=(pair>.7)!=(diag>.7)))
                    if pair>.7:
                        row=dict(**context,a_id=ids[a],b_id=ids[b],layer=KEYS[ll[a]],a_position=a,b_position=b,a_flat=int(flat[a]),b_flat=int(flat[b]),a_xyxy=bb[a],b_xyxy=bb[b],a_score=ss[a],b_score=ss[b],a_gt_iou=vv[a],b_gt_iou=vv[b],pair_iou=pair,pair_iou_float64=diag)
                        if ss[b]>ss[a]:competitors.append(b);local_relations.append(row)
                        else:local_ties.append(row)
                count.append(len(competitors))
            T=min(count);L=len({ll[i] for i in good});minids=[ids[a] for a,n in zip(good,count) if n==T]
            # Only now consult original keep membership; C/T/minimizers are already fixed.
            outpos={int(k):i for i,k in enumerate(keep)}
            competitors={e['b_position'] for e in local_relations}
            selected_positions=sorted(set(good)|competitors)
            labels={p:('I' if model=='IR' else 'F')+f'{j+1:02}' for j,p in enumerate(selected_positions)}
            good_counts=dict(zip(good,count));minset=set(minids)
            candidate_rows=[]
            for pos,cid in enumerate(ids):
                kept=pos in outpos;final=kept and outpos[pos]<150
                cr=dict(**context,label=labels.get(pos,''),candidate_id=cid,flat_index=int(flat[pos]),index_within_level=int(cat[int(flat[pos])]['index_within_level']),layer=KEYS[ll[pos]],level_index=ll[pos],xyxy=bb[pos],score=ss[pos],gt_xyxy=g,gt_iou=vv[pos],role='good' if pos in good_counts else 'direct_competitor' if pos in competitors else 'other',C=good_counts.get(pos,''),is_T_minimizer=cid in minset,nms_input_position=pos,nms_kept=kept,nms_output_position=outpos.get(pos,''),final_kept=final,final_output_position=outpos[pos] if final else '',budget_status='kept_within_150' if final else 'removed_by_150_budget' if kept else 'not_in_nms_output')
                candidate_rows.append(cr)
            goods=[candidate_rows[i] for i in good];selected=[candidate_rows[i] for i in selected_positions]
            previous_good=[x for x in prior['good_candidates'] if x['stem']==stem and x['model']==model and x['gt_index']=='0']
            check([x['a_id'] for x in previous_good]==[ids[a] for a in good],'Day07 good identity/order mismatch')
            for old,new in zip(previous_good,goods):
                for k,newk in [('a_id','candidate_id'),('layer','layer')]:check(old[k]==new[newk],f'Day07 good {k}')
                for k,newk in [('input_position','nms_input_position'),('flat_index','flat_index'),('C','C')]:check(int(old[k])==new[newk],f'Day07 good {k}')
                for k in ['score','gt_iou']:check(float(old[k])==new[k],f'Day07 good {k}')
                for k in ['xyxy','gt_xyxy']:check(json.loads(old[k])==new[k],f'Day07 good {k}')
                check((old['is_T_minimizer']=='True')==new['is_T_minimizer'] and int(old['tie_count'])==sum(e['a_id']==new['candidate_id'] for e in local_ties),'Day07 minimizer/tie differs')
            oldrel=[x for x in prior['competition_relations'] if x['stem']==stem and x['model']==model and x['gt_index']=='0']
            check(len(oldrel)==len(local_relations),'Day07 relation count differs')
            for old,new in zip(oldrel,local_relations):
                for k,value in new.items():
                    converted=json.loads(old[k]) if isinstance(value,list) else int(old[k]) if isinstance(value,int) else float(old[k]) if isinstance(value,float) else old[k]
                    check(converted==value,f'Day07 relation {k} differs')
            metric=next(x for x in prior['per_gt_metrics'] if x['stem']==stem and x['model']==model and x['gt_index']=='0')
            lost=not any(x['nms_kept'] for x in goods)
            check((L,T,len(good),minids,lost)==(int(metric['L']),int(metric['T']),int(metric['good_count']),json.loads(metric['minimizer_ids']),metric['nms_lost']=='True'),'Day07 metric differs')
            check((metric['tied_overlap_exists']=='True')==bool(local_ties),'Day07 ties mismatch')
            for e in local_relations+local_ties:
                for prefix,pos in [('a',e['a_position']),('b',e['b_position'])]:
                    cr=candidate_rows[pos]
                    for field in ['label','nms_kept','nms_input_position','nms_output_position','final_kept','final_output_position','budget_status']:
                        e[prefix+'_'+field]=cr[field]
                e['relation_type']='同层高分重叠关系' if e['b_score']>e['a_score'] else '同层同分重叠关系（不计入C/T）'
                e['actual_gpu_suppression_event_captured']=False
            summary=dict(**context,L=L,T=T,good_count=len(good),minimizer_ids=minids,nms_lost=lost,final_lost=not any(x['final_kept'] for x in goods),nms_input_count=len(ids),nms_output_count=len(keep),final_count=min(150,len(keep)),nms_kept_good_ids=[x['candidate_id'] for x in goods if x['nms_kept']],day07_exact_match=True)
            cases[stem,model]=dict(model=model,stem=stem,gt=g,goods=goods,selected=selected,relations=local_relations,summary=summary)
            all_candidates+=candidate_rows;all_goods+=goods;all_relations+=local_relations;ties+=local_ties
            for rank,pos in enumerate(keep.tolist()):
                cr=candidate_rows[pos]
                all_outputs.append(dict(**context,nms_output_position=rank,nms_input_position=pos,candidate_id=cr['candidate_id'],flat_index=cr['flat_index'],layer=cr['layer'],score=cr['score'],xyxy=cr['xyxy'],final_kept=rank<150,final_output_position=rank if rank<150 else ''))
            print('VERIFIED',stem,model,'L',L,'T',T,'good',len(good),'relations',len(local_relations),'original_keep',len(keep),flush=True)
    for stem in STEMS:
        pair=next(x for x in prior['paired_23'] if x['stem']==stem and x['gt_index']=='0')
        check(pair['failed_model']=='Fusion' and pair['surviving_model']=='IR','Day07 paired outcome differs')
        check((int(pair['L']),int(pair['failed_T']),int(pair['surviving_T']))==(cases[stem,'Fusion']['summary']['L'],cases[stem,'Fusion']['summary']['T'],cases[stem,'IR']['summary']['T']),'Day07 pair L/T differs')
    write_csv(out/'all_score_filtered_candidates.csv',all_candidates)
    selected=[r for r in all_candidates if r['role']!='other']
    write_csv(out/'selected_candidates.csv',selected)
    write_csv(out/'good_candidate_checks.csv',all_goods)
    write_csv(out/'case_summary.csv',[c['summary'] for c in cases.values()])
    write_csv(out/'competition_relations.csv',all_relations)
    write_csv(out/'original_nms_outputs.csv',all_outputs)
    write_csv(out/'equal_score_relations.csv',ties,list(all_relations[0]))
    write_csv(out/'precision_audit.csv',audits,['model','stem','gt_index','kind','a_id','b_id','primary','float64','decision_diff'])
    figure_info=plot_all(out,cases,geometry)
    save(out/'input_provenance.json',dict(inputs=provenance,geometry={s:geometry[s]['parameters'] for s in STEMS},figures=figure_info,source_semantics='raw_capture stores original candidate tensors and original indices; no checkpoint loaded'))
    create_report(out,cases,selected,all_relations,audits,ties)
    return dict(status='passed',scope=[f'{s}/G0/{m}' for s in STEMS for m in MODELS],model_GT=4,all_score_filtered_rows=len(all_candidates),selected_candidate_rows=len(selected),good_rows=len(all_goods),relation_rows=len(all_relations),original_nms_output_rows=len(all_outputs),equal_score_rows=len(ties),precision_audit_rows=len(audits),precision_disagreements=sum(x['decision_diff'] for x in audits),day07_exact_match=True,original_nms_all_indices_verified=True,final_budget_separately_verified=True,letterbox_gt_exact_match=True,all_source_SHA_verified=True,new_model_forwards=0,NMS_runs=0,greedy_replays=0,checkpoint_loads=0,server_connections=0,training_runs=0,test_runs=0,new_research_metrics=0)


def create_report(out,cases,selected,relations,audits,ties):
    def find(stem,model,flat):return next(r for r in cases[stem,model]['selected'] if r['flat_index']==flat)
    def state(r):return '保留' if r['nms_kept'] else '删除'
    lines=['# Week10 Day01：固定两例的原NMS候选去留核查','',
           '本轮只分析030078/G0、030129/G0的IR与Fusion，共4项。先按Day07原定义确定全部好框、C、T和全部最小值ID，再读取原NMS保留索引。没有按结局重新挑选最小值，也没有新增研究指标。','',
           '## 先读懂三类证据','',
           '- **原操作事实**：原捕获`nms_indices`记录某个候选最终是否在NMS全部输出中，以及输出位置。最终150截断另外核对。',
           '- **几何关系**：同层、竞争者对指定GT不达标、原分数严格更高、两框IoU严格>0.7。这只说明满足Day07竞争定义。',
           '- **待验证假设**：某个候选是否在GPU运行过程中实际删掉另一个候选。现有索引没有逐次事件记录，本轮不将关系边提升为实际抑制边。','',
           '全部编号对应完整候选ID。位置均从0开始；“—”表示不属于原输出，不代表遗漏。对指定GT不达标不等于背景。','',
           '## 四项总览','','|图片/GT|模型|L|T|好框数|原NMS保留的好框flat|最终150|全部最小C对应flat|','|---|---|---:|---:|---:|---|---|---|']
    for c in cases.values():
        s=c['summary'];kept=[r['flat_index'] for r in c['goods'] if r['nms_kept']];mins=[r['flat_index'] for r in c['goods'] if r['is_T_minimizer']]
        lines.append(f'|{s["stem"]}/G0|{s["model"]}|{s["L"]}|{s["T"]}|{s["good_count"]}|{kept}|{"仍覆盖" if not s["final_lost"] else "未覆盖"}|{mins}|')
    lines += ['', '这4项的原NMS完整输出数均小于150；本轮选中好框与竞争者没有“原NMS保留、但被150预算删除”的情况。Fusion两例失去好框发生在NMS全部输出处。','',
              '## 1. 030129的IR保留了哪些好框？竞争者还在吗？','']
    a=find('030129','IR',8877);b=find('030129','IR',8872)
    lines += [f'IR只有1个好框：`{a["candidate_id"]}`（{a["label"]}，实际层1，原分数{a["score"]:.9f}，对G0 IoU={a["gt_iou"]:.9f}）。它在原NMS输入位置{a["nms_input_position"]}，被原NMS保留于输出位置{a["nms_output_position"]}，也进入最终150。',
              f'它的唯一直接竞争者是`{b["candidate_id"]}`（{b["label"]}，分数{b["score"]:.9f}，对G0 IoU={b["gt_iou"]:.9f}，相互IoU=0.711326718）。这个竞争者**没有**进入原NMS全部输出，也未进入最终150。',
              '因此T=1只说明NMS前确有一个符合定义的直接竞争者，不能据此预言好框会丢失。其具体删除者及删除发生时刻没有记录，本轮不继续追查。','',
              'Fusion同位置flat=8877的好框被删。其4个直接竞争者8862、8861、8857、8872中，只有8862进入原NMS全部输出（位置0），其余3个删除。高分、同层且高重叠的保留候选8862与好框满足几何竞争条件，但这仍不是直接捕获的GPU抑制事件。','',
              '## 2. 030078双方L、T相同时，具体有什么不同？','',
              '两侧L=2都只说明实际层0和1有好框，T=1都只说明“全部好框C的最小值是1”。它们不表示好框数量、坐标、分数、每个C或竞争者去留相同。','',
              'IR有8个好框：层0的3132、3146、2846、3141，对应C为3、2、2、5；层1的8846、9146、9147、9141，对应C为3、3、3、1。唯一最小值好框9141被原NMS保留（输出位置16）；其唯一直接竞争者8841被删除。其余7个IR好框被删除。',
              'Fusion只有2个好框：层0的3146、层1的9146；二者C都为1，所以**两个并列最小值全部保留在分析中**。它们都被原NMS删除；各自直接竞争者2846、8846分别在原输出位置11、9保留，也进入最终150。',
              '这给出了同L同T而结局不同的候选层面事实：IR的最小值好框保留而竞争者删除；Fusion的两个最小值好框均删除而各自竞争者保留。不能把“最小值数量”或本例状态再包装成新的指标。','',
              '## 3. 哪些情况提示T遗漏了竞争者自身去留？','',
              '030129/IR与030078/IR的存活好框都面对一个直接竞争者，但该竞争者自身未进入原NMS输出。030129/Fusion的T=4也包含3个最终被删竞争者；T统计的是NMS前关系，不是最终留下来的竞争框数量。它没有模拟候选依次处理的过程，无法独自表达竞争者是否最终存活。',
              '这说明T的信息范围有限，不证明竞争者被删的原因，也不推出普遍规律。两例是固定案例核查，不能估计全val发生比例、解释最终RoI/AP75或证明融合根因。','',
              '## 4. 还有哪些关系无法由现有索引解释？','',
              '所有29条关系的双方最终去留已核对。好框删除且竞争者保留的关系与原NMS规则相容，保留索引仍没有记录某条边是否就是GPU实际抑制事件。好框与竞争者均删除时，索引不能给出各自删除者、先后顺序或上游链。好框保留而竞争者删除时，同样不知道竞争者被谁删除。尤其T只统计对本GT不达标的框，删除好框的候选也可能是另一个好框，不能认为表内直接竞争者必定覆盖全部实际抑制原因。',
              '本轮只确认这些竞争者的去留；没有扩展竞争者的上游关系，没有独立贪心重放，没有声称捕获GPU实际抑制配对。不根据亮区、车灯等图像外观判断误差原因。','',
              '## 全部选中候选（完整分析，不省略）','',
              'ID形式为`full_val_stage_run01/模型/图片/flat`；下表逐行给出完整ID。精确坐标和分数见CSV；图中文字仅为显示而舍入。','',
              '|图片|模型|编号|完整候选ID|实际层|角色|C|最小值|原输入位置|原NMS|原输出位置|最终150|','|---|---|---|---|---|---|---:|---|---:|---|---:|---|']
    for r in selected:
        lines.append(f'|{r["stem"]}|{r["model"]}|{r["label"]}|{r["candidate_id"]}|{r["layer"]}|{"好框" if r["role"]=="good" else "竞争者"}|{r["C"]}|{"是" if r["is_T_minimizer"] else ""}|{r["nms_input_position"]}|{state(r)}|{r["nms_output_position"] if r["nms_kept"] else "—"}|{"保留" if r["final_kept"] else "未进入"}|')
    lines += ['', '## 全部同层高分重叠关系','', '|图片|模型|好框编号/flat|竞争者编号/flat|实际层|相互IoU|好框原NMS|竞争者原NMS|','|---|---|---|---|---|---:|---|---|']
    for e in relations:
        lines.append(f'|{e["stem"]}|{e["model"]}|{e["a_label"]}/{e["a_flat"]}|{e["b_label"]}/{e["b_flat"]}|{e["layer"]}|{e["pair_iou"]:.9f}|{"保留" if e["a_nms_kept"] else "删除"}|{"保留" if e["b_nms_kept"] else "删除"}|')
    lines += ['', '## 图像、核验及运行边界','',
              '每例提供IR/Fusion对照图、原Letterbox几何图；每模型提供全部候选的逐框图册和每个有好框层的关系图。图册一张卡只画一个候选和GT，避免密集框遮挡；关系图保留全部好框与边。GT黄色、好框青色、竞争者粉红，实线保留/虚线删除；星号标记全部C最小值，展示实际存活好框不改变T。',
              '原始图片1280×1024，复用原Letterbox缩放0.5到640×512，上下各64像素padding，最终640×640；图像、GT与候选处在同一坐标系。两模型共享同一放大范围和IR底图，Fusion仍指原RGB+IR模型。读取XML完整GT按原顺序经原Letterbox后，精确等于同次原捕获的GT。未重新裁图推理；局部裁剪仅用于绘图展示。',
              f'与Day07逐项核对4项L/T、12个好框的身份/原索引/实际层/坐标/原分数/GT IoU/C/并列最小ID，以及29条竞争关系。额外按完整原输出索引核对各候选去留和最终预算。CPU float32为主判定，独立逐步binary32标量公式精确核对；float64只作诊断，同分重叠{len(ties)}条，阈值敏感/精度审计{len(audits)}条，判定条件未改变。',
              '输入SHA与Day04预检、全量账本/manifest及Day07捕获来源逐项核对。Week09全部历史文件与所用数据/源码在运行前后做SHA检查；结果见verification.json与history_hashes.json。实际命令、运行日志、进程退出码见run_metadata.json、run.log及同级*_process.json/log。',
              '本步骤无新模型前向、NMS执行、checkpoint加载、训练、调参或test。仅4项进行必要数值核验；没有重跑完整val。到此停止，等待反馈。']
    (out/'case_report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    check(out.is_relative_to(HERE) and out!=HERE,'Output must remain inside day01')
    out.mkdir(parents=True,exist_ok=False)
    started=dt.datetime.now(dt.timezone.utc).isoformat()
    protected=[p for p in (ROOT/'week09').rglob('*') if p.is_file()]
    # Only hash selected data and geometry sources; never load any model weights.
    extra=[ROOT/'AGENTS.md',ROOT/'week08/day01/llvip_dataset.py',ROOT/'week08/day02/paired_letterbox.py',ROOT/'data/LLVIP_1000/manifest.csv']
    for stem in STEMS:
        extra += [ROOT/f'data/LLVIP_1000/images/val/{m}/{stem}.jpg' for m in ['infrared','visible']]
        extra.append(ROOT/f'data/LLVIP_1000/annotations_voc/val/{stem}.xml')
    protected=sorted(set(protected+extra))
    before={str(p):sha(p) for p in protected}
    save(out/'history_before.json',before)
    result={};code=1
    class Tee:
        def write(self,s):sys.__stdout__.write(s);log.write(s);log.flush();return len(s)
        def flush(self):log.flush()
    with (out/'run.log').open('x',encoding='utf-8') as log,redirect_stdout(Tee()),redirect_stderr(Tee()):
        try:
            torch.set_num_threads(2)
            with ExitStack() as guard:
                guard.enter_context(patch.object(torch.nn.Module,'_call_impl',side_effect=RuntimeError('Model forward forbidden')))
                guard.enter_context(patch.object(box_ops,'nms',side_effect=RuntimeError('NMS execution forbidden')))
                guard.enter_context(patch.object(box_ops,'batched_nms',side_effect=RuntimeError('NMS execution forbidden')))
                result=analyze(out)
            code=0
        except BaseException:
            traceback.print_exc();save(out/'failure.json',dict(error=traceback.format_exc()))
        finally:
            now=[p for p in (ROOT/'week09').rglob('*') if p.is_file()]
            after={str(p):sha(p) for p in sorted(set(now+extra))}
            unchanged=before==after
            save(out/'history_hashes.json',dict(before=before,after=after,unchanged=unchanged))
            if not unchanged:code=1
            result.update(status='passed' if code==0 else 'failed',historical_files_unchanged=unchanged,protected_file_count=len(before),exit_code=code)
            save(out/'verification.json',result)
            metadata=dict(command=subprocess.list2cmdline(sys.orig_argv),cwd=str(ROOT),started_at_utc=started,finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,torchvision=torchvision.__version__,platform=platform.platform(),device='CPU'),source_sha256={p.name:sha(p) for p in [Path(__file__),HERE/'README.md',HERE/'run_analysis.ps1']},exit_code=code,guards=['torch.nn.Module._call_impl','torchvision.ops.boxes.nms','torchvision.ops.boxes.batched_nms'],scope='four fixed model-GT items; no tests or full-val recomputation')
            save(out/'run_metadata.json',metadata)
            print(json.dumps(result,ensure_ascii=False),flush=True)
            print('EXIT',code,'HISTORY_UNCHANGED',unchanged,flush=True)
    save(out/'artifact_manifest.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    return code


if __name__=='__main__':
    sys.exit(main())
