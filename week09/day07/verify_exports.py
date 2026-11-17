"""Independent exported C/T, pairing and summary audit; raw geometry separately checked in main run."""
import argparse,csv,json,hashlib,sys,traceback,subprocess
from pathlib import Path
from collections import Counter,defaultdict
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert a.output.resolve().is_relative_to(Path(__file__).resolve().parent)
    a.output.mkdir(parents=True,exist_ok=False);code=1
    try:
        d=a.input;meta=json.loads((d/'run_metadata.json').read_text(encoding='utf-8'));assert meta['exit_code']==0 and meta['historical_unchanged']
        assert all(sha(p)==h for p,h in meta['historical_before'].items())
        assert all(sha(p)==h for p,h in meta['input_sha256'].items())
        assert sha(Path(__file__).with_name('analyze_competition.py'))==meta['source_sha256']
        gt=read(d/'per_gt_metrics.csv');good=read(d/'good_candidates.csv');rels=read(d/'competition_relations.csv');ties=read(d/'equal_score_relations.csv')
        key=lambda r:(r['model'],r['stem'],r['gt_index'])
        G={key(r):r for r in gt};assert len(G)==len(gt)==594
        count=Counter();tc=Counter();seen=set()
        for rows,target in [(rels,count),(ties,tc)]:
          for r in rows:
            rk=key(r)+(r['a_id'],r['b_id']);assert rk not in seen;seen.add(rk)
            assert float(r['a_gt_iou'])>=.75 and float(r['b_gt_iou'])<.75 and float(r['pair_iou'])>.7
            assert (float(r['b_score'])>float(r['a_score'])) if rows is rels else (float(r['b_score'])==float(r['a_score']))
            target[key(r)+(r['a_id'],)]+=1
        groups=defaultdict(list);goodkeys=set()
        for r in good:
            k=key(r)+(r['a_id'],);assert k not in goodkeys;goodkeys.add(k)
            assert int(r['C'])==count[k] and int(r['tie_count'])==tc[k]
            assert r['layer'] in ('0','1','pool');groups[key(r)].append(r)
        assert set(groups)==set(G)
        for k,rr in groups.items():
            g=G[k];T=min(int(r['C']) for r in rr);assert int(g['T'])==T and len(rr)==int(g['good_count'])
            assert json.loads(g['minimizer_ids'])==[r['a_id'] for r in rr if int(r['C'])==T]
            assert (g['tied_overlap_exists']=='True')==any(int(r['tie_count']) for r in rr)
        pairs=read(d/'paired_23.csv')
        source=read(Path('week09/day06/outputs/layer_support_run01/all_common_gt_pairs.csv'))
        selected=[r for r in source if r['failed_model'] and int(r['IR_L'])==int(r['Fusion_L'])]
        assert len(pairs)==len(selected)==23
        for p,s in zip(pairs,selected):
            assert (p['stem'],p['gt_index'],p['failed_model'])==(s['stem'],s['gt_index'],s['failed_model'])
            f=G[p['failed_model'],p['stem'],p['gt_index']];v=G[p['surviving_model'],p['stem'],p['gt_index']]
            delta=int(f['T'])-int(v['T']);assert int(p['delta_T'])==delta
            assert p['direction']==('positive' if delta>0 else 'negative' if delta<0 else 'equal')
        for s in read(d/'paired_summary.csv'):
            rr=[p for p in pairs if (s['scene_id']=='ALL' or s['scene_id']==p['scene_id']) and (s['failed_model']=='ALL' or s['failed_model']==p['failed_model'])]
            assert int(s['GT'])==len(rr)
            for direction in ('positive','equal','negative'):
                n=sum(p['direction']==direction for p in rr);assert int(s[direction])==n
                assert float(s[direction+'_rate'])==n/len(rr) if rr else s[direction+'_rate']==''
        for s in read(d/'model_scene_T_summary.csv'):
            rr=[g for g in gt if g['model']==s['model'] and g['T']==s['T'] and (s['scene_id']=='ALL' or s['scene_id']==g['scene_id'])]
            n=len(rr);lost=sum(g['nms_lost']=='True' for g in rr)
            assert int(s['risk_GT'])==n and int(s['nms_lost_GT'])==lost and float(s['loss_rate'])==lost/n
        info=dict(status='passed',model_GT=594,pairs=23,good_rows=len(good),strict_relations=len(rels),tie_relations=len(ties),directions=dict(Counter(p['direction'] for p in pairs)),historical_files_unchanged=len(meta['historical_before']),source_sha256=sha(__file__))
        code=0
    except BaseException:info=dict(status='failed',error=traceback.format_exc())
    info.update(exit_code=code,command=subprocess.list2cmdline(sys.orig_argv))
    (a.output/'verification.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'run.log').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2));return code
if __name__=='__main__':sys.exit(main())
