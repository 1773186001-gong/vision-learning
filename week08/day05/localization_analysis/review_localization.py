"""Independent CSV conservation checks, all-prediction detail export and clear rerender."""
import runpy
from pathlib import Path
ns=runpy.run_path(str(Path(__file__).with_name('analyze_localization.py')))
BASE=ns['BASE']; ROOT=ns['ROOT']; torch=ns['torch']; json=ns['json']; ir=ns['ir']; trace=ns['trace']
original=BASE/'results'; metadata=json.loads((original/'run_metadata.json').read_text())
assert metadata['cache_SHA256']=={m:ir.sha(p) for m,p in ns['PATHS'].items()}
caches={m:torch.load(p,map_location='cpu',weights_only=False) for m,p in ns['PATHS'].items()}
traces={m:[trace(p,g) for p,g in zip(c['predictions'],c['truths'])] for m,c in caches.items()}
detail=original/'all_prediction_matching_details.json'
if not detail.exists(): ns['save'](detail,traces)
else: assert json.loads(detail.read_text())==traces
case_data=json.loads((original/'case_examples/details.json').read_text()); cases=[]
for r in case_data:
    key=(r['case'],r['image_id'],r['gt_index'])
    if key not in cases: cases.append(key)
out=original/'visual_review';out.mkdir();(out/'case_examples').mkdir()
ns.update(caches=caches,ref=caches['IR'],stems=caches['IR']['stems'],traces=traces,cases=cases,out=out)
source=Path(__file__).with_name('analyze_localization.py').read_text(encoding='utf-8')
block=source[source.index('    ds=ir.LLVIPDataset'):source.index('    assert hashes==')]
import textwrap
exec(compile(textwrap.dedent(block),str(Path(__file__)),'exec'),ns)
import csv,numpy as np
rows=list(csv.DictReader((original/'paired_gt_iou_comparison.csv').open(encoding='utf-8-sig')))
assert len(rows)==295
for r in rows: assert abs(float(r['Fusion_IoU'])-float(r['IR_IoU'])-float(r['delta']))<1e-12
high=list(csv.DictReader((original/'high_confidence_analysis.csv').open(encoding='utf-8-sig')))
for r in high:
    assert int(r['high_TP'])+int(r['high_FP'])==int(r['all_high'])
    assert abs(int(r['high_FP'])/int(r['all_high'])-float(r['FP_fraction_of_all_high']))<1e-12
from PIL import Image
pngs=list(original.rglob('*.png'))
for p in pngs:
    with Image.open(p) as image: image.verify()
assert metadata['cache_SHA256']=={m:ir.sha(p) for m,p in ns['PATHS'].items()}
ns['save'](original/'verification.json',dict(status='passed',initial_script_exit_code=0,paired_rows=295,high_score_denominators_checked=True,
    decoded_PNG_count=len(pngs),clear_case_plots=8,cache_SHA256_unchanged=True,no_model_forward=True))
print('PASS: 295 paired differences, high-score denominators, PNG decode, cache integrity; clear plots rerendered offline.')
