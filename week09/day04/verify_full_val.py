"""Downloaded full-val independent coordinate/statistics audit; never loads weights/forwards/NMS."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch
sys.dont_write_bytecode=True
import torch
from torchvision.ops import boxes as ops
import verify_pilot as independent
from offline_full_gt import ROOT,load_json,sha,read_csv,save_json

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capture-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.capture_dir.resolve();dest=args.output.resolve()
    if not dest.is_relative_to(Path(__file__).resolve().parent):raise ValueError('Day04 output only')
    dest.mkdir(parents=True,exist_ok=False);code=1
    try:
        before={str(p):sha(p) for p in source.rglob('*') if p.is_file()}
        manifest=load_json(source/'artifact_manifest.json')
        assert all(sha(source/n)==h for n,h in manifest.items())
        original_meta=load_json(source/'run_metadata.json')
        assert original_meta['status']=='completed' and original_meta['exit_code']==0 and original_meta['new_forward_calls']==288
        assert original_meta['historical_sha256_before']==original_meta['historical_sha256_after']
        local_history=load_json(ROOT/'week09/day04/outputs/six_image_all_gt_run02/run_metadata.json')['historical_sha256_before']
        assert all(sha(p)==h for p,h in local_history.items())
        # Existing independent numeric audit is reused without altering its source or any input file.
        # Only its loaded path/protection view is adapted for server->local transfer.
        original_load=independent.load
        def normalized_load(path):
            data=original_load(path)
            if path==source/'run_metadata.json':
                data=copy.deepcopy(data)
                data['historical_sha256_before']=local_history
                data['historical_sha256_after']=local_history
            elif path==source/'reuse_evidence.json':
                data=copy.deepcopy(data)
                for c in data['captures']:c['raw_path']=str(source/c['relative_raw'])
            return data
        with patch.object(independent,'load',side_effect=normalized_load), \
             patch.object(torch.nn.Module,'_call_impl',side_effect=RuntimeError('Model forward forbidden')), \
             patch.object(ops,'nms',side_effect=RuntimeError('NMS forbidden')), \
             patch.object(ops,'batched_nms',side_effect=RuntimeError('NMS forbidden')):
            info=independent.verify(source)
        entries=read_csv(source/'forward_ledger.csv')
        assert len(entries)==300 and len({(e['model'],e['stem']) for e in entries})==300
        assert sum(int(e['new_forward_calls']) for e in entries)==288
        assert sum(e['origin']=='reused_day03' for e in entries)==12
        for e in entries:
            raw=source/e['relative_raw'];assert sha(raw)==e['raw_sha256']
            if e['origin']=='reused_day03':
                relative=e['source_raw'].replace('\\','/').split('week09/',1)[1]
                assert sha(ROOT/'week09'/relative)==sha(raw)
            else:
                folder=source/e['ledger_relative'];data=torch.load(raw,weights_only=True,map_location='cpu')
                started=load_json(folder/'started.json');captured=load_json(folder/'captured.json');receipt=load_json(folder/'verified.json')
                assert data['signature']==started['signature']==captured['signature']==receipt['signature']==e['signature']
                assert captured['raw_sha256']==receipt['raw_sha256']==sha(raw)
                v=data['verification']
                assert v['forward_calls']==1 and not v['forward_error'] and v['state_sha256_before']==v['state_sha256_after']
                assert all(v[k] for k in ('state_equal_to_initial','config_unchanged','restoration_verified','same_device_exact_replay'))
                assert data['records'][0]['gt_all'].shape==data['observed']['transform_gt_all'].shape
        v=load_json(source/'verification.json')
        assert v['actual']=={'IR':{'images':150,'valid_gt':416,'covered_075':302},'Fusion':{'images':150,'valid_gt':416,'covered_075':285}}
        assert len(v['day01_comparisons'])==832 and all(c['coverage_equal'] and abs(c['actual_difference'])<=1e-6 for c in v['day01_comparisons'])
        assert info['per_gt_stage_rows_verified']==5824 and info['model_gt_items']==832
        assert before=={str(p):sha(p) for p in source.rglob('*') if p.is_file()}
        info.update(scope='full_validation_150_images_per_model',actual=v['actual'],new_forward_calls_verified=288,reused_model_image_items=12,
            archive_artifact_files_sha256_verified=len(manifest),server_protected_files_unchanged=len(original_meta['historical_sha256_before']),
            local_historical_files_unchanged=len(local_history),
            path_normalization='In-memory raw path rebasing; numeric audit local protection view uses preserved pilot snapshot. Original server before/after snapshot equality checked separately.',
            max_day01_abs_difference=v['max_abs_day01_difference'],
            independent_numeric_source_sha256=sha(Path(independent.__file__)),verification_source_sha256=sha(Path(__file__)))
        code=0
    except BaseException:info={'status':'failed','error':traceback.format_exc()}
    info.update(command=subprocess.list2cmdline(sys.orig_argv),exit_code=code)
    save_json(dest/'verification.json',info)
    with (dest/'run.log').open('x',encoding='utf-8') as f:f.write(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(info,ensure_ascii=False,indent=2));return code
if __name__=='__main__':sys.exit(main())
