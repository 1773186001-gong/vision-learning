"""Refresh Day07 using saved evidence only. Python standard library; no tensor loads."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def save(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    out = ap.parse_args().output.resolve()
    if not out.is_relative_to(HERE / 'outputs') or out.exists():
        raise ValueError('Require a new output directory under Day07/outputs')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    old = {str(p): sha(p) for p in HERE.rglob('*') if p.is_file()}
    # This is the already inspected Day07 standard-library aggregation entry.
    # No Day01-Day06 analysis entry is executed.
    command = [sys.executable, '-B', '-X', 'utf8', str(HERE / 'build_review.py'), '--output', str(out)]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    print(proc.stdout, end='')
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end='')
    if not out.exists():
        raise RuntimeError('Aggregation failed before output creation')
    save(out / 'aggregation_process.json', dict(command=command, exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr))
    code = proc.returncode
    try:
        if code:
            raise RuntimeError('Aggregation did not pass; preserve evidence and stop')
        evidence = load(out / 'evidence_index.json')
        for day, folder, fields in [
            (4, 'roi_trace_run01', ['status', 'day03_exact_reproduced', 'identity_chain_verified']),
            (6, 'success_case_trace_run01', ['status', 'day03_exact_reproduced', 'identity_chain_verified'])
        ]:
            base = ROOT / f'week10/day{day:02}/outputs' / folder
            v = load(base / 'verification.json')
            # Preserve exact recorded keys; differing schemas are not silently filled.
            matches = {k: value for k, value in v.items() if 'reproduc' in k.lower() or 'identity' in k.lower()}
            if not any(value is True for k, value in matches.items() if 'reproduc' in k.lower()):
                raise ValueError(f'Day{day} reproduction evidence missing')
            evidence['claims'].append(dict(
                id=f'E{len(evidence["claims"])+1:02}', conclusion=f'Day{day:02}完整复现Day03且身份链通过',
                scope=dict(model=['IR'], image=['030129' if day == 4 else '030078'], GT='G0'),
                run=str(base), source_path=str(base / 'verification.json'),
                row_identity='top-level exact reproduction and identity fields', field_names=list(matches),
                exact_original_values=matches, evidence_type='1_actual_capture_identity_or_index_fact',
                boundary='依据已完成的完整序列/张量核验记录；本轮只复核SHA和记录，不重新执行张量分析'))
        for claim in evidence['claims']:
            if claim.get('day') in [4, 6] and claim['source_path'].endswith(('stage_coverage.csv', 'trace_8877.csv', 'trace_9141.csv')):
                claim['mixed_stage_note'] = '实际阶段与经核对的离线重建并存。background_removed_expanded、score_filtered和预算切片是重建；实际RPN、decode/clip、NMS完整输入/索引与模型返回是捕获事实。详见原表evidence列及原报告。'
        save(out / 'evidence_index.json', evidence)
        # Confirm the proposed extension's fixed scope from the historical ledger,
        # without counting its Week09 executions as Week10 executions.
        historical = ROOT / 'week09/day04/outputs/full_val_stage_run01/verification.json'
        full = load(historical)
        assert all(full['actual'][m]['images'] == 150 and full['actual'][m]['valid_gt'] == 416 for m in ['IR', 'Fusion'])
        scope = dict(source_path=str(historical), sha256=sha(historical), fields={'actual.IR': full['actual']['IR'], 'actual.Fusion': full['actual']['Fusion']}, note='Only scope evidence; historical Week09 execution counts excluded')
        save(out / 'week11_scope_source.json', scope)
        planpath = out / 'week11_plan.md'
        plan = planpath.read_text(encoding='utf-8')
        plan = plan.replace('若不精确复现，保存差异，标为本次新运行；不增加前向尝试、不调整容差。身份链或模型状态等必要核验失败则停止并交付未通过项。', '若任一完整复现核验不一致，立即保存差异与同次捕获，标为本次新运行，并停止本步骤；第一图失败时不继续第二图。不得追加前向尝试或调整容差，也不把本次内部过程写成Day03的过程。身份链或模型状态等必要核验失败同样停止，交付未通过项。')
        plan += '\n\n范围数字来源：`week09/day04/outputs/full_val_stage_run01/verification.json`的`actual.IR/Fusion.images=150`、`valid_gt=416`，SHA与精确字段另见week11_scope_source.json。这些历史执行不计入Week10账本。若第一步与后续扩展分开执行，第一步Fusion恢复1次、结构探测1次；后续296项按IR/Fusion各恢复1次计新增checkpoint加载2次、Fusion结构探测1次。分块导致的额外恢复必须执行前单列预算，不自行扩大前向次数。\n'
        planpath.write_text(plan, encoding='utf-8')
        handpath = out / 'WEEK10_GPT_HANDOFF.md'
        hand = handpath.read_text(encoding='utf-8')
        hand += '\n\n## 本轮重新核验\n\n本轮在已有Day07结果基础上新建run04，不覆盖run01—run03。原报告、表格、来源、日志及源码均已只读复核。Day04/Day06完整复现记录补充见证据E19/E20。记忆文件仍称Week10未实施，仅作为过期索引，本轮没有修改它。不同任务Day不是自然日期；本次整理时间记录在refresh_metadata.json，不反推各天工作日期或时长。\n'
        handpath.write_text(hand, encoding='utf-8')
        (out / 'README.md').write_text('''# Week10 Day07：本轮正式整理交付

本目录为新建的week10_review_run04。已有run01—run03及Day01—Day06均保持不变。本轮只读汇总和SHA/字段核验，模型前向、checkpoint加载、NMS重放均为0；未执行Week11计划，未生成Word。

建议依次阅读：WEEK10_GPT_HANDOFF.md（技术交接）、learning_review_zh.md（初学者总结）、weekly_report_notes_zh.md（周报素材）、week11_plan.md（未执行计划）。精确结论、来源路径与字段见evidence_index.json；每日执行及计数依据见execution_ledger.json。来源SHA见input_provenance.json，历史前后检查见history_hashes.json和previous_day07_hashes.json。

验证入口：verification.json、source_integrity.json、cross_document_checks.json及artifact_manifest.json。aggregation_process.json记录本轮只读汇总子进程命令和退出码；refresh_metadata.json记录最终交付脚本命令、时间、环境和源码SHA。两者都是整理进程，不是新增研究实验。

项目根目录使用以下入口，再次执行必须选择尚不存在的新目录：

```powershell
& D:\\python\\.venv\\Scripts\\python.exe -B -X utf8 week10/day07/refresh_review.py --output week10/day07/outputs/week10_review_run05
```

核验通过仅表示本轮整理及既有证据一致，不代表模型已改进。缺失的历史起止时间、未统一记录的原模型NMS/捕获读取计数保持缺口，不能用推测填充。Fusion内部RoI尚未追踪，尚无独立确认性评估。本轮完成后停止。
''', encoding='utf-8')
    except BaseException:
        code = 1
        save(out / 'refresh_failure.json', dict(error=traceback.format_exc()))
        print(traceback.format_exc(), file=sys.stderr)
    same = all(Path(p).exists() and sha(p) == h for p, h in old.items())
    save(out / 'previous_day07_hashes.json', dict(before=old, after={p: sha(p) if Path(p).exists() else None for p in old}, unchanged=same))
    if not same:
        code = 1
    v = load(out / 'verification.json')
    v.update(status='passed' if code == 0 else 'failed', exit_code=code, previous_day07_unchanged=same, evidence_claims=len(load(out / 'evidence_index.json')['claims']), aggregation_exit_code=proc.returncode, refresh_completed=code == 0)
    if code:
        v.setdefault('unpassed_items', []).append('refresh finalization or historical Day07 check')
    save(out / 'verification.json', v)
    save(out / 'refresh_metadata.json', dict(command=subprocess.list2cmdline(sys.orig_argv), started_at_utc=started, finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), timezone_note='Timestamps stored as UTC; user timezone Asia/Shanghai', environment=dict(python=sys.version, platform=platform.platform(), imports='Python standard library only'), source_sha256={str(HERE / 'refresh_review.py'): sha(HERE / 'refresh_review.py'), str(HERE / 'build_review.py'): sha(HERE / 'build_review.py')}, exit_code=code))
    with (out / 'run.log').open('a', encoding='utf-8') as f:
        f.write(f'\nRefresh finalization: previous_day07_unchanged={same}; evidence_claims={v["evidence_claims"]}; EXIT {code}\n')
    save(out / 'artifact_manifest.json', {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'})
    print(json.dumps(v, ensure_ascii=False))
    print('REFRESH EXIT', code)
    return code

if __name__ == '__main__':
    sys.exit(main())
