"""Read-only Week10 synthesis. Standard library only; no historic entry point execution."""
import argparse,csv,datetime,hashlib,json,platform,subprocess,sys,traceback
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUNS=['candidate_review_run01','upstream_replay_run01','rpn_final_comparison_run01','roi_trace_run01','roi_nms_replay_run01','success_case_trace_run01']
PROVS=['input_provenance.json','input_provenance.json','input_provenance.json','provenance.json','input_provenance.json','input_provenance.json']
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def require(c,n):
 if not c:raise ValueError(n)
def write(out,n,s):
 with (out/n).open('x',encoding='utf-8') as f:f.write(s.strip()+'\n')
def run(out):
 bases={i:ROOT/f'week10/day{i:02}/outputs'/n for i,n in enumerate(RUNS,1)};records={};sources={};integrity=[];tables={};code={}
 def remember(p):sources[str(p)]=sha(p)
 for i,b in bases.items():
  day=b.parents[1];mf=load(b/'artifact_manifest.json');remember(b/'artifact_manifest.json')
  bad=[n for n,h in mf.items() if not (b/n).is_file() or sha(b/n)!=h]
  require(not bad,f'Day{i} manifest mismatch {bad}')
  v=load(b/'verification.json');m=load(b/'run_metadata.json');log=(b/'run.log').read_text(encoding='utf-8-sig')
  require(v['status']=='passed' and v['exit_code']==m['exit_code']==0 and 'EXIT 0' in log,f'Day{i} completion evidence failed')
  for n in ['README.md']:
   (day/n).read_text(encoding='utf-8-sig');remember(day/n)
  (b/'case_report_zh.md').read_text(encoding='utf-8-sig')
  for n in mf:remember(b/n)
  prov=load(b/PROVS[i-1])
  if isinstance(prov,dict):prov=prov['inputs']
  badprov=[x['path'] for x in prov if sha(x['path'])!=x['sha256']];require(not badprov,f'Day{i} provenance changed')
  for x in prov:remember(Path(x['path']))
  sh=m['source_sha256']
  if isinstance(sh,dict):
   for name,h in sh.items():require(sha(day/name)==h,f'Day{i} code changed');remember(day/name)
  else:
   candidates=[p for p in day.glob('*.py') if sha(p)==sh];require(bool(candidates),f'Day{i} source SHA missing')
  code[i]='\n'.join(p.read_text(encoding='utf-8-sig') for p in day.glob('*.py'))
  hist=load(b/'history_hashes.json');require(hist['unchanged'] and hist['before']==hist['after'],f'Day{i} historical snapshot failure')
  for p in b.glob('*.csv'):tables[i,p.name]=read(p)
  process=b.parent/(b.name+'_process.json');process_record=load(process) if process.exists() else None
  if process_record:require(process_record['exit_code']==0,f'Day{i} outer exit');remember(process)
  integrity.append(dict(day=i,status=v['status'],exit_code=v['exit_code'],metadata_exit_code=m['exit_code'],log_exit_zero=True,manifest_files_verified=len(mf),provenance_files_verified=len(prov),source_SHA_verified=True,process_record=process_record))
  records[i]=(v,m,process_record)
 save(out/'source_integrity.json',integrity)
 evidence=[]
 def ev(day,name,conclusion,selector,etype,boundary,fields=None):
  rows=[r for r in tables[day,name] if all(r.get(k)==str(v) for k,v in selector.items())]
  require(bool(rows),f'Evidence selector absent {day}/{name} {selector}')
  evidence.append(dict(id=f'E{len(evidence)+1:02}',conclusion=conclusion,day=day,scope=dict(model=selector.get('model','IR'),image=selector.get('stem','030129' if day in [4,5] else '030078' if day==6 else 'two fixed images'),GT='G0'),run=str(bases[day]),source_path=str(bases[day]/name),row_identity=selector,field_names=fields or list(rows[0]),exact_original_values=rows if fields is None else [{k:r[k] for k in fields} for r in rows],evidence_type=etype,boundary=boundary))
 actual='1_actual_capture_identity_or_index_fact';rebuild='2_reconstruction_checked_against_actual_tensor';replay='3_full_output_matched_independent_replay';observe='4_descriptive_observation_or_unverified_hypothesis'
 ev(1,'case_summary.csv','同L/T不等于完整结构与去留相同；030078双方L2 T1',{'stem':'030078'},observe,'L/T是描述性汇总，不是存活保证')
 ev(1,'selected_candidates.csv','030129/IR好框8877保留，直接竞争者8872删除',{'model':'IR','stem':'030129'},actual,'只确认原集合，不给出删除事件')
 ev(1,'selected_candidates.csv','030078/IR好框9141保留，竞争者8841删除',{'model':'IR','stem':'030078'},actual,'T不含竞争者自身去留')
 ev(2,'case_relations.csv','两项上游几何与原去留',{},actual,'几何不是GPU事件')
 ev(2,'replay_steps.csv','完整RPN重放的步骤与首次删除者',{},replay,'步号与归因仅属于重放，原GPU未保存逐次事件')
 ev(2,'case_verification.csv','80/65完整输入分别匹配51/36完整输出序列',{},replay,'同输出不证明内核事件相同')
 ev(3,'summary.csv','4项CPU同次RPN与最终检测端点',{},actual,'不同端点最大IoU不是身份映射；不是AP75')
 ev(3,'historical_differences.csv','历史GPU与CPU坐标/分数及排序差异',{'stem':'030129','model':'IR'},observe,'逐位置差不等于按身份差，不拼接历史链')
 ev(4,'trace_8877.csv','8877回归定位下降后在类别NMS删除',{},actual,'分类分数不是objectness，不说明根因')
 ev(4,'stage_coverage.csv','另外5框回归达标，NMS处全部失去覆盖',{},actual,'只限030129/IR/G0')
 ev(4,'stage_candidates.csv','分数与背景展开重建精确接到实际小框输入',{'stage':'score_filtered'},rebuild,'该阶段索引不是直接捕获，已与下一实际张量核对')
 ev(4,'final_sources.csv','最终最贴合来自2571，最高分类分数来自8862',{},actual,'来源按索引链，非几何猜测')
 ev(5,'focus_attributions.csv','5好框首次删除分为8862四项、2571一项，8877非好框参照',{},replay,'不能称原CPU内核事件')
 ev(5,'output_comparison.csv','44输入完整重放匹配3个原输出及顺序',{},replay,'完整序列核验后才作归因')
 ev(6,'trace_9141.csv','9141回归改善到0.7943789958953857，随后被NMS删除',{},actual,'不能据索引单独命名实际抑制者')
 ev(6,'final_sources.csv','最终达标来源8847，原RPN不达标，经回归达标',{'output_index':0},actual,'不是9141的后续框')
 ev(6,'stage_coverage.csv','保住0.75覆盖不等于保住最高IoU框',{},observe,'单例描述，不估计总体率')
 ev(6,'two_case_comparison.csv','两例IR回归后各5好框，NMS后1与0',{},observe,'两图不是控制变量实验；Fusion内部尚未追踪')
 for e in evidence:
  rr=e['exact_original_values']
  e['scope']['model']=sorted({r['model'] for r in rr if 'model' in r}) or ['IR']
  e['scope']['image']=sorted({r['stem'] for r in rr if 'stem' in r}) or (['030129'] if e['day'] in [4,5] else ['030078'] if e['day']==6 else ['030078','030129'])
 save(out/'evidence_index.json',dict(evidence_types=[actual,rebuild,replay,observe],numeric_precision='CSV original numeric strings retained without rounding; prose may round',claims=evidence))
 # One authoritative field per count. Duplicates in top-level/counts are checked, never added.
 specs={1:('new_model_forwards','checkpoint_loads','greedy_replays',None),2:('new_model_forwards','checkpoint_loads','main_replay_calls',None),3:('validation_full_forwards','checkpoint_loads',None,'structure_backbone_probes'),4:('counts.full_model_forwards','counts.checkpoint_loads','counts.independent_nms_replays','counts.structure_probes'),5:('counts.model_forwards','counts.checkpoint_loads','counts.main_replays',None),6:('counts.full_model_forward_calls','counts.checkpoint_loads','counts.independent_nms_replays','counts.structure_probes')}
 def field(v,k):
  if k is None:return 0
  for t in k.split('.'):v=v[t]
  return int(v)
 ledger=[]
 for i,(v,m,process) in records.items():
  f,c,n,p=specs[i];counts=dict(validation_full_forwards=field(v,f),checkpoint_loads=field(v,c),independent_main_nms_replays=field(v,n),structure_probes=field(v,p))
  duplicate=[]
  for k,value in v.get('counts',{}).items():
   if k in v:require(v[k]==value,'Duplicate count disagreement');duplicate.append(k)
  for k,value in m.get('counts',{}).items():
   if k in v.get('counts',{}):require(v['counts'][k]==value,'Metadata count disagreement')
  for k in ['training','training_runs','test_inference','test_runs','server_connections']:
   if k in v:require(v[k]==0,'Nonzero prohibited operation')
   if k in v.get('counts',{}):require(v['counts'][k]==0,'Nonzero prohibited operation')
  ledger.append(dict(day=i,run=str(bases[i]),counts=counts,count_sources=dict(validation_full_forwards=f,checkpoint_loads=c,independent_main_nms_replays=n or '0: source contains no independent replay entry',structure_probes=p or '0: no restore/probe in source'),source_file=str(bases[i]/'verification.json'),duplicate_fields_counted_once=duplicate,command=m['command'],environment=m.get('environment',m.get('current_environment')),actual_started_at=m.get('started_at_utc',(process or {}).get('started_at')),actual_finished_at=m.get('finished_at_utc',(process or {}).get('finished_at')),time_note='null means not recorded; task Day number is not calendar date',candidate_capture_reads_note='saved tensor reads are not checkpoint loads; no aggregate guessed where counter absent',original_model_nms_note='Day06 explicit RPN=1 and RoI=1' if i==6 else 'model forwards execute original RPN/RoI NMS; separate call counter not uniformly recorded, do not treat independent replay=0 as no original NMS',zero_operations_evidence='explicit zero fields where present; otherwise read-only source control-flow and completed scoped run, not an independent universal counter',training=0,test_inference=0,server_connections=0))
 totals={k:sum(r['counts'][k] for r in ledger) for k in ledger[0]['counts']}
 require(totals==dict(validation_full_forwards=6,checkpoint_loads=4,independent_main_nms_replays=3,structure_probes=1),'Ledger differs from expected; do not force values')
 save(out/'execution_ledger.json',dict(scope='Week10 Day01-Day06 task chain only; Week09 executions excluded',days=ledger,total=totals,unique_model_image_combinations=4,actual_execution_distribution={'IR/030078':2,'IR/030129':2,'Fusion/030078':1,'Fusion/030129':1},training=0,test_inference=0,server_connections=0,limitations='Some original-model NMS calls and saved tensor reads lack uniform standalone counters; preserved as notes, not fabricated counts',day07_new=dict(model_forwards=0,checkpoint_loads=0,nms_replays=0)))
 # Exact-value consistency checks across exports (no tensors loaded or analyses re-executed).
 row=lambda day,name,**kw:next(r for r in tables[day,name] if all(r.get(k)==str(v) for k,v in kw.items()))
 t4=row(4,'trace_8877.csv',stage='person_decoded');t6=row(6,'trace_9141.csv',stage='person_decoded');f6=row(6,'final_sources.csv',output_index=0)
 require(t4['gt_iou']==str(records[4][0]['target_person_gt_iou']),'Day04 exact number mismatch')
 require(f6['gt_iou']==str(records[6][0]['final_good_sources'][0]['gt_iou']),'Day06 final number mismatch')
 require(f6['flat']=='8847' and row(6,'trace_9141.csv',stage='model_output')['present']=='False','Identity contradiction')
 require(row(5,'focus_attributions.csv',source_flat=8571)['first_deleter_flat']=='2571','Replay attribution mismatch')
 for f in [8576,9156,12831,2576,8877]:require(row(5,'focus_attributions.csv',source_flat=f)['first_deleter_flat']=='8862','Replay attribution mismatch')
 consistency=[]
 for r in tables[6,'two_case_comparison.csv']:
  d=row(3,'summary.csv',model=r['model'],stem=r['stem'])
  for a,b in [('rpn_max_iou','rpn_max_iou'),('final_best_iou','final_correct_max_iou'),('final_best_iou_score','final_best_iou_score'),('final_count','final_count')]:
   require(r[a]==d[b],f'Cross-day exact field mismatch {r["stem"]}/{a}')
   consistency.append(dict(stem=r['stem'],comparison_field=a,day03_field=b,value=r[a],exact_equal=True))
 for r in tables[2,'case_verification.csv']:
  require(all(r[k]=='True' for k in ['full_sequence_equal','layer_nms_fields_equal','final150_equal']),'RPN replay incomplete')
 for r in tables[5,'focus_attributions.csv']:require(r['original_and_replay_agree']==r['attribution_validated']=='True','RoI replay invalid')
 save(out/'cross_document_checks.json',dict(exact_cross_day_fields=consistency,identity_checks=True,replay_full_output_checks=True))
 plan='''# Week11计划（本轮未执行）

第一步：只补齐030078/G0、030129/G0的Fusion RoI身份链，最多新增2次validation完整前向。原Fusion best路径为`D:\\python\\test\\vision-learning\\week08\\day05\\outputs\\fusion_concat_run01\\best.pth`，SHA256为`9f7bbdbb374861e3f8e191b3e03dff6f64502beee30b4dc754458ebce49f963a`；来源依据为Day03恢复源码/来源SHA、严格加载及模型配置记录。执行前重新核对文件SHA，沿用原FrozenBN、RGB+IR配对输入和strict=True，模型只恢复一次，依次处理两图。Fusion恢复自带一次零张量backbone结构探测应单列，不能计入2次验证图前向。不要重新推理已有IR。

捕获原RPN实际输出及flat顺序、RoI实际输入、predictor logits/regression、背景/person概率、原decode/clip、去背景展开、严格分类分数索引、小框索引、完整类别NMS输入与返回、100预算及模型返回。原函数原样调用；离线重建须与下一实际张量精确核对。固定RPN 0.05/0.7/pre150/post150和RoI 0.001/0.5/max100，不加argmax，不调参。

与Day03同图同模型的Fusion_030078_raw_capture.pth、Fusion_030129_raw_capture.pth核对完整RPN序列、按flat对齐框与分数、完整最终boxes/scores/labels及顺序、GT、坐标、配置、checkpoint与状态。不得只比最大IoU。若不精确复现，保存差异，标为本次新运行；不增加前向尝试、不调整容差。身份链或模型状态等必要核验失败则停止并交付未通过项。

核验后用真实表与已有IR同图同GT对照：原RPN好框、回归后好框、分数过滤/小框/NMS前后覆盖、最高分类分数框、最高IoU框的实际身份与去留。不能预先假设Fusion失败原因。此步骤仍是已选案例的探索性诊断，不是独立验证。先审阅结果，再决定是否需要另行授权的离线NMS解释；本计划不默认追加重放。

第二步（另行立项与预算）：若有价值，预先冻结原manifest中全部150张validation图片、IR/Fusion各一次，完整300个模型×图片项及既有416个有效GT/模型。固定阶段、IoU≥0.75口径、空集合约定和主要问题：好框在何阶段出现/消失，最高分类分数分支与保留下来的覆盖有何对应。采用现有字段，不持续按结果选例或增设新指标。先保存manifest/SHA、字段清单和执行预算再运行。

执行预算应按捕获是否完整计算：目前IR已有2份完整RoI捕获；第一步再得到2份Fusion完整捕获后，可复用4项，剩余296项需最多296次新增完整前向。以当前状态合计第一步2次加后续296次，共最多298次新增；不是300次全部重跑。实际兼容性核对失败的旧捕获不能自动重跑，先列缺口再重新授权。原Week09 Day04的300份捕获只有RPN流水线，Day03终点捕获没有完整RoI内部身份阶段，均可作为参照，不能替代剩余RoI捕获。预算按每个模型一次恢复；如分块因资源需多次恢复，应事先明确块数与对应checkpoint加载/结构探测次数，不伪装为图片前向。

同一validation已反复用于选模、选例和提出假设，不能改称独立验证集。上述扩展仍属探索性描述。当前尚无真正独立数据的确认性评估；如未来获得独立数据，应先冻结假设与方案，再单独设计确认性评估。不把test用于反复探索或调方案。训练、改阈值、删候选反事实和模型改进均不在本轮计划执行范围。'''
 write(out,'week11_plan.md',plan)
 daily=[
 ('Day01','为何同L/T仍可有不同结局？','还原实际候选比继续增设指标更能回答问题。复用Day04 GPU候选，仅离线核对4项。','T只计NMS前直接高分竞争关系，未表达竞争者自身是否保留。030078双方L2/T1，IR有8好框而Fusion有2；不能把摘要值当成完整结构。','几何边不能命名GPU实际抑制事件。'),
 ('Day02','被删竞争者如何与好框存活相容？','完整输入独立重放后比较完整序列，避免只验证目标。两项RPN主重放，无新前向。','030129的8862在step0首次删除8872；030078的8847在step4首次删除8841。原输出分别51/36，全序列匹配；各自好框8877/9141保留。','步号与首次删除者是重放解释，不是GPU事件。'),
 ('Day03','RPN覆盖是否传递到最终检测？','两图×IR/Fusion4次CPU同次捕获RPN与完整输出。','030129/IR的RPN好框仍在，最终最大IoU仅0.700329，且对应分类分数0.002493；最高分框IoU约0.598007。历史GPU/CPU数值与部分排序不同。','两个最大IoU不是同一候选身份，不能拼接历史链或推断RoI根因。'),
 ('Day04','030129/IR的8877到底去了哪里？','单次CPU前向用真实RoI行与实际索引建立身份链，精确复现Day03。','8877从0.766334回归到0.625646，分数0.995233，随后在类别NMS删除；另外5框回归达标，最高0.908178，经过分数和小框过滤，NMS后全部丢失；最终最贴合来自2571。','定位下降和筛选删除不同；仍不能只凭原索引给出内核实际抑制者。'),
 ('Day05','5个回归好框为何不在原NMS输出？','读取44个原RoI类别NMS输入，按分类分数与类别分组独立重放一次。','完整匹配3个原索引[0,7,23]：4个好框由8862在step0首次删除，8571由2571在step2首次删除；8877为非好框参照也由8862删除。','这只是经完整输出核验的独立重放归因，不是内核事件跟踪，不是干预因果证据。'),
 ('Day06','成功覆盖是否就是原RPN好框一直保留？','030078/IR单次前向精确复现Day03，030129仅读表对照。','9141从0.757896改善到0.794379，分类分数0.945614，但被NMS删除。最终达标来自8847：RPN0.723833→回归0.758343，分数0.981489。两例NMS前各5好框，之后1与0。','保住0.75覆盖不等于保住最高IoU框；两例均IR，不能写成Fusion内部失败根因已解释。')]
 hand=['# Week10权威技术交接','', '范围：只含这条Week10固定任务链。Day01—Day06实际产物、日志、退出码、manifest与来源已复核后才整理。Day编号是学习任务编号，实际时间/环境/命令见execution_ledger.json；缺失起止时间保留null，不编造工作时长。','']
 for title,q,why,fact,limit in daily:hand.extend([f'## {title}',f'问题：{q}',f'为什么与方法：{why}',f'事实：{fact}',f'边界：{limit}',''])
 hand+=['## 计数与状态','',f'原字段去重后累计：完整validation前向{totals["validation_full_forwards"]}次、checkpoint加载{totals["checkpoint_loads"]}次、Fusion恢复结构探测{totals["structure_probes"]}次、独立主NMS重放{totals["independent_main_nms_replays"]}次。唯一模型×图片组合4项，实际执行6次（两项IR各复现一次）；候选张量读取不是checkpoint加载。原模型内部NMS与独立重放分开，原NMS缺少统一独立计数时不编造总数。训练/test推理/服务器连接均为0；部分天没有统一零计数字段，依据执行源码控制流与限定运行，详见账本限制说明。','', '## 精确证据入口','']
 for e in evidence:hand.append(f'- {e["id"]}：{e["conclusion"]}；源文件`{e["source_path"]}`；行身份`{json.dumps(e["row_identity"],ensure_ascii=False)}`；完整精确字段见evidence_index.json。')
 hand+=['','## 统一解释规则','', '候选与GT的IoU≥0.75用于指定GT覆盖；候选与候选的NMS IoU阈值是RPN按层0.7、RoI按类别0.5，不能混用。RPN objectness不是RoI分类分数。重建已与实际张量相符不等于直接捕获；重放全输出相符不等于内核事件记录。两个端点最大IoU不能构造身份链。只保住覆盖不等于保住最佳定位。两例不构成总体率、因果效应或AP75结果。','', '## 未解决与下一步','', 'Fusion内部RoI尚未追踪；模型为什么形成这些分数与回归量、对不同场景能否成立、模型改进能否提升性能、真正独立数据上的效果，均未解决。Week10诊断步骤完成不等于研究目标实现。优先补齐两图Fusion的最多2次前向，见week11_plan.md；本轮未执行该计划。']
 hand+=['','## 同次CPU终点对照（原表精度）','','历史GPU值、CPU值分列；下表是端点诊断，不建立候选回归身份。', '', '| 模型/图片/G0 | 历史RPN最大IoU | CPU RPN最大IoU | 最终最大IoU | 对应分类分数 | RPN/最终覆盖 |','|---|---:|---:|---:|---:|---|']
 for r in tables[3,'summary.csv']:hand.append(f'| {r["model"]}/{r["stem"]}/G0 | {r["historical_rpn_max_iou"]} | {r["rpn_max_iou"]} | {r["final_correct_max_iou"]} | {r["final_best_iou_score"]} | {r["rpn_covered"]}/{r["final_covered"]} |')
 hand+=['','## 两例身份追踪对照（原表精度）','','| IR图片/G0 | 原RPN好框 | 该框回归IoU/分类分数 | 该框最终去留 | 最终最贴合来源 | 最终IoU/分类分数 | NMS前/后好框数 |','|---|---|---|---|---|---|---|']
 for r in tables[6,'two_case_comparison.csv']:hand.append(f'| {r["stem"]}/G0 | {r["original_rpn_good_identity"]} | {r["original_rpn_good_decoded_iou"]} / {r["original_rpn_good_person_score"]} | 删除 | {r["final_best_iou_identity"]} | {r["final_best_iou"]} / {r["final_best_iou_score"]} | {r["nms_before_good_count"]}/{r["nms_after_good_count"]} |')
 hand+=['','## 按天执行账本','','| 学习任务 | validation完整前向 | checkpoint加载 | 结构探测 | 独立主NMS重放 |','|---|---:|---:|---:|---:|']
 for r in ledger:
  c=r['counts'];hand.append(f'| Day{r["day"]:02} | {c["validation_full_forwards"]} | {c["checkpoint_loads"]} | {c["structure_probes"]} | {c["independent_main_nms_replays"]} |')
 hand+=['| 合计 | 6 | 4 | 1 | 3 |','','精确计数字段、命令、环境、时间缺口与去重依据见execution_ledger.json。Day07读取整理不增加上述执行次数。']
 write(out,'WEEK10_GPT_HANDOFF.md','\n'.join(hand))
 learning='''# 用两个真实案例串起本周所学

GT是真实标注框，是我们判断定位是否贴合的参照。本周始终问指定G0有没有至少一个IoU达到0.75的框。这个问题不同于“模型是否把所有人都检测正确”，也不是AP75：我们没有在这里进行正式的一对一匹配或数据集级精度汇总。

IoU有两种比较对象。框和GT比较，回答定位贴合程度；框和另一个候选比较，则服务于NMS的重复框筛选。一个框可能比另一个框更贴合G0，但只要分类分数较低，又与高分候选高度重叠，仍会在NMS过程中被删。RPN按特征层分组、用objectness和0.7重叠阈值；RoI按类别分组、用分类分数和0.5阈值。这两套分数与条件不能互换。

RPN先产生值得进一步看的候选区域。RoI再针对每个区域提取特征，输出类别logits和各类别的回归偏移。回归把原框调整为类别分支框，分类logits经softmax成为类别概率。分类和回归是不同输出：推理时先算出的分类概率不会因为随后解码框对GT的IoU变高就自动重新计算或升高。GT根本没有参与推理，所以模型也不能当场知道哪个框最贴合G0。训练中两个任务有关联，不意味着每个候选的分数与最终IoU严格一致。

回归是模型预测的调整，不是由GT提供的保证。对030129/IR，原RPN好框8877的IoU约0.766334，回归后反而降到0.625646；它的person分类分数仍高达0.995233。另一方面，其他候选经回归变为5个好框，其中2576达到0.908178，但分类分数仅0.006389。所有5个框均通过分数和小框过滤，却在类别NMS处失去覆盖。经完整输出核验的离线重放说明：4个首次由8862删除，另一个由2571删除。这是能重现原输出的过程解释，不是直接看到了CPU内核里的每一次抑制事件。

030078/IR则提醒我们不要把“最终成功”理解为“最好的那个框保住了”。原RPN唯一好框9141从0.757896回归到0.794379，定位改善，但它也在类别NMS中被删除。最终对G0达标的框来自另一候选8847：原RPN只有0.723833，回归后达到0.758343，分类分数为0.981489，并作为最高分类分数输入保留。最终框和原RPN最大IoU接近，并不能证明它们有相同来源；本周通过真实RoI行和索引确认了来源不同。

两例都没有保住NMS前定位最贴合的框，030078仍留下了一个达到0.75的框，而030129没有。这种差别只能由已选两个案例具体说明，不能概括为成功图所有候选的分类与定位都更一致，更不能归纳为Fusion根因。内部追踪目前只完成IR。

本周真正学会的是逐层缩小问题：先看覆盖在哪个阶段变化，再确认候选真实身份，最后用完整输出核验重放解释。我们增加了过程理解和可审阅证据，尚未通过训练、改结构或独立数据评估证明模型得到改进。下一步先补齐同两张图的Fusion身份链，未来计划与已经完成的结果保持分开。
'''
 write(out,'learning_review_zh.md',learning)
 notes='''# Week10周报Markdown素材

## 本周工作

本周围绕两个固定validation案例，逐步核查了RPN候选竞争、完整NMS输出以及RoI后处理中的候选身份。先复用已有捕获还原L/T，再对两个IR案例完成输出匹配的RPN独立重放；随后通过同次CPU前向比较RPN与最终检测，分别追踪失败和成功IR案例的实际RoI分支。最后对失败例的44个类别NMS输入完成一次独立重放。研究范围保持固定，没有训练或调阈值。

## 实际结果

T没有包含竞争者本身能否存活的信息，同L/T并不意味着相同结局。030129/IR的原RPN好框回归后定位下降，但另有5个分支回归达标；这些好框在类别NMS后全部消失。030078/IR的原RPN好框回归改善后仍被删除，最终保留覆盖的是原本RPN不达标、回归后达标的8847。两例均未保住最高IoU框，最终覆盖分别为有和无。相关身份均由真实行号和索引链确认，重放归因经过完整输出序列核验。

本链条实际累计6次validation完整前向、4次checkpoint加载、1次Fusion恢复结构探测、3次独立主NMS重放；训练、test推理和服务器连接均为0。任务Day编号不是自然日期，本材料不虚构每天工作时长。准确原始数值和日志位置附在证据索引、执行账本中。

## 目前问题

Fusion内部RoI追踪尚未补齐；同例IR与Fusion的内部差异尚不能直接解释。本周观察限于选定案例，没有形成全validation比例、因果贡献或AP75提升结论。分类分数与定位质量不完全一致的形成原因、可行干预及独立数据上的效果仍待验证。

## 下周计划（未执行）

优先严格复用原Fusion best，对同两张固定图补齐最多2次前向的RoI身份链，与Day03终点及同图IR表核对。若复现失败则保留差异，不增加尝试。之后再评估是否冻结全150图validation、两模型的完整阶段诊断；先固定样本和口径、明确预算，再执行探索性扩展。已有validation不能改称独立验证集，不使用test反复调方案。Week10诊断完成不代表整体研究目标已实现。
'''
 write(out,'weekly_report_notes_zh.md',notes)
 # Preserve a compact table of exact values rather than relying on rounded prose.
 save(out/'exact_case_values.json',dict(failure_8877=t4,success_9141=t6,success_final=f6,day03_endpoints=tables[3,'summary.csv'],day05_attributions=tables[5,'focus_attributions.csv'],day06_comparison=tables[6,'two_case_comparison.csv']))
 save(out/'input_provenance.json',[dict(path=p,sha256=h) for p,h in sorted(sources.items())])
 return dict(status='passed',days_verified=6,manifest_files_verified=sum(x['manifest_files_verified'] for x in integrity),evidence_claims=len(evidence),execution_totals=totals,document_consistency_checks=True,unpassed_items=[],limitations=['Day03 start/finish timestamps absent where null in ledger','Some days zero operations and original-model NMS lack uniform direct counters; code/scoped-run evidence distinguished','Fusion internal RoI not yet traced','No independent confirmatory evaluation'],new_model_forwards=0,new_checkpoint_loads=0,new_nms_replays=0,week11_executed=False)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();require(out.is_relative_to(HERE) and out!=HERE,'output scope');out.mkdir(parents=True,exist_ok=False)
 folders=[ROOT/'week08',ROOT/'week09']+[ROOT/f'week10/day{i:02}' for i in range(1,7)]
 extra=[ROOT/'AGENTS.md',Path(r'C:\Users\GXBBBXB\.codex\memory\research-notes.md')]
 def snapshot():return {str(p):sha(p) for p in sorted(set(extra+[p for b in folders for p in b.rglob('*') if p.is_file()]))}
 before=snapshot();save(out/'history_before.json',before);result={};code=1;started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 class Tee:
  def write(self,s):sys.__stdout__.write(s);log.write(s);log.flush();return len(s)
  def flush(self):log.flush()
 with (out/'run.log').open('x',encoding='utf-8') as log,redirect_stdout(Tee()),redirect_stderr(Tee()):
  try:result=run(out);code=0
  except BaseException:traceback.print_exc();save(out/'failure.json',dict(error=traceback.format_exc()))
  finally:
   after=snapshot();unchanged=before==after;save(out/'history_hashes.json',dict(before=before,after=after,unchanged=unchanged))
   if not unchanged:code=1
   result.update(status='passed' if code==0 else 'failed',historical_files_and_memory_unchanged=unchanged,protected_files=len(before),exit_code=code);save(out/'verification.json',result)
   save(out/'run_metadata.json',dict(command=subprocess.list2cmdline(sys.orig_argv),cwd=str(ROOT),started_at_utc=started,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),environment=dict(python=sys.version,platform=platform.platform(),imports='Python standard library only'),source_sha256=sha(Path(__file__)),exit_code=code));print(json.dumps(result,ensure_ascii=False),flush=True);print('EXIT',code,flush=True)
 save(out/'artifact_manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()});return code
if __name__=='__main__':sys.exit(main())
