"""Pure queue/relationship/GT-summary logic. Never invokes NMS or a model."""
from collections import defaultdict
import json
import struct

SCENES=('03','08','13','17')
def f32(v):return struct.unpack('f',struct.pack('f',v))[0]
THRESHOLD=.7
THRESHOLD32=f32(THRESHOLD)
NEAR=1e-6  # Diagnostic flag only, never a threshold relaxation.
def ratio(n,d):return n/d if d else None
def truth(value):
    if value not in ('True','False',True,False):raise ValueError('Invalid bool')
    return value is True or value=='True'

def relationship(same_level,deleted_score,retained_score,iou32,iou64,shifted_iou32):
    a,b,c=f32(iou32)>THRESHOLD32,iou64>THRESHOLD,f32(shifted_iou32)>THRESHOLD32
    eligible=same_level and retained_score>=deleted_score
    near=eligible and min(abs(f32(iou32)-THRESHOLD32),abs(iou64-THRESHOLD),abs(f32(shifted_iou32)-THRESHOLD32))<=NEAR
    return {'same_level':same_level,'score_relation':'equal' if retained_score==deleted_score else 'higher' if retained_score>deleted_score else 'lower',
        'score_tie':retained_score==deleted_score,'eligible_same_layer_score':eligible,
        'raw_float32_strict_overlap':a,'raw_float64_strict_overlap':b,'shifted_float32_strict_overlap':c,
        'primary_possible_relation':eligible and a,
        'diagnostic_only_relation':eligible and not a and (b or c),
        'near_threshold_diagnostic_only':near,
        'relation_predicate_disagreement':eligible and len({a,b,c})>1}

def build_queue(rows):
    grouped=defaultdict(dict)
    for row in rows:
        key=row['model'],row['stem'],int(row['gt_index'])
        if row['stage'] in grouped[key]:raise ValueError('Duplicate model/GT/stage')
        grouped[key][row['stage']]=row
    events=[];paired=[]
    keys={k[1:] for k in grouped if k[0]=='IR'}
    if keys!={k[1:] for k in grouped if k[0]=='Fusion'}:raise ValueError('Unpaired GT keys')
    for stem,gi in sorted(keys):
        states={};scene=None;coords=None
        for model in ('IR','Fusion'):
            ss=grouped[model,stem,gi]
            if len(ss)!=7 or 'score_filtered' not in ss or 'layer_nms' not in ss:raise ValueError('Incomplete stages')
            pre,post=ss['score_filtered'],ss['layer_nms']
            if scene is not None and (scene!=pre['scene_id'] or coords!=pre['gt_xyxy']):raise ValueError('Paired GT identity differs')
            scene,coords=pre['scene_id'],pre['gt_xyxy']
            a,b=truth(pre['covered_075_float32_cpu']),truth(post['covered_075_float32_cpu'])
            if b and not a:raise ValueError('NMS deletion created coverage')
            states[model]=(a,b,a and not b)
            if a and not b:
                events.append({'event_id':f'{model}/{stem}/G{gi}','model':model,'stem':stem,'gt_index':gi,'scene_id':scene,
                    'image_index':int(pre['image_index']),'gt_xyxy':coords,
                    'day04_pre_max_iou_f32':float(pre['max_iou_float32_cpu']),
                    'day04_nms_max_iou_f32':float(post['max_iou_float32_cpu']),
                    'day04_pre_high_count':int(pre['count_iou_ge_075_float32_cpu']),
                    'day04_nms_high_count':int(post['count_iou_ge_075_float32_cpu'])})
        a,b=states['IR'],states['Fusion']
        paired.append({'stem':stem,'gt_index':gi,'scene_id':scene,'IR_precovered':a[0],'Fusion_precovered':b[0],
            'IR_after_nms_covered':a[1],'Fusion_after_nms_covered':b[1],'IR_nms_lost':a[2],'Fusion_nms_lost':b[2],
            'both_precovered':a[0] and b[0],
            'IR_risk_state':'not_at_risk' if not a[0] else 'lost' if a[2] else 'still_covered',
            'Fusion_risk_state':'not_at_risk' if not b[0] else 'lost' if b[2] else 'still_covered',
            'loss_pair_category':'both_lost' if a[2] and b[2] else 'IR_only_lost' if a[2] else 'Fusion_only_lost' if b[2] else 'neither_lost'})
    return events,paired

def pair_summary(paired):
    rows=[]
    for scene in ('ALL',)+SCENES:
        for cohort in ('all_gt','both_precovered'):
            pp=[r for r in paired if (scene=='ALL' or r['scene_id']==scene) and (cohort=='all_gt' or r['both_precovered'])]
            n=len(pp);ir=sum(r['IR_nms_lost'] for r in pp);fu=sum(r['Fusion_nms_lost'] for r in pp)
            ip=sum(r['IR_precovered'] for r in pp);fp=sum(r['Fusion_precovered'] for r in pp)
            rows.append({'scene_id':scene,'cohort':cohort,'GT_denominator':n,'IR_precovered_gt':ip,'Fusion_precovered_gt':fp,
                'IR_lost_gt':ir,'Fusion_lost_gt':fu,'IR_conditional_loss_rate':ratio(ir,ip),'Fusion_conditional_loss_rate':ratio(fu,fp),
                **{cat:sum(r['loss_pair_category']==cat for r in pp) for cat in ('both_lost','IR_only_lost','Fusion_only_lost','neither_lost')},
                'Fusion_minus_IR_lost_gt':fu-ir,
                'same_risk_loss_rate_difference':ratio(fu-ir,n) if cohort=='both_precovered' else None,
                'interpretation':'same_GT_comparison_not_causation' if cohort=='both_precovered' else 'all_GT_loss_states_with_risk_flags; own-risk rates have different denominators'})
    return rows

def model_summary(events,deleted,relations,paired):
    rows=[]
    for scene in ('ALL',)+SCENES:
      for model in ('IR','Fusion'):
        selected=lambda r:r['model']==model and (scene=='ALL' or r['scene_id']==scene)
        ee=[e for e in events if selected(e)];dd=[r for r in deleted if selected(r)];rr=[r for r in relations if selected(r)]
        pp=[p for p in paired if scene=='ALL' or p['scene_id']==scene]
        pre=sum(p[model+'_precovered'] for p in pp)
        rows.append({'scene_id':scene,'model':model,'all_valid_gt':len(pp),'nms_precovered_gt':pre,'nms_lost_gt':len(ee),
            'conditional_loss_rate':ratio(len(ee),pre),
            'gt_with_primary_relation':sum(e['has_primary_relation'] for e in ee),
            'gt_every_deleted_good_has_primary_relation':sum(e['all_deleted_good_have_primary_relation'] for e in ee),
            'gt_partly_or_fully_unexplained_primary':sum(e['unexplained_deleted_good_count']>0 for e in ee),
            'gt_without_any_primary_relation':sum(not e['has_primary_relation'] for e in ee),
            'gt_with_primary_score_tie':sum(e['has_primary_score_tie'] for e in ee),
            'gt_with_eligible_score_tie':sum(e['has_eligible_score_tie'] for e in ee),
            'gt_near_nms_threshold':sum(e['near_nms_threshold'] for e in ee),
            'gt_relation_predicate_disagreement':sum(e['relation_predicate_disagreement'] for e in ee),
            'gt_numerically_sensitive':sum(e['numerically_sensitive'] for e in ee),
            'gt_with_diagnostic_only_relation':sum(e['has_diagnostic_only_relation'] for e in ee),
            'gt_with_related_retained_other_gt_overlap_larger':sum(e['any_related_retained_other_gt_overlap_larger'] for e in ee),
            'deleted_good_candidate_GT_rows':len(dd),'unique_deleted_good_candidates':len({r['deleted_candidate_id'] for r in dd}),
            'primary_relation_GT_rows':len(rr),
            'unique_primary_candidate_pairs':len({(r['deleted_candidate_id'],r['retained_candidate_id']) for r in rr}),
            'selection_warning':'Post-NMS max IoU < .75 by queue definition; worse-retained IoU is not independent evidence'})
    return rows
