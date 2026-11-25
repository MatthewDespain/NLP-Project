#!/usr/bin/env python3
"""Consolidate six frozen-prompt Ollama attribution runs. Standard library only."""
import csv, json, collections, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parent / "model_evaluation"
INPUTS={
'qwen3:1.7b':'attribution_qwen3_1_7b_57facts_v2.csv',
'qwen3:8b':'attribution_qwen3_8b_57facts_v2.csv',
'llama3.2:3b':'attribution_llama3_2_3b_57facts_v2.csv',
'llama3.1:8b':'attribution_llama3_1_8b_57facts_v2.csv',
'gemma3:4b':'attribution_gemma3_4b_57facts_v2.csv',
'gemma3:12b':'attribution_gemma3_12b_57facts_v2.csv',
}
LABELS=['SUPPORTED','PARTIALLY_SUPPORTED','INSUFFICIENT','CONTRADICTED']
SCORED=LABELS[:3]
DEV={'A001','A002','A003','A004','A005','A006','A007','A008','A009','A012','A014','A025','A026','A028'}
def read(name):
 with (ROOT/name).open(newline='',encoding='utf-8-sig') as h:return list(csv.DictReader(h))
def write(name, rows, columns):
 with (ROOT/name).open('w',newline='',encoding='utf-8') as h:
  w=csv.DictWriter(h,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
def metrics(rows):
 n=len(rows); cm=collections.Counter((r['reference_label'],r['predicted_label']) for r in rows)
 result=[]
 for label in SCORED:
  tp=cm[label,label]; support=sum(v for (a,b),v in cm.items() if a==label)
  predicted=sum(v for (a,b),v in cm.items() if b==label)
  precision=tp/predicted if predicted else 0
  recall=tp/support if support else 0
  f1=2*tp/(support+predicted) if support+predicted else 0
  result.append({'class':label,'support':support,'predicted_count':predicted,'tp':tp,'precision':precision,'recall':recall,'f1':f1})
 return {'n':n,'correct':sum(a==b for (a,b),v in cm.items() for _ in range(v)),'accuracy':sum(v for (a,b),v in cm.items() if a==b)/n if n else 0,'macro_f1':sum(x['f1'] for x in result)/3,'per_class':result,'cm':cm}
allrows=[]; summary=[]; classes=[]; conf=[]; errors=[]; ref=None
for model,name in INPUTS.items():
 rows=read(name)
 assert len(rows)==57, (model,len(rows))
 assert len({r['sample_id'] for r in rows})==57
 assert all(r['model']==model and r['prompt_version']=='attribution_top5_v2' for r in rows)
 assert all(r['reference_label'] in SCORED and r['predicted_label'] in LABELS and not r['error'].strip() for r in rows)
 mapping={r['sample_id']:(r['fact_id'],r['reference_label']) for r in rows}
 if ref is None:ref=mapping
 assert mapping==ref,model
 for r in rows:
  r['evaluation_subset']='development_14' if r['sample_id'] in DEV else 'untouched_43'
  r['is_correct']=int(r['reference_label']==r['predicted_label'])
  allrows.append(r)
  if not r['is_correct']:
   errors.append({k:r.get(k,'') for k in ['model','sample_id','fact_id','reference_label','predicted_label','similarity_stratum','explanation','quality_flags']})
 m=metrics(rows); dev=metrics([r for r in rows if r['sample_id'] in DEV]); untouched=metrics([r for r in rows if r['sample_id'] not in DEV]); times=[float(r['elapsed_seconds']) for r in rows]
 summary.append({'model':model,'n':m['n'],'correct':m['correct'],'accuracy':round(m['accuracy'],6),'macro_f1_3class':round(m['macro_f1'],6),'development_correct':dev['correct'],'development_n':dev['n'],'development_accuracy':round(dev['accuracy'],6),'untouched_correct':untouched['correct'],'untouched_n':untouched['n'],'untouched_accuracy':round(untouched['accuracy'],6),'pred_supported':sum(r['predicted_label']=='SUPPORTED' for r in rows),'pred_partial':sum(r['predicted_label']=='PARTIALLY_SUPPORTED' for r in rows),'pred_insufficient':sum(r['predicted_label']=='INSUFFICIENT' for r in rows),'pred_contradicted':sum(r['predicted_label']=='CONTRADICTED' for r in rows),'quality_warning_rows':sum(bool(r['quality_flags'].strip()) for r in rows),'total_seconds':round(sum(times),2),'mean_seconds':round(statistics.mean(times),2),'median_seconds':round(statistics.median(times),2)})
 for x in m['per_class']:
  classes.append({'model':model,**{k:round(v,6) if isinstance(v,float) else v for k,v in x.items()}})
 for actual in SCORED:
  for predicted in LABELS:
   conf.append({'model':model,'reference_label':actual,'predicted_label':predicted,'count':m['cm'][actual,predicted]})
assert len(allrows)==342 and len(DEV)==14 and len(ref)==57
write('attribution_all_342_predictions_v2.csv',allrows,list(allrows[0]))
write('attribution_model_summary_v2.csv',summary,list(summary[0]))
write('attribution_class_metrics_v2.csv',classes,list(classes[0]))
write('attribution_confusion_matrices_v2.csv',conf,list(conf[0]))
write('attribution_misclassifications_v2.csv',errors,list(errors[0]))
# One row per fact, all model predictions, and a disagreement count.
wide=[]
by_id=collections.defaultdict(dict)
for r in allrows:by_id[r['sample_id']][r['model']]=r
for sid in sorted(ref):
 rr=by_id[sid]; predictions=[rr[model]['predicted_label'] for model in INPUTS]
 wide.append({'sample_id':sid,'fact_id':ref[sid][0],'reference_label':ref[sid][1],'subset':'development_14' if sid in DEV else 'untouched_43','distinct_prediction_labels':len(set(predictions)),'correct_models':sum(p==ref[sid][1] for p in predictions),**{model:rr[model]['predicted_label'] for model in INPUTS}})
write('attribution_per_fact_model_comparison_v2.csv',wide,list(wide[0]))
lines=['# SPARK attribution evaluation — six Ollama models','', '## Evaluation setup','', 'All models used the same frozen `attribution_top5_v2` prompt and five evidence windows per fact. Reference: 57 adjudicated facts (36 SUPPORTED, 14 PARTIALLY_SUPPORTED, 7 INSUFFICIENT); 3 excluded facts were not scored. Six models × 57 facts = 342 predictions. The 14 previously inspected development examples are separated from 43 previously untouched examples.','', '## Model results','', '| Model | Correct / 57 | Accuracy | Macro-F1 (3 classes) | Untouched correct / 43 | Mean seconds / fact | Warnings |','|---|---:|---:|---:|---:|---:|---:|']
for s in sorted(summary,key=lambda x:-x['accuracy']):
 lines.append(f"| {s['model']} | {s['correct']}/57 | {s['accuracy']:.1%} | {s['macro_f1_3class']:.1%} | {s['untouched_correct']}/43 ({s['untouched_accuracy']:.1%}) | {s['mean_seconds']:.2f} | {s['quality_warning_rows']} |")
lines+=['','## Per-class metrics','','| Model | Class | Precision | Recall | F1 |','|---|---|---:|---:|---:|']
for c in classes:lines.append(f"| {c['model']} | {c['class']} | {c['precision']:.1%} | {c['recall']:.1%} | {c['f1']:.1%} |")
lines+=['','## Confusion matrices','', 'Each row is a reference class; columns are predicted SUPPORTED / PARTIALLY_SUPPORTED / INSUFFICIENT / CONTRADICTED.','']
for model in INPUTS:
 lines+=['### '+model,'','| Reference | SUPPORTED | PARTIALLY_SUPPORTED | INSUFFICIENT | CONTRADICTED |','|---|---:|---:|---:|---:|']
 for label in SCORED:
  vals=[next(x['count'] for x in conf if x['model']==model and x['reference_label']==label and x['predicted_label']==p) for p in LABELS]
  lines.append('| '+label+' | '+' | '.join(map(str,vals))+' |')
 lines.append('')
lines+=['## Methodological notes','', '- The reference dataset has no CONTRADICTED examples. Predictions of CONTRADICTED are counted as errors, and three-class macro-F1 averages F1 over the three reference classes. Undefined precision for unpredicted classes is set to zero.','- These are descriptive results from a small, stratified sample; the 14 development cases were inspected during prompt development. Untouched-subset scores are reported separately, but the sample is not a population-representative benchmark.','- A quality flag marks a possibly truncated explanation, not necessarily an invalid label. Raw model responses are retained in the 342-row predictions CSV.','- The per-fact comparison table enables disagreement analysis; classification correctness alone does not independently validate the human reference or source evidence.','- Inference times reflect the local execution environment and should not be treated as hardware-independent benchmarks.','']
(ROOT/'attribution_evaluation_report_v2.md').write_text('\n'.join(lines),encoding='utf-8')
(ROOT/'attribution_evaluation_manifest_v2.json').write_text(json.dumps({'inputs':INPUTS,'prompt_version':'attribution_top5_v2','facts':57,'models':6,'predictions':342,'development_ids':sorted(DEV),'reference_distribution':dict(collections.Counter(x[1] for x in ref.values())),'outputs':['attribution_all_342_predictions_v2.csv','attribution_model_summary_v2.csv','attribution_class_metrics_v2.csv','attribution_confusion_matrices_v2.csv','attribution_misclassifications_v2.csv','attribution_per_fact_model_comparison_v2.csv','attribution_evaluation_report_v2.md']},indent=2),encoding='utf-8')
print('Validated:',len(INPUTS),'models,',len(ref),'facts,',len(allrows),'predictions')
for s in summary:print(f"{s['model']:15} {s['correct']:2}/57 accuracy={s['accuracy']:.1%} macro-F1={s['macro_f1_3class']:.1%} untouched={s['untouched_correct']}/43")
print('Output files created in',ROOT)
