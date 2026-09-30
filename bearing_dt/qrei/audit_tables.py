"""Compare printed table cells with their independent aggregate CSV rows."""
import ast
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT=Path("QREI submission")
TABLES=ROOT/"manuscript/tables"
OUT=ROOT/"manuscript/tmp/presubmission_audit_20260929/reports"


def read(path):
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))


def literal(module, name):
    node=ast.parse(Path(module).read_text(encoding="utf-8"))
    return next(ast.literal_eval(v.value) for v in node.body if isinstance(v,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in v.targets))


def clean(cell):
    return cell.strip().replace("$-$","-")


def main():
    names=literal("bearing_dt/qrei/summarize.py","NAMES")
    criterion_labels=literal("bearing_dt/qrei/build_revision.py","CRITERION_LABELS")
    trigger_labels=literal("bearing_dt/qrei/build_revision.py","TRIGGER_LABELS")
    control_names=literal("bearing_dt/qrei/build_revision.py","CONTROL_NAMES")
    group_text=literal("bearing_dt/qrei/build_supplement.py","GROUP_TEXT")
    checks=[]
    def table(name):
        rows=[[clean(c) for c in line.rstrip().removesuffix(r"\\").split(" & ")]
              for line in (TABLES/(name+".tex")).read_text(encoding="utf-8").splitlines() if " & " in line]
        return rows[1:]
    def check(file, row, col, paper, data, source):
        assert paper == data,(file,row,col,paper,data)
        checks.append({"table":file,"row":row,"column":col,"paper":paper,"data":data,"source":str(source),"status":"PASS"})
    result=ROOT/"results/endpoint_v3"
    src=result/"model_summary.csv";values={names[v['model']]:v for v in read(src)}
    columns=[("nMAE",".3f"),("MAE_hours",".2f"),("late_MAE_hours",".2f"),("coverage",".3f"),("median_slope",".2f"),("bearings_nMAE_le_0.20","d")]
    rows=table("results"); assert len(rows)==14
    for row in rows:
        v=values[row[0]]
        for col,(key,fmt) in enumerate(columns,1):
            expected=str(int(v[key])) if fmt=='d' else format(float(v[key]),fmt)
            check('results',row[0],key,row[col],expected,src)
    src=result/"comparison/paired_model_changes.csv";values={names[v['model']]:v for v in read(src)}
    rows=table('paired'); assert len(rows)==7
    for row in rows:
        v=values[row[0]]
        expected=[f"{float(v['nMAE_v2']):.3f}",f"{float(v['nMAE_v3']):.3f}",f"{float(v['delta_v3_minus_v2']):+.3f}",f"[{float(v['CI_low']):+.3f}, {float(v['CI_high']):+.3f}]",str(int(v['improved_bearings']))]
        for j,(a,b) in enumerate(zip(row[1:],expected)): check('paired',row[0],j,a,b,src)
    ledger={v['bearing_id']:v for v in read(ROOT/"evidence/endpoint_ledger.csv")}
    src=result/"per_bearing_metrics.csv";values={v['bearing_id']:v for v in read(src) if v['model']=='representation'}
    rows=table('bearings'); assert len(rows)==8
    for row in rows:
        v=values[row[0]]
        cause='Temperature' if 'temperature' in ledger[row[0]]['reason'] else 'Vibration'
        check('bearings',row[0],'stop cause',row[1],cause,ROOT/"evidence/endpoint_ledger.csv")
        for j,(key,fmt) in enumerate([('nMAE','.3f'),('MAE_hours','.2f'),('coverage','.3f'),('slope','.2f')],2): check('bearings',row[0],key,row[j],format(float(v[key]),fmt),src)
    src=result/"rank_probabilities.csv";best={}
    for v in read(src):
        if v['criterion'] not in best or float(v['P_rank_1'])>float(best[v['criterion']]['P_rank_1']): best[v['criterion']]=v
    label_to_key={label:key for key,label in criterion_labels.items()}
    rows=table('ranks'); assert len(rows)==len(best)==9
    for row in rows:
        v=best[label_to_key[row[0]]]
        expected='Tied (all zero)' if v['criterion']=='prognostic_horizon_fraction' else names[v['model']]
        check('ranks',row[0],'winner',row[1],expected,src)
        check('ranks',row[0],'fraction',row[2],f"{float(v['P_rank_1']):.4f}",src)
    src=result/"maintenance_summary.csv";values={}
    for v in read(src):
        if float(v['required_lead_hours'])==1 and float(v['failure_cost_ratio'])==10:
            name=control_names.get(v['model'],names.get(v['model'],v['model']))
            values[(name,trigger_labels[v['policy']])]=v
    rows=table('policy'); assert len(rows)==14
    for row in rows:
        v=values[(row[0],row[1])]
        for j,key in enumerate(('loss','too_late','unused_life_fraction'),2): check('policy',row[0]+'/'+row[1],key,row[j],f"{float(v[key]):.3f}",src)
    src=ROOT/"evidence/endpoint_ledger.csv"
    defects={"IR":"inner race","OR":"outer race","B":"ball"}
    rows=table('assets'); assert len(rows)==10
    for row in rows:
        v=ledger[row[0]]
        damage=", ".join(defects[d] for d in v['postmortem_defect'].split('/'))
        for j,key,expected in ((1,'records',v['records']),(2,'hours',f"{float(v['duration_hours']):.3f}"),(3,'kHz',str(round(float(v['raw_fs_Hz'])/1000))),
                               (5,'damage',damage),(6,'scored','Yes' if v['event_observed']=='True' else 'No')):
            check('assets',row[0],key,row[j],expected,src)
    cfg=json.loads((ROOT/"protocol_endpoint_v3.json").read_text());models=cfg['learned_models']+cfg['controls']
    src=result/"per_bearing_metrics.csv";values={(v['model'],v['bearing_id']):v for v in read(src)}
    rows=[]
    for line in (TABLES/'allmetrics.tex').read_text(encoding="utf-8").splitlines():
        if line.startswith('M'): rows.append([clean(c) for c in line.removesuffix(r'\\').split(' & ')])
    assert len(rows)==112
    for row in rows:
        v=values[(models[int(row[0][1:])-1],row[1])]
        for j,key in enumerate(('nMAE','MAE_hours','late_MAE_hours','coverage','slope'),2): check('allmetrics','/'.join(row[:2]),key,row[j],f"{float(v[key]):.3f}",src)
    src=result/"comparison/latent_seed_summary.csv";values={v['seed']:v for v in read(src)}
    for row in table('seeds'):
        v=values[row[0]]
        for j,key in enumerate(('nMAE','late_MAE_hours','median_slope','coverage'),2): check('seeds',row[0],key,row[j],f"{float(v[key]):.3f}",src)
        check('seeds',row[0],'count',row[6],v['bearings_nMAE_le_0.20'],src)
    src=result/'comparison/conditional_regime_summary.csv'
    values={v['physical_regime']:v for v in read(src) if v['model']=='representation'}
    for row in table('conditional'):
        v=values[row[0]]
        keys=('independent_bearings','records','largest_bearing_window_fraction','bearing_equal_nMAE','bearing_equal_coverage')
        for j,key in enumerate(keys,2):
            expected=str(int(v[key])) if j<4 else f"{float(v[key]):.3f}"
            check('conditional',row[0],key,row[j],expected,src)
        expected='Not estimable' if not v['coverage_CI_low'] else f"[{float(v['coverage_CI_low']):.3f}, {float(v['coverage_CI_high']):.3f}]"
        check('conditional',row[0],'interval',row[7],expected,src)
    src=result/'comparison/threshold_crossing_diagnostics.csv';values=read(src)
    def crossing(value):
        return 'No crossing' if value is None or not math.isfinite(value) else f'{value:.3f}'
    for row in table('crossings'):
        g=[v for v in values if v['bearing_id']==row[0]]
        thermal=next(v for v in g if v['quantity']=='ep_temperature_max_C' and float(v['threshold'])==110)
        rms=[float(v['first_crossing_elapsed_hours']) for v in g if v['quantity'] in ('A_rms_g','C_rms_g') and float(v['threshold'])==8 and v['crossed']=='True']
        expected=[crossing(float(thermal[k]) if thermal[k] else None) for k in ('first_crossing_elapsed_hours','first_crossing_remaining_hours')]+[crossing(min(rms) if rms else None)]
        for j,(a,b) in enumerate(zip(row[1:],expected),1): check('crossings',row[0],j,a,b,src)
    src=result/'comparison/fitting_cause_support.csv';support={v['test_bearing']:v for v in read(src)}
    for row in table('roles'):
        splitpath=ROOT/'results/endpoint_neural'/row[0]/'split.json'
        split=json.loads(splitpath.read_text())
        expected=[', '.join(split['training_bearings']),split['validation_bearing'],split['calibration_bearing'],support[row[0]]['thermal_fitting_bearings'],support[row[0]]['vibration_fitting_bearings']]
        assert len(set(split['training_bearings']+[split['validation_bearing'],split['calibration_bearing'],row[0]]))==8
        for j,(a,b) in enumerate(zip(row[1:],expected),1): check('roles',row[0],j,a,b,src if j>3 else splitpath)
    for row in table('codes'):
        check('codes',row[0],'model',row[1],names[models[int(row[0][1:])-1]],ROOT/'protocol_endpoint_v3.json')
    src=Path('data/processed/phme_tvoc_10b_endpoint_v3/manifest.json')
    features=json.loads(src.read_text())['feature_columns']
    from bearing_dt.qrei.build_supplement import feature_group
    counts=Counter(feature_group(n) for n in features)
    printed={row[0]:row[1] for row in table('features') if row[0] in group_text}
    assert set(printed)==set(group_text)==set(counts)
    for group,count in printed.items(): check('features',group,'count',count,str(counts[group]),src)
    assert sum(counts.values())==104
    src=ROOT/"evidence/operating_channel_check.csv";values={v['bearing_id']:v for v in read(src)}
    rows=table('channels'); assert len(rows)==10
    for row in rows:
        v=values[row[0]]
        expected=[format(float(v['median_measured_to_set_speed']),'.3f'),format(float(v['spectral_peak_near_set_speed']),'.2f'),
                  format(float(v['spectral_peak_near_measured_speed']),'.2f'),str(int(v['static_load_above_set_max_by_500N'])),
                  format(float(v['median_measured_to_set_dynamic_load']),'.2f')]
        for j,(a,b) in enumerate(zip(row[1:],expected)): check('channels',row[0],j,a,b,src)
    # Secondary analyses (protocol_sensitivity_addendum.json): recompute from per-bearing rows where possible.
    sens=result/'sensitivity'
    src=sens/'seed_model_summary.csv';values={(v['model'],v['seed']):v for v in read(src)}
    rows=table('allseeds'); assert len(rows)==8
    seeds=[str(s) for s in cfg['seeds']]
    for row in rows:
        model=next(k for k,v in names.items() if v==row[0])
        for j,seed in enumerate(seeds,1):
            v=values[(model,seed)]
            check('allseeds',row[0],seed,row[j],f"{float(v['nMAE']):.3f} ({int(v['bearings_nMAE_le_0.20'])})",src)
    per={'Primary':[v for v in read(result/'per_bearing_metrics.csv')],
         'Swapped':read(sens/'roles_swap_per_bearing.csv'),'Reversed':read(sens/'roles_reverse_per_bearing.csv')}
    rows=table('allocations'); assert len(rows)==11
    for j,label in enumerate(('Primary','Swapped','Reversed'),1):
        rep=[float(v['nMAE']) for v in per[label] if v['model']=='representation']
        att=[float(v['nMAE']) for v in per[label] if v['model']=='attention']
        assert len(rep)==len(att)==8
        check('allocations','latent nMAE',label,rows[0][j],f"{sum(rep)/8:.3f}",'per-bearing rows')
        check('allocations','attention nMAE',label,rows[1][j],f"{sum(att)/8:.3f}",'per-bearing rows')
        check('allocations','bearings <= 0.20',label,rows[2][j],str(sum(x<=.2 for x in rep)),'per-bearing rows')
    src=sens/'phase_split_per_bearing.csv';vals={}
    for v in read(src): vals.setdefault(v['model'],[]).append(v)
    for row in table('phase'):
        g=vals[next(k for k,v in names.items() if v==row[0])]; assert len(g)==8
        for j,key in enumerate(('early_error_over_life','final40_error_over_life'),1):
            check('phase',row[0],key,row[j],f"{sum(float(v[key]) for v in g)/8:.3f}",src)
    src=sens/'population_controls_per_bearing.csv';vals={}
    for v in read(src): vals.setdefault(v['control'],[]).append(v)
    for row in table('population'):
        g=vals[row[0][0].lower()+row[0][1:]]; assert len(g)==8
        check('population',row[0],'nMAE',row[1],f"{sum(float(v['nMAE']) for v in g)/8:.3f}",src)
        check('population',row[0],'late',row[2],f"{sum(float(v['late_MAE_hours']) for v in g)/8:.2f}",src)
    src=sens/'rank_scenarios.csv';best={}
    for v in read(src):
        if v['scenario'] not in best or float(v['first_rank_fraction'])>float(best[v['scenario']]['first_rank_fraction']): best[v['scenario']]=v
    labels=literal("bearing_dt/qrei/build_supplement.py","SCENARIO_LABELS")
    rows=table('rankscenarios'); assert len(rows)==15
    for row in rows:
        key=next((k for k,l in labels.items() if l==row[0]),None) or 'endpoint_all8_without_'+row[0].rsplit(' ',1)[1]
        v=best[key]
        check('rankscenarios',row[0],'largest',row[1],f"{float(v['first_rank_fraction']):.3f} ({names[v['model']]})",src)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'table_cells_checked.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=checks[0].keys());w.writeheader();w.writerows(checks)
    print(json.dumps({'printed_cells_compared':len(checks),'tables':sorted(set(v['table'] for v in checks)),'status':'PASS'}),flush=True)


if __name__=='__main__': main()
