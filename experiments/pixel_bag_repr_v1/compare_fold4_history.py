"""Compare the replayed common arms with the original full-CV predictions."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from run import write_json,metrics

root=Path(__file__).resolve().parents[2]
out=root/'results/pixel_bag_repr_v1/received_outer_fold4'
new=pd.read_csv(out/'run/predictions.csv')
with zipfile.ZipFile(root/'pixel_bag_followup_v11_return.zip') as z:
    old=pd.read_csv(io.BytesIO(z.read('reports/spatial/predictions.csv')))
historical_rows=[]
for (arm,seed),ff in old[old.fold==4].groupby(['arm','seed']):
    score=metrics(ff.label.to_numpy(),ff.p.to_numpy(),ff.area_m2.to_numpy(),ff.zone.to_numpy(dtype=bool),ff.unit_id.to_numpy())
    historical_rows.extend(dict(arm=arm,seed=int(seed),domain=domain,**values) for domain,values in score.items())
historical=pd.DataFrame(historical_rows)
historical.to_csv(out/'analysis/historical_fold4_controls.csv',index=False)
print('HISTORICAL CONTROLS\n'+historical.groupby(['domain','arm'])[['auc','pr_auc','coverage_at_area_0.2']].mean().to_string())
old=old[(old.fold==4)&old.arm.isin(['stats_xgb','concat_mil'])].copy()
old['arm']=old.arm.replace({'concat_mil':'concat'})
both=new.merge(old,on=['unit_id','arm','seed'],suffixes=('_new','_old'),validate='one_to_one')
assert len(both)==3770*2*3
for field in ['label','fold','area_m2','zone','has_pixels']:
    if field=='area_m2':
        np.testing.assert_allclose(both[field+'_new'],both[field+'_old'],rtol=1e-12,atol=1e-6)
    else:
        assert (both[field+'_new']==both[field+'_old']).all(),field
comparison=both.assign(abs_difference=np.abs(both.p_new-both.p_old)).groupby(['arm','seed']).abs_difference.agg(['max','mean'])
comparison.to_csv(out/'analysis/historical_prediction_replay.csv')
print(comparison.to_string())
write_json(out/'analysis/historical_prediction_replay.json',dict(paired_rows=len(both),
    max_absolute_difference=float(np.abs(both.p_new-both.p_old).max()),
    interpretation='Replayed common arms, not an independent new split or dataset'))
development=pd.read_csv(root/'results/pixel_bag_repr_v1/received_development/analysis/comparison.csv')
test=pd.read_csv(out/'analysis/comparison.csv')
dd=development[development.domain=='drawdown_zone_internal_ranking'].set_index('arm')
tt=test[test.domain=='drawdown_zone_internal_ranking'].set_index('arm')
shift=pd.DataFrame({'development_macro_auc':dd.macro_auc,'outer_fold4_auc':tt.auc})
shift['development_delta_vs_terrain']=shift.development_macro_auc-shift.loc['terrain','development_macro_auc']
shift['outer_delta_vs_terrain']=shift.outer_fold4_auc-shift.loc['terrain','outer_fold4_auc']
shift.to_csv(out/'analysis/development_and_outer.csv')
print(shift.to_string())
