"""One-time migration of original factor caches into an independent V3 input store."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.v3.common import sha,write_json,seal
from src.v3.assemble import STATIC
from src.v3.temporal import compute,NAMES


def main():
    dest=ROOT/'data/v3'
    if dest.exists():raise FileExistsError(dest)
    old=ROOT/'features/v2'
    counts=pd.read_csv(ROOT/'data/slope_units/slope_units_final_train_count.csv').sort_values('unit_id')
    ids=counts.unit_id.to_numpy()
    base=counts[['unit_id']].assign(label=counts.landslide_count_study.gt(0).astype(int))
    groups=['geometry','terrain','water_network','road','factors_selected']
    for name in groups:
        frame=pd.read_csv(old/f'groups/{name}.csv').rename(columns={'area_m2':'area'})
        needed=[c for c in STATIC if c in frame]
        if not frame.unit_id.is_unique:raise ValueError(name)
        base=base.merge(frame[['unit_id',*needed]],on='unit_id',how='left',validate='one_to_one')
    if set(STATIC)-set(base):raise ValueError('Missing static fields')
    mapping=pd.read_csv(old/'unit_id_map_v2.csv').set_index('unit_id').reindex(ids)
    old_ids=mapping.src_fixed_Id.to_numpy(dtype=int)
    nd=pd.read_csv(old/'sources/ndvi_unit_matrix.csv').set_index('unit_id').reindex(old_ids)
    rn=pd.read_csv(old/'sources/rain_unit_matrix.csv').set_index('unit_id').reindex(old_ids)
    lu=pd.read_csv(old/'groups/landuse.csv').set_index('unit_id').reindex(ids)
    years=range(2000,2022)
    arrays={
        'ndvi':nd[[f'ndvi_{y}' for y in years]].to_numpy(float),
        **{k:rn[[f'rain_{source}_{y}' for y in years]].to_numpy(float) for k,source in
           [('maxdaily','maxdaily'),('max30d','max30d'),('heavydays','heavydays'),('cumulative','cumulative')]},
        'monthly':rn[[f'rain_m{m:02d}_{y}' for y in years for m in range(1,13)]].to_numpy(float),
        'crop':lu[[f'lu_cropland_{y}' for y in years]].to_numpy(float),
        'built':lu[[f'lu_builtup_{y}' for y in years]].to_numpy(float)}
    if np.isinf(np.column_stack(list(arrays.values()))).any():raise ValueError('Infinite source value')
    positive=base.label.eq(1).to_numpy()
    dates=pd.to_datetime(counts.loc[positive,'study_first_landslide_date']).dt.to_period('M').dt.to_timestamp()
    fresh=compute({k:v[positive] for k,v in arrays.items()},dates)
    historical=pd.read_csv(old/'features_v2_train_38.csv').set_index('unit_id').reindex(ids[positive])
    audit=[]
    for j,c in enumerate(NAMES):
        a,b=fresh[:,j],historical[c].to_numpy()
        both=np.isfinite(a)&np.isfinite(b)
        difference=float(np.max(np.abs(a[both]-b[both]))) if both.any() else None
        if difference is not None and difference>1e-7:raise ValueError(f'Positive temporal definition changed: {c}: {difference}')
        audit.append(dict(feature=c,positive_rows=661,comparable=int(both.sum()),max_abs_difference=difference,
                          missing_v3=int(np.isnan(a).sum()),missing_historical=int(np.isnan(b).sum())))
    # An independent scalar reconstruction checks negative reference months too.
    refs=pd.read_csv(ROOT/'features/v3/fold_water/spatial/fold0/reference_dates.csv',parse_dates=['reference_month'])
    samples=np.linspace(0,len(ids)-1,160,dtype=int)
    vector=compute({k:v[samples] for k,v in arrays.items()},refs.reference_month.iloc[samples])
    for j,i in enumerate(samples):
        d=refs.reference_month.iloc[i]; t=d.year-2000; g=t*12+d.month-1
        ndv=arrays['ndvi'][i]; crop=arrays['crop'][i]; built=arrays['built'][i]
        scalar=[ndv[t-2:t].mean(),ndv[t-2:t].mean()-ndv[:t].mean(),
          arrays['maxdaily'][i,t-2:t].max(),arrays['max30d'][i,t-2:t].max(),
          arrays['heavydays'][i,t-2:t].sum(),arrays['cumulative'][i,t-2:t].mean(),
          *[arrays['monthly'][i,g-k:g].sum() for k in [1,3,6]],
          arrays['monthly'][i,(t-1)*12+4:(t-1)*12+9].sum()/arrays['cumulative'][i,t-1],
          crop[t-1],built[t-1],built[t-1]-built[t-2],crop[t-1]-crop[t-2],
          float(abs(built[t-1]-built[t-2])>.02 or abs(crop[t-1]-crop[t-2])>.02)]
        np.testing.assert_allclose(vector[j],scalar,atol=1e-8,rtol=1e-12,equal_nan=True)
    (dest/'groups').mkdir(parents=True);(dest/'sources').mkdir();(dest/'audit').mkdir()
    base[['unit_id','label',*STATIC]].to_csv(dest/'groups/static.csv',index=False,encoding='utf-8-sig')
    counts.to_csv(dest/'events.csv',index=False,encoding='utf-8-sig')
    mapping.reset_index().to_csv(dest/'unit_identity.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(dest/'sources/temporal_2000_2021.npz',unit_id=ids,**arrays)
    shutil.copytree(ROOT/'features/v3/fold_water',dest/'fold_water')
    shutil.copytree(ROOT/'features/v3/station_matching',dest/'stations')
    shutil.copyfile(ROOT/'data/water/干流站点水位-0908.xlsx',dest/'sources/干流站点水位-0908.xlsx')
    # Keep the exact historical contexts as identity plans, never old feature values.
    reference=json.loads((ROOT/'features/pixel_tabpfn_diagnostics_v1/reference/manifest.json').read_text(encoding='utf-8'))
    contexts={p:reference['protocols'][p]['contexts'] for p in ['spatial','admin']}
    write_json(dest/'context_ids.json',dict(rule='Frozen near-first 4 km / far Bernoulli .2 / cap 4000; unchanged IDs',
         seeds=[42,43,44],contexts=contexts))
    for protocol in ['spatial','admin']:
        target=ROOT/'splits/v3';target.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/f'splits/pixel_tabpfn/{protocol}5.csv',target/f'{protocol}5.csv')
    pd.DataFrame(audit).to_csv(dest/'audit/positive_temporal_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(dict(feature=STATIC,missing=base[STATIC].isna().sum().to_numpy())).to_csv(dest/'audit/static_missing.csv',index=False,encoding='utf-8-sig')
    write_json(dest/'audit/validation.json',dict(positive_temporal_rows=661,temporal_features=15,
        independent_scalar_rows=160,positive_feature_tolerance=1e-7,passed=True,
        interpretation='Change fields are differences, not percentage rates; time scales preserved',
        source_missing_counts={k:int(np.isnan(v).sum()) for k,v in arrays.items()}))
    source_files=[old/f'groups/{g}.csv' for g in groups]+[old/'groups/landuse.csv',old/'sources/ndvi_unit_matrix.csv',old/'sources/rain_unit_matrix.csv',old/'unit_id_map_v2.csv',old/'features_v2_train_38.csv',ROOT/'features/v3/fold_water/manifest.json']
    write_json(dest/'lineage.json',dict(note='Historical paths below are provenance, not runtime dependencies',
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source_files}))
    seal(dest,version='V3',complete=True,role='Unimputed canonical factor sources and frozen fold-water operators',
         builder_sha256=sha(Path(__file__)))
    print('V3 data store ready; positive temporal audit:',audit,flush=True)


if __name__=='__main__':main()
