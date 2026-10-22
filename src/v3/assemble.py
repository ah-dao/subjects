"""Assemble 10 self-contained V3 84-column views from canonical factor groups."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, sha, write_json, seal, verify_files
from .temporal import NAMES as TEMPORAL, compute, load_arrays, definitions
from .water_background import WATER_COLUMNS, SUMMARY_COLUMNS

STATIC = ['elevation_mean', 'slope_mean', 'aspect_sin', 'aspect_cos', 'TRI_mean',
          'area', 'shape_index', 'river_dist_m', 'mainstream_dist_m', 'drainage_density',
          'road_dist_m', 'road_density', 'road_major_dist_m', 'road_local_dist_m',
          'soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac']
# Keep the historical table order, with explicit V3 water names.
TABLE = STATIC[:7]+[WATER_COLUMNS[0]]+STATIC[7:10]+TEMPORAL[:10]+TEMPORAL[10:]+STATIC[10:14]+[
    WATER_COLUMNS[2], WATER_COLUMNS[1], WATER_COLUMNS[3], WATER_COLUMNS[4]]+STATIC[14:]
FEATURES = TABLE+['log1p_valid_pixels', 'has_valid_pixels']+SUMMARY_COLUMNS


def assemble(data, out):
    if out.exists(): raise FileExistsError(out)
    manifest = verify_files(data)
    base = pd.read_csv(data/'groups/static.csv').sort_values('unit_id').reset_index(drop=True)
    if len(base)!=25509 or not base.unit_id.is_unique or base.label.sum()!=661:
        raise ValueError('Unexpected modeling population')
    arrays = load_arrays(data/'sources/temporal_2000_2021.npz', base.unit_id)
    if len(FEATURES)!=84 or len(set(FEATURES))!=84 or len(TABLE)!=38:
        raise ValueError('Feature definition invalid')
    out.mkdir(parents=True)
    all_audits=[]
    for protocol in ['spatial','admin']:
        for fold in range(5):
            path=data/'fold_water'/protocol/f'fold{fold}'
            references=pd.read_csv(path/'reference_dates.csv',parse_dates=['reference_month'])
            if not np.array_equal(references.unit_id,base.unit_id) or not np.array_equal(references.label,base.label):
                raise ValueError('Reference IDs/labels differ')
            with np.load(path/'water_pixel_packet.npz',allow_pickle=False) as z:
                if not np.array_equal(z['unit_id'],base.unit_id) or not np.array_equal(z['reference_month'],references.reference_month.to_numpy(dtype='datetime64[D]')):
                    raise ValueError('Water dates/IDs differ')
                water,summary,counts=z['water_table'],z['summary'],z['valid_pixels']
                geometry={k:z[k] for k in ['area','cent','county']}
            temporal=compute(arrays,references.reference_month)
            table=base[['unit_id',*STATIC]].copy()
            for j,c in enumerate(WATER_COLUMNS):table[c]=water[:,j]
            for j,c in enumerate(TEMPORAL):table[c]=temporal[:,j]
            x=np.column_stack([table[TABLE].to_numpy(),np.log1p(counts),counts>0,summary]).astype('float32')
            if x.shape!=(25509,84) or np.isinf(x).any():raise ValueError('Invalid V3 matrix')
            target=out/protocol/f'fold{fold}'; target.mkdir(parents=True)
            readable=pd.DataFrame(x,columns=FEATURES)
            readable.insert(0,'label',base.label); readable.insert(0,'unit_id',base.unit_id)
            readable.insert(2,'role',references.role); readable.insert(3,'reference_month',references.reference_month)
            readable.to_csv(target/'features84.csv',index=False,encoding='utf-8-sig')
            np.savez_compressed(target/'features84.npz',unit_id=base.unit_id.to_numpy(),y=base.label.to_numpy(),x=x,
                fold=references.fold.to_numpy(),reference_month=references.reference_month.to_numpy(dtype='datetime64[D]'),
                background_known=np.isfinite(water[:,0]),zone=np.isfinite(water[:,0])&(water[:,0]>0),
                valid_pixels=counts,**geometry)
            references.to_csv(target/'reference_dates.csv',index=False,encoding='utf-8-sig')
            audit=pd.DataFrame(dict(feature=FEATURES,missing=np.isnan(x).sum(axis=0)))
            audit.to_csv(target/'missing_features.csv',index=False,encoding='utf-8-sig')
            all_audits.append(dict(protocol=protocol,fold=fold,units=len(base),positives=int(base.label.sum()),
                train=int(references.role.eq('train').sum()),test=int(references.role.eq('test').sum()),
                missing_values=int(np.isnan(x).sum()),background_unknown=int((~np.isfinite(water[:,0])).sum())))
            print('ASSEMBLED',protocol,fold,x.shape,flush=True)
    write_json(out/'schema.json',dict(version='V3',ready_for_training=True,table=TABLE,features=FEATURES,
        temporal_definitions=definitions(),missing='Kept as NaN until context-only imputation/scaling',
        zone='preevent_inundation_fraction > 0; missing background excluded from this subgroup, all-population metric retains every unit',
        purpose='Retrospective within-study-area geographic transfer; not prospective risk'))
    pd.DataFrame(all_audits).to_csv(out/'fold_summary.csv',index=False,encoding='utf-8-sig')
    seal(out,version='V3',complete=True,ready_for_training=True,input_manifest_sha256=sha(data/'manifest.json'),
         operator_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('temporal.py'),Path(__file__).with_name('water_background.py')]})


def verify(out):
    manifest=verify_files(out)
    for name,digest in manifest.get('operator_sha256',{}).items():
        if sha(Path(__file__).with_name(name))!=digest:
            raise ValueError('Feature operator changed; rebuild before training: '+name)
    schema=json.loads((out/'schema.json').read_text(encoding='utf-8'))
    if schema['features']!=FEATURES or manifest.get('version')!='V3' or not manifest.get('ready_for_training'):
        raise ValueError('Not a train-ready V3 feature package')
    for protocol in ['spatial','admin']:
        for f in range(5):
            with np.load(out/protocol/f'fold{f}'/'features84.npz',allow_pickle=False) as z:
                if z['x'].shape!=(25509,84) or z['y'].sum()!=661 or len(np.unique(z['unit_id']))!=25509 or np.isinf(z['x']).any():
                    raise ValueError('Invalid feature packet')
    print('PASS ten V3 84-feature views',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['build','verify'])
    p.add_argument('--data',type=Path,default=ROOT/'data/v3')
    p.add_argument('--out',type=Path,default=ROOT/'features/v3/model_inputs')
    args=p.parse_args()
    if args.action=='build':assemble(args.data,args.out)
    else:verify(args.out)


if __name__=='__main__':main()
