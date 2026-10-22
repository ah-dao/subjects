"""Natural-break adapter for frozen OOF mapping. Never edits explanation sources."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(HERE))
from jenks import natural_breaks


def base_module():
    source = ROOT/'experiments/pixel_tabpfn_diagnostics_v1/mapping.py'
    spec = importlib.util.spec_from_file_location('frozen_oof_mapping_v1',source)
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    # Runtime strategy substitution only. Old files and running explanation hashes stay intact.
    base.levels = natural_breaks
    return base


def main():
    base = base_module()
    if '--help' in sys.argv or '-h' in sys.argv:
        base.main(); return
    p = argparse.ArgumentParser(add_help=False); p.add_argument('--out',required=True)
    args,_ = p.parse_known_args()
    base.main()
    out = Path(args.out)
    manifest = json.loads((out/'map_manifest.json').read_text(encoding='utf-8'))
    manifest['grading'] = 'exact Fisher-Jenks natural breaks, five classes; all 25,509 model units, no sampling or area weighting; river forced level 1'
    manifest['boundary_rule'] = 'breaks are inclusive upper class maxima; searchsorted side=left; labels do not select breaks'
    manifest['classification_sources'] = {f'pixel_tabpfn_mapping_v2/{f.name}':base.fp(f) for f in HERE.glob('*.py')}
    comparison = []
    for protocol in ['spatial','admin']:
        table = pd.read_csv(out/f'{protocol}_oof_all25939.csv')
        train = table.is_train.eq(1)
        score = table.loc[train,'score_stats84'].to_numpy()
        new,breaks = natural_breaks(score)
        if not np.array_equal(new,table.loc[train,'level']): raise ValueError('Natural-break readback mismatch')
        stats = table.groupby('level').agg(units=('unit_id','size'),area_m2=('geometry_area_m2','sum'),
            model_units=('is_train','sum'),raw_inventory=('raw_label','sum'))
        model = table[train].groupby('level').agg(model_area_m2=('geometry_area_m2','sum'),model_positives=('label','sum'))
        stats = stats.join(model)
        stats['area_fraction_full'] = stats.area_m2/table.geometry_area_m2.sum()
        stats['model_area_fraction'] = stats.model_area_m2/table.loc[train,'geometry_area_m2'].sum()
        stats['model_positive_fraction'] = stats.model_positives/661
        stats.to_csv(out/f'{protocol}_level_statistics.csv',encoding='utf-8-sig')
        old = np.searchsorted(np.quantile(score,[.2,.4,.6,.8]),score,side='right')+1
        area = table.loc[train,'geometry_area_m2'].to_numpy()
        for method,lev in [('previous_quantiles',old),('natural_breaks',new)]:
            high = lev>=4
            comparison.append(dict(protocol=protocol,method=method,model_units=25509,
                high_very_high_units=int(high.sum()),high_very_high_unit_fraction=float(high.mean()),
                high_very_high_area_km2=float(area[high].sum()/1e6),
                high_very_high_model_area_fraction=float(area[high].sum()/area.sum())))
        pd.DataFrame(dict(level=range(1,6),upper_inclusive=[*breaks,float(score.max())])).to_csv(
            out/f'{protocol}_natural_breaks.csv',index=False,encoding='utf-8-sig')
        manifest['protocols'][protocol]['thresholds'] = breaks.tolist()
        import rasterio
        for suffix in ['score','level']:
            with rasterio.open(out/f'{protocol}_oof_{suffix}.tif','r+') as raster:
                raster.update_tags(grading='Fisher-Jenks natural breaks, 5 classes',boundary_rule='inclusive upper maxima',
                                   class_breaks=json.dumps(breaks.tolist()))
    pd.DataFrame(comparison).to_csv(out/'grading_comparison.csv',index=False,encoding='utf-8-sig')
    base.write_json(out/'map_manifest.json',manifest)
    print('PASS exact natural breaks; river rule and continuous scores retained',flush=True)


if __name__=='__main__': main()
