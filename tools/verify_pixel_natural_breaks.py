"""Check natural-break optimal grouping, unchanged scores and full GIS readback."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/pixel_tabpfn_mapping_v2'))
from jenks import natural_breaks
from verify_pixel_diagnostics_maps import verify


def main():
    folder = ROOT/'results/pixel_tabpfn_diagnostics_v1/natural_breaks_20261020'
    old = ROOT/'results/pixel_tabpfn_diagnostics_v1/initial_release/maps'
    verify(folder)
    manifest = json.loads((folder/'map_manifest.json').read_text(encoding='utf-8'))
    assert 'Fisher-Jenks' in manifest['grading']
    for protocol in ['spatial','admin']:
        table = pd.read_csv(folder/f'{protocol}_oof_all25939.csv').sort_values('unit_id').reset_index(drop=True)
        prior = pd.read_csv(old/f'{protocol}_oof_all25939.csv').sort_values('unit_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(table.drop(columns='level'),prior.drop(columns='level'),check_exact=True)
        mask = table.is_train.eq(1)
        lev,breaks = natural_breaks(table.loc[mask,'score_stats84'].to_numpy())
        np.testing.assert_array_equal(lev,table.loc[mask,'level'])
        np.testing.assert_array_equal(breaks,manifest['protocols'][protocol]['thresholds'])
        import rasterio
        with rasterio.open(folder/f'{protocol}_oof_score.tif') as new, rasterio.open(old/f'{protocol}_oof_score.tif') as original:
            np.testing.assert_array_equal(new.read(1),original.read(1))
    (folder/'natural_breaks_audit.json').write_text(json.dumps(dict(exact_natural_breaks=True,
        continuous_scores_unchanged=True,all_nonlevel_csv_columns_unchanged=True,river_rule_retained=True),indent=2),encoding='utf-8')
    print('PASS natural breaks and unchanged continuous score/diagnostic columns')


if __name__=='__main__': main()
