"""Read back delivered vectors/rasters and verify explicit ID joins and river rules."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if sys.platform=='win32': sys.path.insert(0,str(ROOT/'.geo_deps'))
import rasterio
import shapefile


def verify(folder):
    folder = Path(folder)
    result = {}
    with shapefile.Reader(str(ROOT/'data/slope_units/slope_units_final.shp'),encoding='utf-8') as src:
        names = [f[0] for f in src.fields[1:]]
        original = {int(item.record[names.index('unit_id')]):item.shape for item in src.iterShapeRecords()}
    for protocol in ['spatial','admin']:
        table = pd.read_csv(folder/f'{protocol}_oof_all25939.csv').set_index('unit_id')
        assert len(table)==25939 and table.index.is_unique and set(table.index)==set(original)
        assert table.is_train.sum()==25509
        river = table.is_train.eq(0)
        assert river.sum()==430 and table.loc[river,'level'].eq(1).all() and table.loc[river,'score_stats84'].eq(0).all()
        assert table.loc[~river,'label'].sum()==661 and table.raw_label.sum()==662
        assert table.loc[16490,'raw_label']==1 and table.loc[16490,'level']==1 and table.loc[16490,'label']==-1
        assert np.isfinite(table.score_stats84).all()
        ids = []
        with shapefile.Reader(str(folder/f'{protocol}_oof_all25939.shp'),encoding='utf-8') as src:
            fields = [f[0] for f in src.fields[1:]]
            for item in src.iterShapeRecords():
                row = dict(zip(fields,item.record)); uid = int(row['unit_id']); ids.append(uid)
                assert np.array_equal(item.shape.points,original[uid].points)
                assert list(item.shape.parts)==list(original[uid].parts)
                assert abs(row['score']-table.loc[uid,'score_stats84'])<1e-11
                assert row['level']==table.loc[uid,'level'] and row['raw_label']==table.loc[uid,'raw_label']
            assert len(ids)==len(set(ids))==25939
        with rasterio.open(folder/f'{protocol}_oof_level.tif') as src:
            levels = src.read(1); profile = (src.crs,src.transform,src.shape)
            assert src.crs.to_epsg()==32649 and src.nodata==0
            assert set(np.unique(levels))=={0,1,2,3,4,5}
        with rasterio.open(folder/f'{protocol}_oof_score.tif') as src:
            assert (src.crs,src.transform,src.shape)==profile and src.nodata==-9999
            scores = src.read(1)
            assert np.array_equal(levels==0,scores==-9999)
            assert np.isfinite(scores).all() and ((scores[levels>0]>=0)&(scores[levels>0]<=1)).all()
        result[protocol] = dict(units=25939,model_units=25509,river_units=430,model_positives=661,
            raw_positives=662,geometry_preserved=True,vector_csv_match=True,raster_grid_and_nodata=True,
            nonmissing_scores=25939)
    (folder/'readback_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PASS full vector, CSV and raster readback:',folder)


if __name__=='__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('folder'); a = p.parse_args(); verify(a.folder)
