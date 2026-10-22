"""Complete-population retrospective OOF map; excluded river units get rule level 1."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(HERE))
from run import fp, write_json


def levels(score):
    boundaries = np.quantile(score,[.2,.4,.6,.8])
    if not np.isfinite(score).all() or len(np.unique(boundaries))!=4:
        raise ValueError('Invalid scores or collapsed quintile boundaries')
    return np.searchsorted(boundaries,score,side='right')+1,boundaries


def complete_table(oof,idmap,labels):
    if (len(idmap)!=25939 or not idmap.unit_id.is_unique or not oof.unit_id.is_unique
            or len(oof)!=25509 or set(oof.unit_id)!=set(idmap.loc[idmap.is_train.eq(1),'unit_id'])
            or not labels.unit_id.is_unique or set(labels.unit_id)!=set(idmap.unit_id)):
        raise ValueError('Full/train/label population differs')
    table = idmap[['unit_id','is_train','is_submerged']].merge(oof,on='unit_id',how='left',validate='one_to_one')
    table = table.merge(labels[['unit_id','label']].rename(columns={'label':'raw_label'}),on='unit_id',validate='one_to_one')
    train = table.is_train.eq(1)
    excluded = ~train
    if excluded.sum()!=430 or not table.loc[excluded,'is_submerged'].eq(1).all():
        raise ValueError('Expected exactly 430 excluded river units')
    if not table.loc[train,'label'].eq(table.loc[train,'raw_label']).all() or table.loc[train,'label'].sum()!=661:
        raise ValueError('Model labels differ from full inventory')
    lev,boundaries = levels(table.loc[train,'score_stats84'].to_numpy())
    table['level'] = 1
    table.loc[train,'level'] = lev
    for col in ['score_table40','score_means48','score_distribution80','score_stats84','score_seed_sd','delta84_48','delta84_40','rank_percentile']:
        if not np.isfinite(table.loc[train,col]).all(): raise ValueError('Missing model result: '+col)
        table.loc[excluded,col] = 0.
    for col in ['fold','label','valid_pixels']:
        table.loc[excluded,col] = -1
        table[col] = table[col].astype(int)
    for col in ['low_rank_inventory','high_rank_unlabelled','area_top10','area_top20']:
        table.loc[excluded,col] = 0
        table[col] = table[col].astype(int)
    table['assignment'] = np.where(train,'three_seed_mean_OOF','excluded_river_rule')
    if not table.level.between(1,5).all() or not np.isfinite(table.score_stats84).all():
        raise ValueError('Blank map results')
    return table,boundaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--full-shp',default=str(ROOT/'data/slope_units/slope_units_final.shp'))
    parser.add_argument('--id-map',default=str(ROOT/'features/v2/unit_id_map_v2.csv'))
    parser.add_argument('--labels',default=str(ROOT/'features/v2/features_v2_38.csv'))
    parser.add_argument('--resolution',type=float,default=90)
    a = parser.parse_args()
    if a.resolution<=0: raise ValueError('Positive raster resolution required')
    # Existing optional local GIS runtime. Server explanation has no GIS dependency.
    if sys.platform=='win32' and (ROOT/'.geo_deps').is_dir(): sys.path.insert(0,str(ROOT/'.geo_deps'))
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from pyproj import CRS,Transformer
    import shapefile
    from shapely.geometry import shape,mapping
    from shapely.ops import transform
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    out = Path(a.out); out.mkdir(parents=True,exist_ok=False)
    idmap,labels = pd.read_csv(a.id_map),pd.read_csv(a.labels,usecols=['unit_id','label'])
    with shapefile.Reader(a.full_shp,encoding='utf-8') as reader:
        names = [f[0] for f in reader.fields[1:]]
        if 'unit_id' not in names: raise ValueError('Explicit unit_id field required')
        records = list(reader.iterShapeRecords())
        ids = [int(record.record[names.index('unit_id')]) for record in records]
        geoms = [shape(record.shape.__geo_interface__) for record in records]
    if len(geoms)!=25939 or len(set(ids))!=25939 or set(ids)!=set(idmap.unit_id):
        raise ValueError('Geometry population differs')
    crs = CRS.from_wkt(Path(a.full_shp).with_suffix('.prj').read_text(encoding='utf-8'))
    project = Transformer.from_crs(crs,32649,always_xy=True)
    projected = [transform(project.transform,g) for g in geoms]
    if any(g.is_empty or not g.is_valid or g.area<=0 for g in projected):
        raise ValueError('Invalid geometry; inspect original source rather than silently repairing')
    bounds = np.array([g.bounds for g in projected])
    res = a.resolution
    x0,y0 = np.floor(bounds[:,:2].min(axis=0)/res)*res
    x1,y1 = np.ceil(bounds[:,2:].max(axis=0)/res)*res
    width,height = int(round((x1-x0)/res)),int(round((y1-y0)/res))
    if width*height>100_000_000: raise ValueError('Raster too large; use coarser --resolution')
    tr = from_origin(x0,y1,res,res)
    raster_geoms = [mapping(g) for g in projected]
    area = np.array([g.area for g in projected])
    manifest = dict(kind='retrospective OOF validation map, not final refit or prospective forecast',
        population=dict(full=25939,modelled=25509,excluded_river=430,model_positives=661),
        grading='training-population unit-count quintiles; thresholds independent of labels; river forced level 1',
        score='mean of seed 42/43/44 OOF TabPFN scores; not calibrated event probabilities',
        geometry='EPSG:32649; vector is authoritative, small polygons can disappear at raster resolution',
        resolution_m=res,source_files={str(Path(a.full_shp).name):fp(a.full_shp),'id_map':fp(a.id_map),'labels':fp(a.labels)},
        sources={p.name:fp(p) for p in HERE.glob('*.py')},protocols={})
    colors = ['#dcecc9','#a8d18d','#f6e59a','#ee9a59','#c84435']
    cmap = ListedColormap(colors)
    level_names = ['Very low','Low','Moderate','High','Very high']
    for protocol in ['spatial','admin']:
        oof_path = Path(a.diagnostics)/protocol/'oof_units.csv'
        oof = pd.read_csv(oof_path)
        table,thresholds = complete_table(oof,idmap,labels)
        table = table.set_index('unit_id').loc[ids].reset_index()
        table['geometry_area_m2'] = area
        table.to_csv(out/f'{protocol}_oof_all25939.csv',index=False,encoding='utf-8-sig')
        shp_out = out/f'{protocol}_oof_all25939.shp'
        columns = [('unit_id','unit_id','N',10,0),('score_stats84','score','F',18,12),('level','level','N',2,0),
            ('is_train','is_train','N',1,0),('is_submerged','is_river','N',1,0),('fold','fold','N',3,0),
            ('raw_label','raw_label','N',2,0),('label','model_lab','N',3,0),('score_means48','score48','F',18,12),
            ('score_seed_sd','seed_sd','F',18,12),('delta84_48','delta48','F',18,12),
            ('low_rank_inventory','low_inv','N',1,0),('high_rank_unlabelled','high_unlab','N',1,0)]
        with shapefile.Writer(str(shp_out),shapeType=shapefile.POLYGON,encoding='utf-8') as writer:
            for _,name,kind,size,dec in columns: writer.field(name,kind,size=size,decimal=dec)
            for i,row in table.iterrows():
                writer.shape(records[i].shape)
                writer.record(*[int(row[c]) if kind=='N' else float(row[c]) for c,_,kind,_,_ in columns])
        shp_out.with_suffix('.prj').write_text(crs.to_wkt(),encoding='utf-8')
        shp_out.with_suffix('.cpg').write_text('UTF-8',encoding='ascii')
        classes = rasterize(zip(raster_geoms,table.level.astype(int)),out_shape=(height,width),transform=tr,fill=0,dtype='uint8')
        scores = rasterize(zip(raster_geoms,table.score_stats84.astype(float)),out_shape=(height,width),transform=tr,fill=-9999,dtype='float32')
        for suffix,arr,nodata in [('level',classes,0),('score',scores,-9999)]:
            with rasterio.open(out/f'{protocol}_oof_{suffix}.tif','w',driver='GTiff',height=height,width=width,count=1,
                    dtype=arr.dtype,crs='EPSG:32649',transform=tr,nodata=nodata,compress='deflate') as dst:
                dst.write(arr,1)
                dst.update_tags(purpose='retrospective OOF validation',score_not_calibrated='true',river_assignment='level1 score0 rule')
        fig,ax = plt.subplots(figsize=(12,6),constrained_layout=True)
        ax.imshow(np.ma.masked_equal(classes,0),extent=[x0/1000,x1/1000,y0/1000,y1/1000],origin='upper',cmap=cmap,norm=BoundaryNorm(np.arange(.5,6),5),interpolation='nearest')
        cent = np.array([[g.centroid.x,g.centroid.y] for g in projected])/1000
        known = table.is_train.eq(1)&table.raw_label.eq(1)
        ax.scatter(cent[known,0],cent[known,1],s=3,c='black',alpha=.6,linewidths=0,label='661 model-population inventory positives')
        river_inventory = table.is_train.eq(0)&table.raw_label.eq(1)
        if river_inventory.any(): ax.scatter(cent[river_inventory,0],cent[river_inventory,1],marker='x',s=24,c='magenta',label='Excluded river inventory point (rule-assigned)')
        handles = [Patch(facecolor=color,label=name) for color,name in zip(colors,level_names)]
        h,l = ax.get_legend_handles_labels(); ax.legend(handles=handles+h,fontsize=7,loc='best')
        ax.set(xlabel='UTM easting (km)',ylabel='UTM northing (km)',aspect='equal',
            title=f'{protocol.capitalize()} five-fold OOF | 84-feature TabPFN\nRetrospective susceptibility; 25,939 units including 430 river rule assignments')
        ax.annotate('',xy=(.96,.93),xytext=(.96,.79),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
        ax.text(.96,.95,'N',transform=ax.transAxes,ha='center')
        bx=x0/1000+30; by=y1/1000-90
        ax.plot([bx,bx+20],[by,by],color='black',linewidth=2); ax.text(bx+10,by+2,'20 km',ha='center',fontsize=8)
        fig.savefig(out/f'{protocol}_oof_preview.png',dpi=220)
        fig.savefig(out/f'{protocol}_oof_preview.pdf'); plt.close(fig)
        stats = table.groupby('level').agg(units=('unit_id','size'),area_m2=('geometry_area_m2','sum'),
            raw_inventory=('raw_label','sum'),model_units=('is_train','sum'))
        stats['model_positives'] = table.assign(model_positive=known.astype(int)).groupby('level').model_positive.sum()
        stats.to_csv(out/f'{protocol}_level_statistics.csv',encoding='utf-8-sig')
        manifest['protocols'][protocol] = dict(thresholds=thresholds.tolist(),oof_source=fp(oof_path),
            units=len(table),nonmissing_scores=int(table.score_stats84.notna().sum()),
            excluded_inventory_ids=table.loc[river_inventory,'unit_id'].tolist())
        print('DONE complete OOF map:',protocol,flush=True)
    write_json(out/'map_manifest.json',manifest)


if __name__=='__main__': main()
