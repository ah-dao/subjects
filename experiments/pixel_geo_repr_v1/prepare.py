"""Label-free geology/exposure summaries on the ORIGINAL valid pixel-bag support."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pixel_tabpfn_v1'))
import bag_core as bag
import numpy as np
import pandas as pd

BANDS=('rare_le10','intermittent','frequent_ge90')
CLASSES={'hardness':('hard','medhard','medsoft','soft','unknown'),
         'structure':('flat','dip','oblique','anti','unknown')}


def summarize(wet,area,hardness,structure):
    """All fractions use TOTAL valid bag area; unknown and empty bands are explicit."""
    if len(wet)==0:
        return np.zeros(10),np.zeros(3),np.zeros(30)
    if (not np.isfinite(wet).all() or ((wet<0)|(wet>1)).any()
            or not np.isfinite(area).all() or (area<=0).any()):
        raise ValueError('Invalid exposure or area')
    w=area.astype('float64')/area.sum(dtype='float64')
    band=np.where(wet<=.1,0,np.where(wet>=.9,2,1))
    b=np.bincount(band,weights=w,minlength=3)
    globals_,joint=[],[]
    for codes in (hardness,structure):
        if not np.isin(codes,np.arange(5)).all():
            raise ValueError('Expected normalized category codes 0..4')
        globals_.extend(np.bincount(codes,weights=w,minlength=5))
        joint.extend(np.bincount(band*5+codes,weights=w,minlength=15))
    return np.asarray(globals_),b,np.asarray(joint)


def feature_names():
    g=[f'{kind}_{c}_area' for kind,classes in CLASSES.items() for c in classes]
    b=[f'band_{x}_area' for x in BANDS]
    j=[f'{kind}_{x}_{c}_joint_area' for kind,classes in CLASSES.items() for x in BANDS for c in classes]
    return dict(global_geo=g,band=b,joint=j)


def independence(global_geo,band):
    """Explicit NON-observed control: product of marginals, same 30-column size."""
    return np.concatenate([(band[:,:,None]*global_geo[:,None,k:k+5]).reshape(len(band),15)
                           for k in (0,5)],axis=1)


def sample_structure(path,x,y,crs):
    import rasterio
    from pyproj import Transformer
    with rasterio.open(path) as ds:
        tf=Transformer.from_crs(crs,ds.crs,always_xy=True)
        raster=ds.read(1,masked=True)
        result=np.full(len(x),4,dtype='uint8')
        for start in range(0,len(x),500000):
            end=min(start+500000,len(x))
            xx,yy=tf.transform(x[start:end],y[start:end])
            rows,cols=rasterio.transform.rowcol(ds.transform,xx,yy)
            rows,cols=np.asarray(rows),np.asarray(cols)
            valid=(rows>=0)&(cols>=0)&(rows<ds.height)&(cols<ds.width)
            ii=np.flatnonzero(valid)
            values=raster[rows[ii],cols[ii]]
            good=~np.ma.getmaskarray(values)&np.isin(values.data,[0,1,2,3])
            result[start+ii[good]]=values.data[good].astype('uint8')
        return result,dict(crs=str(ds.crs),shape=list(ds.shape),resolution=list(ds.res),
                           resampling='nearest native cell at bag pixel centre; no new geological resolution')


def hardness_grid(path,grid):
    import rasterio
    from rasterio.features import rasterize
    from pyproj import CRS,Transformer
    import shapefile
    from shapely.geometry import shape
    from shapely.ops import transform
    # Read the original numeric third DBF column, as in the audited historical extractor.
    import struct
    raw=path.with_suffix('.dbf').read_bytes()
    n,hlen,rlen=struct.unpack('<IHH',raw[4:12])
    fields=[]
    for p in range(32,hlen-1,32):
        if raw[p]==13:
            break
        fields.append(raw[p:p+32])
    if len(fields)<3 or chr(fields[2][11])!='N':
        raise ValueError('Unexpected hardness DBF schema')
    offset=1+sum(f[16] for f in fields[:2]); width=fields[2][16]
    codes=[int(raw[hlen+i*rlen+offset:hlen+i*rlen+offset+width].strip() or b'-1') for i in range(n)]
    source=CRS.from_wkt(path.with_suffix('.prj').read_text(encoding='utf-8'))
    forward=Transformer.from_crs(grid.crs,source,always_xy=True)
    back=Transformer.from_crs(source,grid.crs,always_xy=True)
    bounds=forward.transform_bounds(*grid.bounds,densify_pts=41)
    shapes=[]
    with shapefile.Reader(shp=str(path),shx=str(path.with_suffix('.shx'))) as reader:
        for geom in reader.iterShapes(bbox=bounds):
            code=codes[geom.oid]
            if code in (1,2,3,4):
                poly=transform(back.transform,shape(geom.__geo_interface__))
                if not poly.is_valid:
                    poly=poly.buffer(0)
                if not poly.is_empty:
                    shapes.append((poly,code))
    if not shapes:
        raise ValueError('No hardness polygons intersect bag grid')
    result=rasterize(shapes,out_shape=grid.shape,transform=grid.transform,fill=0,dtype='uint8',all_touched=False)
    return result,dict(polygons=len(shapes),source_crs=str(source),field_index=2,
                       classes='1 hard / 2 medium-hard / 3 medium-soft / 4 soft; 0 unknown',
                       rasterization='pixel centre, source record order, original bag grid')


def prepare(args):
    # Local optional GIS dependencies are isolated from the training environment.
    if (ROOT/'.geo_deps').is_dir():
        sys.path.insert(0,str(ROOT/'.geo_deps'))
    import rasterio
    from run import verify_cache
    manifest,d,pixels,areas=verify_cache(args.prepared)
    with np.load(args.bags,allow_pickle=True) as z:
        good,_=bag.encode_terrain(z['X'],[str(x) for x in z['bands']])
        use=good&np.isin(z['uid'],d['unit_id'])
        uid=z['uid'][use].astype('int64'); rows=z['row'][use].astype('int64'); cols=z['col'][use].astype('int64')
        elevation=z['X'][use,[str(x) for x in z['bands']].index('elevation')]
    order=np.argsort(uid,kind='stable')
    uid,rows,cols,elevation=(x[order] for x in (uid,rows,cols,elevation))
    if (len(uid)!=len(pixels) or not np.allclose(elevation,pixels[:,0],rtol=0,atol=1e-5)
            or not np.array_equal(np.searchsorted(uid,d['unit_id'],'left'),d['starts'])
            or not np.array_equal(np.searchsorted(uid,d['unit_id'],'right'),d['ends'])):
        raise ValueError('Raw pixel order/support does not match prepared cache')
    out=bag.fresh_dir(args.out)
    with rasterio.open(args.grid) as grid:
        if grid.crs is None or (rows<0).any() or (cols<0).any() or (rows>=grid.height).any() or (cols>=grid.width).any():
            raise ValueError('Invalid source grid coordinates')
        identities=grid.read(1)
        if not np.array_equal(identities[rows,cols],uid):
            raise ValueError('Full pixel/grid identity mismatch')
        del identities
        print('Rasterizing hardness polygons on original bag grid',flush=True)
        hard,hmeta=hardness_grid(Path(args.hardness),grid)
        hard=hard[rows,cols]
        hard=np.where(np.isin(hard,[1,2,3,4]),hard-1,4).astype('uint8')
        x,y=rasterio.transform.xy(grid.transform,rows,cols)
        print('Sampling native slope-structure raster',flush=True)
        structure,smeta=sample_structure(args.structure,np.asarray(x),np.asarray(y),grid.crs)
    global_geo,band,joint=[],[],[]
    anchors=[]
    for s,e in zip(d['starts'],d['ends']):
        g,b,j=summarize(pixels[s:e,4],areas[s:e],hard[s:e],structure[s:e])
        global_geo.append(g); band.append(b); joint.append(j)
        anchors.append([float(np.mean(pixels[s:e,4])) if e>s else 0.])
    output=out/'features.npz'
    np.savez_compressed(output,unit_id=d['unit_id'],label=d['y'],starts=d['starts'],ends=d['ends'],
                        tab_reference=d['tab'],summary_reference=d['summary'],wet_anchor=anchors,
                        global_geo=np.asarray(global_geo,dtype='float32'),band=np.asarray(band,dtype='float32'),
                        joint=np.asarray(joint,dtype='float32'))
    pd.DataFrame(dict(unit_id=d['unit_id'],pixels=d['ends']-d['starts'],
        hardness_coverage=np.where(d['ends']>d['starts'],1-np.asarray(global_geo)[:,4],0),
        structure_coverage=np.where(d['ends']>d['starts'],1-np.asarray(global_geo)[:,9],0),
        rare_area=np.asarray(band)[:,0],intermittent_area=np.asarray(band)[:,1],frequent_area=np.asarray(band)[:,2])
    ).to_csv(out/'coverage.csv',index=False)
    source_files=[Path(args.bags),Path(args.grid),Path(args.hardness),Path(args.hardness).with_suffix('.dbf'),
                  Path(args.hardness).with_suffix('.shx'),Path(args.hardness).with_suffix('.prj')]
    source_files+=sorted(p for p in Path(args.structure).iterdir() if p.is_file())
    provenance=dict(schema='geo_exposure_area_v1',units=len(d['unit_id']),positives=int(d['y'].sum()),
        pixels=len(pixels),support='original terrain-valid pixel bag; not whole polygon outside bag support',
        exposure='historical wet-frequency <=0.1 / (0.1,0.9) / >=0.9; predeclared, label-free',
        missing='fifth categorical class; empty bag all added features zero, existing has_pixels retained',
        names=feature_names(),hardness=hmeta,structure=smeta,prepared_manifest=manifest,
        sources={str(p):bag.fingerprint(p) for p in source_files},prepare_source=bag.fingerprint(__file__),
        files={p.name:bag.fingerprint(p) for p in (output,out/'coverage.csv')})
    bag.write_json(out/'manifest.json',provenance)
    print('DONE prepare:',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True); p.add_argument('--bags',required=True); p.add_argument('--grid',required=True)
    p.add_argument('--hardness',default=str(ROOT/'data/geology/lithology/中国岩性分布1.shp'))
    p.add_argument('--structure',default=str(ROOT/'data/geology/attitude/slopetype'))
    p.add_argument('--out',required=True)
    prepare(p.parse_args())
