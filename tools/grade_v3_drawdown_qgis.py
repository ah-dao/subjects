"""Run with QGIS Python: independent five-grade Jenks for drawdown OOF scores."""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path

from qgis.core import (Qgis, QgsApplication, QgsClassificationJenks, QgsVectorLayer,
    QgsField, QgsFeature, QgsGeometry, QgsCoordinateReferenceSystem,
    QgsCoordinateTransform, QgsProject, QgsVectorFileWriter, QgsSymbol,
    QgsRendererCategory, QgsCategorizedSymbolRenderer)
from qgis.PyQt.QtCore import QVariant, Qt
from qgis.PyQt.QtGui import QColor

ROOT = Path(__file__).resolve().parents[1]
NAMES = {1:'极低',2:'低',3:'中',4:'高',5:'极高'}
COLORS = {1:'#dcecc9',2:'#a8d18d',3:'#f6e59a',4:'#ee9a59',5:'#c84435'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(rows):
    n, p = len(rows), sum(r['label'] for r in rows)
    area = sum(r['area_m2'] for r in rows)
    output = dict(units=n,positives=p,area_km2=area/1e6)
    for name, minimum in [('high',4),('medium_plus',3)]:
        selected = [r for r in rows if r['level']>=minimum]
        found = sum(r['label'] for r in selected)
        selected_area = sum(r['area_m2'] for r in selected)
        coverage, share = found/p if p else None, selected_area/area if area else None
        output.update({name+'_units':len(selected),name+'_positives':found,
            name+'_coverage':coverage,name+'_area_fraction':share,
            name+'_inventory_fraction':found/len(selected) if selected else None,
            name+'_area_enrichment':coverage/share if coverage is not None and share else None})
    return output


def main(args):
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory')
    summary = json.loads((args.results/'summary.json').read_text())
    config = json.loads((args.results/'config.json').read_text())
    if (not summary['complete'] or not summary['scientific_result']
            or summary['completed_folds'] != list(range(5))
            or config['version'] != 'V3_drawdown84_spatial_s42_v1'):
        raise ValueError('Expected genuine complete fivefold drawdown84 scores')
    source = args.results/'predictions.csv'
    with source.open(encoding='utf-8-sig',newline='') as stream:
        rows = [dict(unit_id=int(r['unit_id']),score=float(r['p']),label=int(r['label']),
            county=r['county'],area_m2=float(r['area_m2']),fold=int(r['fold'])) for r in csv.DictReader(stream)]
    if (len(rows)!=7101 or len({r['unit_id'] for r in rows})!=7101
            or sum(r['label'] for r in rows)!=616 or {r['fold'] for r in rows}!=set(range(5))
            or any(not math.isfinite(r['score']) or not 0<=r['score']<=1 or r['label'] not in [0,1] for r in rows)):
        raise ValueError('Unexpected drawdown population/scores')
    # Classifier sees scores only: no labels, areas or county-specific thresholds.
    classifier = QgsClassificationJenks()
    ranges = classifier.classes([r['score'] for r in rows],5)
    if len(ranges)!=5:
        raise ValueError('QGIS returned fewer than five classes')
    boundaries = [float(r.upperBound()) for r in ranges[:-1]]
    if boundaries!=sorted(set(boundaries)):
        raise ValueError('Non-increasing Jenks thresholds')
    for row in rows:
        row['level'] = bisect.bisect_left(boundaries,row['score'])+1
        row['grade'] = NAMES[row['level']]
    if {r['level'] for r in rows}!=set(range(1,6)):
        raise ValueError('Empty grade')
    args.out.mkdir(parents=True)
    write_csv(args.out/'drawdown84_qgis_grades.csv',rows)
    write_csv(args.out/'unit_id_level.csv',[dict(unit_id=r['unit_id'],level=r['level']) for r in rows])
    overall = evaluate(rows)
    write_csv(args.out/'overall_metrics.csv',[overall])
    counties = [dict(county=name,**evaluate([r for r in rows if r['county']==name])) for name in sorted({r['county'] for r in rows})]
    write_csv(args.out/'county_metrics.csv',counties)
    stats = []
    for level in range(1,6):
        part = [r for r in rows if r['level']==level]
        found, area = sum(r['label'] for r in part),sum(r['area_m2'] for r in part)
        stats.append(dict(level=level,grade=NAMES[level],lower_bound=float(ranges[level-1].lowerBound()),
            upper_bound=float(ranges[level-1].upperBound()),units=len(part),positives=found,
            area_km2=area/1e6,area_fraction=area/(overall['area_km2']*1e6),
            positive_coverage=found/616,inventory_fraction=found/len(part)))
    write_csv(args.out/'five_grade_statistics.csv',stats)
    print('BREAKS',boundaries,flush=True)
    print('OVERALL',json.dumps(overall,ensure_ascii=False),flush=True)

    # Native QGIS geometry join; retain complete modeled polygons only.
    source_shape = ROOT/'data/slope_units/slope_units_final.shp'
    source_layer = QgsVectorLayer(str(source_shape),'source','ogr')
    if not source_layer.isValid():
        raise ValueError('Cannot read current slope-unit geometry')
    memory = QgsVectorLayer('MultiPolygon?crs=EPSG:32649','消落区专用模型_独立Jenks五级','memory')
    provider = memory.dataProvider()
    fields = [QgsField('unit_id',QVariant.Int),QgsField('score',QVariant.Double),
        QgsField('label',QVariant.Int),QgsField('county',QVariant.String),
        QgsField('area_m2',QVariant.Double),QgsField('fold',QVariant.Int),
        QgsField('level',QVariant.Int),QgsField('grade',QVariant.String)]
    provider.addAttributes(fields)
    memory.updateFields()
    project = QgsProject.instance()
    crs = QgsCoordinateReferenceSystem('EPSG:32649')
    project.setCrs(crs)
    transform = QgsCoordinateTransform(source_layer.crs(),crs,project)
    lookup = {r['unit_id']:r for r in rows}
    features = []
    worst_area_error = 0.
    for feature in source_layer.getFeatures():
        uid = int(feature['unit_id'])
        if uid not in lookup:
            continue
        row = lookup[uid]
        geometry = QgsGeometry(feature.geometry())
        geometry.transform(transform)
        geometry.convertToMultiType()
        if geometry.isEmpty() or not geometry.isGeosValid():
            raise ValueError('Invalid selected geometry: '+str(uid))
        error = abs(geometry.area()-row['area_m2'])/row['area_m2']
        worst_area_error = max(worst_area_error,error)
        if error>1e-6:
            raise ValueError('Projected area differs: '+str(uid))
        item = QgsFeature(memory.fields())
        item.setGeometry(geometry)
        item.setAttributes([row[f.name()] for f in fields])
        features.append(item)
    if len(features)!=7101 or not provider.addFeatures(features)[0]:
        raise ValueError('Incomplete geometry join')
    memory.updateExtents()
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = 'GPKG'
    options.layerName = 'drawdown84_jenks5'
    options.fileEncoding = 'UTF-8'
    gpkg = args.out/'drawdown84_jenks5.gpkg'
    result = QgsVectorFileWriter.writeAsVectorFormatV3(memory,str(gpkg),project.transformContext(),options)
    if result[0]!=QgsVectorFileWriter.NoError:
        raise ValueError('GPKG write failed: '+str(result))
    layer = QgsVectorLayer(str(gpkg)+'|layername=drawdown84_jenks5','消落区专用模型_独立Jenks五级','ogr')
    if not layer.isValid() or layer.featureCount()!=7101:
        raise ValueError('GPKG reload failed')
    categories = []
    for level in [5,4,3,2,1]:
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol.setColor(QColor(COLORS[level]))
        symbol.symbolLayer(0).setStrokeStyle(Qt.NoPen)
        categories.append(QgsRendererCategory(level,symbol,NAMES[level]))
    layer.setRenderer(QgsCategorizedSymbolRenderer('level',categories))
    for feature in layer.getFeatures():
        row = lookup[int(feature['unit_id'])]
        if feature['level']!=row['level'] or float(feature['score'])!=row['score']:
            raise ValueError('Saved geometry score/grade differs')
    message, success = layer.saveNamedStyle(str(args.out/'drawdown84_jenks5.qml'))
    if not success:
        raise ValueError('QML save failed: '+message)
    layer.saveStyleToDatabase('drawdown84_jenks5','Independent drawdown QGIS Jenks five grades',True,'')
    project.addMapLayer(layer)
    metadata = dict(scope='drawdown-only; independent of whole-study-area grade system',
        qgis_version=Qgis.QGIS_VERSION,algorithm=classifier.id(),classes=5,
        algorithm_documentation='https://api.qgis.org/api/3.44/qgsclassificationjenks_8cpp_source.html',
        boundaries=boundaries,minimum=min(r['score'] for r in rows),maximum=max(r['score'] for r in rows),
        inclusion='7,101 modeled drawdown units; excludes 430 nonmodeled river units',
        threshold_rule='All counties share drawdown-wide thresholds; upper bounds inclusive',
        weighting='One score per slope unit; area/labels not supplied to classifier',
        classifier_implementation='Native QGIS Jenks; default internal sampling behavior retained',
        score_source='Complete fivefold OOF, seed42; not three-seed mean',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [source,args.results/'config.json',args.results/'summary.json',source_shape]},
        maximum_relative_geometry_area_error=worst_area_error,
        whole_area='Existing whole-study-area grades and approved atlas not modified')
    (args.out/'grading_method.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('QGIS_OUTPUT',args.out,flush=True)
    print('COUNTIES',json.dumps(counties,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results/v3_received_drawdown84_spatial_fivefold_s42')
    parser.add_argument('--out',type=Path,default=ROOT/'results/v3_drawdown84_qgis_jenks_s42')
    args = parser.parse_args()
    app = QgsApplication([],False)
    app.initQgis()
    main(args)
    # Project-owned providers are closed before shutdown.
    QgsProject.instance().clear()
    app.exitQgis()
