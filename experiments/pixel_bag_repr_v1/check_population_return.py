"""Independently cross-check development membership/labels against local originals."""
import io
import json
from pathlib import Path
import zipfile
import hashlib
import pandas as pd
from run import write_json

root = Path(__file__).resolve().parents[2]
out = root/'results/pixel_bag_repr_v1/received_development'
protocol = json.loads((out/'run/population_protocol.json').read_text())
config = json.loads((out/'run/config.json').read_text())
with zipfile.ZipFile(root/'pixel_bag_v1_spatial_s42_e12_return.zip') as z:
    matches = [n for n in z.namelist() if n.endswith('/folds.csv')]
    assert len(matches) == 1
    raw = z.read(matches[0])
    assert hashlib.sha256(raw).hexdigest() == protocol['reference']['sha256']
    outer = pd.read_csv(io.BytesIO(raw))
assert set(outer.loc[outer.fold==4,'unit_id']) == set(protocol['excluded_ids'])
assert set(outer.loc[outer.fold!=4,'unit_id']) == set(protocol['included_ids'])
labels = pd.read_csv(root/'features/v2/features_v2_train_38.csv')
pred = pd.read_csv(out/'run/predictions.csv').drop_duplicates('unit_id')
joined = pred.merge(labels[['unit_id','label','inundation_fraction']],on='unit_id',suffixes=('', '_original'),validate='one_to_one')
assert len(joined)==len(pred)
assert (joined.label==joined.label_original).all()
assert (joined.zone==(joined.inundation_fraction>0)).all()
losses = []
for path in (out/'run').glob('fold*/loss.json'):
    values = json.loads(path.read_text())
    assert len(values)==12 and all(pd.notna(x) and abs(x)<float('inf') for x in values)
    losses.append(dict(model=path.parent.name,first=values[0],last=values[-1]))
assert len(losses)==45
pd.DataFrame(losses).to_csv(out/'analysis/loss_endpoints.csv',index=False)
write_json(out/'analysis/population_crosscheck.json',dict(original_outer_fold_hash_matches=True,
    excluded_fold4_exact=True,labels_match=True,zone_definition_matches=True,neural_runs=45,epochs=12))
print('PASS: original outer fold4 excluded exactly; original labels and zone match; 45 neural runs x 12 epochs')
