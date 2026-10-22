"""Descriptive fold analysis and exploratory fixed-prediction spatial bootstrap."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run import ROOT, write_json


def auc_setup(y, p):
    order = np.argsort(p, kind="stable")
    groups = np.r_[0, np.flatnonzero(np.diff(p[order]))+1]
    yy = y[order]

    def score(weight):
        w = weight[order]
        positive = np.add.reduceat(w*yy, groups)
        negative = np.add.reduceat(w*(1-yy), groups)
        denom = positive.sum()*negative.sum()
        if denom == 0:
            return np.nan
        return float(np.sum(positive*(np.cumsum(negative)-.5*negative))/denom)
    return score


def main(args):
    received = Path(args.received)
    out = received / "analysis"
    out.mkdir(exist_ok=False)
    pred = pd.read_csv(received / "return/run/predictions.csv")
    fold_rows = json.loads((received / "return/run/fold_metrics.json").read_text(encoding="utf-8"))
    table = pd.DataFrame([dict(fold=r["fold"], seed=r["seed"], arm=r["arm"], domain=domain, **m)
                          for r in fold_rows for domain, m in r["metrics"].items()])
    table.to_csv(out / "fold_metrics.csv", index=False)
    macro = table.groupby(["domain", "arm"])[["auc", "pr_auc", "coverage_at_area_0.1",
                                               "coverage_at_area_0.2"]].mean()
    macro.to_csv(out / "macro_metrics.csv")
    # Explicitly align geometry IDs; no reuse of supplied or returned Python code.
    import rasterio  # rasterio before pyproj on the local Windows environment
    from pyproj import CRS, Transformer
    from shapely.ops import transform
    sys.path.insert(0, str(ROOT / "tills"))
    from shp_io import read_polygons
    geom, attrs, wkt = read_polygons(str(ROOT / "data/slope_units/slope_units_final_train.shp"),
                                    want_fields=["unit_id"])
    tf = Transformer.from_crs(CRS.from_wkt(wkt), 32649, always_xy=True)
    centers = {}
    for uid, g in zip(attrs["unit_id"], geom):
        gg = transform(tf.transform, g)
        centers[int(uid)] = [gg.centroid.x, gg.centroid.y]
    ref = pred.loc[pred.arm == "table_xgb"].sort_values("unit_id")
    ids = ref.unit_id.to_numpy()
    yy = ref.label.to_numpy()
    zone = ref.zone.to_numpy(dtype=bool)
    xy = np.array([centers[int(u)] for u in ids])
    blocks = np.floor(xy/args.block_m).astype(int)
    _, inv = np.unique(blocks, axis=0, return_inverse=True)
    n_blocks = inv.max()+1
    arms = pred.arm.unique()
    probs = {arm: pred.loc[pred.arm == arm].set_index("unit_id").p.reindex(ids).to_numpy() for arm in arms}
    domains = {"all": np.ones(len(ids), bool), "drawdown_zone": zone}
    setups = {}
    for domain, mask in domains.items():
        for arm in arms:
            setup = auc_setup(yy[mask], probs[arm][mask])
            np.testing.assert_allclose(setup(np.ones(mask.sum())), roc_auc_score(yy[mask], probs[arm][mask]))
            setups[domain, arm] = setup
    comparisons = [("concat_mil", "table_xgb"), ("concat_mil", "stats_xgb"),
                   ("concat_mil", "mil"), ("water_mil", "concat_mil")]
    draws = {(domain, a, b): [] for domain in domains for a, b in comparisons}
    rng = np.random.default_rng(args.seed)
    for _ in range(args.reps):
        count = np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks)
        w = count[inv]
        for domain, mask in domains.items():
            scores = {arm: setups[domain, arm](w[mask]) for arm in arms}
            for a, b in comparisons:
                draws[domain, a, b].append(scores[a]-scores[b])
    result = []
    for (domain, a, b), samples in draws.items():
        mask = domains[domain]
        delta = setups[domain, a](np.ones(mask.sum()))-setups[domain, b](np.ones(mask.sum()))
        valid = np.asarray(samples)
        valid = valid[np.isfinite(valid)]
        result.append({"domain": domain, "a": a, "b": b, "delta_auc": delta,
                       "percentile_95_interval": np.quantile(valid, [.025, .975]).tolist(),
                       "valid_reps": len(valid)})
    write_json(out / "spatial_bootstrap.json", {"block_m": args.block_m, "origin_utm": [0, 0],
        "blocks": int(n_blocks), "seed": args.seed, "reps": args.reps,
        "interpretation": "exploratory paired block bootstrap of fixed OOF predictions; no refitting; not seed/training uncertainty; no multiplicity correction",
        "comparisons": result})
    print(json.dumps(result, indent=2))
    print("macro metrics\n", macro.round(6).to_string())
    print("Saved:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--received", required=True)
    parser.add_argument("--block-m", type=float, default=3000)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261015)
    main(parser.parse_args())
