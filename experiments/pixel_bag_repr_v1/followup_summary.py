"""Separate seed variation, fold variation and probability ensembling."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run import fingerprint, fresh_dir, metrics, write_json

MEASURES = ["auc", "pr_auc", "brier", "coverage_at_area_0.1", "coverage_at_area_0.2"]
COMPARISONS = [(a, b) for a in ("relative", "level", "wet", "concat")
               for b in ("terrain", "stats_xgb")] + [("relative", "concat"), ("level", "concat"), ("wet", "concat")]


def write_reports(out, pred, population):
    out = Path(out)
    if pred.duplicated(["unit_id", "arm", "seed"]).any() or pred.isna().any().any():
        raise ValueError("Duplicate/missing predictions")
    if not pred.p.between(0, 1).all():
        raise ValueError("Invalid probabilities")
    metadata = pred[["unit_id", "label", "fold", "area_m2", "zone", "has_pixels"]].drop_duplicates()
    if not metadata.unit_id.is_unique:
        raise ValueError("Mismatched labels/folds/area/domain across model runs")
    population = set(population)
    seed_rows, fold_rows, ensemble_rows = [], [], []
    for (arm, seed), frame in pred.groupby(["arm", "seed"]):
        frame = frame.sort_values("unit_id")
        complete = set(frame.unit_id) == population
        result = metrics(frame.label.to_numpy(), frame.p.to_numpy(), frame.area_m2.to_numpy(),
                         frame.zone.to_numpy(dtype=bool), frame.unit_id.to_numpy())
        for domain, values in result.items():
            seed_rows.append(dict(arm=arm, seed=int(seed), domain=domain, complete_oof=complete, **values))
        for fold, ff in frame.groupby("fold"):
            result = metrics(ff.label.to_numpy(), ff.p.to_numpy(), ff.area_m2.to_numpy(),
                             ff.zone.to_numpy(dtype=bool), ff.unit_id.to_numpy())
            for domain, values in result.items():
                fold_rows.append(dict(arm=arm, seed=int(seed), fold=int(fold), domain=domain, **values))
    for arm, frame in pred.groupby("arm"):
        seeds = sorted(frame.seed.unique())
        if not all(set(x.unit_id) == set(frame.unit_id) for _, x in frame.groupby("seed")):
            raise ValueError("Seed ensembles require identical evaluated unit sets")
        ff = frame.groupby("unit_id", sort=True).agg(label=("label", "first"), p=("p", "mean"),
            area=("area_m2", "first"), zone=("zone", "first"))
        result = metrics(ff.label.to_numpy(), ff.p.to_numpy(), ff.area.to_numpy(),
                         ff.zone.to_numpy(dtype=bool), ff.index.to_numpy())
        for domain, values in result.items():
            ensemble_rows.append(dict(arm=arm, seeds=",".join(str(int(s)) for s in seeds),
                                      seed_count=len(seeds), domain=domain, **values))
    seed_df = pd.DataFrame(seed_rows)
    fold_df = pd.DataFrame(fold_rows)
    seed_df.to_csv(out / "seed_metrics.csv", index=False)
    fold_df.to_csv(out / "fold_metrics.csv", index=False)
    macro = fold_df.groupby(["arm", "seed", "domain"], as_index=False)[MEASURES].mean()
    macro.to_csv(out / "macro_metrics.csv", index=False)
    pd.DataFrame(ensemble_rows).to_csv(out / "ensemble_metrics.csv", index=False)
    variability = seed_df.groupby(["arm", "domain"])[MEASURES].agg(["mean", "std", "count"])
    variability.to_csv(out / "seed_variability.csv")
    deltas = []
    for a, b in COMPARISONS:
        aa = seed_df.loc[seed_df.arm == a].set_index(["seed", "domain"])
        bb = seed_df.loc[seed_df.arm == b].set_index(["seed", "domain"])
        for key in aa.index.intersection(bb.index):
            # Pair only exactly the same held-out population within a seed.
            pa = pred.loc[(pred.arm == a) & (pred.seed == key[0])].unit_id
            pb = pred.loc[(pred.arm == b) & (pred.seed == key[0])].unit_id
            if set(pa) != set(pb):
                raise ValueError("Cannot pair different evaluated populations")
            row = dict(a=a, b=b, seed=int(key[0]), domain=key[1])
            for measure in MEASURES:
                row["delta_"+measure] = aa.loc[key, measure]-bb.loc[key, measure]
            ma = macro.loc[(macro.arm == a) & (macro.seed == key[0]) & (macro.domain == key[1])].iloc[0]
            mb = macro.loc[(macro.arm == b) & (macro.seed == key[0]) & (macro.domain == key[1])].iloc[0]
            row["delta_macro_auc"] = ma.auc-mb.auc
            fa = fold_df.loc[(fold_df.arm == a) & (fold_df.seed == key[0]) & (fold_df.domain == key[1])].set_index("fold")
            fb = fold_df.loc[(fold_df.arm == b) & (fold_df.seed == key[0]) & (fold_df.domain == key[1])].set_index("fold")
            row["auc_winning_folds"] = int((fa.auc > fb.auc).sum())
            row["fold_count"] = len(fa)
            deltas.append(row)
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(out / "paired_seed_differences.csv", index=False)
    if len(delta_df):
        delta_df.groupby(["a", "b", "domain"])[[c for c in delta_df if c.startswith("delta_")]].agg(
            ["mean", "std", "count"]).to_csv(out / "paired_seed_variability.csv")
    write_json(out / "report_notes.json", {
        "pr_auc": "average precision (AP), not trapezoidal PR integral",
        "seed_variability": "mean/std of separately evaluated trained seeds; not ensemble performance",
        "ensemble": "average probabilities across the listed seeds, evaluated once",
        "macro": "equal mean of fold metrics; separate from pooled OOF ranking",
        "significance": "No p-values or independent-fold assumptions; seeds are not independent new datasets"})


def combine(args):
    configs, frames, populations = [], [], []
    invariants = ["method", "folds", "fold_seed", "only_fold", "sampling_seed", "k", "hidden",
                  "batch", "epochs", "trees", "eval_reps", "lr", "neg_sampling", "neg_km", "neg_lam"]
    for dirname in args.runs:
        directory = Path(dirname)
        c = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        manifest = c["prepared_manifest"]
        if configs:
            for key in invariants:
                if c["args"].get(key) != configs[0]["args"].get(key):
                    raise ValueError("Cannot combine different experimental settings: " + key)
            if manifest != configs[0]["prepared_manifest"]:
                raise ValueError("Prepared cache manifests differ")
            if c["versions"] != configs[0]["versions"]:
                raise ValueError("Environment versions differ; require explicit separate analysis")
        if c["scope"] != "full CV":
            raise ValueError("This combiner only accepts full CV; do not mix smoke results")
        configs.append(c)
        fold = pd.read_csv(directory / "folds.csv").sort_values("unit_id").reset_index(drop=True)
        if populations:
            pd.testing.assert_frame_equal(fold, populations[0])
        populations.append(fold)
        frame = pd.read_csv(directory / "predictions.csv")
        for _, f in frame.groupby(["arm", "seed"]):
            if set(f.unit_id) != set(fold.unit_id):
                raise ValueError("Incomplete arm/seed population")
        frames.append(frame)
    pred = pd.concat(frames, ignore_index=True)
    out = fresh_dir(args.out)
    write_reports(out, pred, populations[0].unit_id)
    pred.to_csv(out / "predictions.csv", index=False)
    populations[0].to_csv(out / "folds.csv", index=False)
    write_json(out / "provenance.json", {"runs": [str(Path(x)) for x in args.runs],
        "configs": configs, "prediction_files": [fingerprint(Path(x) / "predictions.csv") for x in args.runs]})
    print("Combined reports:", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    combine(p.parse_args())
