"""Round-one input representation controls; see README.md. No v1 files are modified."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VERSION = "pixel_bag_v1.0"
IMPLEMENTATION = "pixel_bag_representation_v1.0"
BASE_ARMS = ("stats_xgb", "terrain", "relative", "level", "wet", "concat")
CONTROL_ARMS = ()
ARM_COLUMNS = {"terrain": [0,1,2,3], "relative": [0,1,2,3,8,9],
               "level": [0,1,2,3,6], "wet": [0,1,2,3,4], "concat": list(range(8))}
TERRAIN_NAMES = ["elevation", "slope", "aspect_sin", "aspect_cos"]
WATER_NAMES = ["historical_wet_fraction", "relative_level_p10", "relative_level_p50",
               "relative_level_p90"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"bytes": Path(path).stat().st_size, "sha256": h.hexdigest()}


def fresh_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def encode_terrain(raw, bands):
    """No learned preprocessing here. Curvature and TRI are deliberately unused."""
    ix = [bands.index(x) for x in ("elevation", "slope", "aspect")]
    a = raw[:, ix]
    good = (np.isfinite(a).all(axis=1) & (a[:, 1] >= 0) & (a[:, 1] <= 90)
            & (a[:, 2] >= 0) & (a[:, 2] < 360))
    a = a[good]
    rad = np.deg2rad(a[:, 2])
    # A flat pixel has no directional aspect.
    nonflat = a[:, 1] > 0
    x = np.column_stack((a[:, 0], a[:, 1], np.sin(rad) * nonflat,
                         np.cos(rad) * nonflat)).astype("float32")
    return good, x


def fit_scaler(x):
    """Explicit training-only imputation/scaling, including all-missing columns."""
    x = np.asarray(x, dtype="float64")
    finite = np.isfinite(x)
    count = finite.sum(axis=0)
    mean = np.divide(np.where(finite, x, 0).sum(axis=0), count,
                     out=np.zeros(x.shape[1]), where=count > 0)
    filled = np.where(finite, x, mean)
    std = filled.std(axis=0)
    std[std < 1e-6] = 1
    return mean, std


def apply_scaler(x, mean, std):
    return ((np.where(np.isfinite(x), x, mean) - mean) / std).astype("float32")


def weighted_summary(x, weights):
    """Area-weighted descriptive controls; deterministic, no population fit."""
    if not len(x):
        return np.full(x.shape[1] * 5, np.nan, dtype="float32")
    weights = weights / weights.sum()
    mean = np.sum(x * weights[:, None], axis=0)
    sd = np.sqrt(np.sum((x - mean) ** 2 * weights[:, None], axis=0))
    qs = []
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind="stable")
        cdf = np.cumsum(weights[order])
        qs.append(x[order[np.minimum(np.searchsorted(cdf, [.1, .5, .9]), len(x)-1)], j])
    return np.concatenate((mean, sd, np.asarray(qs).T.ravel())).astype("float32")


def relative_features(z, area):
    """Area-weighted mid-CDF and robust relative elevation; full bag, no labels."""
    z, area = np.asarray(z, dtype="float64"), np.asarray(area, dtype="float64")
    if not len(z):
        return np.empty((0, 2), dtype="float32")
    if not np.isfinite(z).all() or not np.isfinite(area).all() or (area <= 0).any():
        raise ValueError("Invalid full-bag elevation/area")
    values, inverse = np.unique(z, return_inverse=True)
    mass = np.bincount(inverse, weights=area)
    cumulative = np.cumsum(mass) / mass.sum()
    mid = cumulative - .5 * mass / mass.sum()
    q10, q50, q90 = values[np.searchsorted(cumulative, [.1, .5, .9])]
    # One metre floor avoids explosion for flat/near-flat bags; never fit on other units.
    relative = (z-q50) / max(float(q90-q10), 1.)
    return np.column_stack((mid[inverse], relative)).astype("float32")


def prepare(args):
    import shutil
    source = Path(args.source)
    manifest, d, pixels, areas = load_prepared(source)
    if manifest.get("pixel_names") != TERRAIN_NAMES + WATER_NAMES:
        raise ValueError("Source pixel channel names/order mismatch")
    if pixels.ndim != 2 or pixels.shape[1] != 8:
        raise ValueError("Expected original eight-channel v1 cache")
    if not np.array_equal(d["starts"], np.r_[0, d["ends"][:-1]]) or d["ends"][-1] != len(pixels):
        raise ValueError("Bag offsets must partition all pixels")
    out = fresh_dir(args.out)
    derived = np.lib.format.open_memmap(out / "pixels.npy", mode="w+", dtype="float32", shape=(len(pixels), 10))
    for i, (start, end) in enumerate(zip(d["starts"], d["ends"])):
        if not np.isfinite(pixels[start:end]).all():
            raise ValueError("Nonfinite source pixels")
        derived[start:end, :8] = pixels[start:end]
        derived[start:end, 8:] = relative_features(pixels[start:end, 0], areas[start:end])
    derived.flush()
    del derived, pixels, areas
    for name in ("units.npz", "pixel_area.npy", "unit_audit.csv"):
        shutil.copyfile(source / name, out / name)
    manifest = dict(manifest, representation_schema="relative_midcdf_iqr_v1",
                    relative_columns=["area_mid_cdf", "relative_elevation_1m_floor"],
                    representation_source={name: fingerprint(source / name) for name in
                        ("manifest.json", "units.npz", "pixels.npy", "pixel_area.npy")})
    manifest["pixel_names"] = TERRAIN_NAMES + WATER_NAMES + manifest["relative_columns"]
    manifest["representation_script"] = fingerprint(__file__)
    manifest["derived_files"] = {name: fingerprint(out / name) for name in
                                ("units.npz", "pixels.npy", "pixel_area.npy", "unit_audit.csv")}
    write_json(out / "manifest.json", manifest)
    print("Prepared:", out, flush=True)


def development_population(d, path, excluded):
    fixed = pd.read_csv(path)
    if not fixed.unit_id.is_unique or set(fixed.unit_id) != set(d["unit_id"]):
        raise ValueError("Development reference population mismatch")
    fold = fixed.set_index("unit_id").fold.reindex(d["unit_id"]).to_numpy()
    if not np.isfinite(fold).all() or excluded not in fold:
        raise ValueError("Invalid excluded fold")
    keep = fold != excluded
    n = len(keep)
    reduced = {k: v[keep] if v.ndim and len(v) == n else v for k, v in d.items()}
    return reduced, d["unit_id"][~keep]


def sample_bag(pixels, areas, s, e, k, seed):
    """Area-proportional draws with replacement; padding only for short bags."""
    bag = np.zeros((k, pixels.shape[1]), dtype="float32")
    mask = np.zeros(k, dtype=bool)
    weights = np.zeros(k, dtype="float32")
    if e <= s:
        return bag, mask, weights
    n = e-s
    if n <= k:
        take = np.arange(s, e)
        w = np.asarray(areas[s:e], dtype="float64")
        w /= w.sum()
    else:
        p = np.asarray(areas[s:e], dtype="float64")
        p /= p.sum()
        rng = np.random.default_rng(seed)
        take = s + rng.choice(n, k, replace=True, p=p)
        w = np.full(k, 1/k)
    bag[:len(take)] = pixels[take]
    mask[:len(take)] = True
    weights[:len(take)] = w
    return bag, mask, weights


def build_model(tab_dim, arm, hidden):
    import torch
    from torch import nn

    class BagModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.arm = arm
            self.hidden = hidden
            channels = 8
            self.local = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.attention = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            # An explicit table-only fallback for empty bags.
            self.table_head = nn.Sequential(nn.Linear(tab_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.bag_head = nn.Sequential(nn.Linear(hidden + tab_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, bag, tab, mask, weights):
            selected = bag[:, :, ARM_COLUMNS[self.arm]]
            encoded = bag.new_zeros((*bag.shape[:2], 8))
            encoded[:, :, :selected.shape[-1]] = selected
            h = self.local(encoded)
            logits = self.attention(h).squeeze(-1)
            logits = logits + torch.log(weights.clamp_min(1e-12))
            logits = logits.masked_fill(~mask, -1e9)
            attn = torch.softmax(logits, dim=1) * mask
            attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-12)
            pooled = (attn.unsqueeze(-1) * h).sum(dim=1)
            has = mask.any(dim=1).float()
            base = self.table_head(tab).squeeze(-1)
            delta = self.bag_head(torch.cat((pooled, tab), dim=1)).squeeze(-1)
            return base + has * delta

    return BagModel()


def metrics(y, p, area, zone, ids):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    def one(ii):
        yy, pp, aa = y[ii], p[ii], area[ii]
        result = {"units": len(ii), "positives": int(yy.sum())}
        result["auc"] = float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else None
        result["pr_auc"] = float(average_precision_score(yy, pp)) if yy.sum() else None
        result["brier"] = float(brier_score_loss(yy, pp)) if len(ii) else None
        order = np.lexsort((ids[ii], -pp))
        cum = np.cumsum(aa[order])
        for frac in (.1, .2):
            # Strict budget: do not include a whole unit that crosses the area limit.
            n = int(np.searchsorted(cum, aa.sum() * frac, side="right"))
            selected = order[:n]
            result[f"coverage_at_area_{frac}"] = float(yy[selected].sum()/yy.sum()) if yy.sum() else None
            result[f"actual_area_{frac}"] = float(aa[selected].sum()/aa.sum()) if aa.sum() else None
        return result
    idx = np.arange(len(y))
    return {"all": one(idx), "drawdown_zone_internal_ranking": one(idx[zone])}


def load_prepared(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != VERSION:
        raise ValueError("Prepared schema version mismatch")
    with np.load(path / "units.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    return manifest, d, np.load(path / "pixels.npy", mmap_mode="r"), np.load(path / "pixel_area.npy", mmap_mode="r")


def fold_pixel_scaler(pixels, d, train):
    # Stream sufficient statistics; no validation pixels enter these statistics.
    total = np.zeros(pixels.shape[1], dtype="float64")
    squares = total.copy()
    count = 0
    for i in train:
        a = np.asarray(pixels[d["starts"][i]:d["ends"][i]], dtype="float64")
        total += a.sum(axis=0)
        squares += (a*a).sum(axis=0)
        count += len(a)
    if not count:
        raise ValueError("No training pixels")
    mean = total/count
    std = np.sqrt(np.maximum(squares/count - mean**2, 0))
    std[std < 1e-6] = 1
    return mean, std


def nn_fit_predict(args, arm, d, pixels, areas, train, test, sw, seed, model_dir):
    import torch
    from torch.nn import functional as F
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    model = build_model(d["tab"].shape[1], arm, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tm, ts = fit_scaler(d["tab"][train])
    tab = apply_scaler(d["tab"], tm, ts)
    pm, ps = fold_pixel_scaler(pixels, d, train)
    np.savez(model_dir / "scaler.npz", tab_mean=tm, tab_std=ts, pixel_mean=pm, pixel_std=ps)

    def batch(ii, epoch=0, rep=0, training=False):
        bags, masks, ww = [], [], []
        for i in ii:
            # No model seed in bag draws: identical sampled inputs for comparison arms.
            offset = (epoch+1) if training else rep
            draw_seed = int((args.sampling_seed + int(d["unit_id"][i])*1009
                             + offset*1000003) % (2**32))
            b, m, w = sample_bag(pixels, areas, d["starts"][i], d["ends"][i], args.k, draw_seed)
            b[m] = apply_scaler(b[m], pm, ps)
            bags.append(b); masks.append(m); ww.append(w)
        return (torch.as_tensor(np.asarray(bags), device=device),
                torch.as_tensor(tab[ii], device=device),
                torch.as_tensor(np.asarray(masks), device=device),
                torch.as_tensor(np.asarray(ww), device=device))

    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_weight = 0., 0.
        for ii in np.array_split(rng.permutation(train), max(1, int(np.ceil(len(train)/args.batch)))):
            optimizer.zero_grad()
            logits = model(*batch(ii, epoch=epoch, training=True))
            yy = torch.as_tensor(d["y"][ii], dtype=torch.float32, device=device)
            weight = torch.as_tensor(sw[ii], dtype=torch.float32, device=device)
            losses = F.binary_cross_entropy_with_logits(logits, yy, reduction="none")
            loss = (losses*weight).sum()/weight.sum()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float((losses*weight).sum().detach().cpu())
            total_weight += float(weight.sum().cpu())
        history.append(total_loss/total_weight)
        print(f"{arm} seed={seed} epoch={epoch+1}/{args.epochs} loss={history[-1]:.6f}", flush=True)
    model.eval()
    pred = np.zeros(len(test), dtype="float64")
    with torch.no_grad():
        for rep in range(args.eval_reps):
            for s in range(0, len(test), args.batch):
                ii = test[s:s+args.batch]
                p = torch.sigmoid(model(*batch(ii, rep=rep))).cpu().numpy()
                pred[s:s+len(ii)] += p/args.eval_reps
    torch.save(model.state_dict(), model_dir / "weights.pt")
    write_json(model_dir / "loss.json", history)
    return pred


def train(args):
    from sklearn.cluster import KMeans
    from src.dataset import admin_folds, proximity_weights
    from xgboost import XGBClassifier

    manifest, d, pixels, areas = load_prepared(args.prepared)
    if manifest.get("representation_schema") != "relative_midcdf_iqr_v1" or pixels.shape[1] != 10:
        raise ValueError("Run representation prepare first")
    for name, expected in manifest["derived_files"].items():
        if fingerprint(Path(args.prepared) / name) != expected:
            raise ValueError("Derived cache changed: " + name)
    if args.development_folds:
        d, excluded_ids = development_population(d, args.development_folds, args.exclude_fold)
    else:
        excluded_ids = np.array([], dtype=int)
    out = fresh_dir(args.out)
    write_json(out / "population_protocol.json", {"mode": "development_inner_cv" if args.development_folds else "full_cv_revalidation",
        "excluded_ids": excluded_ids.tolist(), "included_ids": d["unit_id"].tolist(),
        "reference": fingerprint(args.development_folds) if args.development_folds else None,
        "independent_external_validation": False})
    n = len(d["y"])
    if args.method == "spatial_kmeans":
        # No silent change to a different folding method on failure.
        fold = KMeans(n_clusters=args.folds, random_state=args.fold_seed, n_init=10).fit_predict(d["cent"])
    else:
        county_path = out / "county.csv"
        pd.DataFrame({"unit_id": d["unit_id"], "county": d["county"]}).to_csv(county_path, index=False)
        fold = admin_folds(county_path, d["unit_id"], d["y"], args.folds)
    if args.frozen_folds:
        fixed = pd.read_csv(args.frozen_folds)
        if not fixed.unit_id.is_unique or set(fixed.unit_id) != set(d["unit_id"]):
            raise ValueError("Frozen fold population mismatch")
        expected = fixed.set_index("unit_id").fold.reindex(d["unit_id"]).to_numpy()
        if not np.array_equal(expected, fold):
            raise ValueError("Fold assignments differ from frozen reference; stop rather than silently change splits")
    pd.DataFrame({"unit_id": d["unit_id"], "fold": fold}).to_csv(out / "folds.csv", index=False)
    selected = list(range(args.folds)) if args.only_fold is None else [args.only_fold]
    if any(f < 0 or f >= args.folds for f in selected):
        raise ValueError("Invalid --only-fold")
    arms = args.arms.split(",")
    if not set(arms) <= set(BASE_ARMS + CONTROL_ARMS) or len(set(arms)) != len(arms):
        raise ValueError("Invalid/duplicate arms")
    seeds = [int(x) for x in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds")
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "torch", "rasterio", "pyproj", "shapely"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    write_json(out / "config.json", {"version": VERSION,
                                     "implementation": IMPLEMENTATION,
                                     "args": {k: v for k, v in vars(args).items() if k != "func"},
                                     "versions": versions,
                                     "prepared_manifest": manifest, "script": fingerprint(__file__),
                                     "report_helper": fingerprint(Path(__file__).with_name("followup_summary.py")),
                                     "preprocessing": "training-fold only; fixed epochs; no global OOF base",
                                     "scope": "development inner CV" if args.development_folds else "full CV revalidation",
                                     "arm_columns": ARM_COLUMNS, "neural_input_slots": 8,
                                     "budget_rule": "whole units, strict prefix <= budget; ties by unit_id"})
    (out / "run_source.py").write_bytes(Path(__file__).read_bytes())
    (out / "followup_summary_source.py").write_bytes(Path(__file__).with_name("followup_summary.py").read_bytes())
    rows, predictions = [], []
    for f in selected:
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        if len(np.unique(d["y"][tr])) != 2 or len(np.unique(d["y"][te])) != 2:
            raise ValueError("Both classes required in train/test folds")
        sw = np.ones(n)
        if args.neg_sampling == "soft":
            sw = proximity_weights(d["cent"], d["y"], tr[d["y"][tr] == 1],
                                   args.neg_km, args.neg_lam, candidate_idx=tr)
        print(f"Fold {f}: train={len(tr)} test={len(te)} positive_test={d['y'][te].sum()}", flush=True)
        for seed in seeds:
            for arm in arms:
                md = out / f"fold{f}_seed{seed}_{arm}"
                md.mkdir()
                if arm.endswith("xgb"):
                    features = d["tab"] if arm == "table_xgb" else np.column_stack((d["tab"], d["summary"]))
                    mean, std = fit_scaler(features[tr])
                    xx = apply_scaler(features, mean, std)
                    model = XGBClassifier(n_estimators=args.trees, max_depth=5, learning_rate=.05,
                                          subsample=.8, colsample_bytree=.8, eval_metric="logloss",
                                          random_state=seed+f, n_jobs=args.threads, tree_method="hist")
                    model.fit(xx[tr], d["y"][tr], sample_weight=sw[tr])
                    p = model.predict_proba(xx[te])[:, 1]
                    model.save_model(str(md / "model.json"))
                    np.savez(md / "scaler.npz", mean=mean, std=std)
                else:
                    p = nn_fit_predict(args, arm, d, pixels, areas, tr, te, sw, seed+f, md)
                if not np.isfinite(p).all():
                    raise ValueError("Non-finite predictions")
                score = metrics(d["y"][te], p, d["area"][te], d["zone"][te], d["unit_id"][te])
                rows.append({"fold": f, "seed": seed, "arm": arm, "metrics": score})
                for j, i in enumerate(te):
                    predictions.append({"unit_id": int(d["unit_id"][i]), "label": int(d["y"][i]),
                                        "fold": f, "seed": seed, "arm": arm, "p": float(p[j]),
                                        "area_m2": float(d["area"][i]), "zone": int(d["zone"][i]),
                                        "has_pixels": int(d["ends"][i] > d["starts"][i])})
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
                pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
                write_json(out / "fold_metrics.json", rows)
    pred = pd.DataFrame(predictions)
    # Pool only held-out rows; seed ensemble then one score per unit.
    aggregate = []
    for arm in arms:
        frame = pred.loc[pred.arm == arm].groupby("unit_id", sort=True).agg(
            label=("label", "first"), p=("p", "mean"), area=("area_m2", "first"), zone=("zone", "first"))
        aggregate.append({"arm": arm, "evaluated_units": len(frame),
                          "complete_oof": len(frame) == n,
                          "metrics": metrics(frame.label.to_numpy(), frame.p.to_numpy(), frame.area.to_numpy(),
                                             frame.zone.to_numpy(dtype=bool), frame.index.to_numpy())})
    write_json(out / "summary.json", aggregate)
    from followup_summary import write_reports
    write_reports(out, pred, d["unit_id"])
    print("DONE:", out, flush=True)


def export(args):
    """Small return package by default. Raw bags are an explicit optional backup."""
    prepared, run = Path(args.prepared), Path(args.run)
    if not (prepared / "manifest.json").exists() or not (run / "summary.json").exists():
        raise ValueError("Only export a completed prepare/run")
    if json.loads((prepared / "manifest.json").read_text(encoding="utf-8")) != json.loads((run / "config.json").read_text(encoding="utf-8"))["prepared_manifest"]:
        raise ValueError("Export cache differs from training cache")
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    selected = [(p, "run/" + str(p.relative_to(run)).replace("\\", "/"))
                for p in run.rglob("*") if p.is_file() and p.suffix in (".csv", ".json")]
    # XGB model JSON is not needed for review; checkpoints remain on the server.
    selected = [(p, name) for p, name in selected if p.name != "model.json"]
    selected += [(prepared / x, "prepared/"+x) for x in ("manifest.json", "unit_audit.csv")]
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    source_options = [run / "run_source.py", Path(__file__), Path(__file__).with_name("run_v1_frozen.py")]
    source = next((p for p in source_options if p.exists() and fingerprint(p) == config["script"]), None)
    if source is None:
        raise ValueError("Cannot find the exact source used for training; refusing to export wrong code")
    selected.append((source, "code/run.py"))
    helper = run / "followup_summary_source.py"
    if helper.exists():
        if fingerprint(helper) != config["report_helper"]:
            raise ValueError("Report helper source snapshot mismatch")
        selected.append((helper, "code/followup_summary.py"))
    if args.raw_bags:
        selected.append((Path(args.raw_bags), "raw/unit_bags.npz"))
    hashes = {}
    with zipfile.ZipFile(dest, "x", zipfile.ZIP_DEFLATED) as z:
        for p, name in selected:
            z.write(p, name)
            hashes[name] = fingerprint(p)
        z.writestr("checksums.json", json.dumps(hashes, indent=2))
    print("Export:", dest, "bytes:", dest.stat().st_size)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--source", required=True, help="Existing v1 prepared cache")
    a.add_argument("--out", required=True)
    a.set_defaults(func=prepare)
    a = sub.add_parser("train")
    a.add_argument("--prepared", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--arms", default=",".join(BASE_ARMS))
    a.add_argument("--method", choices=["spatial_kmeans", "admin"], default="spatial_kmeans")
    a.add_argument("--folds", type=int, default=5)
    a.add_argument("--fold-seed", type=int, default=42)
    mode = a.add_mutually_exclusive_group(required=True)
    mode.add_argument("--development-folds", help="Outer folds.csv; exclude one region before inner CV")
    mode.add_argument("--full-cv", action="store_true", help="Explicit full-population revalidation")
    a.add_argument("--exclude-fold", type=int, default=4)
    a.add_argument("--frozen-folds", help="Verify recomputed folds against an existing folds.csv")
    a.add_argument("--only-fold", type=int)
    a.add_argument("--seeds", default="42")
    a.add_argument("--sampling-seed", type=int, default=731)
    a.add_argument("--k", type=int, default=64)
    a.add_argument("--hidden", type=int, default=32)
    a.add_argument("--batch", type=int, default=128)
    a.add_argument("--epochs", type=int, default=12)
    a.add_argument("--trees", type=int, default=300)
    a.add_argument("--eval-reps", type=int, default=3)
    a.add_argument("--lr", type=float, default=.001)
    a.add_argument("--threads", type=int, default=4)
    a.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a.add_argument("--neg-sampling", choices=["soft", "none"], default="soft")
    a.add_argument("--neg-km", type=float, default=4.)
    a.add_argument("--neg-lam", type=float, default=.2)
    a.set_defaults(func=train)
    a = sub.add_parser("export")
    a.add_argument("--prepared", required=True)
    a.add_argument("--run", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--raw-bags", help="Optional raw bag backup in the return zip")
    a.set_defaults(func=export)
    args = p.parse_args()
    if args.command == "train":
        if min(args.k, args.hidden, args.batch, args.epochs, args.trees, args.eval_reps, args.threads) < 1:
            p.error("Counts must be positive")
        if args.folds < 2 or not 0 < args.neg_lam <= 1 or args.lr <= 0 or args.neg_km <= 0:
            p.error("Invalid folds/weight/rate/radius")
    args.func(args)


if __name__ == "__main__":
    main()
