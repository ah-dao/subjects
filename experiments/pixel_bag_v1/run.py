"""Independent retrospective pixel-bag experiment. See README.md in this directory."""
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
IMPLEMENTATION = "pixel_bag_followup_v1.1"
BASE_ARMS = ("table_xgb", "stats_xgb", "mil", "concat_mil", "water_mil")
CONTROL_ARMS = ("table_mlp", "mean_mil", "concat_mean_mil")
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


def prepare(args):
    import rasterio
    from pyproj import CRS, Transformer
    from shapely.ops import transform
    sys.path.insert(0, str(ROOT / "tills"))
    from shp_io import read_polygons

    out = fresh_dir(args.out)
    paths = {
        "bags": Path(args.bags),
        "features": ROOT / "features/v2/features_v2_train_38.csv",
        "shp": ROOT / "data/slope_units/slope_units_final_train.shp",
        "county": ROOT / "features/v2/county_units_v2_train.csv",
        "levels": ROOT / "features/v2/sources/daily_station_levels.csv",
        "stations": ROOT / "features/v2/sources/station_assign_v2.csv",
        "grid": ROOT / "features/v2/unit_id_grid.tif",
    }
    f = pd.read_csv(paths["features"])
    if not f.unit_id.is_unique or not set(f.label.unique()) <= {0, 1}:
        raise ValueError("Invalid/duplicate feature IDs or labels")
    tab_names = [c for c in f if c not in ("unit_id", "label")]
    if len(tab_names) != 38:
        raise ValueError("Expected the explicit 38-feature table")
    ids = f.unit_id.to_numpy(dtype="int64")
    with np.load(paths["bags"], allow_pickle=True) as z:
        raw = z["X"]
        uid = z["uid"].astype("int64")
        rows = z["row"].astype("int64")
        cols = z["col"].astype("int64")
        bands = [str(b) for b in z["bands"]]
    if not (len(raw) == len(uid) == len(rows) == len(cols)):
        raise ValueError("Bag arrays have different lengths")
    good, x = encode_terrain(raw, bands)
    raw_counts = pd.Series(uid).value_counts()
    used = np.isin(uid[good], ids)
    uid, rows, cols, x = uid[good][used], rows[good][used], cols[good][used], x[used]
    del raw
    order = np.argsort(uid, kind="stable")
    uid, rows, cols, x = uid[order], rows[order], cols[order], x[order]
    with rasterio.open(paths["grid"]) as grid:
        tr, crs = grid.transform, grid.crs
        if tr.b != 0 or tr.d != 0:
            raise ValueError("Rotated grids not supported")
        if (rows < 0).any() or (cols < 0).any() or (rows >= grid.height).any() or (cols >= grid.width).any():
            raise ValueError("Pixel row/column out of grid bounds")
        # Verify the exact grid identity at deterministic sample locations.
        for k in np.linspace(0, len(uid)-1, min(100, len(uid)), dtype=int):
            from rasterio.windows import Window
            v = grid.read(1, window=Window(int(cols[k]), int(rows[k]), 1, 1))[0, 0]
            if int(v) != int(uid[k]):
                raise ValueError("Bag ID/grid identity mismatch")
        if crs.is_geographic:
            geod = CRS.from_user_input(crs).get_geod()
            row_area = np.empty(grid.height)
            for r in range(grid.height):
                left, top = tr * (0, r)
                right, bottom = tr * (1, r+1)
                row_area[r] = abs(geod.polygon_area_perimeter(
                    [left, right, right, left], [top, top, bottom, bottom])[0])
            pixel_area = row_area[rows].astype("float32")
        elif crs.is_projected:
            factor = CRS.from_user_input(crs).axis_info[0].unit_conversion_factor
            pixel_area = np.full(len(uid), abs(tr.a * tr.e) * factor**2, dtype="float32")
        else:
            raise ValueError("Unrecognized grid CRS")

    # Historical fixed conditions, NOT event-time/forecast conditions.
    levels = pd.read_csv(paths["levels"], parse_dates=["date"])
    if not np.isfinite(levels.level).all() or levels.duplicated(["station", "date"]).any():
        raise ValueError("Invalid or duplicate station water-level observations")
    assign = pd.read_csv(paths["stations"])
    if not assign.unit_id.is_unique:
        raise ValueError("Duplicate station assignment")
    station = f[["unit_id"]].merge(assign[["unit_id", "station"]],
                                    validate="one_to_one", how="left").station
    if station.isna().any():
        raise ValueError("Missing station assignment")
    water = np.zeros((len(x), 4), dtype="float32")
    station_dates = {}
    for st in station.unique():
        lev = levels.loc[levels.station == st].sort_values("date")
        if not len(lev):
            raise ValueError(f"No levels for station {st}")
        H = np.sort(lev.level.to_numpy())
        pix = np.isin(uid, ids[station.to_numpy() == st])
        elev = x[pix, 0]
        water[pix, 0] = 1 - np.searchsorted(H, elev, side="left") / len(H)
        water[pix, 1:] = (elev[:, None] - np.quantile(H, [.1, .5, .9])) / 30.0
        station_dates[str(st)] = {"first": str(lev.date.min()), "last": str(lev.date.max()),
                                  "n": len(H)}
    allx = np.column_stack((x, water))
    del water, x
    starts = np.searchsorted(uid, ids, side="left")
    ends = np.searchsorted(uid, ids, side="right")
    counts = ends - starts
    summaries = np.empty((len(ids), allx.shape[1] * 5 + 4), dtype="float32")
    for i, (s, e) in enumerate(zip(starts, ends)):
        xx, ww = allx[s:e], pixel_area[s:e]
        summaries[i, :-4] = weighted_summary(xx, ww)
        if e > s:
            w = ww / ww.sum()
            wet = xx[:, 4]
            mean_slope = np.sum(w * xx[:, 1])
            mean_wet = np.sum(w * wet)
            joint = [np.sum(w * wet * xx[:, j]) / max(mean_wet, 1e-8) for j in (1, 2, 3)]
            joint.append(np.sum(w * (wet-mean_wet) * (xx[:, 1]-mean_slope)))
            summaries[i, -4:] = joint
        else:
            summaries[i, -4:] = np.nan
    # Geometry explicitly aligned by ID, not by shapefile row assumptions.
    geoms, attrs, wkt = read_polygons(str(paths["shp"]), want_fields=["unit_id"])
    geo_ids = np.asarray(attrs["unit_id"], dtype=int)
    if len(np.unique(geo_ids)) != len(geo_ids) or set(geo_ids) != set(ids):
        raise ValueError("Training geometry/feature population mismatch")
    transformer = Transformer.from_crs(CRS.from_wkt(wkt), 32649, always_xy=True)
    gg = [transform(transformer.transform, g) for g in geoms]
    position = {int(u): i for i, u in enumerate(geo_ids)}
    indices = [position[int(u)] for u in ids]
    cent = np.array([[gg[i].centroid.x, gg[i].centroid.y] for i in indices])
    area = np.array([gg[i].area for i in indices])
    if not np.isfinite(cent).all() or not np.isfinite(area).all() or (area <= 0).any():
        raise ValueError("Invalid centroid/area")
    county = pd.read_csv(paths["county"])
    county = f[["unit_id"]].merge(county[["unit_id", "county"]], validate="one_to_one", how="left")
    if county.county.isna().any():
        raise ValueError("Missing county")
    # Same availability information supplied to every arm, including table-only controls.
    availability = np.column_stack((np.log1p(counts), counts > 0))
    tab = np.column_stack((f[tab_names].to_numpy(dtype="float32"), availability))
    audit = f[["unit_id", "label"]].copy()
    audit["n_raw"] = [int(raw_counts.get(u, 0)) for u in ids]
    audit["n_valid"] = counts
    audit["empty_bag"] = counts == 0
    audit.to_csv(out / "unit_audit.csv", index=False)
    np.save(out / "pixels.npy", allx)
    np.save(out / "pixel_area.npy", pixel_area)
    np.savez(out / "units.npz", unit_id=ids, y=f.label.to_numpy(dtype=int), tab=tab,
             summary=summaries, starts=starts, ends=ends, cent=cent, area=area,
             zone=f.inundation_fraction.to_numpy() > 0, county=county.county.to_numpy(dtype=str))
    summary_names = [f"{stat}_{name}" for stat in ("mean", "sd", "p10", "p50", "p90")
                     for name in TERRAIN_NAMES + WATER_NAMES]
    manifest = {"version": VERSION, "purpose": "retrospective spatial susceptibility experiment",
                "not_prospective_forecast": True, "units": len(ids), "positives": int(f.label.sum()),
                "valid_training_pixels": len(uid), "raw_bag_pixels": int(raw_counts.sum()),
                "empty_units": int((counts == 0).sum()),
                "empty_positive_units": int(((counts == 0) & (f.label.to_numpy() == 1)).sum()),
                "low_pixel_positive_units": int(((counts < 8) & (f.label.to_numpy() == 1)).sum()),
                "excluded_channels": ["curvature", "TRI"], "pixel_names": TERRAIN_NAMES + WATER_NAMES,
                "tab_names": tab_names + ["log1p_valid_pixels", "has_valid_pixels"],
                "summary_names": summary_names + ["wet_weighted_slope", "wet_weighted_aspect_sin",
                                                     "wet_weighted_aspect_cos", "wet_slope_covariance"],
                "station_dates": station_dates,
                "area_weighting": "ellipsoidal row pixel areas for geographic grid",
                "sources": {k: fingerprint(v) for k, v in paths.items()},
                "script": fingerprint(__file__)}
    for ext in (".dbf", ".shx", ".prj"):
        p = paths["shp"].with_suffix(ext)
        manifest["sources"]["shp"+ext] = fingerprint(p)
    write_json(out / "manifest.json", manifest)  # completion marker written last
    print(json.dumps({k: manifest[k] for k in ("units", "positives", "valid_training_pixels",
                                              "empty_units", "empty_positive_units")}, indent=2))
    print("Prepared:", out)


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
            channels = 8 if arm in ("concat_mil", "concat_mean_mil") else 4
            self.local = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            self.condition = nn.Linear(4, hidden * 2) if arm == "water_mil" else None
            self.attention = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            # An explicit table-only fallback for empty bags.
            self.table_head = nn.Sequential(nn.Linear(tab_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.bag_head = nn.Sequential(nn.Linear(hidden + tab_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, bag, tab, mask, weights):
            if self.arm == "table_mlp":
                # Matched table paths: same two heads and empty-bag fallback as MIL,
                # but no pixel values, water conditions or learned bag representation.
                pooled = tab.new_zeros((len(tab), self.hidden))
                base = self.table_head(tab).squeeze(-1)
                delta = self.bag_head(torch.cat((pooled, tab), dim=1)).squeeze(-1)
                return base + mask.any(dim=1).float() * delta
            channels = 8 if self.arm in ("concat_mil", "concat_mean_mil") else 4
            h = self.local(bag[:, :, :channels])
            if self.condition is not None:
                gamma, beta = self.condition(bag[:, :, 4:]).chunk(2, dim=-1)
                h = torch.relu(h * (1 + .5 * torch.tanh(gamma)) + .5 * torch.tanh(beta))
            if self.arm in ("mean_mil", "concat_mean_mil"):
                attn = weights * mask
            else:
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
    pm, ps = (np.zeros(pixels.shape[1]), np.ones(pixels.shape[1])) if arm == "table_mlp" else fold_pixel_scaler(pixels, d, train)
    np.savez(model_dir / "scaler.npz", tab_mean=tm, tab_std=ts, pixel_mean=pm, pixel_std=ps)

    def batch(ii, epoch=0, rep=0, training=False):
        bags, masks, ww = [], [], []
        for i in ii:
            if arm == "table_mlp":
                b = np.zeros((args.k, pixels.shape[1]), dtype="float32")
                m = np.zeros(args.k, dtype=bool)
                w = np.zeros(args.k, dtype="float32")
                if d["ends"][i] > d["starts"][i]:
                    m[0], w[0] = True, 1.
                bags.append(b); masks.append(m); ww.append(w)
                continue
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
    out = fresh_dir(args.out)
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
                                     "scope": "partial screening" if args.only_fold is not None else "full CV",
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
    a.add_argument("--bags", default=str(ROOT / "features/v2/unit_bags.npz"))
    a.add_argument("--out", required=True)
    a.set_defaults(func=prepare)
    a = sub.add_parser("train")
    a.add_argument("--prepared", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--arms", default="table_xgb,stats_xgb,mil,concat_mil,water_mil")
    a.add_argument("--method", choices=["spatial_kmeans", "admin"], default="spatial_kmeans")
    a.add_argument("--folds", type=int, default=5)
    a.add_argument("--fold-seed", type=int, default=42)
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
