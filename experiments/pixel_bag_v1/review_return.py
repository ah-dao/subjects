"""Validate and archive a server return without executing any returned code."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import zipfile

import numpy as np
import pandas as pd

from run import ROOT, fingerprint, metrics, write_json


def review(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError("Use a new output directory; existing results are never overwritten")
    out.mkdir(parents=True)
    report = {"return_archive": fingerprint(args.return_zip), "raw_archive": fingerprint(args.raw_tgz)}
    with zipfile.ZipFile(args.return_zip) as z:
        if z.testzip() is not None:
            raise ValueError("Bad ZIP CRC")
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP names")
        checksums = json.loads(z.read("checksums.json"))
        if set(names) != set(checksums) | {"checksums.json"}:
            raise ValueError("ZIP member/checksum mismatch")
        for name, expected in checksums.items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe archive path")
            data = z.read(name)
            if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise ValueError("Return checksum failure: " + name)
            dest = out / "return" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        manifest = json.loads(z.read("prepared/manifest.json"))
        config = json.loads(z.read("run/config.json"))
        summary = json.loads(z.read("run/summary.json"))
        pred = pd.read_csv(io.BytesIO(z.read("run/predictions.csv")))
        audit = pd.read_csv(io.BytesIO(z.read("prepared/unit_audit.csv")))
        folds = pd.read_csv(io.BytesIO(z.read("run/folds.csv")))
        code_hash = hashlib.sha256(z.read("code/run.py")).hexdigest()
    assert code_hash == config["script"]["sha256"]
    assert manifest == config["prepared_manifest"]
    report["prepare_and_training_code_identical"] = code_hash == manifest["script"]["sha256"]
    report["server_matches_local_experiment_code"] = code_hash == fingerprint(Path(__file__).with_name("run.py"))["sha256"]
    frozen = Path(__file__).with_name("run_v1_frozen.py")
    report["server_matches_frozen_first_release"] = frozen.exists() and code_hash == fingerprint(frozen)["sha256"]
    report["source_file_matches"] = {}
    source_paths = {"features": "features/v2/features_v2_train_38.csv",
                    "shp": "data/slope_units/slope_units_final_train.shp",
                    "county": "features/v2/county_units_v2_train.csv",
                    "levels": "features/v2/sources/daily_station_levels.csv",
                    "stations": "features/v2/sources/station_assign_v2.csv"}
    for key, name in source_paths.items():
        report["source_file_matches"][key] = fingerprint(ROOT / name) == manifest["sources"][key]
    expected_raw = {"features/v2/unit_bags.npz", "features/v2/unit_id_grid.tif",
                    "results/unit_grid_report.txt", "results/unit_bags_report.txt"}
    with tarfile.open(args.raw_tgz, "r:gz") as t:
        members = t.getmembers()
        if {m.name for m in members} != expected_raw or len(members) != 4 or not all(m.isfile() for m in members):
            raise ValueError("Unexpected raw backup contents")
        for m in members:
            dest = out / "raw_backup" / m.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with t.extractfile(m) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    for key, name in [("bags", "features/v2/unit_bags.npz"), ("grid", "features/v2/unit_id_grid.tif")]:
        if fingerprint(out / "raw_backup" / name) != manifest["sources"][key]:
            raise ValueError("Raw backup does not match training source: " + key)
    report["all_return_checksums_valid"] = True
    report["raw_bag_and_grid_match_training_manifest"] = True
    # Inspect only the known, generated NumPy numeric arrays; no returned script is imported.
    with np.load(out / "raw_backup/features/v2/unit_bags.npz", allow_pickle=False) as z:
        x = z["X"]
        ids = z["uid"]
        report["raw_bag"] = {"pixels": len(ids), "units": len(np.unique(ids)),
                              "shape": list(x.shape), "dtype": str(x.dtype),
                              "nonfinite_per_band": (~np.isfinite(x)).sum(axis=0).tolist()}
    assert audit.unit_id.is_unique and folds.unit_id.is_unique
    assert len(audit) == manifest["units"] and int(audit.label.sum()) == manifest["positives"]
    assert not pred.duplicated(["unit_id", "seed", "arm"]).any()
    assert not pred.isna().any().any()
    assert pred.p.between(0, 1).all() and (pred.area_m2 > 0).all()
    mapping = pred[["unit_id", "label", "fold"]].drop_duplicates()
    assert mapping.unit_id.is_unique
    assert mapping.set_index("unit_id").label.equals(audit.set_index("unit_id").label.reindex(mapping.unit_id))
    assert np.array_equal(mapping.fold, folds.set_index("unit_id").fold.reindex(mapping.unit_id))
    arms = config["args"]["arms"].split(",")
    assert set(pred.arm) == set(arms)
    ref = None
    for arm in arms:
        frame = pred.loc[pred.arm == arm].sort_values(["seed", "unit_id"])
        keys = frame[["unit_id", "seed", "label", "fold", "area_m2", "zone", "has_pixels"]].reset_index(drop=True)
        if ref is None:
            ref = keys
        else:
            pd.testing.assert_frame_equal(keys, ref)
    recomputed = []
    for s in summary:
        frame = pred.loc[pred.arm == s["arm"]].groupby("unit_id", sort=True).agg(
            label=("label", "first"), p=("p", "mean"), area=("area_m2", "first"), zone=("zone", "first"))
        score = metrics(frame.label.to_numpy(), frame.p.to_numpy(), frame.area.to_numpy(),
                        frame.zone.to_numpy(dtype=bool), frame.index.to_numpy())
        for domain in score:
            for key, value in score[domain].items():
                old = s["metrics"][domain][key]
                if value is None:
                    assert old is None
                else:
                    np.testing.assert_allclose(value, old, rtol=1e-10, atol=1e-12)
        recomputed.append({"arm": s["arm"], "metrics": score})
    report["metrics_recomputed_from_predictions_match"] = True
    report["prediction_rows"] = len(pred)
    report["evaluated_units"] = int(pred.unit_id.nunique())
    report["evaluated_positives"] = int(pred.loc[pred.arm == arms[0]].drop_duplicates("unit_id").label.sum())
    report["training_population"] = {"units": len(audit), "positives": int(audit.label.sum()),
        "raw_training_pixels": int(audit.n_raw.sum()), "valid_training_pixels": int(audit.n_valid.sum()),
        "removed_training_pixels": int((audit.n_raw-audit.n_valid).sum()),
        "positive_pixels_removed": int((audit.loc[audit.label == 1].n_raw-audit.loc[audit.label == 1].n_valid).sum()),
        "minimum_valid_positive_pixels": int(audit.loc[audit.label == 1].n_valid.min()),
        "empty_units": int((audit.n_valid == 0).sum())}
    report["recomputed"] = recomputed
    report["scope"] = config["scope"]
    report["config"] = config["args"]
    write_json(out / "review.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("recomputed", "config")}, indent=2))
    print("Saved:", out / "review.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--return-zip", required=True)
    p.add_argument("--raw-tgz", required=True)
    p.add_argument("--out", required=True)
    review(p.parse_args())
