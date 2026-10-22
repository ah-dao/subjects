"""Meaningful invariants plus a complete tiny CPU run; not a scientific validation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).with_name("run.py")
spec = importlib.util.spec_from_file_location("pixel_bag_run", SCRIPT)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class DataContracts(unittest.TestCase):
    def test_legacy_models_exactly_preserved(self):
        import torch
        frozen_spec = importlib.util.spec_from_file_location("frozen_pixel_bag", SCRIPT.with_name("run_v1_frozen.py"))
        frozen = importlib.util.module_from_spec(frozen_spec)
        frozen_spec.loader.exec_module(frozen)
        for arm in ("mil", "concat_mil", "water_mil"):
            torch.manual_seed(42)
            old = frozen.build_model(3, arm, 8)
            torch.manual_seed(42)
            new = run.build_model(3, arm, 8)
            self.assertEqual(list(old.state_dict()), list(new.state_dict()))
            for key, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
            bag, tab = torch.randn(2, 4, 8), torch.randn(2, 3)
            mask = torch.tensor([[True, True, False, False], [False]*4])
            w = mask.float() / mask.sum(dim=1, keepdim=True).clamp_min(1)
            torch.testing.assert_close(old(bag, tab, mask, w), new(bag, tab, mask, w), rtol=0, atol=0)

    def test_controls_remove_only_the_intended_information(self):
        import torch
        bag, tab = torch.randn(2, 4, 8), torch.randn(2, 3)
        mask = torch.ones(2, 4, dtype=torch.bool)
        w = torch.tensor([[.1, .2, .3, .4]]*2)
        table = run.build_model(3, "table_mlp", 8)
        torch.testing.assert_close(table(bag, tab, mask, w), table(bag*100, tab, mask, w))
        for arm in ("mean_mil", "concat_mean_mil"):
            mean = run.build_model(3, arm, 8)
            expected = mean(bag, tab, mask, w).detach()
            with torch.no_grad():
                for parameter in mean.attention.parameters():
                    parameter.add_(100)
            torch.testing.assert_close(expected, mean(bag, tab, mask, w))
            permutation = [3, 1, 0, 2]
            torch.testing.assert_close(expected, mean(bag[:, permutation], tab, mask[:, permutation], w[:, permutation]))

    def test_seed_report_does_not_confuse_ensemble_with_seed_mean(self):
        from followup_summary import write_reports
        import pandas as pd
        root = SCRIPT.parents[2] / "results" / "pixel_bag_tests"
        root.mkdir(parents=True, exist_ok=True)
        rows = []
        for seed, pp in [(42, [.1, .9, .6, .4]), (43, [.6, .4, .1, .9])]:
            for i, p in enumerate(pp):
                rows.append(dict(unit_id=i+1, label=i % 2, fold=i//2, seed=seed, arm="table_xgb",
                                 p=p, area_m2=1., zone=1, has_pixels=1))
        with tempfile.TemporaryDirectory(dir=root) as td:
            write_reports(td, pd.DataFrame(rows), [1, 2, 3, 4])
            separate = pd.read_csv(Path(td) / "seed_metrics.csv")
            ensemble = pd.read_csv(Path(td) / "ensemble_metrics.csv")
            self.assertAlmostEqual(separate.loc[separate.domain == "all", "auc"].mean(), .75)
            self.assertAlmostEqual(ensemble.loc[ensemble.domain == "all", "auc"].iloc[0], 1.)

    def test_followup_plan_is_frozen_and_ordered(self):
        import argparse
        from followup import jobs, command
        args = argparse.Namespace(out_root="test-followup", reference="old-seed42", prepared="old-cache",
                                  threads=4, device="cuda")
        plan = jobs(args)
        self.assertEqual([j["stage"] for j in plan], ["1", "2", "3", "3"])
        self.assertEqual(plan[0]["seeds"], "43,44")
        self.assertEqual(plan[1]["method"], "admin")
        self.assertEqual(set(plan[2]["arms"].split(",")), set(run.CONTROL_ARMS))
        cmd = command(args, plan[0])
        self.assertIn("--frozen-folds", cmd)
        self.assertEqual(cmd[cmd.index("--epochs")+1], "12")
        self.assertEqual(cmd[cmd.index("--trees")+1], "300")

    def test_prepare_with_real_grid_and_shuffled_geometry_ids(self):
        import argparse
        import rasterio
        from rasterio.transform import from_origin
        from pyproj import CRS
        import shapefile
        import pandas as pd
        root = SCRIPT.parents[2] / "results" / "pixel_bag_tests"
        root.mkdir(parents=True, exist_ok=True)
        old_root = run.ROOT
        sys.path.insert(0, str(old_root / "tills"))
        with tempfile.TemporaryDirectory(dir=root) as td:
            td = Path(td).resolve()
            self.assertTrue(td.is_relative_to(root.resolve()))
            try:
                run.ROOT = td
                for name in ("features/v2/sources", "data/slope_units"):
                    (td / name).mkdir(parents=True)
                # Grid IDs ordered 1,2,3,4; geometry deliberately ordered 4,3,2,1.
                grid = np.array([[1, 2], [3, 4]], dtype="int32")
                transform = from_origin(109., 31., .001, .001)
                gp = td / "features/v2/unit_id_grid.tif"
                with rasterio.open(gp, "w", driver="GTiff", width=2, height=2, count=1,
                                   dtype="int32", crs="EPSG:4326", transform=transform) as ds:
                    ds.write(grid, 1)
                shp = td / "data/slope_units/slope_units_final_train.shp"
                with shapefile.Writer(str(shp)) as writer:
                    writer.field("unit_id", "N", decimal=0)
                    for u in (4, 3, 2, 1):
                        r, c = divmod(u-1, 2)
                        left, top = transform * (c, r)
                        right, bottom = transform * (c+1, r+1)
                        writer.poly([[[left, top], [right, top], [right, bottom],
                                      [left, bottom], [left, top]]])
                        writer.record(u)
                shp.with_suffix(".prj").write_text(CRS.from_epsg(4326).to_wkt())
                frame = pd.DataFrame({"unit_id": [1, 2, 3, 4], "label": [0, 1, 0, 1],
                                      "inundation_fraction": [0, .1, 0, .1]})
                for j in range(37):
                    frame[f"x{j}"] = j
                frame.to_csv(td / "features/v2/features_v2_train_38.csv", index=False)
                pd.DataFrame({"unit_id": [1,2,3,4], "county": ["A","A","B","B"]}).to_csv(
                    td / "features/v2/county_units_v2_train.csv", index=False)
                pd.DataFrame({"unit_id": [1,2,3,4], "station": ["S"]*4}).to_csv(
                    td / "features/v2/sources/station_assign_v2.csv", index=False)
                pd.DataFrame({"date": ["2003-01-01", "2003-01-02"], "station": ["S"]*2,
                              "level": [145., 175.]}).to_csv(
                    td / "features/v2/sources/daily_station_levels.csv", index=False)
                bags = td / "bags.npz"
                np.savez(bags, uid=[1,2,3,4], row=[0,0,1,1], col=[0,1,0,1],
                         X=np.array([[150, np.nan, 0, 0, 0], [160, 20, 359, 0, 6200],
                                     [180, 10, 1, 0, 0], [155, 30, 90, 0, 0]], dtype="float32"),
                         bands=np.array(["elevation", "slope", "aspect", "curvature", "TRI"]))
                out = td / "prepared"
                run.prepare(argparse.Namespace(out=str(out), bags=str(bags)))
                manifest, d, pix, area = run.load_prepared(out)
                self.assertEqual(manifest["empty_units"], 1)
                self.assertEqual(manifest["empty_positive_units"], 0)
                np.testing.assert_array_equal(d["unit_id"], [1,2,3,4])
                self.assertTrue(d["cent"][0, 0] < d["cent"][1, 0])
                self.assertTrue((d["area"] > 0).all())
                np.testing.assert_allclose(pix[:, 4], [.5, 0, .5])
                self.assertTrue((area > 0).all())
                # Release Windows memory maps before TemporaryDirectory cleanup.
                del pix, area
            finally:
                run.ROOT = old_root

    def test_excluded_channels_and_circular_aspect(self):
        raw = np.array([[100, 20, 359, np.nan, 6200], [100, 20, 1, 0, 0],
                        [100, np.nan, 1, 0, 0], [100, 0, 50, 0, 0]], dtype=float)
        good, x = run.encode_terrain(raw, ["elevation", "slope", "aspect", "curvature", "TRI"])
        np.testing.assert_array_equal(good, [True, True, False, True])
        self.assertLess(np.linalg.norm(x[0, 2:] - x[1, 2:]), .04)
        np.testing.assert_array_equal(x[-1, 2:], [0, 0])

    def test_training_only_scaler_and_all_missing(self):
        tr = np.array([[1, np.nan], [3, np.nan]])
        mean, std = run.fit_scaler(tr)
        np.testing.assert_array_equal(mean, [2, 0])
        test = run.apply_scaler([[1000, np.nan]], mean, std)
        self.assertEqual(float(test[0, 0]), 998)
        self.assertTrue(np.isfinite(test).all())

    def test_fixed_sampling_and_short_bag_mask(self):
        x = np.arange(80).reshape(10, 8).astype("float32")
        area = np.arange(1, 11).astype("float32")
        a = run.sample_bag(x, area, 0, 10, 4, 71)
        b = run.sample_bag(x, area, 0, 10, 4, 71)
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)
        short, mask, w = run.sample_bag(x, area, 0, 2, 4, 3)
        np.testing.assert_array_equal(mask, [True, True, False, False])
        np.testing.assert_allclose(w[:2], [1/3, 2/3])
        self.assertEqual(short[2:].sum(), 0)

    def test_pixel_scaler_ignores_heldout_pixels(self):
        x = np.array([[1]*8, [3]*8, [10000]*8], dtype="float32")
        d = {"starts": np.array([0, 2]), "ends": np.array([2, 3])}
        mean, std = run.fold_pixel_scaler(x, d, [0])
        np.testing.assert_array_equal(mean, [2]*8)
        np.testing.assert_array_equal(std, [1]*8)

    def test_empty_bag_is_finite_table_fallback(self):
        import torch
        for arm in ("mil", "concat_mil", "water_mil", "table_mlp", "mean_mil", "concat_mean_mil"):
            m = run.build_model(3, arm, 8)
            tab = torch.randn(2, 3)
            mask = torch.zeros((2, 4), dtype=torch.bool)
            weights = torch.zeros((2, 4))
            pred = m(torch.randn(2, 4, 8), tab, mask, weights)
            torch.testing.assert_close(pred, m.table_head(tab).squeeze(-1))
            pred.sum().backward()
            self.assertTrue(torch.isfinite(pred).all())

    def test_budget_does_not_exceed_target(self):
        result = run.metrics(np.array([1, 0, 1]), np.array([.9, .8, .1]),
                             np.array([20., 20., 60.]), np.ones(3, bool), np.arange(3))
        self.assertEqual(result["all"]["actual_area_0.1"], 0)
        self.assertEqual(result["all"]["actual_area_0.2"], .2)

    def test_end_to_end_eight_arms_and_export(self):
        root = SCRIPT.parents[2] / "results" / "pixel_bag_tests"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as td:
            td = Path(td).resolve()
            self.assertTrue(td.is_relative_to(root.resolve()))
            prepared = td / "prepared"
            prepared.mkdir()
            rng = np.random.default_rng(42)
            n = 48
            counts = np.full(n, 12)
            counts[0] = 0
            starts = np.r_[0, counts.cumsum()[:-1]]
            ends = counts.cumsum()
            pix = rng.normal(size=(counts.sum(), 8)).astype("float32")
            np.save(prepared / "pixels.npy", pix)
            np.save(prepared / "pixel_area.npy", np.ones(len(pix), dtype="float32"))
            y = np.arange(n) % 2
            tab = rng.normal(size=(n, 40)).astype("float32")
            tab[0, 0] = np.nan
            np.savez(prepared / "units.npz", unit_id=np.arange(1, n+1), y=y, tab=tab,
                     summary=rng.normal(size=(n, 44)).astype("float32"), starts=starts, ends=ends,
                     cent=np.c_[np.repeat([0., 10000.], n//2), np.tile(np.arange(n//2), 2)],
                     area=np.full(n, 100.), zone=np.ones(n, dtype=bool),
                     county=np.repeat(["A", "B", "C", "D"], n//4))
            run.write_json(prepared / "manifest.json", {"version": run.VERSION, "synthetic": True, "units": n})
            (prepared / "unit_audit.csv").write_text("unit_id,label\n1,0\n", encoding="utf-8")
            out = td / "run"
            cmd = [sys.executable, str(SCRIPT), "train", "--prepared", str(prepared), "--out", str(out),
                   "--arms", ",".join(run.BASE_ARMS + run.CONTROL_ARMS),
                   "--folds", "2", "--epochs", "1", "--trees", "3", "--k", "8", "--hidden", "8",
                   "--batch", "16", "--threads", "1", "--eval-reps", "2", "--device", "cpu"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summaries = json.loads((out / "summary.json").read_text())
            self.assertEqual(len(summaries), 8)
            self.assertTrue(all(x["complete_oof"] for x in summaries))
            import pandas as pd
            pred = pd.read_csv(out / "predictions.csv")
            self.assertEqual(len(pred), n*8)
            self.assertFalse(pred.duplicated(["unit_id", "seed", "arm"]).any())
            self.assertTrue((out / "seed_metrics.csv").exists())
            self.assertTrue((out / "macro_metrics.csv").exists())
            export_cmd = [sys.executable, str(SCRIPT), "export", "--prepared", str(prepared),
                          "--run", str(out), "--out", str(td / "export.zip")]
            result = subprocess.run(export_cmd, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            import zipfile
            with zipfile.ZipFile(td / "export.zip") as z:
                self.assertIsNone(z.testzip())
                self.assertIn("checksums.json", z.namelist())
                self.assertNotIn("raw/unit_bags.npz", z.namelist())
                self.assertIn("code/followup_summary.py", z.namelist())
            # Follow-up aggregation can read a complete run, but must reject
            # accidentally supplying the same run twice (duplicate OOF keys).
            import argparse
            from followup_summary import combine
            combined = td / "combined"
            combine(argparse.Namespace(runs=[str(out)], out=str(combined)))
            self.assertTrue((combined / "macro_metrics.csv").exists())
            with self.assertRaises(ValueError):
                combine(argparse.Namespace(runs=[str(out), str(out)], out=str(td / "duplicate")))
            # Exercise the real ordered launcher on a tiny, shortened fixture.
            # The production FIXED settings remain unchanged after this context.
            import followup as wf
            from unittest.mock import patch
            reference = td / "reference_base"
            reference.mkdir()
            c = json.loads((out / "config.json").read_text())
            c["args"]["arms"] = ",".join(run.BASE_ARMS)
            run.write_json(reference / "config.json", c)
            run.write_json(reference / "summary.json", [x for x in summaries if x["arm"] in run.BASE_ARMS])
            pred.loc[pred.arm.isin(run.BASE_ARMS)].to_csv(reference / "predictions.csv", index=False)
            (reference / "folds.csv").write_bytes((out / "folds.csv").read_bytes())
            settings = dict(folds=2, epochs=1, trees=3, k=8, hidden=8, batch=16, eval_reps=2)
            task = argparse.Namespace(prepared=str(prepared), reference=str(reference),
                out_root=str(td / "ordered"), stage="1", dry_run=False, threads=1, device="cpu")
            with patch.dict(wf.FIXED, settings):
                wf.run_stage(task)
                wf.run_stage(task)  # completed runs are checked and skipped
                wf.summarize(task)
                task.out = str(td / "ordered_return.zip")
                wf.export(task)
                with zipfile.ZipFile(task.out) as z:
                    self.assertIsNone(z.testzip())
                    self.assertEqual(json.loads(z.read("stages.json"))["completed"], ["spatial_s43_s44"])
                    checks = json.loads(z.read("checksums.json"))
                    self.assertEqual(set(z.namelist()), set(checks) | {"checksums.json"})
            admin = td / "admin"
            cmd = [sys.executable, str(SCRIPT), "train", "--prepared", str(prepared), "--out", str(admin),
                   "--method", "admin", "--folds", "2", "--only-fold", "0", "--arms", "table_xgb",
                   "--trees", "3", "--threads", "1"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(json.loads((admin / "summary.json").read_text())[0]["complete_oof"])
            frozen = td / "bad_folds.csv"
            fold = pd.read_csv(out / "folds.csv")
            fold["fold"] = 1-fold["fold"]
            fold.to_csv(frozen, index=False)
            bad_cmd = [sys.executable, str(SCRIPT), "train", "--prepared", str(prepared),
                       "--out", str(td / "bad_reference"), "--folds", "2", "--arms", "table_xgb",
                       "--frozen-folds", str(frozen), "--trees", "1"]
            result = subprocess.run(bad_cmd, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Fold assignments differ", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
