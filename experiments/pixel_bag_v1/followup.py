"""Ordered, frozen follow-up experiment stages; no mainline/data overwrites."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from run import BASE_ARMS, CONTROL_ARMS, fingerprint, write_json

HERE = Path(__file__).resolve().parent
FIXED = {"folds": 5, "fold_seed": 42, "epochs": 12, "trees": 300, "k": 64,
         "hidden": 32, "batch": 128, "eval_reps": 3, "lr": .001,
         "sampling_seed": 731, "neg_sampling": "soft", "neg_km": 4., "neg_lam": .2}


def jobs(args):
    root = Path(args.out_root)
    reference = Path(args.reference)
    return [
        dict(stage="1", name="spatial_s43_s44", method="spatial_kmeans", seeds="43,44",
             arms=",".join(BASE_ARMS), frozen=str(reference / "folds.csv")),
        dict(stage="2", name="admin_s42_s43_s44", method="admin", seeds="42,43,44",
             arms=",".join(BASE_ARMS), frozen=None),
        dict(stage="3", name="spatial_controls_s42_s43_s44", method="spatial_kmeans", seeds="42,43,44",
             arms=",".join(CONTROL_ARMS), frozen=str(reference / "folds.csv")),
        dict(stage="3", name="admin_controls_s42_s43_s44", method="admin", seeds="42,43,44",
             arms=",".join(CONTROL_ARMS), frozen=str(root / "admin_s42_s43_s44/folds.csv")),
    ]


def command(args, job):
    cmd = [sys.executable, str(HERE / "run.py"), "train", "--prepared", args.prepared,
           "--out", str(Path(args.out_root) / job["name"]), "--method", job["method"],
           "--seeds", job["seeds"], "--arms", job["arms"], "--threads", str(args.threads),
           "--device", args.device]
    for key, value in FIXED.items():
        cmd += ["--"+key.replace("_", "-"), str(value)]
    if job["frozen"]:
        cmd += ["--frozen-folds", job["frozen"]]
    return cmd


def validate_run(path, method, seeds, arms, prepared_manifest):
    path = Path(path)
    c = json.loads((path / "config.json").read_text(encoding="utf-8"))
    expected = dict(FIXED, method=method, seeds=seeds, arms=arms, only_fold=None)
    for key, value in expected.items():
        if c["args"].get(key) != value:
            raise ValueError(f"Frozen setting mismatch: {path} {key}")
    if c["prepared_manifest"] != prepared_manifest:
        raise ValueError("Prepared manifest differs: " + str(path))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if {r["arm"] for r in summary} != set(arms.split(",")) or not all(r["complete_oof"] for r in summary):
        raise ValueError("Run is not complete: " + str(path))
    # A seed-averaged summary alone does not establish complete per-seed coverage.
    import pandas as pd
    pred = pd.read_csv(path / "predictions.csv")
    population = set(pd.read_csv(path / "folds.csv").unit_id)
    if len(population) != prepared_manifest["units"]:
        raise ValueError("Run population count mismatch")
    if pred.duplicated(["unit_id", "arm", "seed"]).any():
        raise ValueError("Duplicate predictions")
    for arm in arms.split(","):
        for seed in [int(s) for s in seeds.split(",")]:
            if set(pred.loc[(pred.arm == arm) & (pred.seed == seed)].unit_id) != population:
                raise ValueError("Missing arm/seed held-out predictions")
    return c


def collect(args):
    manifest = json.loads((Path(args.prepared) / "manifest.json").read_text(encoding="utf-8"))
    validate_run(args.reference, "spatial_kmeans", "42", ",".join(BASE_ARMS), manifest)
    available = []
    for job in jobs(args):
        p = Path(args.out_root) / job["name"]
        if (p / "summary.json").exists():
            validate_run(p, job["method"], job["seeds"], job["arms"], manifest)
            available.append((job, p))
    return manifest, available


def run_stage(args):
    selected = [j for j in jobs(args) if args.stage == "all" or j["stage"] == args.stage]
    if args.dry_run:
        for job in selected:
            print("stage", job["stage"], "|", " ".join(command(args, job)))
        return
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    manifest, _ = collect(args)
    for job in selected:
        # Require completed earlier stages when starting at a later step.
        for previous in jobs(args):
            if int(previous["stage"]) < int(job["stage"]):
                validate_run(root / previous["name"], previous["method"], previous["seeds"],
                             previous["arms"], manifest)
        out = root / job["name"]
        if (out / "summary.json").exists():
            validate_run(out, job["method"], job["seeds"], job["arms"], manifest)
            print("Already complete, skip:", out, flush=True)
            continue
        if out.exists():
            raise ValueError("Incomplete directory exists; preserve it and use a new --out-root: " + str(out))
        cmd = command(args, job)
        write_json(root / (job["name"]+"_launch.json"), {"command": cmd, "fixed": FIXED,
                    "runner": fingerprint(HERE / "run.py"), "no_hyperparameter_selection": True})
        log = logs / (job["name"]+".log")
        print("START", job["name"], "log:", log, flush=True)
        with log.open("w", encoding="utf-8") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Training failed ({result.returncode}); inspect {log}")
        validate_run(out, job["method"], job["seeds"], job["arms"], manifest)
        print("DONE", job["name"], flush=True)


def export(args):
    _, available = collect(args)
    if not available:
        raise ValueError("No completed follow-up stage to export")
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    selected = []
    for name, directory in [("spatial_reference_s42", Path(args.reference))] + [(j["name"], p) for j, p in available]:
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix in (".csv", ".json", ".py") and p.name != "model.json":
                selected.append((p, "runs/"+name+"/"+p.relative_to(directory).as_posix()))
    for p in (Path(args.out_root) / "logs").glob("*.log"):
        selected.append((p, "logs/"+p.name))
    for p in (Path(args.out_root)).glob("*_launch.json"):
        selected.append((p, "launch/"+p.name))
    for p in (Path(args.out_root) / "reports").rglob("*"):
        if p.is_file() and p.suffix in (".csv", ".json"):
            selected.append((p, "reports/"+p.relative_to(Path(args.out_root) / "reports").as_posix()))
    for name in ("manifest.json", "unit_audit.csv"):
        selected.append((Path(args.prepared) / name, "prepared/"+name))
    for name in ("run.py", "run_v1_frozen.py", "followup.py", "followup_summary.py"):
        selected.append((HERE / name, "code/"+name))
    hashes = {}
    with zipfile.ZipFile(dest, "x", zipfile.ZIP_DEFLATED) as z:
        for p, name in selected:
            z.write(p, name)
            hashes[name] = fingerprint(p)
        stages = json.dumps({"completed": [j["name"] for j, _ in available],
            "expected": [j["name"] for j in jobs(args)], "fixed": FIXED}, indent=2).encode("utf-8")
        z.writestr("stages.json", stages)
        hashes["stages.json"] = {"bytes": len(stages), "sha256": hashlib.sha256(stages).hexdigest()}
        z.writestr("checksums.json", json.dumps(hashes, indent=2))
    print("Exported:", dest)


def summarize(args):
    from followup_summary import combine
    _, available = collect(args)
    root = Path(args.out_root)
    spatial = [str(Path(args.reference))] + [str(p) for j, p in available if j["method"] == "spatial_kmeans"]
    admin = [str(p) for j, p in available if j["method"] == "admin"]
    report_root = root / "reports"
    if report_root.exists():
        raise ValueError("Reports already exist; preserve them or use followup_summary.py with a new --out")
    if len(spatial) < 2:
        raise ValueError("Complete stage 1 before summarizing")
    # Each domain combines only a single split method and matching environment/settings.
    combine(argparse.Namespace(runs=spatial, out=str(report_root / "spatial")))
    if admin:
        combine(argparse.Namespace(runs=admin, out=str(report_root / "admin")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for action in ("run", "export", "summarize"):
        a = sub.add_parser(action)
        a.add_argument("--prepared", default="results/pixel_bag_v1/prepared")
        a.add_argument("--reference", default="results/pixel_bag_v1/spatial_s42_e12")
        a.add_argument("--out-root", default="results/pixel_bag_v1/followup_v11")
        if action == "run":
            a.add_argument("--stage", choices=["1", "2", "3", "all"], required=True)
            a.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
            a.add_argument("--threads", type=int, default=4)
            a.add_argument("--dry-run", action="store_true")
            a.set_defaults(func=run_stage)
        elif action == "export":
            a.add_argument("--out", required=True)
            a.set_defaults(func=export)
        else:
            a.set_defaults(func=summarize)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
