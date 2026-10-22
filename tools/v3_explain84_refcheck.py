"""One-fold GPU reference replay; diagnose before changing PFI or tolerances."""
import argparse
import gc
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.v3 import train as base
from src.v3.common import sha,seal,write_json
from src.v3.controls import reference
from src.v3.explain import reference_environment,subset_indices,TOLERANCE,output_lock


def compare(a,b):
    if a.shape!=b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid comparison arrays')
    diff=np.abs(a-b)
    return dict(units=len(a),max_abs=float(diff.max()),mean_abs=float(diff.mean()),
        p99_abs=float(np.quantile(diff,.99)),over_tolerance=int((diff>TOLERANCE).sum()))


def conclusion(checks):
    good=lambda k:checks[k]['max_abs']<=TOLERANCE
    stable=good('full_repeat') and good('subset_repeat')
    if not stable:return 'same_model_prediction_not_stable_stop'
    if good('full_vs_reference') and not good('subset_vs_reference'):
        return 'original_query_replay_passes_subset_query_fails'
    if good('full_vs_reference') and good('subset_vs_reference'):
        return 'both_reproduce_check_reported_protocol_fold_and_runtime'
    return 'original_query_replay_also_fails_stop_environment_or_refit_unresolved'


def runtime_info():
    import torch
    info=dict(python=platform.python_version(),platform=platform.platform(),torch_full=torch.__version__,
        cuda_runtime=torch.version.cuda,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark)
    if torch.cuda.is_available():
        info.update(gpu=torch.cuda.get_device_name(0),compute_capability=list(torch.cuda.get_device_capability(0)))
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
        info['gpu_driver_query']=p.stdout.strip() if p.returncode==0 else 'unavailable'
    except (OSError,subprocess.TimeoutExpired):info['gpu_driver_query']='unavailable'
    return info


def run(args):
    contexts=base.check_inputs(args);ref=reference()
    versions=base.environment(args)
    if versions!=reference_environment():raise ValueError('Use original V3 environment')
    training=json.loads((ROOT/'features/v3/explain_reference_s42/training_config.json').read_text())
    if args.predict_batch!=training['parameters']['predict_batch']:raise ValueError('Use reference batch32')
    with output_lock(args.out):
        args.out.mkdir(parents=True,exist_ok=False)
        path=args.features/args.protocol/f'fold{args.fold}'/'features84.npz'
        with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        keep=base.context_indices(d['unit_id'],d['y'],d['fold'],args.fold,contexts[args.protocol][f'{args.fold}_42'])
        test=np.flatnonzero(d['fold']==args.fold)
        sub=subset_indices(len(test),args.max_per_fold,args.protocol,args.fold);take=test[sub]
        mean,std,counts=base.scaler(d['x'][keep]);xx=base.transform(d['x'],mean,std)
        model=base.make_model(args,42+args.fold)
        report=dict(version='V3_explain84_refcheck',diagnosis_only=True,tolerance=TOLERANCE,
            protocol=args.protocol,fold=args.fold,model_seed=42+args.fold,versions=versions,runtime=runtime_info(),
            checkpoint_sha256=sha(args.model_path),features_sha256=sha(path),context_sha256=sha(args.contexts),
            sources=base.source_hashes(),code_sha256=sha(__file__),context_ids=d['unit_id'][keep].tolist(),
            original_test_units=len(test),sample_units=len(take),predict_batch=args.predict_batch,
            original_query='full heldout order chunked32; identical to V3 train.predict',
            experiment_query='same1024 subset as explain84, chunked32',
            note='No tolerance changes, no original outputs changed, not a PFI/scientific model result')
        write_json(args.out/'config.json',report)
        try:
            print('FIT',args.protocol,args.fold,'one4000context model, fourprediction passes only',flush=True)
            model.fit(xx[keep],d['y'][keep])
            sampled=base.predict(model,xx[take],args.predict_batch);print('SUBSET PASS1 DONE',flush=True)
            sampled2=base.predict(model,xx[take],args.predict_batch);print('SUBSET PASS2 DONE',flush=True)
            full=base.predict(model,xx[test],args.predict_batch);print('FULL PASS1 DONE',flush=True)
            full2=base.predict(model,xx[test],args.predict_batch);print('FULL PASS2 DONE',flush=True)
            expected=ref[ref.protocol.eq(args.protocol)].set_index('unit_id').loc[d['unit_id'][test],'p'].to_numpy()
            checks=dict(full_vs_reference=compare(full,expected),subset_vs_reference=compare(sampled,expected[sub]),
                subset_vs_full_same_ids=compare(sampled,full[sub]),full_repeat=compare(full2,full),
                subset_repeat=compare(sampled2,sampled))
            result=dict(checks=checks,conclusion=conclusion(checks),tolerance=TOLERANCE,
                actual_gpu_diagnosis=True,permission_to_skip_baseline_check=False)
            write_json(args.out/'report.json',result)
            np.savez_compressed(args.out/'predictions.npz',unit_id=d['unit_id'][test],reference=expected,
                full=full,full_repeat=full2,subset_indices=sub,subset=sampled,subset_repeat=sampled2)
            delta=np.abs(full-expected);order=np.argsort(-delta)[:20]
            pd.DataFrame(dict(unit_id=d['unit_id'][test][order],reference=expected[order],replay=full[order],
                abs_error=delta[order])).to_csv(args.out/'worst_full_reference_differences.csv',index=False)
            print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        except Exception as exc:
            write_json(args.out/'diagnostic_error.json',dict(type=type(exc).__name__,message=str(exc)))
            raise
        finally:
            del model;gc.collect()
            import torch
            if torch.cuda.is_available():torch.cuda.empty_cache()
            seal(args.out,version='V3_refcheck_diagnostic',not_scientific_result=True)
            dest=args.out.parent/(args.out.name+'_return.zip')
            with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
                for p in args.out.iterdir():z.write(p,p.name)
            print('DIAGNOSTIC ZIP',dest,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=ROOT/'features/v3/model_inputs')
    p.add_argument('--contexts',type=Path,default=ROOT/'data/v3/context_ids.json')
    p.add_argument('--model-path',type=Path,default=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')
    p.add_argument('--protocol',choices=['spatial','admin'],default='spatial')
    p.add_argument('--fold',type=int,choices=range(5),default=0)
    p.add_argument('--predict-batch',type=int,default=32)
    p.add_argument('--max-per-fold',type=int,default=1024)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--out',type=Path,default=ROOT/'results/v3_explain84_refcheck')
    a=p.parse_args()
    if a.max_per_fold<0 or a.predict_batch<1:p.error('Invalid numeric arguments')
    run(a)
