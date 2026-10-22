"""Independently reproducible refit, never an override of legacy OOF matching."""
import gc
import random
import numpy as np
from . import train as base

POLICY='independent_seeded_refit_v1; legacy OOF for comparison only; not exact historical model explanation'
TOLERANCE=2e-5


def seed_fit(seed,injected):
    random.seed(seed);np.random.seed(seed)
    if not injected:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def cleanup(injected):
    gc.collect()
    if not injected:
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()


def checked_max(a,b):
    if a.shape!=b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid refit stability predictions')
    error=float(np.max(np.abs(a-b)))
    if error>TOLERANCE:raise ValueError(f'Independent refit not stable: {error} > {TOLERANCE}; stop, no PFI')
    return error


def stable_fit(args,context_x,y,test_x,seed,factory,injected):
    """Two sequential fits (one resident model), each with repeated full predictions."""
    seed_fit(seed,injected);probe=factory(args,seed)
    try:
        probe.fit(context_x,y)
        first=base.predict(probe,test_x,args.predict_batch)
        repeat=base.predict(probe,test_x,args.predict_batch)
        checked_max(first,repeat)
    finally:
        del probe;cleanup(injected)
    seed_fit(seed,injected);model=factory(args,seed)
    try:
        model.fit(context_x,y)
        full=base.predict(model,test_x,args.predict_batch)
        final_repeat=base.predict(model,test_x,args.predict_batch)
        checked_max(full,final_repeat);checked_max(first,full)
    except Exception:
        del model;cleanup(injected);raise
    return model,full,dict(first_fit=first,first_repeat=repeat,final_repeat=final_repeat)


def validate_replay(saved,ids,full,reference,fit):
    np.testing.assert_array_equal(saved['unit_id'],ids)
    np.testing.assert_allclose(saved['reference'],reference,rtol=0,atol=1e-15)
    if not np.isfinite(full).all() or not ((full>=0)&(full<=1)).all():raise ValueError('Invalid full refit baseline')
    errors=dict(first_repeat=checked_max(saved['first_fit'],saved['first_repeat']),
        final_repeat=checked_max(full,saved['final_repeat']),between_fits=checked_max(saved['first_fit'],full))
    if errors!=fit['stability_errors']:raise ValueError('Recorded stability errors differ')
    return errors
