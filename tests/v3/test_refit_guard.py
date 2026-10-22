import unittest
from types import SimpleNamespace
import numpy as np
from src.v3.refit_guard import stable_fit,validate_replay


class Model:
    def __init__(self,offset=0,drift=False):self.offset=offset;self.calls=0;self.drift=drift
    def fit(self,x,y):self.classes_=np.array([0,1])
    def predict_proba(self,x):
        self.calls+=1;p=np.clip(.4+.01*x[:,0]+self.offset+(self.calls*.001 if self.drift else 0),0,1)
        return np.column_stack([1-p,p])


class GuardTests(unittest.TestCase):
    def test_independent_fits_and_repeat_predictions_are_checked(self):
        args=SimpleNamespace(predict_batch=32);x=np.arange(4.)[:,None];y=np.array([0,1,0,1])
        model,full,saved=stable_fit(args,x,y,x,42,lambda a,s:Model(),True)
        saved.update(unit_id=np.arange(4),reference=full+.01)
        fit=dict(stability_errors=dict(first_repeat=0.,final_repeat=0.,between_fits=0.))
        validate_replay(saved,np.arange(4),full,full+.01,fit)
        with self.assertRaises(ValueError):stable_fit(args,x,y,x,42,lambda a,s:Model(drift=True),True)
        calls=[]
        def differing(a,s):calls.append(s);return Model(offset=len(calls)*.001)
        with self.assertRaises(ValueError):stable_fit(args,x,y,x,42,differing,True)
        saved['first_fit']=saved['first_fit']+.001
        with self.assertRaises(ValueError):validate_replay(saved,np.arange(4),full,full+.01,fit)


if __name__=='__main__':unittest.main()
