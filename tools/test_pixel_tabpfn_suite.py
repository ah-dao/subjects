"""Serial orchestration checks; these tests do not train TabPFN or require CUDA."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pixel_tabpfn_suite as suite


class SuiteTests(unittest.TestCase):
    def scratch(self):
        root=suite.ROOT/'results/pixel_tabpfn_suite_tests'
        root.mkdir(parents=True,exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def invoke(self, args):
        with patch.object(sys, 'argv', ['suite']+args), contextlib.redirect_stdout(io.StringIO()):
            suite.main()

    def test_admin_pair_keeps_matched_low_memory_protocol(self):
        with patch.object(suite, 'audit_split'), patch.object(suite.subprocess, 'run') as child:
            self.invoke(['check','--job','admin'])
        commands=[call.args[0] for call in child.call_args_list]
        self.assertEqual(len(commands),2)
        self.assertEqual([c[c.index('--stage')+1] for c in commands],['baseline','fusion'])
        for cmd in commands:
            for flag,value in [('--max-train','4000'),('--n-estimators','2'),
                               ('--predict-batch','32'),('--fit-mode','fit_preprocessors'),
                               ('--seeds','42,43,44')]:
                self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertTrue(cmd[cmd.index('--split-file')+1].endswith('admin5.csv'))

    def test_baseline_failure_stops_before_fusion(self):
        with self.scratch() as tmp:
            with patch.object(suite,'audit_split'), patch.object(suite,'verified_complete',return_value=False), \
                 patch.object(suite.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['train'])) as child:
                with self.assertRaises(subprocess.CalledProcessError):
                    self.invoke(['run','--job','admin','--out-root',tmp])
                self.assertEqual(child.call_count,1)
                self.assertFalse((Path(tmp)/'logs/admin_fusion.log').exists())

    def test_incomplete_and_old_configuration_are_not_silently_skipped(self):
        import argparse
        with self.scratch() as tmp:
            args=argparse.Namespace(out_root=tmp,predict_batch=32,threads=4,max_train=4000,n_estimators=2)
            out=Path(tmp)/'admin_baseline'
            out.mkdir()
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                suite.verified_complete(args,'admin_baseline')
            (out/'summary.json').write_text(json.dumps(dict(complete=True,complete_oof=True,outer_folds=list(range(5)))))
            (out/'config.json').write_text(json.dumps(dict(args=dict(max_train=12000,n_estimators=4))))
            with self.assertRaisesRegex(ValueError,'configuration differs'):
                suite.verified_complete(args,'admin_baseline')

    def test_real_frozen_admin_population_and_county_isolation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            frame=suite.audit_split('admin_fusion')
        self.assertEqual(len(frame),25509)
        self.assertEqual(int(frame.label.sum()),661)
        self.assertEqual(frame.county.nunique(),21)
        self.assertEqual(frame.groupby('county').fold.nunique().max(),1)
        with patch.object(suite,'ADMIN_SPLIT_SHA256','wrong'):
            with self.assertRaisesRegex(ValueError,'frozen'):
                suite.audit_split('admin_fusion')


if __name__=='__main__':
    unittest.main()
