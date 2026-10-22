import unittest
import numpy as np
from tools.v3_explain84_refcheck import compare,conclusion


class RefcheckTests(unittest.TestCase):
    def test_query_difference_does_not_silently_pass(self):
        fixed=np.array([.1,.4,.9]);changed=fixed+np.array([0,.0012,0])
        checks=dict(full_vs_reference=compare(fixed,fixed),subset_vs_reference=compare(changed,fixed),
            full_repeat=compare(fixed,fixed),subset_repeat=compare(changed,changed))
        self.assertEqual(conclusion(checks),'original_query_replay_passes_subset_query_fails')
        self.assertEqual(checks['subset_vs_reference']['over_tolerance'],1)
        checks['full_vs_reference']=compare(changed,fixed)
        self.assertEqual(conclusion(checks),'original_query_replay_also_fails_stop_environment_or_refit_unresolved')
        checks['subset_repeat']=compare(fixed,changed)
        self.assertEqual(conclusion(checks),'same_model_prediction_not_stable_stop')

    def test_invalid_predictions_rejected_and_exact_replay(self):
        a=np.array([.3,.7]);v=compare(a,a)
        self.assertEqual(v['max_abs'],0)
        self.assertEqual(conclusion(dict(full_vs_reference=v,subset_vs_reference=v,full_repeat=v,subset_repeat=v)),
            'both_reproduce_check_reported_protocol_fold_and_runtime')
        for other in [np.array([np.nan,.7]),np.array([.3])]:
            with self.assertRaises(ValueError):compare(a,other)


if __name__=='__main__':unittest.main()
