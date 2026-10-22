import unittest
import numpy as np
import pandas as pd
from tools.analyze_v3_failure import area_top
from src.v3.common import ROOT

class FailureTests(unittest.TestCase):
    def test_area_budget_is_not_unit_count_budget(self):
        p = np.array([.9, .8, .7, .6]); area = np.array([1., 2., 6., 1.]); ids = np.array([2, 1, 3, 4])
        np.testing.assert_array_equal(area_top(p, area, ids), [True, False, False, False])
        # Frozen rule takes the ranked prefix within budget; it does not skip an oversized unit.

    def test_frozen_area_accounting_and_county_integrity(self):
        with np.load(ROOT/'features/v3/model_inputs/spatial/fold0/features84.npz') as z:
            ids = z['unit_id']; area = z['area']; y = z['y']; county = z['county']
        expected = {'spatial': [7357,5802,5243,3337,3770], 'admin': [6109,4915,3044,6995,4446]}
        for protocol in expected:
            fold = pd.read_csv(ROOT/f'splits/v3/{protocol}5.csv').set_index('unit_id').fold.reindex(ids).to_numpy()
            self.assertEqual([int((fold == i).sum()) for i in range(5)], expected[protocol])
            self.assertEqual(int(y.sum()), 661)
            self.assertAlmostEqual(sum(area[fold == i].sum() for i in range(5)), area.sum(), places=4)
            if protocol == 'admin':
                self.assertEqual(pd.DataFrame(dict(county=county, fold=fold)).groupby('county').fold.nunique().max(), 1)

if __name__ == '__main__': unittest.main()
