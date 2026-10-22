import unittest
import numpy as np
from src.v3.temporal import compute


class TemporalTests(unittest.TestCase):
    def arrays(self):
        return {k:(np.arange(264 if k=='monthly' else 22,dtype=float)[None,:]+1)
                for k in ['ndvi','maxdaily','max30d','heavydays','cumulative','monthly','crop','built']}

    def test_multiple_time_scales_not_cumulative_background(self):
        a=self.arrays();x=compute(a,['2004-06-01'])[0]
        # Prior annual values are 3,4; NDVI long background is mean(1,2,3,4).
        np.testing.assert_allclose(x[:6],[3.5,1,4,4,7,3.5])
        # 2004-06 cutoff has monthly index 53, prior months have values 53,52,...
        np.testing.assert_allclose(x[6:9],[53,53+52+51,sum(range(48,54))])
        self.assertEqual(x[10],4);self.assertEqual(x[11],4)
        self.assertEqual(x[12],1);self.assertEqual(x[13],1);self.assertEqual(x[14],1)

    def test_future_years_and_event_month_never_enter(self):
        a=self.arrays();b={k:v.copy() for k,v in a.items()}
        for k,v in b.items():v[:,53 if k=='monthly' else 4:]=999999
        np.testing.assert_array_equal(compute(a,['2004-06-01']),compute(b,['2004-06-01']))

    def test_missing_month_not_zero_or_partial_accumulation(self):
        a=self.arrays();a['monthly'][0,52]=np.nan
        x=compute(a,['2004-06-01'])[0]
        self.assertTrue(np.isnan(x[6:9]).all())

    def test_change_fields_are_differences(self):
        a=self.arrays();a['crop'][0,2:4]=[.2,.3];a['built'][0,2:4]=[.1,.12]
        x=compute(a,['2004-06-01'])[0]
        self.assertAlmostEqual(x[12],.02);self.assertAlmostEqual(x[13],.1)


if __name__=='__main__':unittest.main()
