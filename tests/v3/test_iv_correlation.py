import unittest
import numpy as np
from tools.v3_iv_correlation import information_cells, correlation_tables, county_text
from src.v3.explain84_design import feature_bins


class InformationTests(unittest.TestCase):
    def test_density_denominator_smoothing_and_empty_bins(self):
        bins=np.array([0,0,1,1,-1]); y=np.array([1,0,0,0,1])
        rows=information_cells(bins,{-1:'missing',0:'low',1:'high',2:'unseen'},y,np.full(5,'all'))
        by={r['bin']:r for r in rows}
        self.assertAlmostEqual(by[0]['frequency_ratio'],1.25)
        self.assertAlmostEqual(by[0]['information_value_raw_finite'],np.log(1.25))
        self.assertAlmostEqual(by[1]['information_value_smoothed'],np.log((.5/3)/(2.5/6)))
        self.assertEqual(by[1]['raw_iv_status'],'negative_infinity_zero_positive')
        self.assertTrue(np.isnan(by[2]['information_value_smoothed']))
        self.assertEqual(sum(r['units'] for r in rows),5)
        self.assertEqual(sum(r['positives'] for r in rows),2)

    def test_layer_denominators_and_unseen_missing(self):
        self.assertEqual(county_text('巴南区'.encode('gb18030').decode('latin1')),'巴南区')
        self.assertEqual(county_text('巴南区'),'巴南区')
        context=np.r_[np.zeros(90),np.arange(1,11)]
        bins,labels,plan=feature_bins(context,np.array([0,2,9,np.nan]))
        self.assertEqual(bins[-1],-1);self.assertNotEqual(bins[0],bins[1])
        rows=information_cells(bins,labels,np.array([1,0,1,0]),np.array(['a','a','b','b']))
        self.assertTrue(all(r['stratum_units']==2 and r['stratum_positives']==1 for r in rows))
        one=information_cells(np.array([0,0]),{0:'zero'},np.array([0,0]),np.array(['a','a']))
        self.assertTrue(np.isnan(one[0]['information_value_smoothed']))

    def test_pairwise_finite_constant_and_minimum_support(self):
        x=np.array([[1,4,1],[2,3,1],[3,np.inf,1],[4,1,1],[np.nan,0,1]],float)
        rho,n=correlation_tables(x,['a','b','constant'])
        self.assertEqual(n.loc['a','b'],3)
        self.assertAlmostEqual(rho.loc['a','b'],-1)
        self.assertTrue(np.isnan(rho.loc['a','constant']))


if __name__=='__main__':unittest.main()
