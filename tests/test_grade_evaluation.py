import unittest
import numpy as np
import pandas as pd
from tools.evaluate_v3_grades import summarize, checked_grades


class GradeEvaluationTests(unittest.TestCase):
    def test_count_coverage_and_area_budget_are_distinct(self):
        d=pd.DataFrame(dict(unit_id=range(5),p=[.1,.2,.3,.4,.5],
                            level=[1,2,3,4,5],label=[1,0,0,1,1],area_m2=[1,1,2,2,4]))
        r=summarize(d)
        self.assertAlmostEqual(r['high_recall'],2/3)
        self.assertAlmostEqual(r['high_area_fraction'],.6)
        self.assertEqual(r['high_positive_fraction'],1.)
        self.assertAlmostEqual(r['high_area_enrichment'],(2/3)/.6)
        self.assertEqual(r['missed_positives'],1)
        np.testing.assert_array_equal(checked_grades(d.drop(columns='level'),d).level,d.level)

    def test_no_positive_or_no_high_does_not_invent_metrics(self):
        d=pd.DataFrame(dict(label=[0,0],level=[1,2],area_m2=[1.,1.]))
        r=summarize(d)
        self.assertIsNone(r['high_recall'])
        self.assertIsNone(r['high_positive_fraction'])
        self.assertIsNone(r['high_area_enrichment'])

    def test_reject_missing_duplicate_or_reversed_grades(self):
        base=pd.DataFrame(dict(unit_id=range(5),p=[.1,.2,.3,.4,.5]))
        g=pd.DataFrame(dict(unit_id=range(5),level=[1,2,3,4,5]))
        with self.assertRaises(ValueError): checked_grades(base,g.iloc[:-1])
        with self.assertRaises(ValueError): checked_grades(base,pd.concat([g,g.iloc[:1]]))
        g.level=[5,4,3,2,1]
        with self.assertRaises(ValueError): checked_grades(base,g)


if __name__=='__main__': unittest.main()
