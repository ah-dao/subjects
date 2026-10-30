import unittest
import pandas as pd
from src.v3.evaluation_standard import evaluate


class StandardEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(dict(unit_id=range(10), label=[0, 0, 0, 0, 1, 0, 1, 0, 1, 1],
            p=[.01, .02, .1, .15, .2, .3, .4, .5, .7, .9], area_m2=[1]*10))
        self.grades = self.frame[['unit_id', 'p']].rename(columns={'p': 'score'})
        self.grades['level'] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    def test_grade_coverage_area_and_ba(self):
        ranking, coverage, levels = evaluate(self.frame, self.grades)
        self.assertEqual(ranking.positives.iloc[0], 4)
        middle = levels[levels.minimum_level.eq(3)].iloc[0]
        self.assertEqual(middle.landslide_coverage, 1)
        self.assertEqual(middle.area_fraction, .6)
        self.assertAlmostEqual(middle.ba, (1+4/6)/2)
        high = levels[levels.minimum_level.eq(4)].iloc[0]
        self.assertEqual(high.landslide_coverage, .75)
        self.assertEqual(high.area_fraction, .4)

    def test_ties_in_area_curve_are_indivisible(self):
        frame = self.frame.copy()
        frame.loc[frame.unit_id.isin([4, 5, 6, 7, 8, 9]), 'p'] = .9
        _, coverage, levels = evaluate(frame)
        self.assertIsNone(levels)
        self.assertEqual(coverage.area_fraction.iloc[0], .6)
        self.assertEqual(coverage.attained_recall.iloc[0], 1)

    def test_reject_wrong_scores_repeated_units_and_reversed_grades(self):
        grades = self.grades.copy()
        grades.loc[0, 'score'] = .9
        with self.assertRaises(ValueError): evaluate(self.frame, grades)
        with self.assertRaises(ValueError): evaluate(pd.concat([self.frame, self.frame]))
        grades = self.grades.copy()
        grades['level'] = 6-grades.level
        with self.assertRaises(ValueError): evaluate(self.frame, grades)


if __name__ == '__main__':
    unittest.main()
