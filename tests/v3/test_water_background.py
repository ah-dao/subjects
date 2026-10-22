import unittest
import numpy as np
import pandas as pd
from src.v3.water_background import WaterHistory, transition_counts
from src.v3.reference_dates import assign_reference_dates


class WaterTests(unittest.TestCase):
    def observations(self):
        dates = pd.date_range('2001-01-01', '2003-12-31')
        return pd.DataFrame(dict(station='A', date=dates, level=np.where(np.arange(len(dates)) % 2, 2., 0.)))

    def test_transition_boundaries_match_brute_force(self):
        levels = np.array([0, 2, 0, 3, 3, 1.])
        elevations = np.array([-1, 0, 1, 2, 3, 4.])
        brute = np.abs(np.diff((levels[:, None] >= elevations).astype(int), axis=0)).sum(axis=0)
        np.testing.assert_array_equal(transition_counts(levels, elevations), brute)

    def test_future_observations_cannot_change_features(self):
        original = self.observations()
        changed = original.copy()
        changed.loc[changed.date.ge('2003-02-01'), 'level'] = 9999
        terrain = np.array([[1, 20, 0, 1], [2, 30, 1, 0.]])
        a = WaterHistory(original).unit('A', '2003-02-01', np.array([1., 2]), terrain, np.ones(2))
        b = WaterHistory(changed).unit('A', '2003-02-01', np.array([1., 2]), terrain, np.ones(2))
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])
        self.assertEqual(a[2]['last_allowed_date'], '2003-01-31')
        self.assertEqual(a[2]['expected_days'], 761)
        self.assertEqual(a[0][1], 760)  # starts at 2001; each adjacent day changes state

    def test_pre_2001_observations_are_excluded(self):
        frame = self.observations()
        earlier = pd.DataFrame(dict(station=['A'], date=[pd.Timestamp('2000-12-31')], level=[99999.]))
        a = WaterHistory(frame).profile('A', '2003-02-01')
        b = WaterHistory(pd.concat([earlier, frame])).profile('A', '2003-02-01')
        np.testing.assert_array_equal(a['H'], b['H'])

    def test_long_gap_does_not_bridge_or_backfill(self):
        frame = self.observations()
        frame = frame[~frame.date.between('2002-01-01', '2002-01-10')]
        h = WaterHistory(frame, 3)
        p = h.profile('A', '2003-02-01')
        self.assertFalse(p['complete'])
        self.assertEqual(p['held_days'], 3)
        self.assertEqual(p['missing_days'], 7)
        terrain = np.array([[1., 20, 0, 1]])
        table, summary, _ = h.unit('A', '2003-02-01', np.array([1.]), terrain, np.ones(1))
        self.assertTrue(np.isnan(table[:3]).all())
        self.assertTrue(np.isfinite(table[3:]).all())
        self.assertTrue(np.isfinite(summary[:4]).all())
        self.assertTrue(np.isnan(summary[4:8]).all())

    def test_antecedent_is_one_month_and_drawdown_is_signed(self):
        frame = self.observations()
        frame['level'] = 2.
        frame.loc[frame.date.eq('2002-11-01'), 'level'] = 0.
        h = WaterHistory(frame)
        terrain = np.array([[1., 20, 0, 1]])
        table, _, _ = h.unit('A', '2003-02-01', np.array([1.]), terrain, np.ones(1))
        self.assertEqual(table[3], 31)
        self.assertEqual(table[4], -2)


class ReferenceTests(unittest.TestCase):
    def population(self):
        ids = np.arange(20)
        p = pd.DataFrame(dict(unit_id=ids, label=(ids % 2 == 0).astype(int),
                              study_first_landslide_date=[f'200{3+i%5}-0{1+i%8}-15' if i%2 == 0 else None for i in ids]))
        return p, pd.DataFrame(dict(unit_id=ids, fold=ids % 5))

    def test_joint_dates_reproducible_row_order_independent(self):
        p, f = self.population()
        a, donor = assign_reference_dates(p, f, 'spatial', 0)
        b, _ = assign_reference_dates(p.iloc[::-1], f.iloc[::-1], 'spatial', 0)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(set(a.loc[a.label.eq(0), 'reference_month']).issubset(set(donor.reference_month)))

    def test_test_event_dates_cannot_affect_training_negative_dates(self):
        p, f = self.population()
        a, _ = assign_reference_dates(p, f, 'spatial', 0)
        p.loc[p.unit_id.mod(5).eq(0) & p.label.eq(1), 'study_first_landslide_date'] = '2020-12-01'
        b, _ = assign_reference_dates(p, f, 'spatial', 0)
        mask = a.label.eq(0)
        pd.testing.assert_series_equal(a.loc[mask, 'reference_month'], b.loc[mask, 'reference_month'])


if __name__ == '__main__':
    unittest.main()
