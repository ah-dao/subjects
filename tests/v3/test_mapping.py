import itertools
import unittest
import numpy as np
import pandas as pd
from src.v3.jenks import natural_breaks
from src.v3.mapping import complete_table


class MapTests(unittest.TestCase):
    def test_exact_breaks_match_exhaustive_sse_with_duplicate_values(self):
        rng = np.random.default_rng(42)
        for _ in range(12):
            x = np.round(rng.uniform(size=14), 1)
            labels, boundaries = natural_breaks(x, classes=3)
            values = np.unique(x)
            def sse(labels):
                return sum(((x[labels == k]-x[labels == k].mean())**2).sum() for k in np.unique(labels))
            exhaustive = min(sse(np.searchsorted(values[list(cut)], x, side='left'))
                             for cut in itertools.combinations(range(len(values)-1), 2))
            self.assertAlmostEqual(sse(labels), exhaustive, places=12)
            np.testing.assert_array_equal(np.searchsorted(boundaries, boundaries, side='left')+1, [1, 2])

    def test_full_population_river_rule_preserves_raw_positive(self):
        ids = np.arange(1, 25940)
        identity = pd.DataFrame(dict(unit_id=ids, is_train=(ids <= 25509).astype(int), is_submerged=(ids > 25509).astype(int)))
        inventory = pd.DataFrame(dict(unit_id=ids, raw_label=((ids <= 661) | (ids == 25939)).astype(int)))
        oof = pd.DataFrame(dict(unit_id=ids[:25509], label=(ids[:25509] <= 661).astype(int),
            p=np.linspace(.001, .9, 25509), seed_sd=0., fold=0, zone=0, background_known=1, reference_month='2010-01-01'))
        table, breaks = complete_table(oof, identity, inventory)
        self.assertEqual(len(table), 25939)
        self.assertEqual(table.p.notna().sum(), 25939)
        river = table[table.is_train.eq(0)]
        self.assertTrue(river.level.eq(1).all() and river.p.eq(0).all())
        self.assertEqual(river.raw_label.sum(), 1)
        self.assertTrue(river.label.eq(-1).all())
        self.assertEqual(table[table.is_train.eq(1)].label.sum(), 661)
        bad = oof.copy(); bad.loc[0, 'unit_id'] = 25939
        with self.assertRaises(ValueError): complete_table(bad, identity, inventory)


if __name__ == '__main__': unittest.main()
