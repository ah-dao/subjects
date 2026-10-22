"""Independent exhaustive SSE checks and tie/permutation invariants."""
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from jenks import natural_breaks


def sse(x,labels):
    return sum(float(((x[labels==k]-x[labels==k].mean())**2).sum()) for k in np.unique(labels))


class JenksTests(unittest.TestCase):
    def test_exact_optimum_against_exhaustive_partitions(self):
        for seed in range(12):
            rng = np.random.RandomState(seed)
            values = np.sort(rng.uniform(0,1,8))
            x = np.repeat(values,rng.randint(1,7,8))
            labels,breaks = natural_breaks(x)
            optimum = min(sse(x,np.searchsorted(values[list(cuts)],x,side='left')+1)
                          for cuts in itertools.combinations(range(7),4))
            self.assertAlmostEqual(sse(x,labels),optimum,places=11)
            self.assertTrue(np.all(np.diff(breaks)>0))

    def test_equal_scores_stay_together_and_break_inclusive(self):
        x = np.repeat(np.array([0.,.01,.2,.21,.6,.61,.9,1.]),[40,4,10,2,5,1,2,1])
        labels,breaks = natural_breaks(x)
        for v in np.unique(x): self.assertEqual(len(np.unique(labels[x==v])),1)
        np.testing.assert_array_equal(np.searchsorted(breaks,breaks,side='left')+1,[1,2,3,4])
        permutation = np.random.RandomState(4).permutation(len(x))
        shuffled,other = natural_breaks(x[permutation])
        np.testing.assert_array_equal(other,breaks)
        np.testing.assert_array_equal(shuffled,labels[permutation])

    def test_invalid_input(self):
        for x in [[],[0,1,2],[0,1,np.nan,3,4],[0,1,np.inf,3,4]]:
            with self.assertRaises(ValueError): natural_breaks(x)


if __name__=='__main__': unittest.main()
