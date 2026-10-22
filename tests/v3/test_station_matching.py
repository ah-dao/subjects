"""Scientific regressions: true 2D distance, matching identity, and boundaries."""
import unittest
import numpy as np
import pandas as pd
from src.v3 import config
from src.v3.station_matching import dms, nearest, registry


class StationTests(unittest.TestCase):
    def test_published_location_and_identity(self):
        _, stations = registry(config.REGISTRY)
        self.assertAlmostEqual(stations.loc[1, 'longitude'], 107.4)
        self.assertAlmostEqual(stations.loc[1, 'latitude'], 29.75)
        got = nearest(stations.longitude, stations.latitude, stations)
        self.assertEqual(list(got.station), list(stations.station))
        np.testing.assert_allclose(got.distance_m, 0, atol=1e-6)

    def test_latitude_changes_nearest_station(self):
        stations = pd.DataFrame(dict(station=['north', 'south'], longitude=[108., 108.],
                                     latitude=[31., 29.]))
        got = nearest([108., 108.], [29.1, 30.9], stations)
        self.assertEqual(list(got.station), ['south', 'north'])
        self.assertTrue(got.distance_margin_m.gt(100000).all())

    def test_reject_invalid_or_missing_coordinates(self):
        _, stations = registry(config.REGISTRY)
        for lon, lat in [([float('nan')], [30]), ([108], [100])]:
            with self.assertRaises(ValueError):
                nearest(lon, lat, stations)
        with self.assertRaises(ValueError):
            dms([107, 65, 0])


if __name__ == '__main__':
    unittest.main()
