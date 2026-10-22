import unittest
import numpy as np
from src.v3.map_atlas import county_text, shaded_relief


class AtlasTests(unittest.TestCase):
    def test_county_name_repair_is_reversible_and_preserves_chinese(self):
        for name in ['忠县', '巫山县', '万州区', '石柱土家族自治县']:
            self.assertEqual(county_text(name), name)
            self.assertEqual(county_text(name.encode('gb18030').decode('latin1')), name)
        self.assertEqual(county_text('unknown'), 'unknown')

    def test_relief_does_not_invent_elevation_outside_dem(self):
        dem = np.full((12, 12), np.nan)
        dem[3:9, 3:9] = 100.
        shade, smooth = shaded_relief(dem, 180., 180.)
        np.testing.assert_array_equal(np.isfinite(shade), np.isfinite(dem))
        np.testing.assert_array_equal(np.isfinite(smooth), np.isfinite(dem))
        np.testing.assert_allclose(smooth[3:9, 3:9], 100)
        np.testing.assert_allclose(shade[3:9, 3:9], np.sin(np.pi/4), rtol=1e-6)


if __name__ == '__main__': unittest.main()
