import nearest_dmc
import unittest


class TestNearestDMC(unittest.TestCase):
    def test_loading_colors_csv(self):
        self.assertNotEqual(len(nearest_dmc.dmc_colors), 0)

    def test_find_nearby_colors(self):
        # try an exact color
        black = nearest_dmc.rgb_to_dmc(0, 0, 0)
        self.assertEqual(black["floss"], "#310")

        # try a near color
        not_quite_light_salmon = nearest_dmc.rgb_to_dmc(254, 200, 201)
        self.assertEqual(not_quite_light_salmon["floss"], "#761")
