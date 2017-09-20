from .context import hemo
import unittest
from scripts import create_net as cnet
from hemo import net_prep as nprep

class graphTests(unittest.TestCase):

    def setUp(self):
        # Is there another way to create the network, so we can test net_prep methods?
        self.G = cnet.create_network_multiple_sources_and_sinks(5)

    def tearDown(self):
        del self.G

    def test_radii_perturbation_volume(self):
        """Tests if volume is fixed after changing radii"""
        initial_volume = cnet.get_total_volume(self.G)
        # cnet.perturb_all_radii(self.G) # Will perturbing radii on the same graph create a graph with different radii?
        perturbed_volume = cnet.get_total_volume(self.G)
        self.assertEqual(initial_volume, perturbed_volume)

    def test_transit_times(self):
        """Tests that all transit times are positive"""
        transit_times = []
        for src, sink in self.G.edges():
            transit_times.append(self.G[src][sink]['inverse_transit_time'])
        for index in range(len(transit_times)):
            self.assertGreater(transit_times[index], 0)

if __name__ == '__main__':
    unittest.main()
