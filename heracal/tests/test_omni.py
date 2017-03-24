'''Tests for omni.py'''

import nose.tools as nt
import os
import numpy as np
import aipy as a
from heracal import omni


class AntennaArray(a.fit.AntennaArray):
    def __init__(self, *args, **kwargs):
        a.fit.AntennaArray.__init__(self, *args, **kwargs)
        self.antpos_ideal = kwargs.pop('antpos_ideal')


class Testaatoinfo(object):
    def setUp(self):
        """Set up for basic tests of antenna array to info object."""
        lat = "45:00"
        lon = "90:00"
        self.freqs = np.linspace(.1, .2, 128)
        beam = a.fit.beam(freqs)
        ants = [a.fit.Antenna(0, 0, 0, beam),
                a.fit.Antenna(0, 50, 0, beam),
                a.fit.Antenna(0, 100, 0, beam),
                a.fit.Antenna(0, 150, 0, beam)]
        antpos_ideal = np.array([ant.pos for ant in ants])
        self.aa = AntennaArray((lat, lon), ants, antpos_ideal=antpos_ideal)

    def test_aa_to_info(self):
        self.info = omni.aa_to_info(self.aa)
        reds = [[(0, 1), (1, 2), (2, 3)], [(0, 2), (1, 3)]]
        nt.assert_equal(self.info.subsetant, np.array([0, 1, 2, 3]))
        for rb in self.info.get_reds():
            nt.assert_true(rb in reds)
