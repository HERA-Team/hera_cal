'''Tests for omni.py'''

import nose.tools as nt
import os
import numpy as np
import aipy as a
import heracal.omni as omni
#from heracal import omni


class AntennaArray(a.fit.AntennaArray):
    def __init__(self, *args, **kwargs):
        a.fit.AntennaArray.__init__(self, *args, **kwargs)
        self.antpos_ideal = kwargs.pop('antpos_ideal')


def get_aa(freqs, nants=4):
        lat = "45:00"
        lon = "90:00"
        beam = a.fit.Beam(freqs)
        ants = []
        for i in range(nants):
            ants.append(a.fit.Antenna(0, 50 * i, 0, beam))
        antpos_ideal = np.array([ant.pos for ant in ants])
        aa = AntennaArray((lat, lon), ants, antpos_ideal=antpos_ideal)
        return aa


class TestMethods(object):
    def setUp(self):
        """Set up for basic tests of antenna array to info object."""
        self.freqs = np.linspace(.1, .2, 16)
        self.pols = ['x', 'y']
        self.aa = get_aa(self.freqs)
        self.info = omni.aa_to_info(self.aa, pols=self.pols)
        self.gains = {pol: {ant: np.ones((1, self.freqs.size)) for ant in range(self.info.nant)} for pol in self.pols}

    def test_aa_to_info(self):
        info = omni.aa_to_info(self.aa)
        reds = [[(0, 1), (1, 2), (2, 3)], [(0, 2), (1, 3)]]
        nt.assert_true(np.all(info.subsetant == np.array([0, 1, 2, 3])))
        for rb in info.get_reds():
            nt.assert_true(rb in reds)

    def test_filter_reds(self):
        # exclude ants
        reds = omni.filter_reds(self.info.get_reds(), ex_ants=[0, 4])
        nt.assert_equal(reds, [[(1, 2), (2, 3)], [(1, 6), (2, 7)], [(5, 2), (6, 3)], [(5, 6), (6, 7)]])
        # include ants
        reds = omni.filter_reds(self.info.get_reds(), ants=[0, 1, 4, 5, 6])
        nt.assert_equal(reds, [[(0, 5), (1, 6)], [(4, 5), (5, 6)]])
        # exclued bls
        reds = omni.filter_reds(self.info.get_reds(), ex_bls=[(0, 2), (1, 2)])
        nt.assert_equal(reds, [[(0, 1), (2, 3)], [(0, 6), (1, 7)], [(0, 5), (1, 6), (2, 7)],
                               [(4, 2), (5, 3)], [(4, 1), (5, 2), (6, 3)], [(4, 6), (5, 7)],
                               [(4, 5), (5, 6), (6, 7)]])
        # include bls
        reds = omni.filter_reds(self.info.get_reds(), bls=[(0, 2), (1, 2)])
        nt.assert_equal(reds, [])
        # include ubls
        reds = omni.filter_reds(self.info.get_reds(), ubls=[(0, 2), (1, 4)])
        nt.assert_equal(reds, [[(0, 2), (1, 3)], [(4, 1), (5, 2), (6, 3)]])
        # exclude ubls
        reds = omni.filter_reds(self.info.get_reds(), ex_ubls=[(0, 2), (1, 4), (4, 5), (0, 5), (2, 3)])
        nt.assert_equal(reds, [[(0, 6), (1, 7)], [(4, 2), (5, 3)], [(4, 6), (5, 7)]])
        # exclude crosspols
        # reds = omni.filter_reds(self.info.get_reds(), ex_crosspols=()

    def test_compute_reds(self):
        reds = omni.compute_reds(4, self.pols, self.info.antloc[:self.info.nant])
        for i in reds:
            for k in i:
                for l in k:
                    nt.assert_true(isinstance(l, omni.Antpol))

    def test_wrap_gains_to_antpol(self):
        _gains = omni.wrap_gains_to_antpol(self.info, self.gains)
        nt.assert_equal(len(_gains), self.info.nant * len(self.pols))

#    def test_wrap_vis_to_antpol(self):
#
#    def test_omnical_output(self):
#
#    def test_logcal(self):
#
#    def test_lincal(self):
#
#    def test_remove_degen(self):
#
#    def test_compute_xtalk(self):
#
#    def test_from_fits(self):
#
#    def test_get_phase(self):

class Test_Antpol(object):
    def setUp(self):
        self.pols = ['x', 'y']
        antennas = [0]
        self.antpols = []
        for pol in self.pols:
            self.antpols.append(omni.Antpol(antennas[0], pol, 1))

    def test_antpol(self):
        for i, ant in enumerate(self.antpols):
            nt.assert_equal(ant.antpol(), (0, self.pols[i]))
            nt.assert_equal(ant.ant(), 0)
            nt.assert_equal(ant.pol(), self.pols[i])
            nt.assert_equal(int(ant), i)
            nt.assert_equal(str(ant), '{0}{1}'.format(ant.ant(), ant.pol()))
            nt.assert_true(ant == 0)
            nt.assert_equal({ant: None}.keys()[0], ant)


class Test_Redcal_Basics(object):
    def setUp(self):
        self.freqs = np.array([.1, .125, .150, .175, .2])
        self.aa = get_aa(self.freqs)
        self.info = omni.aa_to_info(self.aa)
        self.times = np.arange(3)
        self.pol = ['x']
        self.data = {}
        for ai, aj in self.info.bl_order():
            self.data[ai.ant(), aj.ant()] = {self.pol[0] * 2: np.ones((self.times.size, self.freqs.size), dtype=np.complex64)}
        self.unitgains = {self.pol[0]: {ant: np.ones((self.times.size, self.freqs.size), dtype=np.complex64) for ant in self.info.subsetant}}

    def test_logcal(self):
        m, g, v = omni.logcal(self.data, self.info, gainstart=self.unitgains)
        nt.assert_equal(np.testing.assert_equal(g, self.unitgains), None)

    def test_lincal(self):
        m1, g1, v1 = omni.logcal(self.data, self.info, gainstart=self.unitgains)
        m, g, v = omni.lincal(self.data, self.info, g1, v1)
        nt.assert_equal(np.testing.assert_equal(g, self.unitgains), None)


class Test_FirstCal(object):
    def setUp(self):
        # set up the basics
        self.freqs = np.linspace(.1, .2, 16)
        self.times = np.arange(4)
        self.aa = get_aa(self.freqs, nants=8)
        self.info = omni.aa_to_info(self.aa, fcal=True)
        self.pol = 'x'
        self.reds = self.info.get_reds()
        self.true_vis = {self.pol * 2: {}}
        # Make true visibilities. These are random complex arrays. Note format is [pol][unique_baseline_representative]
        for i, rg in enumerate(self.reds):
            rd = np.array(np.random.randn(self.times.size, self.freqs.size) + 1j * np.random.randn(self.times.size, self.freqs.size), dtype=np.complex64)
            self.true_vis[self.pol * 2][rg[0]] = rd
        self.true_gains = {self.pol: {}}
        # Make the true gains. Set to unity. Format is [pol][antenna].
        for i in self.info.subsetant:
            self.true_gains[self.pol][i] = np.ones((self.times.size, self.freqs.size), dtype=np.complex64)  # make it more complicated
        # Make noisey gains = true_gains + some phase wraps.
        self.noisey_gains = {self.pol: {i: self.true_gains[self.pol][i] + np.exp(2j*np.pi*(np.random.randint(10,30))*self.freqs) for i in self.true_gains[self.pol].keys()}}
        self.data = {}
        self.bl2red = {}
        for rg in self.reds:
            for r in rg:
                self.bl2red[r] = rg[0]
        for redgp in self.reds:
            for ai, aj in redgp:
                self.data[ai, aj] = {self.pol * 2: self.true_vis[self.pol * 2][self.bl2red[ai, aj]] * self.noisey_gains[self.pol][ai] * np.conj(self.noisey_gains[self.pol][aj])}

        self.wgts = {}
        for redgp in self.reds:
            for ai, aj in redgp:
                self.wgts[ai, aj] = {self.pol * 2: np.ones_like(self.data[ai, aj][self.pol*2], dtype=float)}
