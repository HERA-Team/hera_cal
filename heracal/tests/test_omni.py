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


class TestBasics(object):
    def setUp(self):
        """Set up for basic tests of antenna array to info object."""
        self.freqs = np.linspace(.1, .2, 16)
        self.aa = get_aa(self.freqs)
        self.info = omni.aa_to_info(self.aa, pols=['x', 'y'])

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

    def test_unitgains(self):
        nt.assert_equal(np.testing.assert_equal(omni.create_unitgains(self.data), {self.pol[0]: {ant: np.ones((self.times.size, self.freqs.size), dtype=np.complex64) for ant in self.info.subsetant}}), None)

    def test_logcal(self):
        m, g, v = omni.logcal(self.data, self.info, gainstart=self.unitgains)
        nt.assert_equal(np.testing.assert_equal(g, self.unitgains), None)

    def test_lincal(self):
        m1, g1, v1 = omni.logcal(self.data, self.info, gainstart=self.unitgains)
        m, g, v = omni.lincal(self.data, self.info, gainstart=g1, visstart=v1)
        nt.assert_equal(np.testing.assert_equal(g, self.unitgains), None)


class Test_Redcal_with_Degen(object):
    def setUp(self):
        self.freqs = np.linspace(.1, .2, 16)
        self.times = np.arange(4)
        self.aa = get_aa(self.freqs, nants=20)
        self.info = omni.aa_to_info(self.aa)
        self.pol = 'x'
        self.reds = self.info.get_reds()
        self.true_vis = {self.pol * 2: {}}
        for i, rg in enumerate(self.reds):
            rd = np.array(np.random.randn(self.times.size, self.freqs.size) + 1j * np.random.randn(self.times.size, self.freqs.size), dtype=np.complex64)
            self.true_vis[self.pol * 2][rg[0]] = rd
        self.true_gains = {self.pol: {}}
        for i in self.info.subsetant:
            self.true_gains[self.pol][i] = np.ones((self.times.size, self.freqs.size), dtype=np.complex64)  # make it more complicated
        self.data = {}
        self.bl2red = {}
        for rg in self.reds:
            for r in rg:
                self.bl2red[r] = rg[0]
        for redgp in self.reds:
            for ai, aj in redgp:
                self.data[ai, aj] = {self.pol * 2: self.true_vis[self.pol * 2][self.bl2red[ai, aj]] * self.true_gains[self.pol][ai] * np.conj(self.true_gains[self.pol][aj])}

    def test_redcal(self):
        m1, g1, v1 = omni.logcal(self.data, self.info)
        m, g, v = omni.lincal(self.data, self.info, gainstart=g1, visstart=v1)
        _, g, v = omni.removedegen(self.info, g, v, omni.create_unitgains(self.data))
        nt.assert_equal(np.testing.assert_almost_equal(m['chisq'], np.zeros_like(m['chisq']), decimal=8), None)

        # make sure model visibilities equals true visibilities
        for pol in v.keys():
            for bl in v[pol].keys():
                nt.assert_equal(np.testing.assert_almost_equal(v[pol][bl], self.true_vis[pol][bl], decimal=8), None)

        # make sure gains equal true gains
        for pol in g.keys():
            for ai in g[pol].keys():
                nt.assert_equal(np.testing.assert_almost_equal(g[pol][ai], self.true_gains[pol][ai], decimal=8), None)

        # test to make sure degeneracies keep average amplitudes and phases constant.
        gains = np.array([g[pol][i] for pol in g.keys() for i in g[pol].keys()])
        nt.assert_equal(np.testing.assert_almost_equal(np.mean(np.abs(gains), axis=0), np.ones_like(gains), decimal=8))
        nt.assert_equal(np.testing.assert_almost_equal(np.mean(np.angle(gains), axis=0), np.zeros_like(np.real(gains[0])), decimal=8), None)
