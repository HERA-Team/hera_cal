'''Tests for omni.py'''

import nose.tools as nt
import os
import numpy as np
import aipy as a
import heracal.omni as omni
import heracal.miriad as miriad
from copy import deepcopy
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

#    def test_from_fits(self):
#
    def test_get_phase(self):
        sine = np.exp(-2j*np.pi*self.freqs*10).reshape(-1,1)
        nt.assert_equal(np.testing.assert_equal(sine, omni.get_phase(self.freqs, 10)), None)

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
        self.wgts = {}
        for ai, aj in self.info.bl_order():
            self.data[ai.ant(), aj.ant()] = {self.pol[0] * 2: np.ones((self.times.size, self.freqs.size), dtype=np.complex64)}
        self.unitgains = {self.pol[0]: {ant: np.ones((self.times.size, self.freqs.size), dtype=np.complex64) for ant in self.info.subsetant}}

    def test_run_omnical(self):
        m, g, v = omni.run_omnical(self.data, self.info, gains0=self.unitgains)
        nt.assert_equal(np.testing.assert_equal(g, self.unitgains), None)

    def test_compute_xtalk(self):
        m, g, v = omni.run_omnical(self.data, self.info, gains0=self.unitgains)
        wgts = {self.pol[0]*2 : {}}
        zeros = {self.pol[0]*2: {}}
        for ai, aj in self.info.bl_order():
            wgts[self.pol[0] * 2][ai.ant(), aj.ant()] = np.ones_like(m['res'][self.pol[0]*2][ai.ant(), aj.ant()], dtype=np.bool)
            zeros[self.pol[0] * 2][ai.ant(), aj.ant()] = np.mean(np.zeros_like(m['res'][self.pol[0]*2][ai.ant(), aj.ant()]), axis=0) # need to average over the times
        nt.assert_equal(np.testing.assert_equal(omni.compute_xtalk(m['res'], wgts), zeros), None)
            
        
        

#class Test_FirstCal(object):
#
#    def removedegen2(self, info, gains, vis, gainstart):
#        # divide out by gainstart (e.g. firstcal gains).    
#        g,v = deepcopy(gains),deepcopy(vis)
#        for ant in gains.keys():
#            g[ant] /= gainstart[ant]
#
#        # Calculate matrices used for projecting out degeneracies from antenna locations
#        Rgains =  np.array([np.append(ai,1) for ai in info.antloc])
#        Mgains = np.linalg.pinv(Rgains.T.dot(Rgains)).dot(Rgains.T)
#        Rvis = np.hstack((-info.ubl, np.zeros((len(info.ubl),1))))
#        reds = info.get_reds()
#        ntimes, nfreqs = gains.values()[0].shape
#
#        for t in range(ntimes):
#            for f in range(nfreqs):
#                gainSols = np.array([g[ai][t,f] for ai in info.subsetant])
#                visSols = np.array([vis[rg[0]][t,f] for rg in reds])
#
#                #Fix amplitudes
#                newGainSols = gainSols * np.exp(-1.0j * np.mean(np.angle(gainSols)))
#                newGainSols = newGainSols / np.mean(np.abs(newGainSols))
#                newVisSols = visSols * np.mean(np.abs(gainSols))**2
#
#                #Fix phases
#                degenRemoved = Mgains.dot(np.angle(newGainSols))
#                newGainSols = newGainSols * np.exp(-1.0j * Rgains.dot(degenRemoved))
#                newVisSols = newVisSols * np.exp(-1.0j * Rvis.dot(degenRemoved))
#
#                for i,ant in enumerate(info.subsetant): g[ant][t,f] = newGainSols[i]
#                for i,rg in enumerate(reds): v[rg[0]][t,f] = newVisSols[i]
#
#        # multipy back in gainstart.
#        for ai in g.keys():
#            g[ai] *= gainstart[ai]
#
#        return {}, g, v
#    
#    def setUp(self):
#        antpos = np.array([[ 14.60000038, -25.28794098,   1.], [ 21.89999962, -12.64397049,   1.], [ 14.60000038,  25.28794098,   1.], [-21.89999962, -12.64397049,   1.], [-14.60000038,   0.        ,   1.], [ 21.89999962,  12.64397049,   1.], [ 29.20000076,   0.        ,   1.], [-14.60000038, -25.28794098,   1.], [  0.        ,  25.28794098,   1.], [  0.        , -25.28794098,   1.], [  0.        ,   0.        ,   1.], [ -7.30000019, -12.64397049,   1.], [ -7.30000019,  12.64397049,   1.], [-21.89999962,  12.64397049,   1.], [-29.20000076,   0.        ,   1.], [ 14.60000038,   0.        ,   1.], [-14.60000038,  25.28794098,   1.], [  7.30000019, -12.64397049,   1.]])
#        reds = [[(0, 8), (9, 16)], [(13, 15), (14, 17), (3, 0), (4, 1), (16, 5), (12, 6)], [(3, 17), (4, 15), (7, 0), (11, 1), (16, 2), (12, 5), (10, 6), (14, 10)], [(3, 6), (14, 5)], [(0, 9), (1, 17), (2, 8), (4, 14), (6, 15), (8, 16), (12, 13), (11, 3), (10, 4), (9, 7), (15, 10), (17, 11)], [(3, 8), (11, 2), (9, 5)], [(3, 9), (4, 17), (12, 15), (11, 0), (10, 1), (8, 5), (13, 10), (14, 11)], [(0, 13), (1, 16)], [(0, 4), (1, 12), (6, 8), (9, 14), (15, 16), (17, 13)], [(0, 5), (3, 16), (7, 12), (17, 2), (11, 8)], [(0, 10), (7, 14), (10, 16), (11, 13), (6, 2), (9, 4), (15, 8), (17, 12)], [(1, 9), (2, 12), (5, 10), (6, 17), (8, 13), (12, 14), (10, 3), (17, 7), (15, 11)], [(2, 3), (5, 7)], [(16, 17), (12, 0), (8, 1), (13, 9)], [(0, 17), (1, 15), (3, 14), (4, 13), (9, 11), (10, 12), (12, 16), (5, 2), (7, 3), (11, 4), (6, 5), (17, 10)], [(3, 15), (4, 5), (7, 1), (13, 2), (11, 6)], [(5, 15), (8, 12), (10, 11), (13, 14), (15, 17), (1, 0), (6, 1), (4, 3), (12, 4), (11, 7), (17, 9), (16, 13)], [(0, 15), (1, 5), (3, 13), (4, 16), (9, 10), (11, 12), (15, 2), (7, 4), (10, 8)], [(0, 6), (3, 12), (4, 8), (7, 10), (9, 15), (14, 16), (10, 2), (17, 5)], [(8, 17), (2, 1), (13, 7), (12, 9), (16, 11)], [(0, 2), (7, 16), (9, 8)], [(4, 6), (14, 15), (3, 1), (13, 5)], [(0, 14), (1, 13), (6, 16)], [(2, 14), (6, 7), (5, 3)], [(2, 9), (8, 7)], [(2, 4), (5, 11), (6, 9), (8, 14), (15, 7)], [(1, 14), (6, 13)], [(0, 8), (9, 16)], [(13, 15), (14, 17), (3, 0), (4, 1), (16, 5), (12, 6)], [(3, 17), (4, 15), (7, 0), (11, 1), (16, 2), (12, 5), (10, 6), (14, 10)], [(3, 6), (14, 5)], [(0, 9), (1, 17), (2, 8), (4, 14), (6, 15), (8, 16), (12, 13), (11, 3), (10, 4), (9, 7), (15, 10), (17, 11)], [(3, 8), (11, 2), (9, 5)], [(3, 9), (4, 17), (12, 15), (11, 0), (10, 1), (8, 5), (13, 10), (14, 11)], [(0, 13), (1, 16)], [(0, 4), (1, 12), (6, 8), (9, 14), (15, 16), (17, 13)], [(0, 5), (3, 16), (7, 12), (17, 2), (11, 8)], [(0, 10), (7, 14), (10, 16), (11, 13), (6, 2), (9, 4), (15, 8), (17, 12)], [(1, 9), (2, 12), (5, 10), (6, 17), (8, 13), (12, 14), (10, 3), (17, 7), (15, 11)], [(2, 3), (5, 7)], [(16, 17), (12, 0), (8, 1), (13, 9)], [(0, 17), (1, 15), (3, 14), (4, 13), (9, 11), (10, 12), (12, 16), (5, 2), (7, 3), (11, 4), (6, 5), (17, 10)], [(3, 15), (4, 5), (7, 1), (13, 2), (11, 6)], [(5, 15), (8, 12), (10, 11), (13, 14), (15, 17), (1, 0), (6, 1), (4, 3), (12, 4), (11, 7), (17, 9), (16, 13)], [(0, 15), (1, 5), (3, 13), (4, 16), (9, 10), (11, 12), (15, 2), (7, 4), (10, 8)], [(0, 6), (3, 12), (4, 8), (7, 10), (9, 15), (14, 16), (10, 2), (17, 5)], [(8, 17), (2, 1), (13, 7), (12, 9), (16, 11)], [(0, 2), (7, 16), (9, 8)], [(4, 6), (14, 15), (3, 1), (13, 5)], [(0, 14), (1, 13), (6, 16)], [(2, 14), (6, 7), (5, 3)], [(2, 9), (8, 7)], [(2, 4), (5, 11), (6, 9), (8, 14), (15, 7)], [(1, 14), (6, 13)], [(0, 8), (9, 16)], [(13, 15), (14, 17), (3, 0), (4, 1), (16, 5), (12, 6)], [(3, 17), (4, 15), (7, 0), (11, 1), (16, 2), (12, 5), (10, 6), (14, 10)], [(3, 6), (14, 5)], [(0, 9), (1, 17), (2, 8), (4, 14), (6, 15), (8, 16), (12, 13), (11, 3), (10, 4), (9, 7), (15, 10), (17, 11)], [(3, 8), (11, 2), (9, 5)], [(3, 9), (4, 17), (12, 15), (11, 0), (10, 1), (8, 5), (13, 10), (14, 11)], [(0, 13), (1, 16)], [(0, 4), (1, 12), (6, 8), (9, 14), (15, 16), (17, 13)], [(0, 5), (3, 16), (7, 12), (17, 2), (11, 8)], [(0, 10), (7, 14), (10, 16), (11, 13), (6, 2), (9, 4), (15, 8), (17, 12)], [(1, 9), (2, 12), (5, 10), (6, 17), (8, 13), (12, 14), (10, 3), (17, 7), (15, 11)], [(2, 3), (5, 7)], [(16, 17), (12, 0), (8, 1), (13, 9)], [(0, 17), (1, 15), (3, 14), (4, 13), (9, 11), (10, 12), (12, 16), (5, 2), (7, 3), (11, 4), (6, 5), (17, 10)], [(3, 15), (4, 5), (7, 1), (13, 2), (11, 6)], [(5, 15), (8, 12), (10, 11), (13, 14), (15, 17), (1, 0), (6, 1), (4, 3), (12, 4), (11, 7), (17, 9), (16, 13)], [(0, 15), (1, 5), (3, 13), (4, 16), (9, 10), (11, 12), (15, 2), (7, 4), (10, 8)], [(0, 6), (3, 12), (4, 8), (7, 10), (9, 15), (14, 16), (10, 2), (17, 5)], [(8, 17), (2, 1), (13, 7), (12, 9), (16, 11)], [(0, 2), (7, 16), (9, 8)], [(4, 6), (14, 15), (3, 1), (13, 5)], [(0, 14), (1, 13), (6, 16)], [(2, 14), (6, 7), (5, 3)], [(2, 9), (8, 7)], [(2, 4), (5, 11), (6, 9), (8, 14), (15, 7)], [(1, 14), (6, 13)], [(0, 8), (9, 16)], [(13, 15), (14, 17), (3, 0), (4, 1), (16, 5), (12, 6)], [(3, 17), (4, 15), (7, 0), (11, 1), (16, 2), (12, 5), (10, 6), (14, 10)], [(3, 6), (14, 5)], [(0, 9), (1, 17), (2, 8), (4, 14), (6, 15), (8, 16), (12, 13), (11, 3), (10, 4), (9, 7), (15, 10), (17, 11)], [(3, 8), (11, 2), (9, 5)], [(3, 9), (4, 17), (12, 15), (11, 0), (10, 1), (8, 5), (13, 10), (14, 11)], [(0, 13), (1, 16)], [(0, 4), (1, 12), (6, 8), (9, 14), (15, 16), (17, 13)], [(0, 5), (3, 16), (7, 12), (17, 2), (11, 8)], [(0, 10), (7, 14), (10, 16), (11, 13), (6, 2), (9, 4), (15, 8), (17, 12)], [(1, 9), (2, 12), (5, 10), (6, 17), (8, 13), (12, 14), (10, 3), (17, 7), (15, 11)], [(2, 3), (5, 7)], [(16, 17), (12, 0), (8, 1), (13, 9)], [(0, 17), (1, 15), (3, 14), (4, 13), (9, 11), (10, 12), (12, 16), (5, 2), (7, 3), (11, 4), (6, 5), (17, 10)], [(3, 15), (4, 5), (7, 1), (13, 2), (11, 6)], [(5, 15), (8, 12), (10, 11), (13, 14), (15, 17), (1, 0), (6, 1), (4, 3), (12, 4), (11, 7), (17, 9), (16, 13)], [(0, 15), (1, 5), (3, 13), (4, 16), (9, 10), (11, 12), (15, 2), (7, 4), (10, 8)], [(0, 6), (3, 12), (4, 8), (7, 10), (9, 15), (14, 16), (10, 2), (17, 5)], [(8, 17), (2, 1), (13, 7), (12, 9), (16, 11)], [(0, 2), (7, 16), (9, 8)], [(4, 6), (14, 15), (3, 1), (13, 5)], [(0, 14), (1, 13), (6, 16)], [(2, 14), (6, 7), (5, 3)], [(2, 9), (8, 7)], [(2, 4), (5, 11), (6, 9), (8, 14), (15, 7)], [(1, 14), (6, 13)]]
#        self.freqs = np.linspace(.1,.2,64)
#        self.times = np.arange(1)
#        ants = np.arange(len(antpos))
#
#        self.info = omni.FirstCalRedundantInfo(len(antpos))
#        self.info.init_from_reds(reds, antpos)
#
#        # Simulate unique "true" visibilities
#        np.random.seed(21)
#        self.vis_true = {}
#        i = 0
#        for rg in reds:
#            self.vis_true[rg[0]] = np.array(1.0*np.random.randn(len(self.times),len(self.freqs)) + 1.0j*np.random.randn(len(self.times),len(self.freqs)), dtype=np.complex64)
#
#        # Simulate true gains and then remove degeneracies from true gains so that removedegen will produce exact answers
#        self.gain_true = {}
#        for i in ants:
#            self.gain_true[i] = np.array(1. + (.1*np.random.randn(len(self.times),len(self.freqs)) + .1j*np.random.randn(len(self.times),len(self.freqs))), dtype=np.complex64) 
#        self.g0 = {i: np.ones_like(self.gain_true[i]) for i in ants}
#        #_, self.gain_true, _ = self.removedegen2(self.info, self.gain_true, self.vis_true, self.g0)
#       
#        # Generate and apply firstcal gains
#        self.fcgains = {}
#        self.delays = {}
#        for i in ants:
#            self.delays[i] = np.random.randint(10,50)
#            fcspectrum = np.exp(2.0j * np.pi * self.delays[i] * self.freqs)
#            self.fcgains[i] = np.array([fcspectrum for t in self.times], dtype=np.complex64)
#
#        # Generate fake data 
#        bl2ublkey = {bl: rg[0] for rg in reds for bl in rg}
#        self.data = {}
#        self.wgts = {}
#        for rg in reds:
#            for (i,j) in rg:
#                self.data[(i,j)] = np.array(np.conj(self.gain_true[i]*self.fcgains[i]) * self.gain_true[j]*self.fcgains[j] * self.vis_true[rg[0]], dtype=np.complex64)
#                self.wgts[(i,j)] = np.ones_like(self.data[(i,j)], dtype=np.bool)
#
#    def test_data_to_delays(self):
#        fcal = omni.FirstCal(self.data, self.wgts, self.freqs, self.info)
#        w, ww = fcal.data_to_delays()
#        for (i,k), (l,m) in w.keys():
#            nt.assert_almost_equal(w[(i,k), (l,m)][0], self.delays[i]-self.delays[k]-self.delays[l]+self.delays[m], delta=1)
#            nt.assert_almost_equal(ww[(i,k), (l,m)][0], 0, delta=1)
#
#    def test_get_N(self):
#        fcal = omni.FirstCal(self.data, self.wgts, self.freqs, self.info)
#        nt.assert_equal(fcal.get_N(len(fcal.info.bl_pairs)).shape, (len(fcal.info.bl_pairs), len(fcal.info.bl_pairs)))  # the only requirement on N is it's shape.
#
#    def test_get_M(self):
#        fcal = omni.FirstCal(self.data, self.wgts, self.freqs, self.info)
#        for k in fcal.get_M():
#            nt.assert_equal(k.shape, (len(self.info.bl_pairs), len(self.times)))
#        _M = np.array([ -1*(self.delays[i]*np.ones(len(self.times))-self.delays[k]*np.ones(len(self.times))-self.delays[l]*np.ones(len(self.times))+self.delays[m]*np.ones(len(self.times))) for (i,k),(l,m) in self.info.bl_pairs])
#        _O = np.zeros((len(self.info.bl_pairs), len(self.times)))
#        nt.assert_equal(np.testing.assert_equal(_M, fcal.get_M()[0]), None)
#        nt.assert_equal(np.testing.assert_equal(_O, fcal.get_M()[1]), None)
