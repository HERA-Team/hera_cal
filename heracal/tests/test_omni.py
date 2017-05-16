'''Tests for omni.py'''

import nose.tools as nt
import os
import numpy as np
import aipy as a
from omnical.calib import RedundantInfo
import heracal.omni as omni
import heracal.miriad as miriad
from heracal.data import DATA_PATH
from pyuvdata import UVCal
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

        info = omni.aa_to_info(self.aa, fcal=True)
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

    def test_from_npz(self):
        Ntimes = 3 
        Nchans = 1024  # hardcoded for this file
        meta, gains, vis, xtalk = omni.from_npz(os.path.join(DATA_PATH, 'zen.2457698.50098.xx.pulledtime.npz'))
        for m in meta.keys():
            if m.startswith('chisq'):
                nt.assert_equal(meta[m].shape, (Ntimes,Nchans))
        nt.assert_equal(len(meta['freqs']), Nchans)
        nt.assert_equal(len(meta['jds']), Ntimes)
        nt.assert_equal(len(meta['lsts']), Ntimes)

        nt.assert_equal(gains.keys(), ['x'])
        for ant in gains['x'].keys():
            nt.assert_equal(gains['x'][ant].dtype, np.complex64)
            nt.assert_equal(gains['x'][ant].shape, (Ntimes,Nchans))
        
        nt.assert_equal(vis.keys(), ['xx'])
        for bl in vis['xx'].keys():
            nt.assert_equal(vis['xx'][bl].dtype, np.complex64)
            nt.assert_equal(vis['xx'][bl].shape, (Ntimes, Nchans))

        nt.assert_equal(xtalk.keys(), ['xx'])
        for bl in xtalk['xx'].keys():
            nt.assert_equal(xtalk['xx'][bl].dtype, np.complex64)
            nt.assert_equal(xtalk['xx'][bl].shape, (Ntimes, Nchans))
            for time in range(Ntimes):
                nt.assert_true(np.all(xtalk['xx'][bl][0] == xtalk['xx'][bl][time]))

    def test_get_phase(self):
        freqs = np.linspace(.1,.2,1024).reshape(-1,1)  # GHz
        tau = 10  # ns 
        nt.assert_true(np.all(omni.get_phase(freqs, tau) == np.exp(-2j*np.pi*freqs*tau)))
    def test_from_fits_gain(self):
        Ntimes = 3 * 2  # need 2 here because reading two files
        Nchans = 1024  # hardcoded for this file
        # read in the same file twice to make sure file concatenation works
        meta, gains, vis, xtalk = omni.from_fits([os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits')]*2)
        for m in meta.keys():
            if m.startswith('chisq'):
                nt.assert_equal(meta[m].shape, (Ntimes,Nchans))
        nt.assert_equal(len(meta['freqs']), Nchans)
        nt.assert_equal(len(meta['times']), Ntimes)
        nt.assert_equal(type(meta['history']), str)
        nt.assert_equal(meta['gain_conventions'], 'divide')

        nt.assert_equal(gains.keys(), ['x'])
        for ant in gains['x'].keys():
            nt.assert_equal(gains['x'][ant].dtype, np.complex64)
            nt.assert_equal(gains['x'][ant].shape, (Ntimes,Nchans))
        
        nt.assert_equal(vis.keys(), ['xx'])
        for bl in vis['xx'].keys():
            nt.assert_equal(vis['xx'][bl].dtype, np.complex64)
            nt.assert_equal(vis['xx'][bl].shape, (Ntimes, Nchans))

        nt.assert_equal(xtalk.keys(), ['xx'])
        for bl in xtalk['xx'].keys():
            nt.assert_equal(xtalk['xx'][bl].dtype, np.complex64)
            nt.assert_equal(xtalk['xx'][bl].shape, (Ntimes, Nchans))
            for time in range(Ntimes):
                nt.assert_true(np.all(xtalk['xx'][bl][0] == xtalk['xx'][bl][time]))

    def test_from_fits_delay(self):
        Ntimes = 3 * 2  # need 2 here because reading two files
        Nchans = 1024  # hardcoded for this file
        Ndelay = 1  # number of delays per integration
        # read in the same file twice to make sure file concatenation works
        meta, gains, vis, xtalk = omni.from_fits([os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.firstcal.fits')]*2)
        for m in meta.keys():
            if m.startswith('chisq'):
                nt.assert_equal(meta[m].shape, (Ntimes,))
        nt.assert_equal(len(meta['freqs']), Nchans)
        nt.assert_equal(len(meta['times']), Ntimes)
        nt.assert_equal(type(meta['history']), str)
        nt.assert_equal(meta['gain_conventions'], 'divide')

        nt.assert_equal(gains.keys(), ['x'])
        for ant in gains['x'].keys():
            nt.assert_equal(gains['x'][ant].dtype, np.complex128)
            nt.assert_equal(gains['x'][ant].shape, (Ntimes,Nchans))
        
        nt.assert_equal(vis, {})

        nt.assert_equal(xtalk, {})


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


class Test_RedundantInfo(object):
    def setUp(self):
        self.aa = get_aa(np.linspace(.1,.2,16))
        self.pol = ['x']
        self.info = omni.aa_to_info(self.aa, pols=self.pol)
        self.reds = self.info.get_reds()
        self.nondegenerategains = {}
        self.gains = {}
        self.vis = {}
        self.gains[self.pol[0]] = { gn: np.random.randn(1,16) + 1j*np.random.randn(1,16) for gn in self.info.subsetant }
        self.nondegenerategains[self.pol[0]] = { gn:  np.random.randn(1,16) + 1j*np.random.randn(1,16) for gn in self.info.subsetant}
        self.vis[self.pol[0]*2] = { red[0]: np.random.randn(1,16)+1j*np.random.randn(1,16) for red in self.reds}

    def test_bl_order(self):
        self.bl_order = [(omni.Antpol(self.info.subsetant[i], self.info.nant), omni.Antpol(self.info.subsetant[j], self.info.nant)) for i, j in self.info.bl2d]
        nt.assert_equal(self.bl_order, self.info.bl_order())

    def test_order_data(self):
        self.data = {}
        for red in self.reds:
            for i,j in red:
                # Randomly swap baseline orientation
                if np.random.randint(2):
                    self.data[i,j] = {self.pol[0]*2: np.random.randn(1,16)+1j*np.random.randn(1,16)}
                else:
                    self.data[j,i] = {self.pol[0]*2: np.random.randn(1,16)+1j*np.random.randn(1,16)}

        d = []
        for i, j in self.info.bl_order():
            bl = (i.ant(), j.ant())
            pol = i.pol() + j.pol()
            try: d.append(self.data[bl][pol])
            except(KeyError): d.append(self.data[bl[::-1]][pol[::-1]].conj())
        nt.assert_equal(np.testing.assert_equal(np.array(d).transpose((1,2,0)), self.info.order_data(self.data)), None)
        
    def test_pack_calpar(self):
        calpar = np.zeros((1,16,self.info.calpar_size(4,len(self.info.ubl))))
        calpar2 = np.zeros((1,16,self.info.calpar_size(4,len(self.info.ubl))))
        
        omni_info = RedundantInfo()
        reds = omni.compute_reds(4, self.pol, self.info.antloc[:self.info.nant])
        omni_info.init_from_reds(reds, self.info.antloc)
        _gains = {}
        for pol in self.gains:
            for ant in self.gains[pol]:
                _gains[ant] = self.gains[pol][ant].conj()
                
        _vis = {}
        for pol in self.vis:
            for i,j in self.vis[pol]:
                _vis[i,j] = self.vis[pol][i,j]
        calpar = omni_info.pack_calpar(calpar, gains=_gains, vis=_vis)
        nt.assert_equal(np.testing.assert_equal(self.info.pack_calpar(calpar2, self.gains, self.vis), calpar), None)

        # again with nondegenerate gains
        calpar = np.zeros((1,16,self.info.calpar_size(4,len(self.info.ubl))))
        calpar2 = np.zeros((1,16,self.info.calpar_size(4,len(self.info.ubl))))
        
        omni_info = RedundantInfo()
        reds = omni.compute_reds(4, self.pol, self.info.antloc[:self.info.nant])
        omni_info.init_from_reds(reds, self.info.antloc)
        _gains = {}
        for pol in self.gains:
            for ant in self.gains[pol]:
                _gains[ant] = self.gains[pol][ant].conj()/self.nondegenerategains[pol][ant].conj()
                
        _vis = {}
        for pol in self.vis:
            for i,j in self.vis[pol]:
                _vis[i,j] = self.vis[pol][i,j]
        calpar = omni_info.pack_calpar(calpar, gains=_gains, vis=_vis)
        nt.assert_equal(np.testing.assert_equal(self.info.pack_calpar(calpar2, self.gains, self.vis,nondegenerategains=self.nondegenerategains), calpar), None)

        # test not giving gains and vis to calpar
        calpar = np.zeros((1,16,self.info.calpar_size(4,len(self.info.ubl))))
        calpar_out = omni_info.pack_calpar(calpar)
        nt.assert_equal(np.testing.assert_equal(calpar, calpar_out), None)

    def test_unpack_calpar(self):
        calpar = np.zeros((1,16,self.info.calpar_size(4,len(self.info.ubl))))
        calpar = self.info.pack_calpar(calpar, self.gains, self.vis)
        
        m,g,v = self.info.unpack_calpar(calpar)
        for pol in g.keys():
            for ant in g[pol].keys():
                nt.assert_equal(np.testing.assert_almost_equal(g[pol][ant], self.gains[pol][ant]), None)
        nt.assert_equal(np.testing.assert_equal(v, self.vis), None)
         

        calpar = np.zeros((1,16,self.info.calpar_size(4,len(self.info.ubl))))
        calpar = self.info.pack_calpar(calpar, self.gains, self.vis, nondegenerategains=self.nondegenerategains)
        
        m,g,v = self.info.unpack_calpar(calpar, nondegenerategains=self.nondegenerategains)
        for pol in g.keys():
            for ant in g[pol].keys():
                nt.assert_equal(np.testing.assert_almost_equal(g[pol][ant], self.gains[pol][ant]), None)
        nt.assert_equal(np.testing.assert_equal(v, self.vis), None)


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


class Test_HERACal(UVCal):
    def test_gainHC(self):
        meta, gains, vis, xtalk = omni.from_fits(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits'))
        meta['inttime'] = np.diff(meta['times'])[0]*60*60*24
        optional = {'observer': 'Zaki Ali (zakiali@berkeley.edu)'}
        hc = omni.HERACal(meta, gains, ex_ants=[81], optional=optional)  # the fits file was run with ex_ants=[81] and we need to include it here for the test.
        uv = UVCal()
        uv.read_calfits(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits'))
        for param in hc:
            if param == '_history': continue
            elif param == '_time_range':  # why do we need this?
                nt.assert_equal(np.testing.assert_almost_equal(getattr(hc, param).value, getattr(uv, param).value, 5), None)
            else:
                nt.assert_true(np.all(getattr(hc, param) == getattr(uv, param)))

