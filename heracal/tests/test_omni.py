'''Tests for omni.py'''

import nose.tools as nt
import os, sys
import numpy as np
import aipy as a
from omnical.calib import RedundantInfo
import heracal.omni as omni
from heracal.data import DATA_PATH
from pyuvdata import UVCal, UVData
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

    def test_reds_for_minimal_V(self):
        reds = omni.compute_reds(4, self.pols, self.info.antloc[:self.info.nant])
        mVreds = omni.reds_for_minimal_V(reds)
        # test that the new reds array is shorter by 1/4, as expected
        nt.assert_equal(len(mVreds),len(reds) - len(reds)/4)
        # test that we haven't invented or lost baselines
        cr,cmv = 0,0
        for arr in reds: cr+=len(arr)
        for arr in mVreds: cmv+=len(arr)
        nt.assert_equal(cr,cmv)
        # test that no crosspols are in linpol red arrays and vice versa
        for arr in mVreds:
            p0 = arr[0][0].pol()+arr[0][1].pol()
            for ap in arr:
                p = ap[0].pol()+ap[1].pol()
                nt.assert_equal(len(set(p)),len(set(p0)))
        # test that every xy has its corresponding yx in the array
        for arr in mVreds:
            p0 = arr[0][0].pol()+arr[0][1].pol()
            if len(set(p0))==1: continue #not interested in linpols
            for ap in arr:
                ai,aj = ap
                bi,bj = omni.Antpol(ai.ant(),aj.pol(),self.info.nant),omni.Antpol(aj.ant(),ai.pol(),self.info.nant) 
                if not (bi,bj) in arr and not (bj,bi) in arr:
                    raise ValueError('Something has gone wrong in polarized redundancy calculation (missing crosspols)')
        # test AssertionError
        _reds = reds[:-1]
        try:
            omni.reds_for_minimal_V(_reds)
            assert False, 'should not have gotten here'
        except Exception:
            pass   
    
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

        pol2str = {-5: 'x', -6: 'y'}
        uvcal = UVCal()
        uvcal.read_calfits(os.path.join(DATA_PATH,'zen.2457698.40355.xx.HH.uvc.fits'))
        np.testing.assert_equal(uvcal.freq_array.flatten(), meta['freqs'])
        np.testing.assert_equal(np.resize(uvcal.time_array, (Ntimes,)), meta['times'])  # need repeat here because reading 2 files.
        nt.assert_equal(uvcal.history, meta['history'])
        nt.assert_equal(uvcal.gain_convention, meta['gain_conventions'])
        for ai, ant in enumerate(uvcal.ant_array):
            for ip, pol in enumerate(uvcal.jones_array):
                for nsp in range(uvcal.Nspws):
                    np.testing.assert_equal(np.resize(uvcal.gain_array[ai, nsp, :, :, ip].T, (Ntimes,Nchans)),  gains[pol2str[pol]][ant])

        str2pol = {'xx': -5, 'yy': -6}
        uvd = UVData()
        uvd.read_uvfits(os.path.join(DATA_PATH,'zen.2457698.40355.xx.HH.uvc.vis.fits'))
        # frim_fits turns data into drift
        uvd.unphase_to_drift()
        for pol in vis:
            for i,j in vis[pol]:
                uvpol = list(uvd.polarization_array).index(str2pol[pol])
                uvmask = np.all(np.array(zip(uvd.ant_1_array, uvd.ant_2_array)) == [i,j], axis=1)
                # need to resize because test is reading in 2 files with from_fits.
                np.testing.assert_equal(vis[pol][i,j], np.resize(uvd.data_array[uvmask][:,0,:,uvpol], vis[pol][i,j].shape))
                
        uvd = UVData()
        uvd.read_uvfits(os.path.join(DATA_PATH,'zen.2457698.40355.xx.HH.uvc.xtalk.fits'))
        # from_fits turns data into drift
        uvd.unphase_to_drift()
        for pol in xtalk:
            for i,j in xtalk[pol]:
                uvpol = list(uvd.polarization_array).index(str2pol[pol])
                uvmask = np.all(np.array(zip(uvd.ant_1_array, uvd.ant_2_array)) == [i,j], axis=1)
                # need to resize because test is reading in 2 files with from_fits.
                np.testing.assert_equal(xtalk[pol][i,j], np.resize(uvd.data_array[uvmask][:,0,:,uvpol], xtalk[pol][i,j].shape)) 
 
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

    def test_make_uvdata_vis(self):
        sys.path.append(DATA_PATH)  # append data_path to path so we can find calfile.
        aa = a.cal.get_aa('heratest_calfile', np.array([.15])) # This aa is specific for the fits file below.
        sys.path[:-1]  # remove last entry from path (DATA_PATH)

        # read in meta, gains, vis, xtalk from file.
        meta, gains, vis, xtalk = omni.from_fits([os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits')])
        _xtalk = {}
        # overide xtalk to have single visibility. from fits expands to size of vis data.
        for pol in xtalk.keys():
            _xtalk[pol] = {key : xtalk[pol][key][0,:] for key in xtalk[pol].keys() }
        # write to new file for both vis and xtalk
        uv = omni.make_uvdata_vis(aa, meta, vis)
        uv.write_uvfits(os.path.join(DATA_PATH,'write_vis_test.fits'), force_phase=True, spoof_nonessential=True)
        uv = omni.make_uvdata_vis(aa, meta, _xtalk, xtalk=True)
        uv.write_uvfits(os.path.join(DATA_PATH,'write_xtalk_test.fits'), force_phase=True, spoof_nonessential=True)
        
        # read in old and newly written files and check equality.
        uv_vis_in = UVData()
        uv_vis_in.read_uvfits(os.path.join(DATA_PATH,'zen.2457698.40355.xx.HH.uvc.vis.fits'))
        uv_vis_in.unphase_to_drift()

        uv_xtalk_in = UVData()
        uv_xtalk_in.read_uvfits(os.path.join(DATA_PATH,'zen.2457698.40355.xx.HH.uvc.xtalk.fits'))
        uv_xtalk_in.unphase_to_drift()

        uv_vis_out= UVData()
        uv_vis_out.read_uvfits(os.path.join(DATA_PATH,'write_vis_test.fits'))
        uv_vis_out.unphase_to_drift()

        uv_xtalk_out = UVData()
        uv_xtalk_out.read_uvfits(os.path.join(DATA_PATH,'write_xtalk_test.fits'))
        uv_xtalk_out.unphase_to_drift()

        nt.assert_equal(uv_vis_in, uv_vis_out)
        nt.assert_equal(uv_xtalk_in, uv_xtalk_in)
    
    def test_concatenate_UVCal_on_pol(self):
        calname0 = os.path.join(DATA_PATH,'zen.2457705.41052.xx.HH.uvc.firstcal.fits')
        calname1 = os.path.join(DATA_PATH,'zen.2457705.41052.yy.HH.uvc.firstcal.fits')
        calnameList = [calname0,calname1]
        cal0 = UVCal()
        cal0.read_calfits(calname0)
        cal1 = UVCal()
        cal1.read_calfits(calname1)
        
        # Concatenate and test concatenation
        newcal = omni.concatenate_UVCal_on_pol(calnameList)
        testpath0 = os.path.join(DATA_PATH, 'zen.2457705.41052.yy.HH.uvc.firstcal.test0.fits')
        if os.path.exists(testpath0): os.system('rm %s'%testpath0)
        newcal.write_calfits(testpath0)
        
        nt.assert_equal(newcal.Njones,2)
        nt.assert_equal(sorted(newcal.jones_array), [-6,-5])
        nt.assert_equal(newcal.flag_array.shape[-1], 2)
        nt.assert_equal(newcal.delay_array.shape[-1], 2)
        nt.assert_equal(newcal.quality_array.shape[-1], 2)
        
        cal1.gain_convention = 'multiply'
        testpath1 = os.path.join(DATA_PATH, 'zen.2457705.41052.yy.HH.uvc.firstcal.test1.fits')
        if os.path.exists(testpath1): os.system('rm %s'%testpath1)
        cal1.write_calfits(testpath1)
        
        try:
            failcal = omni.concatenate_UVCal_on_pol([calname0,calname0])
            assert False, 'should not have gotten here'
        except ValueError:
            pass
        try:
            failcal = omni.concatenate_UVCal_on_pol([calname0,testpath0])
            assert False, 'should not have gotten here'
        except ValueError:
            pass
        try:
            failcal = omni.concatenate_UVCal_on_pol([calname0,testpath1])
            assert False, 'should not have gotten here'
        except ValueError:
            pass

    def test_miriad_to_dict(self):
        str2pol = {'xx':-5, 'yy': -6, 'xy':-7, 'yy':-8}
        uvd =  UVData()
        uvd.read_miriad(os.path.join(DATA_PATH,'zen.2457698.40355.xx.HH.uvcAA'))
    
        d,f = omni.miriad_to_dict(os.path.join(DATA_PATH,'zen.2457698.40355.xx.HH.uvcAA'))
        for pol in d:
            for i,j in d[pol]:
                uvpol = list(uvd.polarization_array).index(str2pol[pol])
                uvmask = np.all(np.array(zip(uvd.ant_1_array, uvd.ant_2_array)) == [i,j], axis=1)
                np.testing.assert_equal(d[pol][i,j], uvd.data_array[uvmask][:,0,:,uvpol])
                np.testing.assert_equal(f[pol][i,j], uvd.flag_array[uvmask][:,0,:,uvpol])
        
        
        
        
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
        optional = {'observer': 'heracal'}
        hc = omni.HERACal(meta, gains, optional=optional)  # the fits file was run with ex_ants=[81] and we need to include it here for the test.
        uv = UVCal()
        uv.read_calfits(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.fits'))
        for param in hc:
            if param == '_history': continue
            elif param == '_time_range':  # why do we need this?
                nt.assert_equal(np.testing.assert_almost_equal(getattr(hc, param).value, getattr(uv, param).value, 5), None)
            else:
                nt.assert_true(np.all(getattr(hc, param) == getattr(uv, param)))

