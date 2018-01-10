'''Tests for abscal.py'''
import nose.tools as nt
import os
import json
import numpy as np
import aipy
import optparse
import sys
from pyuvdata import UVCal, UVData
from pyuvdata import utils as uvutils
import hera_cal as hc
from hera_cal.data import DATA_PATH
from collections import OrderedDict as odict
import copy

class Test_AbsCal_Funcs:

    def setUp(self):
        np.random.seed(0)

        # load into pyuvdata object
        self.uvd = UVData()
        self.uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA"))
        self.freq_array = np.unique(self.uvd.freq_array)
        self.antpos, self.ants = self.uvd.get_ENU_antpos(center=True, pick_data_ants=True)
        self.antpos = odict(zip(self.ants, self.antpos))
        self.time_array = np.unique(self.uvd.time_array)

        # configure data into dictionaries
        data, flags = hc.abscal.UVData2AbsCalDict(self.uvd, pop_autos=True)

        # configure wgts
        wgts = copy.deepcopy(flags)
        for k in wgts.keys():
            wgts[k] = (~wgts[k]).astype(np.float)
                
        # configure baselines
        bls = odict([(x, self.antpos[x[1]] - self.antpos[x[0]]) for x in data.keys()])

        abs_gain = 0.5
        TT_phi = np.array([-0.004, 0.006, 0])
        model = odict()
        for i, k in enumerate(data.keys()):
            model[k] = data[k] * np.exp(abs_gain + 1j*np.dot(TT_phi, bls[k]))

        # assign data
        self.data = data
        self.bls = bls
        self.model = model
        self.wgts = wgts

    def test_UVData2AbsCalDict(self):
        # test filename
        fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        data, flags = hc.abscal.UVData2AbsCalDict(fname, pop_autos=False)
        nt.assert_equal(data[(11, 12, 'xx')].shape, (60, 64))
        nt.assert_equal(flags[(11, 12, 'xx')].shape, (60, 64))
        nt.assert_equal((11, 11, 'xx') in data, True)
        data, flags = hc.abscal.UVData2AbsCalDict([fname])
        nt.assert_equal(data[(11, 12, 'xx')].shape, (60, 64))

        # test pop autos
        data, flags = hc.abscal.UVData2AbsCalDict(fname, pop_autos=True)
        nt.assert_equal((11, 11, 'xx') in data, False)

        # test pol select
        data, flags = hc.abscal.UVData2AbsCalDict(fname, pop_autos=False, pol_select=['xx'])
        nt.assert_equal(data[(11, 12, 'xx')].shape, (60, 64))

        # test uvd object
        uvd = UVData()
        uvd.read_miriad(fname)
        data, flags = hc.abscal.UVData2AbsCalDict(uvd)
        nt.assert_equal(data[(11, 12, 'xx')].shape, (60, 64))
        data, flags = hc.abscal.UVData2AbsCalDict([uvd])
        nt.assert_equal(data[(11, 12, 'xx')].shape, (60, 64))

        # test multiple
        fname2 = os.path.join(DATA_PATH, "zen.2458043.13298.xx.HH.uvORA")
        data, flags = hc.abscal.UVData2AbsCalDict([fname, fname2])
        nt.assert_equal(data[(11, 12, 'xx')].shape, (120, 64))
        nt.assert_equal(flags[(11, 12, 'xx')].shape, (120, 64))

    def test_data_key_to_array_axis(self):
        m, pk = hc.abscal.data_key_to_array_axis(self.model, 2)
        nt.assert_equal(m[(11, 12)].shape, (60, 64, 1))
        nt.assert_equal('xx' in pk, True)
        # test w/ avg_dict
        m, ad, pk = hc.abscal.data_key_to_array_axis(self.model, 2, avg_dict=self.bls)
        nt.assert_equal(m[(11, 12)].shape, (60, 64, 1))
        nt.assert_equal(ad[(11, 12)].shape, (3,))
        nt.assert_equal('xx' in pk, True)

    def test_array_axis_to_data_key(self):
        m, pk = hc.abscal.data_key_to_array_axis(self.model, 2)
        m2 = hc.abscal.array_axis_to_data_key(m, 2, ['xx'])
        nt.assert_equal(m2[(11, 12, 'xx')].shape, (60, 64))
        # copy dict
        m, ad, pk = hc.abscal.data_key_to_array_axis(self.model, 2, avg_dict=self.bls)
        m2, cd = hc.abscal.array_axis_to_data_key(m, 2, ['xx'], copy_dict=ad)
        nt.assert_equal(m2[(11, 12, 'xx')].shape, (60, 64))
        nt.assert_equal(cd[(11, 12, 'xx')].shape, (3,))

    def test_interp2d(self):
        # test interpolation
        m, mf = hc.abscal.interp2d_vis(self.data, self.time_array, self.freq_array,
                                       self.time_array, self.freq_array)
        nt.assert_equal(m[(11, 12, 'xx')].shape, (60, 64))
        # downsampling
        m, mf = hc.abscal.interp2d_vis(self.data, self.time_array, self.freq_array,
                                       self.time_array[::2], self.freq_array[::2])
        nt.assert_equal(m[(11, 12, 'xx')].shape, (30, 32))

    def test_compute_reds(self):
        reds = hc.abscal.compute_reds(self.antpos)
        nt.assert_equal(len(reds), 9)
        reds = hc.abscal.compute_reds(self.antpos, ex_ants=[11, 24])
        nt.assert_equal(len(reds), 8)

    def test_gains2calfits(self):
        cfname = os.path.join(DATA_PATH, 'ex.calfits')
        abscal_gains = np.ones((len(self.ants), 60, 1024, 1), dtype=np.complex)
        abscal_gains = odict(zip(self.ants, abscal_gains))
        freq_array = np.linspace(100, 200, 1024)
        time_array = np.linspace(2450842.1, 2450842.4, 60)
        pol_array = np.array(['x'])
        if os.path.exists(cfname):
            os.remove(cfname)
        hc.abscal.gains2calfits(cfname, abscal_gains, freq_array, time_array, pol_array)
        nt.assert_equal(os.path.exists(cfname), True)
        if os.path.exists(cfname):
            os.remove(cfname)

    def test_echo(self):
        hc.abscal.echo('hi', verbose=True)
        hc.abscal.echo('hi', type=1, verbose=True)



class Test_AbsCal:

    def setUp(self):
        np.random.seed(0)

        # load into pyuvdata object
        self.uvd = UVData()
        self.uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA"))
        self.freq_array = self.uvd.freq_array.squeeze()
        self.antpos, self.ants = self.uvd.get_ENU_antpos(center=True, pick_data_ants=True)
        self.time_array = self.uvd.time_array.reshape(self.uvd.Ntimes, self.uvd.Nbls)[:, 0]

        # configure data into dictionaries
        data, flags = hc.abscal.UVData2AbsCalDict(self.uvd, pop_autos=True)

        # configure wgts
        wgts = copy.deepcopy(flags)
        for k in wgts.keys():
            wgts[k] = (~wgts[k]).astype(np.float)
                
        # configure baselines
        bls = OrderedDict([(x, antpos[x[1]] - antpos[x[0]]) for x in data.keys()])

        abs_gain = 0.5
        TT_phi = np.array([-0.004, 0.006, 0])
        model = odict()
        for i, k in enumerate(data.keys()):
            model[k] = data[k] * np.exp(abs_gain + 1j*np.dot(TT_phi, bls[k]))

        # assign data
        self.data = data
        self.bls = bls
        self.model = model
        self.wgts = wgts

        self.AC = hc.abscal.AbsCal(self.model, self.data, wgts=self.wgts,
                                   freqs=self.freq_array, times=self.time_array, pols='xx')
        self.OC = hc.abscal.OmniAbsCal(self.model, self.data, self.antpos, wgts=self.wgts,
                                       freqs=self.freq_array, times=self.time_array, pols='xx')

    def test_init(self):
        # init with no meta
        AC = hc.abscal.AbsCal(self.model, self.data)
        # init with meta
        OC = hc.abscal.OmniAbsCal(self.model, self.data)
        nt.assert_almost_equal(AC.bls[(11,12,'xx')][0], 14.607843358274238)
        nt.assert_almost_equal(OC.bls[(11,12,'xx')][0], 14.607843358274238)


    def test_abs_amp_lincal(self):
        pass

    def test_TT_phs_logcal(self):
        pass

    def test_amp_logcal(self):
        pass

    def test_phs_logcal(self):
        pass

    def test_delay_lincal(self):
        pass


