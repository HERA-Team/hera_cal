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


def noise(size, scale=1.0):
    sig = 1./np.sqrt(2)
    return 1+scale*(np.random.normal(scale=sig, size=size) + 1j*np.random.normal(scale=sig, size=size))


class Test_AbsCal_Funcs:

    def setUp(self):
        # load into pyuvdata object
        self.uvd = UVData()
        self.uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA"))
        self.freq_array = np.unique(self.uvd.freq_array)
        self.antpos, self.ants = self.uvd.get_ENU_antpos(center=True, pick_data_ants=True)
        self.antpos = odict(zip(self.ants, self.antpos))
        self.time_array = np.unique(self.uvd.time_array)

        np.random.seed(0)

        # configure data into dictionaries
        model, flags, pols = hc.abscal.UVData2AbsCalDict(self.uvd)
        for k in model.keys():
            if k[0] == k[1]:
                model.pop(k)
                
        wgts = copy.deepcopy(flags)
        for k in wgts.keys():
            wgts[k] = (~wgts[k]).astype(np.float)
                
        bls = odict()
        for bl_id in np.unique(self.uvd.baseline_array):
            bl = self.uvd.baseline_to_antnums(bl_id)
            if bl[0] == bl[1]:
                continue
            bls[bl] = self.antpos[bl[1]] - self.antpos[bl[0]]
            
        abs_gain = 0.5
        phs_phi = np.array([-0.004, 0.006, 0])
        noise_amp = 0.1
        data = odict()
        for i, k in enumerate(model.keys()):
            data[k] = model[k] + noise(model[k].shape, scale=noise_amp)
            model[k] *= np.exp(abs_gain + 1j*np.dot(phs_phi, bls[k]))

        self.data = data
        self.bls = bls
        self.model = model
        self.wgts = wgts

        self.AC = hc.abscal.AbsCal(self.model, self.data, wgts=self.wgts,
                                   antpos=self.antpos, freqs=self.freq_array, times=self.time_array, pols=pols)

    def test_init(self):
        # init with no antpos
        AC = hc.abscal.AbsCal(self.model, self.data)
        # init with antpos
        AC = hc.abscal.AbsCal(self.model, self.data, antpos=self.antpos)
        nt.assert_almost_equal(AC.bls[(11,12)][0], 14.607843358274238)

    def test_abs_amp_lincal(self):
        # no unravel
        self.AC.abs_amp_lincal(verbose=False)
        nt.assert_equal(self.AC.abs_amp.shape, (60, 64, 1))
        # unravel time
        self.AC.abs_amp_lincal(unravel_time=True, verbose=False)
        nt.assert_equal(self.AC.abs_amp.shape, (1, 64, 1))

    def test_TT_phs_logcal(self):
        # no unravel
        self.AC.TT_phs_logcal(verbose=False)
        nt.assert_equal(self.AC.psi.shape, (60, 64, 1))
        nt.assert_equal(self.AC.TT_PHI.shape, (2, 60, 64, 1))

    def test_delay_lincal(self):
        pass

    def test_phs_logcal(self):
        pass

    def test_amp_logcal(self):
        pass

    def test_UVData2AbsCalDict(self):
        # test filename
        data, flags, pols = hc.abscal.UVData2AbsCalDict([os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")])
        nt.assert_equal(data[(11, 12)].shape, (60, 64, 1))
        nt.assert_equal(flags[(11, 12)].shape, (60, 64, 1))
        nt.assert_equal(pols[0], 'xx')

        # test uvd object
        uvd = UVData()
        uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA"))
        data, flags, pols = hc.abscal.UVData2AbsCalDict(uvd)
        nt.assert_equal(data[(11, 12)].shape, (60, 64, 1))

        # test multiple 
        data, flags, pols = hc.abscal.UVData2AbsCalDict([uvd, uvd])
        nt.assert_equal(len(data), 2)
        nt.assert_equal(len(flags), 2)
        nt.assert_equal(pols[0], ['xx'])

    def test_interp2d(self):
        # test interpolation
        m, mf = hc.abscal.interp2d_vis(self.data, self.time_array, self.freq_array,
                                       self.time_array, self.freq_array)
        nt.assert_equal(m[(11, 12)].shape, (60, 64, 1))
        # downsampling
        m, mf = hc.abscal.interp2d_vis(self.data, self.time_array, self.freq_array,
                                       self.time_array[::2], self.freq_array[::2])
        nt.assert_equal(m[(11, 12)].shape, (30, 32, 1))

    def test_avg_data_across_red_bls(self):
        pass

    def test_avg_file_across_red_bls(self):
        pass

    def test_mirror_data_to_red_bls(self):
        pass

    def test_compute_reds(self):
        pass

    def test_gains2calfits(self):
        pass

    def test_echo(self):
        hc.abscal.echo('hi', verbose=False)

    def test_lst_align(self):
        pass


class Test_AbsCal:

    def setUp(self):
        # load into pyuvdata object
        self.uvd = UVData()
        self.uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA"))
        self.freq_array = self.uvd.freq_array.squeeze()
        self.antpos, self.ants = self.uvd.get_ENU_antpos(center=True, pick_data_ants=True)
        self.time_array = self.uvd.time_array.reshape(self.uvd.Ntimes, self.uvd.Nbls)[:, 0]

        np.random.seed(0)



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

    def test_make_gains(self):
        pass

    def test_write_calfits(self):
        pass



