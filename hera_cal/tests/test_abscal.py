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
from hera_cal.omni import compute_reds
from hera_cal.data import DATA_PATH
from collections import OrderedDict
import copy


def noise(size, scale=1.0):
    sig = 1./np.sqrt(2)
    return 1+scale*(np.random.normal(scale=sig, size=size) + 1j*np.random.normal(scale=sig, size=size))


class Test_AbsCal:

    def setUp(self):
        # load into pyuvdata object
        self.uvd = UVData()
        self.uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA"))
        self.freq_array = self.uvd.freq_array.squeeze()
        self.antpos, self.ants = uvutils.get_ENU_from_UVData(self.uvd, center=True, pick_data_ants=True)
        self.time_array = self.uvd.time_array.reshape(self.uvd.Ntimes, self.uvd.Nbls)[:, 0]

        np.random.seed(0)

        # configure data into dictionaries
        model, flags, pols = hc.abscal.UVData2AbsCalDict([self.uvd])
        for k in model.keys():
            if k[0] == k[1]:
                model.pop(k)
                
        wgts = copy.deepcopy(flags)
        for k in wgts.keys():
            wgts[k] = (~wgts[k]).astype(np.float)
                
        bls = OrderedDict()
        antpos, ants = uvutils.get_ENU_from_UVData(self.uvd, pick_data_ants=True)
        ants = ants.tolist()
        for bl_id in np.unique(self.uvd.baseline_array):
            bl = self.uvd.baseline_to_antnums(bl_id)
            if bl[0] == bl[1]:
                continue
            bls[bl] = antpos[ants.index(bl[1])] - antpos[ants.index(bl[0])]
            
        abs_gain = 0.5
        phs_phi = np.array([-0.04, 0.06, 0])
        noise_amp = 0.1
        data = OrderedDict()
        for i, k in enumerate(model.keys()):
            data[k] = model[k] + noise(model[k].shape, scale=noise_amp)
            model[k] *= np.exp(abs_gain + 1j*np.dot(phs_phi, bls[k]))

        self.data = data
        self.bls = bls
        self.model = model
        self.wgts = wgts

        self.AC = hc.abscal.AbsCal(self.model, self.data, bls=self.bls)

    def test_init(self):
        # init with no bls
        AC = hc.abscal.AbsCal(self.model, self.data)
        nt.assert_almost_equal(AC.data[(11,12)][0,0,0], (1.1247373376201772+0.098716504553859774j))
        # init with bls
        AC = hc.abscal.AbsCal(self.model, self.data, bls=self.bls)
        nt.assert_almost_equal(AC.bls[(11,12)][0], 14.607843358274238)
        # test exceptions
        nt.assert_raises(TypeError, hc.abscal.AbsCal, self.model, self.data, freqs=np.arange(20))
        nt.assert_raises(TypeError, hc.abscal.AbsCal, self.model, self.data, pols=np.arange(20))

    def test_amp_lincal(self):
        # no unravel
        self.AC.amp_lincal()
        nt.assert_equal(self.AC.gain_amp.shape, (60, 64, 1))
        nt.assert_almost_equal(self.AC.gain_amp[10,10,0], 1.9234912147968954)
        # unravel freq
        self.AC.amp_lincal(unravel_freq=True)
        nt.assert_equal(self.AC.gain_amp.shape, (60, 1, 1))
        nt.assert_almost_equal(self.AC.gain_amp[10,0,0], 0.001582025004187968)
        # unravel time
        self.AC.amp_lincal(unravel_time=True)
        nt.assert_equal(self.AC.gain_amp.shape, (1, 64, 1))
        nt.assert_almost_equal(self.AC.gain_amp[0,10,0], 1.5109382164897442)
        # unravel time and freq
        self.AC.amp_lincal(unravel_freq=True, unravel_time=True)
        nt.assert_equal(self.AC.gain_amp.shape, (1, 1, 1))
        nt.assert_almost_equal(self.AC.gain_amp[0,0,0], 0.00027423806417397386)
        # unravel pol
        self.AC.amp_lincal(unravel_pol=True)
        nt.assert_equal(self.AC.gain_amp.shape, (60, 64, 1))
        nt.assert_almost_equal(self.AC.gain_amp[10,10,0], 1.9234912147968954)


    def test_phs_logcal(self):
        # no unravel
        self.AC.phs_logcal(verbose=True)
        nt.assert_equal(self.AC.gain_psi.shape, (60, 64, 1))
        nt.assert_equal(self.AC.gain_phi.shape, (2, 60, 64, 1))
        # unravel
        self.AC.phs_logcal(unravel_time=True, unravel_freq=True, unravel_pol=True)
        nt.assert_equal(self.AC.gain_psi.shape, (1, 1, 1))
        nt.assert_equal(self.AC.gain_phi.shape, (2, 1, 1, 1))
        nt.assert_almost_equal(self.AC.gain_phi[0,0,0,0], -0.0041545500054190705)
        nt.assert_almost_equal(self.AC.gain_phi[1,0,0,0], -0.05688947460975867)
        nt.assert_almost_equal(self.AC.gain_psi[0,0,0], 1.1356928190550697)

    def test_UVData2AbsCalDict(self):
        # test filename
        data, flags, pols = hc.abscal.UVData2AbsCalDict([os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")])
        nt.assert_equal(data[(11, 12)].shape, (60, 64, 1))
        nt.assert_almost_equal(data[(11, 12)][10, 10, 0], (-2.715235+3.0420172j))
        nt.assert_equal(flags[(11, 12)].shape, (60, 64, 1))
        nt.assert_equal(pols[0], 'xx')

        # test uvd object
        uvd = UVData()
        uvd.read_miriad(os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA"))
        data, flags, pols = hc.abscal.UVData2AbsCalDict([uvd])
        nt.assert_equal(data[(11, 12)].shape, (60, 64, 1))
        nt.assert_almost_equal(data[(11, 12)][10, 10, 0], (-2.715235+3.0420172j))

        # test multiple 
        data, flags, pols = hc.abscal.UVData2AbsCalDict([uvd, uvd])
        nt.assert_equal(len(data), 2)
        nt.assert_equal(len(flags), 2)
        nt.assert_equal(pols[0], ['xx'])

    def test_interp_model(self):
        # test interpolation
        m = hc.abscal.interp_model(self.model, self.time_array, self.freq_array, self.time_array+.0001, self.freq_array)
        nt.assert_equal(m[(11, 12)].shape, (60, 64, 1))
        # downsampling
        m = hc.abscal.interp_model(self.model, self.time_array, self.freq_array, self.time_array[::2]+.0001, self.freq_array[::2])
        nt.assert_equal(m[(11, 12)].shape, (30, 32, 1))









