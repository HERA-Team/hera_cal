# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import pytest
import numpy as np
import sys
import os
import glob
import copy
from pyuvdata import UVData

from .. import utils, io, apply_cal, tempcal
from ..data import DATA_PATH


class Test_tempcal():
    def setup_method(self):
        dfiles = sorted(glob.glob(os.path.join(DATA_PATH, "zen.2458043.*.xx.HH.XRAA.uvh5")))
        self.hd = io.HERAData(dfiles)
        self.data, self.flags, self.nsamps = self.hd.read(bls=[(24, 24), (25, 25), (37, 37)])
        self.times = np.unique(self.hd.time_array)

    def test_gains_from_autos(self):
        # test gains from autos
        key = (24, 24, 'ee')
        gkey = (24, 'Jee')
        gain, gfalg, smooth = tempcal.gains_from_autos(self.data, self.times, flags=self.flags, smooth_frate=1.0,
                                                       nl=1e-10, Nmirror=20, edgeflag=10, verbose=False)
        assert isinstance(gain, dict)
        assert len(gain) == 3
        assert gain[gkey].shape == (self.hd.Ntimes, self.hd.Nfreqs)

        # assert gains are constant across freq
        assert np.isclose(gain[gkey][0], gain[gkey][0, 0]).all()

        # assert residual std below a value that is set by-hand when it works properly
        assert np.std((self.data[key] - smooth[key])[:, 10:-10][~self.flags[key][:, 10:-10]]) < 20

        # test applying calibration is a good match to smoothed data
        assert np.std((self.data[key] / (gain[gkey]**2) - smooth[key])[:, 10:-10][~self.flags[key][:, 10:-10]]) < 15

        # test flag propagation
        self.flags[key][:] = True
        g, gf, s = tempcal.gains_from_autos(self.data, self.times, flags=self.flags, smooth_frate=1.0,
                                            nl=1e-10, Nmirror=20, edgeflag=10, verbose=False)
        assert np.all(gf[gkey])

    def test_avg_ants(self):
        key = (24, 24, 'ee')
        gkey = (24, 'Jee')
        gain, gflag, smooth = tempcal.gains_from_autos(self.data, self.times, flags=self.flags, smooth_frate=1.0,
                                                       nl=1e-10, Nmirror=20, edgeflag=10, verbose=False)
        # test avg_ants
        ag, af = tempcal.avg_gain_ants(gain, list(gain.keys()), gflags=gflag, inplace=False)
        assert not np.all(np.isclose(gain[(24, 'Jee')], gain[(25, 'Jee')]))
        assert np.all([np.isclose(ag[_k] - ag[(24, 'Jee')], 0.0).all() for _k in ag])

        # test inplace
        tempcal.avg_gain_ants(gain, list(gain.keys()), gflags=gflag, inplace=True)
        assert np.all([np.isclose(gain[_k] - gain[(24, 'Jee')], 0.0).all() for _k in gain])

    def test_normalize_tempgains(self):
        key = (24, 24, 'ee')
        gkey = (24, 'Jee')
        gain, gflag, smooth = tempcal.gains_from_autos(self.data, self.times, flags=self.flags, smooth_frate=1.0,
                                                       nl=1e-10, Nmirror=20, edgeflag=10, verbose=False)

        normtime = 2458043.41427365
        ng = tempcal.normalize_tempgains(gain, self.times, normtime, inplace=False)
        assert np.isclose(np.abs(ng[gkey][np.argmin(np.abs(self.times - normtime)), :]), 1.0).all()

    def test_gains_from_tempdata(self):
        pytest.raises(NotImplementedError, tempcal.gains_from_tempdata, None, None, None)
