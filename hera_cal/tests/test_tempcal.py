# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import pytest
import numpy as np
import sys
import os
import glob
import six
import copy
from pyuvdata import UVData

from .. import utils, io, apply_cal, tempcal
from ..data import DATA_PATH


class Test_TempCal():
    def setup_method(self):
        dfiles = sorted(glob.glob(os.path.join(DATA_PATH, "zen.2458043.*.xx.HH.XRAA.uvh5")))
        self.T = tempcal.TempCal(dfiles)
        autos = [bl for bl in self.T.bls if (bl[0] == bl[1])]
        self.T.read(bls=autos)

    def test_gains_from_autos(self):
        # mostly already tested in top-level function
        # this just checks outputs are populated etc.
        self.T.gains_from_autos(self.T.data, self.T.times, flags=self.T.flags,
                                smooth_frate=1.0, nl=1e-10, Nmirror=20,
                                edgeflag=10, verbose=False)
        k = (24, 24, 'xx')
        gkey = (24, 'Jxx')
        assert k in self.T.smooth
        assert self.T.smooth[k].shape == self.T.data[k].shape 

        # test avg_ants b/c it needs gains
        self.T.avg_ants(list(self.T.gains.keys()))
        assert np.all([np.isclose(self.T.gains[_k] - self.T.gains[(24, 'Jxx')], 0.0).all() for _k in self.T.gains])

        # test setting abscal time
        caltime = 2458043.41427365
        self.T.set_abscal_time(self.T.times, caltime)
        assert np.isclose(np.abs(self.T.gains[gkey][np.argmin(np.abs(self.T.times - caltime)), :]), 1.0).all()


def test_gains_from_autos():
    # load data
    dfiles = sorted(glob.glob(os.path.join(DATA_PATH, "zen.2458043.*.xx.HH.XRAA.uvh5")))
    uvd = UVData()
    uvd.read(dfiles, bls=[(24, 24), (25, 25)])
    times = np.unique(uvd.time_array)

    # test gains from autos
    d = uvd.get_data(24, 24, 'xx')
    f = uvd.get_flags(24, 24, 'xx')
    g, gf, s = tempcal.gains_from_autos(d, times, flags=f, smooth_frate=1.0, nl=1e-10, Nmirror=20,
                                        edgeflag=10, verbose=False)
    assert isinstance(g, np.ndarray)

    # assert gains are constant across freq
    assert np.isclose(g[0], g[0, 0]).all() 

    # assert residual std below a value that is set by-hand when it works properly
    assert np.std((d - s)[:, 10:-10][~f[:, 10:-10]]) < 20

    # test applying calibration is a good match to smoothed data
    assert np.std((d / (g**2) - s)[:, 10:-10][~f[:, 10:-10]]) < 15

    # test flag propagation
    f[:] = True
    g, gf, s = tempcal.gains_from_autos(d, times, flags=f, smooth_frate=1.0, nl=1e-10, Nmirror=20,
                                        edgeflag=10, verbose=False)
    assert np.all(gf)


def test_gains_from_tempdata():
    pytest.raises(NotImplementedError, tempcal.gains_from_tempdata, None, None, None)
