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

from .. import utils, io, apply_cal
from ..tempcal import TempCal
from ..data import DATA_PATH


def test_tempcal():
    # load data
    dfiles = sorted(glob.glob(os.path.join(DATA_PATH, "zen.2458043.*.xx.HH.XRAA.uvh5")))
    T = TempCal(dfiles)
    autos = [bl for bl in T.bls if bl[0] == bl[1]]
    T.read(bls=autos)

    # test gains from autos
    T.flags[(38, 38, 'xx')][:] = True
    T.gains_from_autos(T.data, T.times, flags=T.flags, smooth_frate=1.0, nl=1e-10, Nmirror=20,
                       edgeflag=10, verbose=False)

    # assert smooth and ratio are populated
    k = (24, 24, 'xx')
    gkey = (24, 'Jxx')
    assert (k in T.ratio) and (k in T.smooth)
    assert T.ratio[k].shape == T.data[k].shape 

    # assert gains are constant across freq
    assert np.isclose(T.gains[gkey][0], T.gains[gkey][0, 0]).all() 

    # assert residual std below a value that is set by-hand when it works properly
    assert np.std((T.data[k] - T.smooth[k])[:, 10:-10][~T.flags[k][:, 10:-10]]) < 20

    # assert flag propagation
    assert np.all(T.gflags[(38, 'Jxx')])

    # test avg_ants: assert all gains are the same
    T.avg_ants(list(T.gains.keys()))
    assert np.all([np.isclose(T.gains[_k] - T.gains[(24, 'Jxx')], 0.0).all() for _k in T.gains])

    # test setting abscal time
    caltime = 2458043.41427365
    T.set_abscal_time(T.times, caltime)
    assert np.isclose(np.abs(T.gains[gkey][np.argmin(np.abs(T.times - caltime)), :]), 1.0).all()

    # test applying calibration
    caldata = copy.deepcopy(T.data)
    apply_cal.calibrate_in_place(caldata, T.gains)
    assert np.std((caldata[k] - T.smooth[k])[:, 10:-10][~T.flags[k][:, 10:-10]]) < 15

    # test write
    T.write_gains("./test_ex.calfits", overwrite=True)
    assert os.path.exists("./test_ex.calfits")
    os.remove("./test_ex.calfits")

    # test exceptions
    pytest.raises(NotImplementedError, T.gains_from_tempdata, None, None, None, None)
