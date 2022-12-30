# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License


import pytest
import numpy as np
import os
import copy
from pyuvdata import UVData

from .. import datacontainer, flag_utils, io, utils
from ..data import DATA_PATH


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
def test_solar_flag():
    data_fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
    uvd = UVData()
    uvd.read_miriad(data_fname)
    uvd.use_future_array_shapes()
    data, flags, antp, ant, f, t, l, p = io.load_vis(uvd, return_meta=True)
    # get solar altitude
    a = utils.get_sun_alt(2458043)
    assert isinstance(a, (float, np.floating, np.float64))
    a = utils.get_sun_alt([2458043, 2458043.5])
    assert isinstance(a, (np.ndarray))
    # test solar flag
    bl = (24, 25, 'ee')
    _flags = flag_utils.solar_flag(flags, times=t, flag_alt=20.0, inplace=False)
    assert _flags[bl][:41].all()
    assert not flags[bl][:41].all()
    # test ndarray
    flag_utils.solar_flag(flags[bl], times=t, flag_alt=20.0, inplace=True)
    assert flags[bl][:41].all()
    # test uvd
    flag_utils.solar_flag(uvd, flag_alt=20.0, inplace=True)
    assert uvd.get_flags(bl)[:41].all()
    # test exception
    pytest.raises(AssertionError, flag_utils.solar_flag, flags)


def test_synthesize_ant_flags():
    flags = datacontainer.DataContainer({(0, 0, 'xx'): np.ones((5, 5), bool),
                                         (0, 1, 'xx'): np.ones((5, 5), bool),
                                         (1, 2, 'xx'): np.zeros((5, 5), bool),
                                         (2, 3, 'xx'): np.zeros((5, 5), bool)})
    flags[(2, 3, 'xx')][:, 4] = True
    # aggressive flagging
    ant_flags = flag_utils.synthesize_ant_flags(flags, threshold=0.0)
    np.testing.assert_array_equal(ant_flags[(0, 'Jxx')], True)
    np.testing.assert_array_equal(ant_flags[(1, 'Jxx')], False)
    np.testing.assert_array_equal(ant_flags[(2, 'Jxx')][:, 0:4], False)
    np.testing.assert_array_equal(ant_flags[(2, 'Jxx')][:, 4], True)
    np.testing.assert_array_equal(ant_flags[(3, 'Jxx')][:, 0:4], False)
    np.testing.assert_array_equal(ant_flags[(3, 'Jxx')][:, 4], True)
    # conservative flagging
    ant_flags = flag_utils.synthesize_ant_flags(flags, threshold=0.75)
    np.testing.assert_array_equal(ant_flags[(2, 'Jxx')][:, 4], False)
    # very conservative flagging
    flags[(1, 2, 'xx')][:3, 4] = True
    ant_flags = flag_utils.synthesize_ant_flags(flags, threshold=1.0)
    np.testing.assert_array_equal(ant_flags[(2, 'Jxx')][:3, 4], True)
    np.testing.assert_array_equal(ant_flags[(2, 'Jxx')][3:, 4], False)


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
def test_factorize_flags():
    # load data
    data_fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
    hd = io.HERAData(data_fname, filetype='miriad')
    data, flags, _ = hd.read(bls=[(24, 25)])

    # run on ndarray
    f = flag_utils.factorize_flags(flags[(24, 25, 'ee')].copy(), time_thresh=1.5 / 60, inplace=False)
    assert f[52].all()
    assert f[:, 60:62].all()

    f = flag_utils.factorize_flags(flags[(24, 25, 'ee')].copy(), spw_ranges=[(45, 60)],
                                   time_thresh=0.5 / 60, inplace=False)
    assert f[:, 48].all()
    assert not np.min(f, axis=1).any()
    assert not f[:, 24].all()

    f = flags[(24, 25, 'ee')].copy()
    flag_utils.factorize_flags(f, time_thresh=0.5 / 60, inplace=True)
    assert f[:, 48].all()
    assert not np.min(f, axis=1).any()

    # run on datacontainer
    f2 = flag_utils.factorize_flags(copy.deepcopy(flags), time_thresh=0.5 / 60, inplace=False)
    np.testing.assert_array_equal(f2[(24, 25, 'ee')], f)

    # test exceptions
    pytest.raises(ValueError, flag_utils.factorize_flags, flags, spw_ranges=(0, 1))
    pytest.raises(ValueError, flag_utils.factorize_flags, 'hi')
