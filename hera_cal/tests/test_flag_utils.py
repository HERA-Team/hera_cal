# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License


from __future__ import print_function, division, absolute_import
import nose.tools as nt
import numpy as np
import os
from pyuvdata import UVData
from hera_cal import flag_utils, utils, io


def test_solar_flag():
    data_fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
    uvd = UVData()
    uvd.read_miriad(data_fname)
    data, flags, antp, ant, f, t, l, p = io.load_vis(uvd, return_meta=True)
    # get solar altitude
    a = utils.get_sun_alt(2458043)
    nt.assert_true(isinstance(a, (float, np.float, np.float64)))
    a = utils.get_sun_alt([2458043, 2458043.5])
    nt.assert_true(isinstance(a, (np.ndarray)))
    # test solar flag
    bl = (24, 25, 'xx')
    _flags = flag_utils.solar_flag(flags, times=t, flag_alt=20.0, inplace=False)
    nt.assert_true(_flags[bl][:41].all())
    nt.assert_false(flags[bl][:41].all())
    # test ndarray
    flag_utils.solar_flag(flags[bl], times=t, flag_alt=20.0, inplace=True)
    nt.assert_true(flags[bl][:41].all())
    # test uvd
    flag_utils.solar_flag(uvd, flag_alt=20.0, inplace=True)
    nt.assert_true(uvd.get_flags(bl)[:41].all())
    # test exception
    nt.assert_raises(AssertionError, flag_utils.solar_flag, flags)


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
