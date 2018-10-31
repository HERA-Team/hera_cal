# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import nose.tools as nt
import os
import shutil
import numpy as np
import sys
from collections import OrderedDict as odict
import copy
import glob
from pyuvdata import UVData
from pyuvdata import utils as uvutils

import hera_cal as hc
from hera_cal.data import DATA_PATH


def test_timeavg_waterfall():
    fname = os.path.join(DATA_PATH, "zen.2458042.12552.xx.HH.uvXA")

    uvd = UVData()
    uvd.read_miriad(fname)

    d = uvd.get_data(24, 25)
    f = uvd.get_flags(24, 25)
    n = uvd.get_nsamples(24, 25)
    t = np.unique(uvd.time_array)
    fr = uvd.freq_array.squeeze()
    lsts = []
    for _l in uvd.lst_array:
        if _l not in lsts:
            lsts.append(_l)
    lsts = np.array(lsts)
    antpos, ants = uvd.get_ENU_antpos()
    blv = antpos[ants.tolist().index(24)] - antpos[ants.tolist().index(25)]

    # test basic execution
    output = hc.frf.timeavg_waterfall(d, 25, verbose=False)
    ad, af, an, al, aea = (output['avg_data'], output['win_flags'], output['avg_nsamples'],
                           output['avg_lsts'], output['avg_extra_arrays'])
    nt.assert_equal(ad.shape, (3, 64))
    nt.assert_equal(af.shape, (3, 64))
    nt.assert_equal(an.shape, (3, 64))
    nt.assert_false(np.any(af))
    nt.assert_almost_equal(an[1, 0], 25.0)
    nt.assert_almost_equal(an[2, 0], 10.0)

    # test rephase
    output = hc.frf.timeavg_waterfall(d, 25, flags=f, rephase=True, lsts=lsts, freqs=fr, bl_vec=blv,
                                      nsamples=n, extra_arrays=dict(times=t), verbose=False)
    ad, af, an, al, aea = (output['avg_data'], output['win_flags'], output['avg_nsamples'],
                           output['avg_lsts'], output['avg_extra_arrays'])

    nt.assert_equal(ad.shape, (3, 64))
    nt.assert_equal(af.shape, (3, 64))
    nt.assert_equal(an.shape, (3, 64))
    nt.assert_true(np.any(af))
    nt.assert_equal(len(al), 3)
    nt.assert_equal(len(aea['avg_times']), 3)
    nt.assert_almost_equal(an.max(), 25.0)

    # test various Navgs
    output = hc.frf.timeavg_waterfall(d, 1, flags=f, rephase=True, lsts=lsts, freqs=fr, bl_vec=blv,
                                      nsamples=n, extra_arrays=dict(times=t), verbose=False)
    ad, af, an, al, aea = (output['avg_data'], output['win_flags'], output['avg_nsamples'],
                           output['avg_lsts'], output['avg_extra_arrays'])

    nt.assert_equal(ad.shape, (60, 64))
    output = hc.frf.timeavg_waterfall(d, 60, flags=f, rephase=True, lsts=lsts, freqs=fr, bl_vec=blv,
                                      nsamples=n, extra_arrays=dict(times=t), verbose=False)
    ad, af, an, al, aea = (output['avg_data'], output['win_flags'], output['avg_nsamples'],
                           output['avg_lsts'], output['avg_extra_arrays'])
    nt.assert_equal(ad.shape, (1, 64))

    # wrap lst
    output = hc.frf.timeavg_waterfall(d, 60, flags=f, rephase=True, lsts=lsts + 1.52917804, freqs=fr, bl_vec=blv,
                                      nsamples=n, extra_arrays=dict(times=t), verbose=False)
    ad2, af2, an2, al2, aea2 = (output['avg_data'], output['win_flags'], output['avg_nsamples'],
                                output['avg_lsts'], output['avg_extra_arrays'])

    nt.assert_equal(ad.shape, (1, 64))
    nt.assert_true(np.isclose(ad, ad2).all())
    nt.assert_true(np.allclose(al, al2 - 1.52917804))


class Test_FRFilter:

    def setUp(self):
        self.fname = os.path.join(DATA_PATH, "zen.2458042.12552.xx.HH.uvXA")
        self.F = hc.frf.FRFilter()
        self.uvd = UVData()
        self.uvd.read_miriad(self.fname)

    def test_load_data(self):
        self.F.load_data(self.fname)
        hd = hc.io.HERAData(self.fname, filetype='miriad')
        hd.read()
        nt.assert_equal(self.F.input_data, hd)
        self.F.load_data(self.uvd)
        nt.assert_equal(self.F.input_data, hd)

    def test_timeavg_data(self):
        self.F.load_data(self.uvd)
        self.F.timeavg_data(35, rephase=True)
        nt.assert_equal(self.F.Navg, 3)

        self.F.timeavg_data(1e10, rephase=True, verbose=False)
        nt.assert_equal(self.F.Navg, 60)

        # exceptions
        nt.assert_raises(AssertionError, self.F.timeavg_data, 1.0)

    def test_write_data(self):
        self.F.load_data(self.uvd)
        self.F.timeavg_data(35, rephase=False, verbose=False)
        u = self.F.write_data("./out.uv", write_avg=True, filetype='miriad', overwrite=True)
        nt.assert_true(os.path.exists("./out.uv"))
        hd = hc.io.HERAData('./out.uv', filetype='miriad')
        hd.read()
        nt.assert_equal(u, hd)

        u = self.F.write_data("./out.uv", overwrite=False)
        nt.assert_equal(u, None)

        u = self.F.write_data("./out.uv", write_avg=False, overwrite=True)
        nt.assert_true(np.isclose(u.data_array, self.uvd.data_array).all())
        if os.path.exists("./out.uv"):
            shutil.rmtree("./out.uv")
