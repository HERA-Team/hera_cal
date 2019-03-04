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
import unittest

import hera_cal as hc
from hera_cal.data import DATA_PATH
from scipy import stats


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
    ad, af, an, al, aea = hc.frf.timeavg_waterfall(d, 25, verbose=False)
    nt.assert_equal(ad.shape, (3, 64))
    nt.assert_equal(af.shape, (3, 64))
    nt.assert_equal(an.shape, (3, 64))
    nt.assert_false(np.any(af))
    nt.assert_almost_equal(an[1, 0], 25.0)
    nt.assert_almost_equal(an[2, 0], 10.0)

    # test rephase
    ad, af, an, al, aea = hc.frf.timeavg_waterfall(d, 25, flags=f, rephase=True, lsts=lsts, freqs=fr, bl_vec=blv,
                                                   nsamples=n, extra_arrays=dict(times=t), verbose=False)

    nt.assert_equal(ad.shape, (3, 64))
    nt.assert_equal(af.shape, (3, 64))
    nt.assert_equal(an.shape, (3, 64))
    nt.assert_true(np.any(af))
    nt.assert_equal(len(al), 3)
    nt.assert_equal(len(aea['avg_times']), 3)
    nt.assert_almost_equal(an.max(), 25.0)

    # test various Navgs
    ad, af, an, al, aea = hc.frf.timeavg_waterfall(d, 1, flags=f, rephase=True, lsts=lsts, freqs=fr, bl_vec=blv,
                                                   nsamples=n, extra_arrays=dict(times=t), verbose=False)

    nt.assert_equal(ad.shape, (60, 64))
    ad, af, an, al, aea = hc.frf.timeavg_waterfall(d, 60, flags=f, rephase=True, lsts=lsts, freqs=fr, bl_vec=blv,
                                                   nsamples=n, extra_arrays=dict(times=t), verbose=False)
    nt.assert_equal(ad.shape, (1, 64))

    # wrap lst
    ad2, af2, an2, al2, aea2 = hc.frf.timeavg_waterfall(d, 60, flags=f, rephase=True, lsts=lsts + 1.52917804, freqs=fr, bl_vec=blv,
                                                        nsamples=n, extra_arrays=dict(times=t), verbose=False)

    nt.assert_equal(ad.shape, (1, 64))
    nt.assert_true(np.isclose(ad, ad2).all())
    nt.assert_true(np.allclose(al, al2 - 1.52917804))


def test_fir_filtering():
    # convert a high-pass frprofile to an FIR filter
    frbins = np.linspace(-40e-3, 40e-3, 1024)
    frp = np.ones(1024)
    frp[512-9:512+10] = 0.0
    fir, tbins = hc.frf.frp_to_fir(frp, delta_bin=np.diff(frbins)[0])
    # confirm its purely real
    nt.assert_false(np.isclose(np.abs(fir.real), 0.0).any())
    nt.assert_true(np.isclose(np.abs(fir.imag), 0.0).all())

    # convert back
    _frp, _frbins = hc.frf.frp_to_fir(fir, delta_bin=np.diff(tbins)[0], undo=True)
    np.testing.assert_array_almost_equal(frp, _frp.real)
    np.testing.assert_array_almost_equal(np.diff(frbins), np.diff(_frbins))
    nt.assert_true(np.isclose(np.abs(_frp.imag), 0.0).all())

    # test noise averaging properties
    frp = np.zeros(1024)
    frp[512] = 1.0
    t_ratio = hc.frf.fr_tavg(frp)
    nt.assert_true(np.isclose(t_ratio, 1024).all())


class Test_FRFilter:

    def setUp(self):
        self.fname = os.path.join(DATA_PATH, "zen.2458042.12552.xx.HH.uvXA")
        self.F = hc.frf.FRFilter(self.fname, filetype='miriad')
        self.F.read()

    def test_timeavg_data(self):
        self.F.timeavg_data(self.F.data, self.F.times, self.F.lsts, 35, rephase=True, keys=[(24, 25, 'xx')])
        nt.assert_equal(self.F.Navg, 3)
        nt.assert_equal(len(self.F.avg_data), 1)
        nt.assert_equal(self.F.avg_data[(24, 25, 'xx')].shape, (20, 64))

        self.F.timeavg_data(self.F.data, self.F.times, self.F.lsts, 1e10, rephase=True, verbose=False, overwrite=False)
        nt.assert_equal(self.F.Navg, 60)
        nt.assert_equal(len(self.F.avg_data), 28)
        nt.assert_equal(self.F.avg_data[(24, 25, 'xx')].shape, (20, 64))
        nt.assert_equal(self.F.avg_data[(24, 37, 'xx')].shape, (1, 64))

        # exceptions
        nt.assert_raises(AssertionError, self.F.timeavg_data, self.F.data, self.F.times, self.F.lsts, 1.0)

    def test_filter_data(self):
        # construct high-pass filter
        frates = np.fft.fftshift(np.fft.fftfreq(self.F.Ntimes, self.F.dtime)) * 1e3
        w = np.ones((self.F.Ntimes, self.F.Nfreqs), dtype=np.float)
        w[np.abs(frates) < 20] = 0.0
        frps = hc.datacontainer.DataContainer(dict([(k, w) for k in self.F.data]))

        # make gaussian random noise
        bl = (24, 25, 'xx')
        window = 'blackmanharris'
        ec = 0
        np.random.seed(0)
        self.F.data[bl] = np.reshape(stats.norm.rvs(0, 1, self.F.Ntimes * self.F.Nfreqs) \
                                    + 1j * stats.norm.rvs(0, 1, self.F.Ntimes * self.F.Nfreqs), (self.F.Ntimes, self.F.Nfreqs))
        # fr filter noise
        self.F.filter_data(self.F.data, frps, overwrite=True, verbose=False, axis=0, keys=[bl])

        # check key continue w/ ridiculous edgecut
        self.F.filter_data(self.F.data, frps, overwrite=False, verbose=False, keys=[bl], edgecut_low=100, axis=0)

        # fft
        self.F.fft_data(data=self.F.data, assign='dfft', ax='freq', window=window, edgecut_low=ec, edgecut_hi=ec, overwrite=True)
        self.F.fft_data(data=self.F.filt_data, assign='rfft', ax='freq', window=window, edgecut_low=ec, edgecut_hi=ec, overwrite=True)

        # ensure drop in noise power is reflective of frf_nsamples
        dfft = np.mean(np.abs(self.F.dfft[bl]), axis=0)
        rfft = np.mean(np.abs(self.F.rfft[bl]), axis=0)
        r = np.mean(dfft / rfft)
        nt.assert_almost_equal(r, np.sqrt(np.mean(self.F.filt_nsamples[bl])), places=1)

    def test_write_data(self):
        self.F.timeavg_data(self.F.data, self.F.times, self.F.lsts, 35, rephase=False, verbose=False)
        self.F.write_data(self.F.avg_data, "./out.uv", filetype='miriad', overwrite=True,
                          add_to_history='testing', times=self.F.avg_times, lsts=self.F.avg_lsts)
        nt.assert_true(os.path.exists("./out.uv"))
        hd = hc.io.HERAData('./out.uv', filetype='miriad')
        hd.read()
        nt.assert_true('testing' in hd.history.replace('\n', '').replace(' ', ''))
        nt.assert_true('Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', ''))
        shutil.rmtree("./out.uv")

        nt.assert_raises(AssertionError, self.F.write_data, self.F.avg_data, "./out.uv", times=self.F.avg_times)
        nt.assert_raises(ValueError, self.F.write_data, self.F.data, "hi", filetype='foo')
