# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import pytest
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
from scipy import stats

from .. import datacontainer, io, frf
from ..data import DATA_PATH


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
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
    ad, af, an, al, aea = frf.timeavg_waterfall(d, 25, verbose=False)
    assert ad.shape == (3, 64)
    assert af.shape == (3, 64)
    assert an.shape == (3, 64)
    assert not np.any(af)
    assert np.allclose(an[1, 0], 25.0)
    assert np.allclose(an[2, 0], 10.0)

    # test rephase
    ad, af, an, al, aea = frf.timeavg_waterfall(d, 25, flags=f, rephase=True, lsts=lsts, freqs=fr, bl_vec=blv,
                                                nsamples=n, extra_arrays=dict(times=t), verbose=False)

    assert ad.shape == (3, 64)
    assert af.shape == (3, 64)
    assert an.shape == (3, 64)
    assert np.any(af)
    assert len(al) == 3
    assert len(aea['avg_times']) == 3
    assert np.allclose(an.max(), 25.0)

    # test various Navgs
    ad, af, an, al, aea = frf.timeavg_waterfall(d, 1, flags=f, rephase=True, lsts=lsts, freqs=fr, bl_vec=blv,
                                                nsamples=n, extra_arrays=dict(times=t), verbose=False)

    assert ad.shape == (60, 64)
    ad, af, an, al, aea = frf.timeavg_waterfall(d, 60, flags=f, rephase=True, lsts=lsts, freqs=fr, bl_vec=blv,
                                                nsamples=n, extra_arrays=dict(times=t), verbose=False)
    assert ad.shape == (1, 64)

    # wrap lst
    ad2, af2, an2, al2, aea2 = frf.timeavg_waterfall(d, 60, flags=f, rephase=True, lsts=lsts + 1.52917804, freqs=fr, bl_vec=blv,
                                                     nsamples=n, extra_arrays=dict(times=t), verbose=False)

    assert ad.shape == (1, 64)
    assert np.allclose(ad, ad2)
    assert np.allclose(al, al2 - 1.52917804)


def test_fir_filtering():
    # convert a high-pass frprofile to an FIR filter
    frbins = np.linspace(-40e-3, 40e-3, 1024)
    frp = np.ones(1024)
    frp[512 - 9:512 + 10] = 0.0
    fir, tbins = frf.frp_to_fir(frp, delta_bin=np.diff(frbins)[0])
    # confirm its purely real
    assert not np.any(np.isclose(np.abs(fir.real), 0.0))
    assert np.allclose(np.abs(fir.imag), 0.0)

    # convert back
    _frp, _frbins = frf.frp_to_fir(fir, delta_bin=np.diff(tbins)[0], undo=True)
    np.testing.assert_array_almost_equal(frp, _frp.real)
    np.testing.assert_array_almost_equal(np.diff(frbins), np.diff(_frbins))
    assert np.allclose(np.abs(_frp.imag), 0.0)

    # test noise averaging properties
    frp = np.zeros(1024)
    frp[512] = 1.0
    t_ratio = frf.fr_tavg(frp)
    assert np.allclose(t_ratio, 1024)


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
class Test_FRFilter(object):
    def setup_method(self):
        self.fname = os.path.join(DATA_PATH, "zen.2458042.12552.xx.HH.uvXA")
        self.F = frf.FRFilter(self.fname, filetype='miriad')
        self.F.read()

    def test_timeavg_data(self):
        # test basic time average
        self.F.timeavg_data(self.F.data, self.F.times, self.F.lsts, 35, rephase=True, keys=[(24, 25, 'ee')])
        assert self.F.Navg == 3
        assert len(self.F.avg_data) == 1
        assert self.F.avg_data[(24, 25, 'ee')].shape == (20, 64)

        # test full time average and overwrite
        self.F.timeavg_data(self.F.data, self.F.times, self.F.lsts, 1e10, rephase=True, verbose=False, overwrite=False)
        assert self.F.Navg == 60
        assert len(self.F.avg_data) == 28
        assert self.F.avg_data[(24, 25, 'ee')].shape == (20, 64)
        assert self.F.avg_data[(24, 37, 'ee')].shape == (1, 64)

        # test weight by nsample
        F = copy.deepcopy(self.F)
        k = (24, 25, 'ee')
        F.nsamples[k][:3] = 0.0
        F.timeavg_data(F.data, F.times, F.lsts, 35, nsamples=F.nsamples, keys=[k], overwrite=True,
                       wgt_by_nsample=True)
        assert np.all(np.isclose(F.avg_data[k][0], 0.0))  # assert data is zero b/c I zeroed nsample
        assert np.all(np.isclose(F.avg_nsamples[k][0], 0.0))  # assert avg_nsample is also zero
        assert np.all(np.isclose(F.avg_nsamples[k][1:], 3.0))  # assert non-zeroed nsample is 3

        # repeat without nsample wgt
        F.timeavg_data(F.data, F.times, F.lsts, 35, nsamples=F.nsamples, keys=[k], overwrite=True,
                       wgt_by_nsample=False)
        assert not np.any(np.isclose(F.avg_data[k][0, 5:-5], 0.0))  # assert non-edge data is now not zero
        assert np.all(np.isclose(F.avg_nsamples[k][0], 0.0))  # avg_nsample should still be zero

        # exceptions
        pytest.raises(AssertionError, self.F.timeavg_data, self.F.data, self.F.times, self.F.lsts, 1.0)

    def test_filter_data(self):
        # construct high-pass filter
        frates = np.fft.fftshift(np.fft.fftfreq(self.F.Ntimes, self.F.dtime)) * 1e3
        w = np.ones((self.F.Ntimes, self.F.Nfreqs), dtype=np.float)
        w[np.abs(frates) < 20] = 0.0
        frps = datacontainer.DataContainer(dict([(k, w) for k in self.F.data]))

        # make gaussian random noise
        bl = (24, 25, 'ee')
        window = 'blackmanharris'
        ec = 0
        np.random.seed(0)
        self.F.data[bl] = np.reshape(stats.norm.rvs(0, 1, self.F.Ntimes * self.F.Nfreqs)
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
        assert np.allclose(r, np.sqrt(np.mean(self.F.filt_nsamples[bl])), atol=1e-1)

    def test_write_data(self):
        self.F.timeavg_data(self.F.data, self.F.times, self.F.lsts, 35, rephase=False, verbose=False)
        self.F.write_data(self.F.avg_data, "./out.uv", filetype='miriad', overwrite=True,
                          add_to_history='testing', times=self.F.avg_times, lsts=self.F.avg_lsts)
        assert os.path.exists("./out.uv")
        hd = io.HERAData('./out.uv', filetype='miriad')
        hd.read()
        assert 'testing' in hd.history.replace('\n', '').replace(' ', '')
        assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
        shutil.rmtree("./out.uv")

        pytest.raises(AssertionError, self.F.write_data, self.F.avg_data, "./out.uv", times=self.F.avg_times)
        pytest.raises(ValueError, self.F.write_data, self.F.data, "hi", filetype='foo')
