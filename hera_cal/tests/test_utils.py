# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import pytest
import numpy as np
import sys
import os
import warnings
import shutil
import glob
from collections import OrderedDict as odict
import copy
from contextlib import contextmanager
from pyuvdata import UVData
from pyuvdata import UVCal
import pyuvdata.tests as uvtest
from sklearn import gaussian_process as gp
from ..redcal import filter_reds
from ..redcal import get_pos_reds

from hera_sim.noise import white_noise
from .. import utils, abscal, datacontainer, io, redcal
from ..calibrations import CAL_PATH
from ..data import DATA_PATH


class Test_Pol_Ops(object):
    def test_comply_pol(self):
        assert utils.comply_pol('XX') == 'xx'
        assert utils.comply_pol('Xx') == 'xx'
        assert utils.comply_pol('xx') == 'xx'
        assert utils.comply_pol('JXX') == 'Jxx'
        assert utils.comply_pol('Jxx') == 'Jxx'
        assert utils.comply_pol('EE') == 'ee'
        assert utils.comply_pol('Ee') == 'ee'
        assert utils.comply_pol('ee') == 'ee'
        assert utils.comply_pol('JEE') == 'Jee'
        assert utils.comply_pol('Jee') == 'Jee'
        assert utils.comply_pol('I') == 'pI'
        pytest.raises(KeyError, utils.comply_pol, 'stuff')
        pytest.raises(KeyError, utils.comply_pol, 'Jxe')

    def test_split_pol(self):
        assert utils.split_pol('xx') == ('Jxx', 'Jxx')
        assert utils.split_pol('xy') == ('Jxx', 'Jyy')
        assert utils.split_pol('XY') == ('Jxx', 'Jyy')

        assert utils.split_pol('ee') == ('Jee', 'Jee')
        assert utils.split_pol('en') == ('Jee', 'Jnn')
        assert utils.split_pol('EN') == ('Jee', 'Jnn')

        pytest.raises(KeyError, utils.split_pol, 'I')
        pytest.raises(KeyError, utils.split_pol, 'pV')

    def test_join_pol(self):
        assert utils.join_pol('Jxx', 'Jxx') == 'xx'
        assert utils.join_pol('Jxx', 'Jyy') == 'xy'
        assert utils.join_pol('Jee', 'Jee') == 'ee'
        assert utils.join_pol('Jee', 'Jnn') == 'en'

    def test_split_bl(self):
        assert utils.split_bl((1, 2, 'xx')) == ((1, 'Jxx'), (2, 'Jxx'))
        assert utils.split_bl((1, 2, 'xy')) == ((1, 'Jxx'), (2, 'Jyy'))
        assert utils.split_bl((1, 2, 'XX')) == ((1, 'Jxx'), (2, 'Jxx'))

        assert utils.split_bl((1, 2, 'ee')) == ((1, 'Jee'), (2, 'Jee'))
        assert utils.split_bl((1, 2, 'en')) == ((1, 'Jee'), (2, 'Jnn'))
        assert utils.split_bl((1, 2, 'EE')) == ((1, 'Jee'), (2, 'Jee'))

        pytest.raises(KeyError, utils.split_bl, (1, 2, 'pQ'))
        pytest.raises(KeyError, utils.split_bl, (1, 2, 'U'))

    def test_join_bl(self):
        assert utils.join_bl((1, 'Jxx'), (2, 'Jxx')) == (1, 2, 'xx')
        assert utils.join_bl((1, 'Jxx'), (2, 'Jyy')) == (1, 2, 'xy')

        assert utils.join_bl((1, 'Jee'), (2, 'Jee')) == (1, 2, 'ee')
        assert utils.join_bl((1, 'Jee'), (2, 'Jnn')) == (1, 2, 'en')

    def test_reverse_bl(self):
        assert utils.reverse_bl((1, 2, 'xx')) == (2, 1, 'xx')
        assert utils.reverse_bl((1, 2, 'xy')) == (2, 1, 'yx')
        assert utils.reverse_bl((1, 2, 'XX')) == (2, 1, 'xx')
        assert utils.reverse_bl((1, 2, 'ee')) == (2, 1, 'ee')
        assert utils.reverse_bl((1, 2, 'en')) == (2, 1, 'ne')
        assert utils.reverse_bl((1, 2, 'EE')) == (2, 1, 'ee')
        assert utils.reverse_bl((1, 2, 'pI')) == (2, 1, 'pI')
        assert utils.reverse_bl((1, 2)) == (2, 1)

    def test_comply_bl(self):
        assert utils.comply_bl((1, 2, 'xx')) == (1, 2, 'xx')
        assert utils.comply_bl((1, 2, 'xy')) == (1, 2, 'xy')
        assert utils.comply_bl((1, 2, 'XX')) == (1, 2, 'xx')

        assert utils.comply_bl((1, 2, 'ee')) == (1, 2, 'ee')
        assert utils.comply_bl((1, 2, 'en')) == (1, 2, 'en')
        assert utils.comply_bl((1, 2, 'EE')) == (1, 2, 'ee')

        assert utils.comply_bl((1, 2, 'pI')) == (1, 2, 'pI')

    def test_make_bl(self):
        assert utils.make_bl((1, 2, 'xx')) == (1, 2, 'xx')
        assert utils.make_bl((1, 2), 'xx') == (1, 2, 'xx')
        assert utils.make_bl((1, 2, 'xy')) == (1, 2, 'xy')
        assert utils.make_bl((1, 2), 'xy') == (1, 2, 'xy')
        assert utils.make_bl((1, 2, 'XX')) == (1, 2, 'xx')
        assert utils.make_bl((1, 2), 'XX') == (1, 2, 'xx')

        assert utils.make_bl((1, 2, 'ee')) == (1, 2, 'ee')
        assert utils.make_bl((1, 2), 'ee') == (1, 2, 'ee')
        assert utils.make_bl((1, 2, 'en')) == (1, 2, 'en')
        assert utils.make_bl((1, 2), 'en') == (1, 2, 'en')
        assert utils.make_bl((1, 2, 'EE')) == (1, 2, 'ee')
        assert utils.make_bl((1, 2), 'EE') == (1, 2, 'ee')

        assert utils.make_bl((1, 2, 'pI')) == (1, 2, 'pI')
        assert utils.make_bl((1, 2), 'pI') == (1, 2, 'pI')


class TestFilterBls(object):

    def test_filter_bls(self):
        bls = [(0, 1, 'ee'), (1, 2, 'ee'), (0, 2, 'ee'), (0, 1, 'ne')]
        antpos = {0: np.array([0, 0, 0]), 1: np.array([1, 0, 0]), 2: np.array([2, 0, 0])}

        assert set(utils.filter_bls(bls, ants=[0, 1])) == set([(0, 1, 'ee'), (0, 1, 'ne')])
        assert set(utils.filter_bls(bls, ants=[(0, 'Jee'), (1, 'Jee')])) == set([(0, 1, 'ee')])

        assert set(utils.filter_bls(bls, ex_ants=[0])) == set([(1, 2, 'ee')])
        assert set(utils.filter_bls(bls, ex_ants=[(0, 'Jee')])) == set([(1, 2, 'ee'), (0, 1, 'ne')])

        assert set(utils.filter_bls(bls, pols=['ee'])) == set([(0, 1, 'ee'), (1, 2, 'ee'), (0, 2, 'ee')])

        assert set(utils.filter_bls(bls, antpos=antpos, min_bl_cut=1.5)) == set([(0, 2, 'ee')])
        assert set(utils.filter_bls(bls, antpos=antpos, max_bl_cut=1.5)) == set([(0, 1, 'ee'), (1, 2, 'ee'), (0, 1, 'ne')])
        with pytest.raises(AssertionError):
            utils.filter_bls(bls, min_bl_cut=1.5)


class TestHistoryVersion():

    def test_history_string(self):
        hs = utils.history_string()
        assert 'function test_history_string() in' in hs
        assert 'test_utils.py' in hs
        hs = utils.history_string('stuff')
        assert 'stuff' in hs
        assert 'Notes' in hs


class TestFftDly(object):

    def setup_method(self):
        np.random.seed(0)
        self.freqs = np.linspace(.1, .2, 1024)

    def test_ideal(self):
        true_dlys = np.random.uniform(-200, 200, size=60)
        true_dlys.shape = (60, 1)
        data = np.exp(2j * np.pi * self.freqs.reshape((1, -1)) * true_dlys)
        df = np.median(np.diff(self.freqs))
        dlys, offs = utils.fft_dly(data, df, f0=self.freqs[0])
        assert np.median(np.abs(dlys - true_dlys)) < 1e-5  # median accuracy of 10 fs
        np.testing.assert_almost_equal(offs, 0, decimal=4)
        dlys, offs = utils.fft_dly(data, df, medfilt=True, f0=self.freqs[0])
        assert np.median(np.abs(dlys - true_dlys)) < 1e-2  # median accuracy of 10 ps

    def test_ideal_offset(self):
        true_dlys = np.random.uniform(-200, 200, size=60)
        true_dlys.shape = (60, 1)
        data = np.exp(2j * np.pi * self.freqs * true_dlys + 1j * 0.123)
        df = np.median(np.diff(self.freqs))
        dlys, offs = utils.fft_dly(data, df, f0=self.freqs[0])
        assert np.median(np.abs(dlys - true_dlys)) < 1e-5  # median accuracy of 10 fs
        np.testing.assert_almost_equal(offs, 0.123, decimal=4)
        mdl = np.exp(2j * np.pi * self.freqs * dlys + 1j * offs)
        np.testing.assert_almost_equal(np.angle(data * mdl.conj()), 0, decimal=5)
        dlys, offs = utils.fft_dly(data, df, edge_cut=100, f0=self.freqs[0])
        assert np.median(np.abs(dlys - true_dlys)) < 1e-4  # median accuracy of 100 fs
        np.testing.assert_almost_equal(offs, 0.123, decimal=4)
        dlys, offs = utils.fft_dly(data, df, medfilt=True, f0=self.freqs[0])
        assert np.median(np.abs(dlys - true_dlys)) < 1e-2  # median accuracy of 10 ps
        np.testing.assert_almost_equal(offs, 0.123, decimal=1)

    def test_noisy(self):
        true_dlys = np.random.uniform(-200, 200, size=60)
        true_dlys.shape = (60, 1)
        data = np.exp(2j * np.pi * self.freqs.reshape((1, -1)) * true_dlys) + 5 * white_noise((60, 1024))
        df = np.median(np.diff(self.freqs))
        dlys, offs = utils.fft_dly(data, df)
        assert np.median(np.abs(dlys - true_dlys)) < 1  # median accuracy of 1 ns
        dlys, offs = utils.fft_dly(data, df, medfilt=True)
        assert np.median(np.abs(dlys - true_dlys)) < 1  # median accuracy of 1 ns

    def test_rfi(self):
        true_dlys = np.random.uniform(-200, 200, size=60)
        true_dlys.shape = (60, 1)
        data = np.exp(2j * np.pi * self.freqs.reshape((1, -1)) * true_dlys)
        data[:, ::16] = 1000.
        df = np.median(np.diff(self.freqs))
        dlys, offs = utils.fft_dly(data, df, medfilt=True)
        assert np.median(np.abs(dlys - true_dlys)) < 1e-2  # median accuracy of 10 ps

    def test_nan(self):
        true_dlys = np.random.uniform(-200, 200, size=60)
        true_dlys.shape = (60, 1)
        data = np.exp(2j * np.pi * self.freqs.reshape((1, -1)) * true_dlys)
        data[:, ::16] = np.nan
        df = np.median(np.diff(self.freqs))
        dlys, offs = utils.fft_dly(data, df)
        assert np.median(np.abs(dlys - true_dlys)) < 1e-3  # median accuracy of 1 ps

    @pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
    def test_realistic(self):
        # load into pyuvdata object
        data_fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        model_fname = os.path.join(DATA_PATH, "zen.2458042.12552.xx.HH.uvXA")
        # make custom gain keys
        d, fl, antpos, a, freqs, t, l, p = io.load_vis(data_fname, return_meta=True, pick_data_ants=False)
        freqs /= 1e9  # in GHz
        # test basic execution
        k1 = (24, 25, 'ee')
        k2 = (37, 38, 'ee')
        flat_phs = d[k1] * d[k2].conj()
        df = np.median(np.diff(freqs))
        # basic execution
        dlys, offs = utils.fft_dly(flat_phs, df, medfilt=True, f0=freqs[0])  # dlys in ns
        assert dlys.shape == (60, 1)
        assert np.all(np.abs(dlys) < 1)  # all delays near zero
        true_dlys = np.random.uniform(-20, 20, size=60)
        true_dlys.shape = (60, 1)
        phs = np.exp(2j * np.pi * freqs.reshape((1, -1)) * (true_dlys + dlys))
        dlys, offs = utils.fft_dly(flat_phs * phs, df, medfilt=True, f0=freqs[0])
        assert np.median(np.abs(dlys - true_dlys)) < 2  # median accuracy better than 2 ns

    def test_error(self):
        true_dlys = np.random.uniform(-200, 200, size=60)
        true_dlys.shape = (60, 1)
        data = np.exp(2j * np.pi * self.freqs.reshape((1, -1)) * true_dlys)
        pytest.raises(ValueError, utils.interp_peak, np.fft.fft(data), method='blah')

    def test_interp_peak(self):
        # code testing is done in TestFftDly, so just check optional parameters here
        # check reject_edges
        x = np.linspace(0, 10, 101)
        y = (x - 5)**2 + np.isclose(x, 5.0).astype(float)
        # check peak is zeroth bin
        inds, bs, peaks, p = utils.interp_peak(y[None, :], method='quadratic', reject_edges=False)
        assert inds[0] == 0
        # check peak is middle bin with reject_edges
        inds, bs, peaks, p = utils.interp_peak(y[None, :], method='quadratic', reject_edges=True)
        assert inds[0] == np.argmax(y - (x - 5)**2)
        # check peak is last bin even w/ reject_edges
        y = x * 1.0
        inds, bs, peaks, p = utils.interp_peak(y[None, :], method='quadratic', reject_edges=True)
        assert inds[0] == np.argmax(y)
        # check peak is zero bin even w/ reject_edges
        y = np.abs(-x * 1.0 + 10)
        inds, bs, peaks, p = utils.interp_peak(y[None, :], method='quadratic', reject_edges=True)
        assert inds[0] == np.argmax(y)


class TestAAFromUV(object):
    def setup_method(self):
        # define test file that is compatible with get_aa_from_uv
        self.test_file = "zen.2457999.76839.xx.HH.uvA"

    def test_get_aa_from_uv(self):
        fn = os.path.join(DATA_PATH, self.test_file)
        uvd = UVData()
        uvd.read_miriad(fn)
        aa = utils.get_aa_from_uv(uvd)
        # like miriad, aipy will pad the aa with non-existent antennas,
        #   because there is no concept of antenna names
        assert len(aa) == 88


class TestAA(object):
    def setup_method(self):
        # define test file that is compatible with get_aa_from_uv
        self.test_file = "zen.2457999.76839.xx.HH.uvA"

    def test_aa_get_params(self):
        # generate aa from file
        fn = os.path.join(DATA_PATH, self.test_file)
        uvd = UVData()
        uvd.read_miriad(fn)
        aa = utils.get_aa_from_uv(uvd)

        # change one antenna position, and read it back in to check it's the same
        antpos = {'x': 0., 'y': 1., 'z': 2.}
        params = aa.get_params()
        for key in antpos.keys():
            params['0'][key] = antpos[key]
        aa.set_params(params)
        new_params = aa.get_params()
        new_top = [new_params['0'][key] for key in antpos.keys()]
        old_top = [antpos[key] for key in antpos.keys()]
        assert np.allclose(old_top, new_top)


def test_JD2LST():
    # test float execution
    jd = 2458042.
    assert np.allclose(utils.JD2LST(jd, longitude=21.), 3.930652307266274)
    # test array execution
    jd = np.arange(2458042, 2458046.1, .5)
    lst = utils.JD2LST(jd, longitude=21.)
    assert len(lst) == 9
    assert np.allclose(lst[3], 0.81486300218170715)


def test_LST2JD():
    # test basic execution
    lst = np.pi
    jd = utils.LST2JD(lst, start_jd=2458042)
    assert np.allclose(jd, 2458042.8708433118)
    # test array execution
    lst = np.arange(np.pi, np.pi + 1.1, 0.2)
    jd = utils.LST2JD(lst, start_jd=2458042)
    assert len(jd) == 6
    assert np.allclose(jd[3], 2458042.9660755517)
    # test allow_other_jd = True
    lsts = np.arange(-np.pi / 4, 7 * np.pi / 2, .1)
    jds = utils.LST2JD(lsts, start_jd=2458042, allow_other_jd=True)
    for jd in jds:
        assert int(np.floor(jd)) in [2458041, 2458042, 2458043, 2458044]
    assert not np.all([np.floor(jd) == 2458042 for jd in jds])
    # test allow_other_jd = False
    lsts = np.arange(-np.pi / 4, 7 * np.pi / 2, .1)
    jds = utils.LST2JD(lsts, start_jd=2458042, allow_other_jd=False)
    for jd in jds:
        assert int(np.floor(jd)) == 2458042
    # test branch cut
    lsts = np.arange(0, 2 * np.pi, .1)
    jds = utils.LST2JD(lsts, start_jd=2458042, allow_other_jd=True, lst_branch_cut=0.0)
    assert jds[0] < jds[-1]
    jds = utils.LST2JD(lsts, start_jd=2458042, allow_other_jd=True, lst_branch_cut=1)
    assert np.min(jds[lsts < 1]) > np.max(jds[lsts > 1])
    # test that lst 0 falls on correct day
    for jd in range(2458042, 2458042 + 365):
        assert np.floor(utils.LST2JD(0, start_jd=jd, allow_other_jd=True)) == jd
    # test that lst of branch cut falls on correct day
    for jd in range(2458042, 2458042 + 365):
        assert np.floor(utils.LST2JD(1, start_jd=jd, allow_other_jd=True, lst_branch_cut=1)) == jd
    # test convert back and forth for a range of days to 1e-8 radians precision
    for start_jd in range(2458042, 2458042 + 365):
        lsts = np.arange(0, 2 * np.pi, .1)
        lsts2 = utils.JD2LST(utils.LST2JD(lsts, start_jd=jd, allow_other_jd=True, lst_branch_cut=1))
        is_close = (np.abs(lsts - lsts2) < 1e-8) | (np.abs(lsts - lsts2 + 2 * np.pi) < 1e-8) | (np.abs(lsts - lsts2 - 2 * np.pi) < 1e-8)
        assert np.all(is_close)


def test_JD2RA():
    # test basic execution
    jd = 2458042.5
    ra = utils.JD2RA(jd)
    assert np.allclose(ra, 46.130897831277629)
    # test array
    jd = np.arange(2458042, 2458043.01, .2)
    ra = utils.JD2RA(jd)
    assert len(ra) == 6
    assert np.allclose(ra[3], 82.229459674026003)
    # test exception
    pytest.raises(ValueError, utils.JD2RA, jd, epoch='foo')
    # test J2000 epoch
    ra = utils.JD2RA(jd, epoch='J2000')
    assert np.allclose(ra[0], 225.37671446615548)


def test_combine_calfits():
    test_file1 = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.abs.calfits')
    test_file2 = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.dly.calfits')
    # test basic execution
    if os.path.exists('ex.calfits'):
        os.remove('ex.calfits')
    utils.combine_calfits([test_file1, test_file2], 'ex.calfits', outdir='./', overwrite=True, broadcast_flags=True)
    # test it exists
    assert os.path.exists('ex.calfits')
    # test antenna number
    uvc = UVCal()
    uvc.read_calfits('ex.calfits')
    assert len(uvc.antenna_numbers) == 7
    # test time number
    assert uvc.Ntimes == 60
    # test gain value got properly multiplied
    uvc_dly = UVCal()
    uvc_dly.read_calfits(test_file1)
    uvc_abs = UVCal()
    uvc_abs.read_calfits(test_file2)
    assert np.allclose(uvc_dly.gain_array[0, 0, 10, 10, 0] * uvc_abs.gain_array[0, 0, 10, 10, 0], uvc.gain_array[0, 0, 10, 10, 0])
    if os.path.exists('ex.calfits'):
        os.remove('ex.calfits')
    utils.combine_calfits([test_file1, test_file2], 'ex.calfits', outdir='./', overwrite=True, broadcast_flags=False)
    assert os.path.exists('ex.calfits')
    if os.path.exists('ex.calfits'):
        os.remove('ex.calfits')


def test_lst_rephase():
    # load point source sim w/ array at latitude = 0
    fname = os.path.join(DATA_PATH, "PAPER_point_source_sim.uv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (data, flags, antpos, ants, freqs, times, lsts, pols) = io.load_vis(fname, return_meta=True)
    data_drift = copy.deepcopy(data)
    transit_integration = 50

    # get integration time in LST, baseline dict
    dlst = np.median(np.diff(lsts))
    bls = odict([(k, antpos[k[0]] - antpos[k[1]]) for k in data.keys()])

    # basic test: single dlst for all integrations
    utils.lst_rephase(data, bls, freqs, dlst, lat=0.0)
    # get phase error on shortest EW baseline
    k = (0, 1, 'ee')
    # check error at transit
    phs_err = np.angle(data[k][transit_integration, 4] / data_drift[k][transit_integration + 1, 4])
    assert np.isclose(phs_err, 0, atol=1e-7)
    # check error across file
    phs_err = np.angle(data[k][:-1, 4] / data_drift[k][1:, 4])
    assert np.abs(phs_err).max() < 1e-4

    # multiple phase term test: dlst per integration
    dlst = np.array([np.median(np.diff(lsts))] * data[k].shape[0])
    data = copy.deepcopy(data_drift)
    utils.lst_rephase(data, bls, freqs, dlst, lat=0.0)
    # check error at transit
    phs_err = np.angle(data[k][transit_integration, 4] / data_drift[k][transit_integration + 1, 4])
    assert np.isclose(phs_err, 0, atol=1e-7)
    # check err across file
    phs_err = np.angle(data[k][:-1, 4] / data_drift[k][1:, 4])
    assert np.abs(phs_err).max() < 1e-4

    # phase all integrations to a single integration
    dlst = lsts[50] - lsts
    data = copy.deepcopy(data_drift)
    utils.lst_rephase(data, bls, freqs, dlst, lat=0.0)
    # check error at transit
    phs_err = np.angle(data[k][transit_integration, 4] / data_drift[k][transit_integration, 4])
    assert np.isclose(phs_err, 0, atol=1e-7)
    # check error across file
    phs_err = np.angle(data[k][:, 4] / data_drift[k][50, 4])
    assert np.abs(phs_err).max() < 1e-4

    # test operation on array
    k = (0, 1, 'ee')
    d = data_drift[k].copy()
    d_phs = utils.lst_rephase(d, bls[k], freqs, dlst, lat=0.0, array=True)
    assert np.allclose(np.abs(np.angle(d_phs[50] / data[k][50])).max(), 0.0)


def test_chisq():
    # test basic case
    data = datacontainer.DataContainer({(0, 1, 'xx'): np.ones((5, 10), dtype=complex)})
    model = datacontainer.DataContainer({(0, 1, 'xx'): 3 * np.ones((5, 10), dtype=complex)})
    chisq, nObs, chisq_per_ant, nObs_per_ant = utils.chisq(data, model)
    assert chisq.shape == (5, 10)
    assert nObs.shape == (5, 10)
    assert chisq.dtype == float
    assert nObs.dtype == int
    np.testing.assert_array_equal(chisq, 4.0)
    np.testing.assert_array_equal(nObs, 1)
    np.testing.assert_array_equal(chisq_per_ant[0, 'Jxx'], 4.0)
    np.testing.assert_array_equal(chisq_per_ant[1, 'Jxx'], 4.0)
    np.testing.assert_array_equal(nObs_per_ant[0, 'Jxx'], 1)
    np.testing.assert_array_equal(nObs_per_ant[1, 'Jxx'], 1)

    # test with reds
    data = datacontainer.DataContainer({(0, 1, 'xx'): np.ones((5, 10), dtype=complex),
                                        (1, 2, 'xx'): np.ones((5, 10), dtype=complex)})
    model = datacontainer.DataContainer({(0, 1, 'xx'): 2 * np.ones((5, 10), dtype=complex)})
    chisq, nObs, chisq_per_ant, nObs_per_ant = utils.chisq(data, model, reds=[[(0, 1, 'xx'), (1, 2, 'xx')]])
    np.testing.assert_array_equal(chisq, 2.0)
    np.testing.assert_array_equal(nObs, 2)
    assert (1, 2, 'xx') not in model

    # test with weights
    data = datacontainer.DataContainer({(0, 1, 'xx'): np.ones((5, 10), dtype=complex)})
    model = datacontainer.DataContainer({(0, 1, 'xx'): 2 * np.ones((5, 10), dtype=complex)})
    data_wgts = datacontainer.DataContainer({(0, 1, 'xx'): np.zeros((5, 10), dtype=float)})
    data_wgts[(0, 1, 'xx')][:, 0] = 1.0
    chisq, nObs, chisq_per_ant, nObs_per_ant = utils.chisq(data, model, data_wgts)
    assert np.sum(chisq) == 5.0
    assert np.sum(nObs) == 5

    # test update case
    data = datacontainer.DataContainer({(0, 1, 'xx'): np.ones((5, 10), dtype=complex)})
    model = datacontainer.DataContainer({(0, 1, 'xx'): 2 * np.ones((5, 10), dtype=complex)})
    data_wgts = datacontainer.DataContainer({(0, 1, 'xx'): np.ones((5, 10), dtype=float)})
    chisq, nObs, chisq_per_ant, nObs_per_ant = utils.chisq(data, model, data_wgts)
    chisq, nObs, chisq_per_ant, nObs_per_ant = utils.chisq(data, model, data_wgts, chisq=chisq, nObs=nObs,
                                                           chisq_per_ant=chisq_per_ant, nObs_per_ant=nObs_per_ant)
    np.testing.assert_array_equal(chisq, 2.0)
    np.testing.assert_array_equal(nObs, 2)
    np.testing.assert_array_equal(chisq_per_ant[0, 'Jxx'], 2.0)
    np.testing.assert_array_equal(chisq_per_ant[1, 'Jxx'], 2.0)
    np.testing.assert_array_equal(nObs_per_ant[0, 'Jxx'], 2)
    np.testing.assert_array_equal(nObs_per_ant[1, 'Jxx'], 2)

    # test with gains and gain flags
    gains = {(0, 'Jxx'): .5**.5 * np.ones((5, 10), dtype=complex),
             (1, 'Jxx'): .5**.5 * np.ones((5, 10), dtype=complex)}
    gain_flags = {(0, 'Jxx'): np.zeros((5, 10), dtype=bool),
                  (1, 'Jxx'): np.zeros((5, 10), dtype=bool)}
    gain_flags[0, 'Jxx'][:, 0] = True
    chisq, nObs, chisq_per_ant, nObs_per_ant = utils.chisq(data, model, data_wgts, gains=gains, gain_flags=gain_flags)
    assert np.isclose(np.sum(chisq), 0.0)
    assert np.sum(nObs) == 45
    assert np.isclose(np.sum(chisq_per_ant[0, 'Jxx']), 0.0)
    assert np.isclose(np.sum(chisq_per_ant[1, 'Jxx']), 0.0)
    assert np.sum(nObs_per_ant[1, 'Jxx']) == 45
    assert np.sum(nObs_per_ant[1, 'Jxx']) == 45

    # test errors
    pytest.raises(ValueError, utils.chisq, data, model, data_wgts, chisq=chisq)
    pytest.raises(ValueError, utils.chisq, data, model, data_wgts, nObs=nObs)
    pytest.raises(AssertionError, utils.chisq, data, model, data_wgts, split_by_antpol=True, chisq={'Jxx': 1}, nObs={})
    pytest.raises(AssertionError, utils.chisq, data, model, data_wgts, split_by_antpol=True, nObs={'Jxx': 1}, chisq={})
    pytest.raises(ValueError, utils.chisq, data, model, data_wgts, chisq_per_ant=chisq_per_ant)
    pytest.raises(ValueError, utils.chisq, data, model, data_wgts, nObs_per_ant=nObs_per_ant)
    pytest.raises(AssertionError, utils.chisq, data, model, data_wgts, chisq_per_ant=chisq_per_ant, nObs_per_ant={})
    pytest.raises(AssertionError, utils.chisq, data, model, data_wgts, chisq_per_ant={}, nObs_per_ant=nObs_per_ant)
    pytest.raises(KeyError, utils.chisq, data, model, data_wgts, gains={(0, 'x'): np.ones((5, 10), dtype=complex)})
    data_wgts = datacontainer.DataContainer({(0, 1, 'xx'): 1.0j * np.ones((5, 10), dtype=float)})
    pytest.raises(AssertionError, utils.chisq, data, model, data_wgts)

    # test by_pol option
    data = datacontainer.DataContainer({(0, 1, 'xx'): np.ones((5, 10), dtype=complex)})
    model = datacontainer.DataContainer({(0, 1, 'xx'): 2 * np.ones((5, 10), dtype=complex)})
    data_wgts = datacontainer.DataContainer({(0, 1, 'xx'): np.ones((5, 10), dtype=float)})
    chisq, nObs, chisq_per_ant, nObs_per_ant = utils.chisq(data, model, data_wgts, split_by_antpol=True)
    assert 'Jxx' in chisq
    assert 'Jxx' in nObs
    assert chisq['Jxx'].shape == (5, 10)
    assert nObs['Jxx'].shape == (5, 10)
    np.testing.assert_array_equal(chisq['Jxx'], 1.0)
    np.testing.assert_array_equal(nObs['Jxx'], 1)
    data = datacontainer.DataContainer({(0, 1, 'xy'): np.ones((5, 10), dtype=complex)})
    model = datacontainer.DataContainer({(0, 1, 'xy'): 2 * np.ones((5, 10), dtype=complex)})
    data_wgts = datacontainer.DataContainer({(0, 1, 'xy'): np.ones((5, 10), dtype=float)})
    chisq, nObs, chisq_per_ant, nObs_per_ant = utils.chisq(data, model, data_wgts, split_by_antpol=True)
    assert len(chisq) == 0
    assert len(nObs) == 0
    assert len(chisq_per_ant) == 0
    assert len(chisq_per_ant) == 0


def test_per_antenna_modified_z_scores():
    metric = {(0, 'Jnn'): 1, (50, 'Jnn'): 0, (2, 'Jnn'): 2,
              (2, 'Jee'): 2000, (0, 'Jee'): -300}
    zscores = utils.per_antenna_modified_z_scores(metric)
    np.testing.assert_almost_equal(zscores[0, 'Jnn'], 0, 10)
    np.testing.assert_almost_equal(zscores[50, 'Jnn'], -0.6745, 10)
    np.testing.assert_almost_equal(zscores[2, 'Jnn'], 0.6745, 10)


def test_gp_interp1d():
    # load data
    dfiles = glob.glob(os.path.join(DATA_PATH, "zen.2458043.4*.xx.HH.XRAA.uvh5"))
    uvd = UVData()
    uvd.read(dfiles, bls=[(37, 39)])
    times = np.unique(uvd.time_array) * 24 * 60
    times -= times.min()
    y = uvd.get_data(37, 39, 'ee')
    f = uvd.get_flags(37, 39, 'ee')

    # interpolate
    yint = utils.gp_interp1d(times, y, length_scale=5.0, Nmirror=20, flags=f, nl=1e-10)

    # check residual
    # plt.imshow(np.real(y-yint),aspect='auto',vmin=-10,vmax=10)
    assert np.std((y - yint)[~f]) < 10

    # now test without feeding flags and see that fit got worse
    yint2 = utils.gp_interp1d(times, y, length_scale=5.0, Nmirror=20, nl=1e-10)
    assert np.std((y - yint2)[~f]) > 10

    # try with custom x_eval: test it is same as starting yint at same times
    yint3 = utils.gp_interp1d(times, y, x_eval=times, length_scale=5.0, Nmirror=20, flags=f, nl=1e-10)
    assert np.all(np.isclose(yint[:], yint3))

    # assert custom kernel with slightly different params gives different results
    kernel = 1 * gp.kernels.RBF(4.0) + gp.kernels.WhiteKernel(1e-10)
    yint4 = utils.gp_interp1d(times, y, kernel=kernel, Nmirror=20, flags=f)
    assert not np.all(np.isclose(yint, yint4))

    # test thinning
    yint_0thin = utils.gp_interp1d(times, y, length_scale=5.0, flags=f, nl=1e-10, xthin=None)
    yint_1thin = utils.gp_interp1d(times, y, length_scale=5.0, flags=f, nl=1e-10, xthin=1)
    yint_2thin = utils.gp_interp1d(times, y, length_scale=5.0, flags=f, nl=1e-10, xthin=2)

    # check 0thin and 1thin are equivalent
    assert np.all(np.isclose(np.abs(yint_0thin - yint_1thin), 0.0))

    # check 1thin and 2thin are *reasonably* close to within noise of original data
    # plt.plot(np.abs(y[:, 10]));plt.plot(np.abs(yint_1thin[:, 10]));plt.plot(np.abs(yint_2thin[:, 10]))
    nstd = np.std(y - yint_0thin, axis=0)  # residual noise after subtraction with unthinned model
    rstd = np.std(yint_1thin - yint_2thin, axis=0)  # error flucturations between 1 and 2 thin models
    assert np.nanmedian(nstd / rstd) > 2.0  # assert model error is on average less then half noise


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
def test_red_average():
    # setup
    hd = io.HERAData(os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5"))
    data, flags, nsamples = hd.read()
    antpos, ants = hd.get_ENU_antpos(pick_data_ants=True)
    antposd = dict(zip(ants, antpos))
    reds = redcal.get_pos_reds(antposd)
    blkey = reds[0][0] + ('ee',)

    # test redundant average
    hda = utils.red_average(hd, reds, inplace=False)

    # assert type and averaging is correct
    assert isinstance(hda, io.HERAData)
    assert hda.Nbls == len(reds)
    nsamp = np.sum([hd.get_nsamples(bl + ('ee',)) * ~hd.get_flags(bl + ('ee',)) for bl in reds[0]], axis=0)
    assert np.isclose(hda.get_nsamples(blkey), nsamp).all()
    d = np.asarray([hd.get_data(bl + ('ee',)) for bl in reds[0]])
    w = np.asarray([(~hd.get_flags(bl + ('ee',))).astype(float) for bl in reds[0]])
    davg = np.sum(d * w, axis=0) / np.sum(w, axis=0).clip(1e-10, np.inf)
    # set flagged data to 1.0+0.0j
    flagged_f = np.all(w == 0, axis=0)
    davg[flagged_f] = 1.0
    assert np.isclose(hda.get_data(blkey), davg).all()

    # try where all weights are unity but user provided flags are preserved.
    user_weights = {}
    for bl in flags:
        user_weights[bl] = np.ones_like(flags[bl], dtype=float)
        if np.all(flags[bl]):
            user_weights[bl][:] = 0.

    hda = utils.red_average(hd, reds, inplace=False, propagate_flags=True)
    w = np.asarray([user_weights[(bl + ('ee',))] for bl in reds[0]])
    f = np.asarray([(~hd.get_flags(bl + ('ee',))).astype(float) * user_weights[(bl + ('ee',))] for bl in reds[0]])
    favg = np.isclose(np.sum(f, axis=0), 0.0)
    assert np.isclose(hda.get_flags(blkey), favg).all()

    # try with DataContainer
    data_avg, flag_avg, _ = utils.red_average(data, reds, flags=flags, inplace=False)
    assert isinstance(data_avg, (datacontainer.DataContainer, dict))
    assert len(data_avg) == len(reds)
    assert np.isclose(data_avg[blkey], davg).all()
    assert np.isclose(flag_avg[blkey], hda.get_flags(blkey)).all()
    # try with no flags
    data_avg2, _, _ = utils.red_average(data, reds, inplace=False)
    assert np.isclose(data_avg2[blkey], np.mean([data[bl + ('ee',)] for bl in reds[0]], axis=0)).all()

    # test inplace
    _hda = copy.deepcopy(hd)
    utils.red_average(_hda, inplace=True)
    assert hda == _hda

    # try with DataContainer
    data2, flags2 = copy.deepcopy(data), copy.deepcopy(flags)
    utils.red_average(data2, flags=flags2, inplace=True)
    assert np.isclose(data2[blkey], data_avg[blkey]).all()
    assert np.isclose(flags2[blkey], flag_avg[blkey]).all()

    # try automatic red calc
    hda2 = utils.red_average(hd, inplace=False)
    assert hda == hda2
    data_avg3, _, _ = utils.red_average(data, flags=flags, inplace=False)
    assert np.isclose(data_avg[blkey], data_avg3[blkey]).all()

    # try with large tolerance
    hda3 = utils.red_average(hd, bl_tol=1000, inplace=False)
    assert hda3.Nbls == 1

    # now try with modified nsamples
    _hd = copy.deepcopy(hd)
    _hd.nsample_array[:] = 0.0
    _hd.nsample_array[hd.antpair2ind(reds[0][0] + ('ee',))] = 1.0
    _hd.flag_array[:] = False
    hda3 = utils.red_average(_hd, inplace=False)
    # averaged data should equal original, unaveraged data due to weighting
    assert np.isclose(hda3.get_data(reds[0][0] + ('ee',)), hd.get_data(reds[0][0] + ('ee',))).all()

    # try with manual weights
    wgts = datacontainer.DataContainer({k: _hd.get_nsamples(k) for k in _hd.get_antpairpols()})
    hda4 = utils.red_average(_hd, wgts=wgts, inplace=False)
    assert hda3 == hda4

    # exceptions
    _data = copy.deepcopy(data)
    _data.antpos = None
    pytest.raises(ValueError, utils.red_average, _data)
    pytest.raises(ValueError, utils.red_average, 'foo')


def test_red_average_conjugate_baseline_case():
    # this test covers the case where there baselines that are redundant
    # sans conjugation.
    to_test = []
    for filenum in range(3):
        # zeroth file is a raw correlator '.sum.uvh5' file.
        # first file is a file with flagged antennas removed and chunked.
        # second file is after foreground / xtalk filtering / time averaging.
        input_file = os.path.join(DATA_PATH, f'red_averaging_conjugate_tester_{filenum}.uvh5')
        hd = io.HERAData(input_file)
        d, f, n = hd.read()
        reds = get_pos_reds(hd.antpos)
        reds = [[bl[:2] for bl in grp if bl in hd.get_antpairs() or bl[::-1] in hd.get_antpairs()] for grp in reds]
        reds = [grp for grp in reds if len(grp) > 0]
        red_grp_keys = [grp[0] for grp in reds]
        # test conjugate sets with fed_datacontainers = False
        hd_red_average = utils.red_average(hd, reds=reds, inplace=False,
                                           red_bl_keys=red_grp_keys)
        # check without specifying reds or red_bl_keys.
        hd_red_average_1 = utils.red_average(hd, inplace=False)
        # test with fed_datacontainers = True
        dr, fr, nr = utils.red_average(data=d, flags=f, nsamples=n, inplace=False)
        d0, f0, n0 = hd_red_average.build_datacontainers()
        d1, f1, n1 = hd_red_average_1.build_datacontainers()
        # make sure all three cases give same nsamples, flags, and data.
        for k in d0:
            assert np.allclose(d0[k], d1[k])
            assert np.allclose(f0[k], f1[k])
            assert np.allclose(n0[k], n1[k])
            assert np.allclose(n0[k], nr[k])
            assert np.allclose(f0[k], fr[k])
            assert np.allclose(d0[k], dr[k])


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_gain_relative_difference():
    # setup
    old_gains = {(0, 'Jxx'): np.ones((10, 10), dtype=complex),
                 (1, 'Jxx'): np.ones((10, 10), dtype=complex)}
    new_gains = {(0, 'Jxx'): 2. * np.ones((10, 10), dtype=complex),
                 (1, 'Jxx'): 4. * np.ones((10, 10), dtype=complex)}
    flags = {(0, 'Jxx'): np.zeros((10, 10), dtype=bool),
             (1, 'Jxx'): np.zeros((10, 10), dtype=bool)}
    flags[(0, 'Jxx')][3, 4:6] = True
    flags[(1, 'Jxx')][3:5, 4] = True

    # standard test with flags
    relative_diff, avg_relative_diff = utils.gain_relative_difference(old_gains, new_gains, flags)
    assert relative_diff[0, 'Jxx'][0, 0] == 1.
    assert relative_diff[1, 'Jxx'][0, 0] == 3.
    assert avg_relative_diff['Jxx'][0, 0] == 2.
    assert avg_relative_diff['Jxx'][3, 4] == 0.  # both flagged
    assert avg_relative_diff['Jxx'][3, 5] == 3.  # ant 0 flagged
    assert avg_relative_diff['Jxx'][4, 3] == 2.  # ant 1 flagged

    # test different denominator
    relative_diff, avg_relative_diff = utils.gain_relative_difference(old_gains, new_gains, flags, denom=new_gains)
    assert relative_diff[0, 'Jxx'][0, 0] == .5
    assert relative_diff[1, 'Jxx'][0, 0] == 3. / 4.
    assert avg_relative_diff['Jxx'][0, 0] == 5. / 8.


def test_echo(capsys):
    utils.echo('hi', verbose=True)
    output = capsys.readouterr().out
    assert output.strip() == 'hi'

    utils.echo('hi', type=1, verbose=True)
    output = capsys.readouterr().out
    assert output[0] == '\n'
    assert output[1:4] == 'hi\n'
    assert output[4:] == '-' * 40 + '\n'


def test_chunck_baselines_by_redundant_group():
    reds_extended = [[(24, 24), (25, 25), (37, 37), (38, 38), (39, 39), (52, 52), (53, 53), (67, 67), (68, 68), (125, 125), (146, 146)],
                     [(24, 37), (25, 38), (38, 52), (39, 53), (39, 125), (125, 146)],
                     [(24, 38), (25, 39), (37, 52), (38, 53), (25, 146)],
                     [(24, 25), (37, 38), (38, 39), (52, 53), (24, 67)],
                     [(24, 52), (25, 53), (67, 68)],
                     [(25, 37), (39, 52), (67, 125)],
                     [(24, 39), (37, 53)],
                     [(37, 39)],
                     [(25, 52)],
                     [(24, 53)],
                     [(24, 125), (52, 68)],
                     [(37, 125), (39, 149)]]
    reds = [[(24, 24), (25, 25), (37, 37), (38, 38), (39, 39), (52, 52), (53, 53)],
            [(24, 37), (25, 38), (38, 52), (39, 53)],
            [(24, 38), (25, 39), (37, 52), (38, 53)],
            [(24, 25), (37, 38), (38, 39), (52, 53)],
            [(24, 52), (25, 53)],
            [(25, 37), (39, 52)],
            [(24, 39), (37, 53)],
            [(37, 39)],
            [(25, 52)],
            [(24, 53)]]
    # add polarizations.
    for grpnum in range(len(reds)):
        for blnum in range(len(reds[grpnum])):
            reds[grpnum][blnum] = reds[grpnum][blnum] + ('ee', )
    # add polarizations.
    for grpnum in range(len(reds_extended)):
        for blnum in range(len(reds_extended[grpnum])):
            reds_extended[grpnum][blnum] = reds_extended[grpnum][blnum] + ('ee', )

    chunked_by_four_expected = [reds[0], reds[1], reds[2], reds[3],
                                reds[4] + reds[5], reds[6] + reds[7] + reds[8], reds[9]]
    # unravel redundant group.
    bls = []
    for grp in reds:
        for bl in grp:
            bls.append(bl)
    chunked_by_four_output = utils.chunk_baselines_by_redundant_groups(reds=filter_reds(reds_extended, bls=bls), max_chunk_size=4)
    for chunk1, chunk2 in zip(chunked_by_four_output, chunked_by_four_expected):
        assert chunk1 == chunk2


def test_select_spw_ranges(tmpdir):
    # validate spw_ranges.
    tmp_path = tmpdir.strpath
    # test that units are propagated from calibration gains to calibrated data.
    new_cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
    uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
    hd = io.HERAData(uvh5)
    nf = hd.Nfreqs
    output = os.path.join(tmp_path, 'test_calibrated_output.uvh5')
    utils.select_spw_ranges(inputfilename=uvh5, outputfilename=output, spw_ranges=[(0, 256), (332, 364), (792, 1000)])
    hdo = io.HERAData(output)
    assert np.allclose(hdo.freq_array, np.hstack([hd.freq_array[:256], hd.freq_array[332:364], hd.freq_array[792:1000]]))
    # test case where no spw-ranges supplied
    utils.select_spw_ranges(inputfilename=uvh5, outputfilename=output, clobber=True)
    hdo = io.HERAData(output)
    assert np.allclose(hdo.freq_array, hd.freq_array)


def test_select_spw_ranges_argparser():
    sys.argv = [sys.argv[0], 'a', 'b', '--clobber', '--spw_ranges', '0 20,30 100,120 150']
    ap = utils.select_spw_ranges_argparser()
    args = ap.parse_args()
    assert args.spw_ranges == [(0, 20), (30, 100), (120, 150)]
    # test tilde formatting.
    sys.argv = [sys.argv[0], 'a', 'b', '--clobber', '--spw_ranges', '0~20,30~100,120~150']
    ap = utils.select_spw_ranges_argparser()
    args = ap.parse_args()
    assert args.spw_ranges == [(0, 20), (30, 100), (120, 150)]


def test_select_spw_ranges_run_script_code(tmpdir):
    # test script code from scripts/test_select_spw_ranges.py
    tmp_path = tmpdir.strpath
    new_cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
    uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
    hd = io.HERAData(uvh5)
    hd.read()
    nf = hd.Nfreqs
    output = os.path.join(tmp_path, 'test_calibrated_output.uvh5')
    # construct bash script command
    select_cmd = f'python ./scripts/select_spw_ranges.py {uvh5} {output} --clobber --spw_ranges 0~256,332~364,792~1000'
    # and excecute inside of python
    os.system(select_cmd)
    # test that output has correct frequencies.
    hdo = io.HERAData(output)
    hdo.read()
    assert np.allclose(hdo.freq_array, np.hstack([hd.freq_array[:256], hd.freq_array[332:364], hd.freq_array[792:1000]]))
    freq_inds = np.hstack([np.arange(0, 256).astype(int), np.arange(332, 364).astype(int), np.arange(792, 1000).astype(int)])
    # and check that data, flags, nsamples make sense.
    assert np.allclose(hdo.data_array, hd.data_array[:, freq_inds, :])
    assert np.allclose(hdo.flag_array, hd.flag_array[:, freq_inds, :])
    assert np.allclose(hdo.nsample_array, hd.nsample_array[:, freq_inds, :])
