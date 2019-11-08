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

from hera_sim.noise import white_noise
from .. import utils, abscal, datacontainer, io
from ..calibrations import CAL_PATH
from ..data import DATA_PATH


class Test_Pol_Ops(object):
    def test_split_pol(self):
        assert utils.split_pol('xx') == ('Jxx', 'Jxx')
        assert utils.split_pol('xy') == ('Jxx', 'Jyy')
        assert utils.split_pol('XY') == ('Jxx', 'Jyy')
        pytest.raises(KeyError, utils.split_pol, 'I')
        pytest.raises(KeyError, utils.split_pol, 'pV')

    def test_join_pol(self):
        assert utils.join_pol('Jxx', 'Jxx') == 'xx'
        assert utils.join_pol('Jxx', 'Jyy') == 'xy'

    def test_split_bl(self):
        assert utils.split_bl((1, 2, 'xx')) == ((1, 'Jxx'), (2, 'Jxx'))
        assert utils.split_bl((1, 2, 'xy')) == ((1, 'Jxx'), (2, 'Jyy'))
        assert utils.split_bl((1, 2, 'XX')) == ((1, 'Jxx'), (2, 'Jxx'))
        pytest.raises(KeyError, utils.split_bl, (1, 2, 'pQ'))
        pytest.raises(KeyError, utils.split_bl, (1, 2, 'U'))

    def test_join_bl(self):
        assert utils.join_bl((1, 'Jxx'), (2, 'Jxx')) == (1, 2, 'xx')
        assert utils.join_bl((1, 'Jxx'), (2, 'Jyy')) == (1, 2, 'xy')

    def test_reverse_bl(self):
        assert utils.reverse_bl((1, 2, 'xx')) == (2, 1, 'xx')
        assert utils.reverse_bl((1, 2, 'xy')) == (2, 1, 'yx')
        assert utils.reverse_bl((1, 2, 'XX')) == (2, 1, 'xx')
        assert utils.reverse_bl((1, 2, 'pI')) == (2, 1, 'pI')
        assert utils.reverse_bl((1, 2)) == (2, 1)

    def test_comply_bl(self):
        assert utils.comply_bl((1, 2, 'xx')) == (1, 2, 'xx')
        assert utils.comply_bl((1, 2, 'xy')) == (1, 2, 'xy')
        assert utils.comply_bl((1, 2, 'XX')) == (1, 2, 'xx')
        assert utils.comply_bl((1, 2, 'pI')) == (1, 2, 'pI')

    def test_make_bl(self):
        assert utils.make_bl((1, 2, 'xx')) == (1, 2, 'xx')
        assert utils.make_bl((1, 2), 'xx') == (1, 2, 'xx')
        assert utils.make_bl((1, 2, 'xy')) == (1, 2, 'xy')
        assert utils.make_bl((1, 2), 'xy') == (1, 2, 'xy')
        assert utils.make_bl((1, 2, 'XX')) == (1, 2, 'xx')
        assert utils.make_bl((1, 2), 'XX') == (1, 2, 'xx')
        assert utils.make_bl((1, 2, 'pI')) == (1, 2, 'pI')
        assert utils.make_bl((1, 2), 'pI') == (1, 2, 'pI')


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
        k1 = (24, 25, 'xx')
        k2 = (37, 38, 'xx')
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
        y = (x - 5)**2 + np.isclose(x, 5.0).astype(np.float)
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
    bls = odict(map(lambda k: (k, antpos[k[0]] - antpos[k[1]]), data.keys()))

    # basic test: single dlst for all integrations
    utils.lst_rephase(data, bls, freqs, dlst, lat=0.0)
    # get phase error on shortest EW baseline
    k = (0, 1, 'xx')
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
    k = (0, 1, 'xx')
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


def test_gp_interp1d():
    # load data
    dfiles = glob.glob(os.path.join(DATA_PATH, "zen.2458043.4*.xx.HH.XRAA.uvh5"))
    uvd = UVData()
    uvd.read(dfiles, bls=[(37, 39)])
    times = np.unique(uvd.time_array) * 24 * 60
    times -= times.min()
    y = uvd.get_data(37, 39, 'xx')
    f = uvd.get_flags(37, 39, 'xx')

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


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_gain_relative_difference():
    # setup
    old_gains = {(0, 'Jxx'): np.ones((10, 10), dtype=np.complex),
                 (1, 'Jxx'): np.ones((10, 10), dtype=np.complex)}
    new_gains = {(0, 'Jxx'): 2. * np.ones((10, 10), dtype=np.complex),
                 (1, 'Jxx'): 4. * np.ones((10, 10), dtype=np.complex)}
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
