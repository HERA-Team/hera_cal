# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import nose.tools as nt
import numpy as np
import sys
import os
import warnings
import shutil
import glob
from collections import OrderedDict as odict
import copy
from contextlib import contextmanager
import six
from pyuvdata import UVData
from pyuvdata import UVCal
import pyuvdata.tests as uvtest

from hera_cal import utils, abscal, datacontainer, io
from hera_cal.redcal import noise
from hera_cal.calibrations import CAL_PATH
from hera_cal.data import DATA_PATH


# define a context manager for checking stdout
# from https://stackoverflow.com/questions/4219717/how-to-assert-output-with-nosetest-unittest-in-python
@contextmanager
def captured_output():
    new_out, new_err = six.StringIO(), six.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class Test_Pol_Ops(object):
    def test_split_pol(self):
        nt.assert_equal(utils.split_pol('xx'), ('Jxx', 'Jxx'))
        nt.assert_equal(utils.split_pol('xy'), ('Jxx', 'Jyy'))
        nt.assert_equal(utils.split_pol('XY'), ('Jxx', 'Jyy'))
        nt.assert_raises(KeyError, utils.split_pol, 'I')
        nt.assert_raises(KeyError, utils.split_pol, 'pV')

    def test_join_pol(self):
        nt.assert_equal(utils.join_pol('Jxx', 'Jxx'), 'xx')
        nt.assert_equal(utils.join_pol('Jxx', 'Jyy'), 'xy')

    def test_split_bl(self):
        nt.assert_equal(utils.split_bl((1, 2, 'xx')), ((1, 'Jxx'), (2, 'Jxx')))
        nt.assert_equal(utils.split_bl((1, 2, 'xy')), ((1, 'Jxx'), (2, 'Jyy')))
        nt.assert_equal(utils.split_bl((1, 2, 'XX')), ((1, 'Jxx'), (2, 'Jxx')))
        nt.assert_raises(KeyError, utils.split_bl, (1, 2, 'pQ'))
        nt.assert_raises(KeyError, utils.split_bl, (1, 2, 'U'))

    def test_join_bl(self):
        nt.assert_equal(utils.join_bl((1, 'Jxx'), (2, 'Jxx')), (1, 2, 'xx'))
        nt.assert_equal(utils.join_bl((1, 'Jxx'), (2, 'Jyy')), (1, 2, 'xy'))

    def test_reverse_bl(self):
        nt.assert_equal(utils.reverse_bl((1, 2, 'xx')), (2, 1, 'xx'))
        nt.assert_equal(utils.reverse_bl((1, 2, 'xy')), (2, 1, 'yx'))
        nt.assert_equal(utils.reverse_bl((1, 2, 'XX')), (2, 1, 'xx'))
        nt.assert_equal(utils.reverse_bl((1, 2, 'pI')), (2, 1, 'pI'))
        nt.assert_equal(utils.reverse_bl((1, 2)), (2, 1))

    def test_comply_bl(self):
        nt.assert_equal(utils.comply_bl((1, 2, 'xx')), (1, 2, 'xx'))
        nt.assert_equal(utils.comply_bl((1, 2, 'xy')), (1, 2, 'xy'))
        nt.assert_equal(utils.comply_bl((1, 2, 'XX')), (1, 2, 'xx'))
        nt.assert_equal(utils.comply_bl((1, 2, 'pI')), (1, 2, 'pI'))

    def test_make_bl(self):
        nt.assert_equal(utils.make_bl((1, 2, 'xx')), (1, 2, 'xx'))
        nt.assert_equal(utils.make_bl((1, 2), 'xx'), (1, 2, 'xx'))
        nt.assert_equal(utils.make_bl((1, 2, 'xy')), (1, 2, 'xy'))
        nt.assert_equal(utils.make_bl((1, 2), 'xy'), (1, 2, 'xy'))
        nt.assert_equal(utils.make_bl((1, 2, 'XX')), (1, 2, 'xx'))
        nt.assert_equal(utils.make_bl((1, 2), 'XX'), (1, 2, 'xx'))
        nt.assert_equal(utils.make_bl((1, 2, 'pI')), (1, 2, 'pI'))
        nt.assert_equal(utils.make_bl((1, 2), 'pI'), (1, 2, 'pI'))


class TestFftDly(object):

    def setUp(self):
        np.random.seed(0)
        self.freqs = np.linspace(.1, .2, 1024)

    def test_ideal(self):
        true_dlys = np.random.uniform(-200, 200, size=60)
        true_dlys.shape = (60, 1)
        data = np.exp(2j * np.pi * self.freqs.reshape((1, -1)) * true_dlys)
        df = np.median(np.diff(self.freqs))
        dlys, offs = utils.fft_dly(data, df)
        np.testing.assert_almost_equal(5 * dlys, 5 * true_dlys, -1)  # accuracy of 2 ns
        np.testing.assert_almost_equal(offs, 0, -2)
        dlys, offs = utils.fft_dly(data, df, medfilt=True)
        np.testing.assert_almost_equal(5 * dlys, 5 * true_dlys, -1)  # accuracy of 2 ns

    def test_ideal_offset(self):
        true_dlys = np.random.uniform(-200, 200, size=60)
        true_dlys.shape = (60, 1)
        data = np.exp(2j * np.pi * self.freqs.reshape((1, -1)) * true_dlys + 1j * 0.123)
        df = np.median(np.diff(self.freqs))
        dlys, offs = utils.fft_dly(data, df)
        np.testing.assert_almost_equal(5 * dlys, 5 * true_dlys, -1)  # accuracy of 2 ns
        np.testing.assert_almost_equal(offs, 0.123, -2)
        dlys, offs = utils.fft_dly(data, df, medfilt=True)
        np.testing.assert_almost_equal(5 * dlys, 5 * true_dlys, -1)  # accuracy of 2 ns
        mdl = np.exp(2j * np.pi * self.freqs.reshape((1, -1)) * dlys + 1j * offs)
        np.testing.assert_almost_equal(np.angle(data * mdl.conj()), 0, -1)

    def test_noisy(self):
        true_dlys = np.random.uniform(-200, 200, size=60)
        true_dlys.shape = (60, 1)
        data = np.exp(2j * np.pi * self.freqs.reshape((1, -1)) * true_dlys) + 5 * noise((60, 1024))
        df = np.median(np.diff(self.freqs))
        dlys, offs = utils.fft_dly(data, df)
        np.testing.assert_almost_equal(1. * dlys, 1. * true_dlys, -1)  # accuracy of 10 ns
        dlys, offs = utils.fft_dly(data, df, medfilt=True)
        np.testing.assert_almost_equal(1. * dlys, 1. * true_dlys, -1)  # accuracy of 10 ns

    def test_rfi(self):
        true_dlys = np.random.uniform(-200, 200, size=60)
        true_dlys.shape = (60, 1)
        data = np.exp(2j * np.pi * self.freqs.reshape((1, -1)) * true_dlys)
        data[:, ::16] = 1000.
        df = np.median(np.diff(self.freqs))
        dlys, offs = utils.fft_dly(data, df, medfilt=True)
        np.testing.assert_almost_equal(5. * dlys, 5. * true_dlys, -1)  # accuracy of 2 ns

    def test_nan(self):
        true_dlys = np.random.uniform(-200, 200, size=60)
        true_dlys.shape = (60, 1)
        data = np.exp(2j * np.pi * self.freqs.reshape((1, -1)) * true_dlys)
        data[:, ::16] = np.nan
        df = np.median(np.diff(self.freqs))
        dlys, offs = utils.fft_dly(data, df)
        np.testing.assert_almost_equal(5. * dlys, 5. * true_dlys, -1)  # accuracy of 2 ns

    def test_realistic(self):
        # load into pyuvdata object
        data_fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        model_fname = os.path.join(DATA_PATH, "zen.2458042.12552.xx.HH.uvXA")
        # make custom gain keys
        d, fl, antpos, a, freqs, t, l, p = io.load_vis(data_fname, return_meta=True, pick_data_ants=False)
        freqs /= 1e9
        # test basic execution
        k1 = (24, 25, 'xx')
        k2 = (37, 38, 'xx')
        flat_phs = d[k1] * d[k2].conj()
        df = np.median(np.diff(freqs))
        # basic execution
        dlys, offs = utils.fft_dly(flat_phs, df, medfilt=True)
        nt.assert_equal(dlys.shape, (60, 1))
        np.testing.assert_almost_equal(dlys, .25, 1)
        true_dlys = np.random.uniform(-20, 20, size=60)
        true_dlys.shape = (60, 1)
        phs = np.exp(2j * np.pi * freqs.reshape((1, -1)) * (true_dlys - .25))
        dlys, offs = utils.fft_dly(flat_phs * phs, df, medfilt=True)
        np.testing.assert_almost_equal(5. * dlys, 5. * true_dlys, -1)


class TestAAFromUV(object):
    def setUp(self):
        # define test file that is compatible with get_aa_from_uv
        self.test_file = "zen.2457999.76839.xx.HH.uvA"

    def test_get_aa_from_uv(self):
        fn = os.path.join(DATA_PATH, self.test_file)
        uvd = UVData()
        uvd.read_miriad(fn)
        aa = utils.get_aa_from_uv(uvd)
        # like miriad, aipy will pad the aa with non-existent antennas,
        #   because there is no concept of antenna names
        nt.assert_equal(len(aa), 88)


class TestAAFromCalfile(object):
    def setUp(self):
        # define frequencies
        self.freqs = np.array([0.15])

        # add directory with calfile
        if CAL_PATH not in sys.path:
            sys.path.append(CAL_PATH)
        self.calfile = "hera_test_calfile"

    def test_get_aa_from_calfile(self):
        aa = utils.get_aa_from_calfile(self.freqs, self.calfile)
        nt.assert_equal(len(aa), 128)


class TestAA(object):
    def setUp(self):
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
        nt.assert_true(np.allclose(old_top, new_top))


def test_JD2LST():
    # test float execution
    jd = 2458042.
    nt.assert_almost_equal(utils.JD2LST(jd, longitude=21.), 3.930652307266274)
    # test array execution
    jd = np.arange(2458042, 2458046.1, .5)
    lst = utils.JD2LST(jd, longitude=21.)
    nt.assert_equal(len(lst), 9)
    nt.assert_almost_equal(lst[3], 0.81486300218170715)


def test_LST2JD():
    # test basic execution
    lst = np.pi
    jd = utils.LST2JD(lst, start_jd=2458042)
    nt.assert_almost_equal(jd, 2458042.8708433118)
    # test array execution
    lst = np.arange(np.pi, np.pi + 1.1, 0.2)
    jd = utils.LST2JD(lst, start_jd=2458042)
    nt.assert_equal(len(jd), 6)
    nt.assert_almost_equal(jd[3], 2458042.9660755517)


def test_JD2RA():
    # test basic execution
    jd = 2458042.5
    ra = utils.JD2RA(jd)
    nt.assert_almost_equal(ra, 46.130897831277629)
    # test array
    jd = np.arange(2458042, 2458043.01, .2)
    ra = utils.JD2RA(jd)
    nt.assert_equal(len(ra), 6)
    nt.assert_almost_equal(ra[3], 82.229459674026003)
    # test exception
    nt.assert_raises(ValueError, utils.JD2RA, jd, epoch='foo')
    # test J2000 epoch
    ra = utils.JD2RA(jd, epoch='J2000')
    nt.assert_almost_equal(ra[0], 225.37671446615548)


def test_combine_calfits():
    test_file1 = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.abs.calfits')
    test_file2 = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.dly.calfits')
    # test basic execution
    if os.path.exists('ex.calfits'):
        os.remove('ex.calfits')
    utils.combine_calfits([test_file1, test_file2], 'ex.calfits', outdir='./', overwrite=True, broadcast_flags=True)
    # test it exists
    nt.assert_true(os.path.exists('ex.calfits'))
    # test antenna number
    uvc = UVCal()
    uvc.read_calfits('ex.calfits')
    nt.assert_equal(len(uvc.antenna_numbers), 7)
    # test time number
    nt.assert_equal(uvc.Ntimes, 60)
    # test gain value got properly multiplied
    uvc_dly = UVCal()
    uvc_dly.read_calfits(test_file1)
    uvc_abs = UVCal()
    uvc_abs.read_calfits(test_file2)
    nt.assert_almost_equal(uvc_dly.gain_array[0, 0, 10, 10, 0] * uvc_abs.gain_array[0, 0, 10, 10, 0], uvc.gain_array[0, 0, 10, 10, 0])
    if os.path.exists('ex.calfits'):
        os.remove('ex.calfits')
    utils.combine_calfits([test_file1, test_file2], 'ex.calfits', outdir='./', overwrite=True, broadcast_flags=False)
    nt.assert_true(os.path.exists('ex.calfits'))
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
    nt.assert_true(np.isclose(phs_err, 0, atol=1e-7))
    # check error across file
    phs_err = np.angle(data[k][:-1, 4] / data_drift[k][1:, 4])
    nt.assert_true(np.abs(phs_err).max() < 1e-4)

    # multiple phase term test: dlst per integration
    dlst = np.array([np.median(np.diff(lsts))] * data[k].shape[0])
    data = copy.deepcopy(data_drift)
    utils.lst_rephase(data, bls, freqs, dlst, lat=0.0)
    # check error at transit
    phs_err = np.angle(data[k][transit_integration, 4] / data_drift[k][transit_integration + 1, 4])
    nt.assert_true(np.isclose(phs_err, 0, atol=1e-7))
    # check err across file
    phs_err = np.angle(data[k][:-1, 4] / data_drift[k][1:, 4])
    nt.assert_true(np.abs(phs_err).max() < 1e-4)

    # phase all integrations to a single integration
    dlst = lsts[50] - lsts
    data = copy.deepcopy(data_drift)
    utils.lst_rephase(data, bls, freqs, dlst, lat=0.0)
    # check error at transit
    phs_err = np.angle(data[k][transit_integration, 4] / data_drift[k][transit_integration, 4])
    nt.assert_true(np.isclose(phs_err, 0, atol=1e-7))
    # check error across file
    phs_err = np.angle(data[k][:, 4] / data_drift[k][50, 4])
    nt.assert_true(np.abs(phs_err).max() < 1e-4)

    # test operation on array
    k = (0, 1, 'xx')
    d = data_drift[k].copy()
    d_phs = utils.lst_rephase(d, bls[k], freqs, dlst, lat=0.0, array=True)
    nt.assert_almost_equal(np.abs(np.angle(d_phs[50] / data[k][50])).max(), 0.0)


def test_chisq():
    # test basic case
    data = datacontainer.DataContainer({(0, 1, 'xx'): np.ones((5, 10), dtype=complex)})
    model = datacontainer.DataContainer({(0, 1, 'xx'): 3 * np.ones((5, 10), dtype=complex)})
    chisq, nObs, chisq_per_ant, nObs_per_ant = utils.chisq(data, model)
    nt.assert_true(chisq.shape == (5, 10))
    nt.assert_true(nObs.shape == (5, 10))
    nt.assert_true(chisq.dtype == float)
    nt.assert_true(nObs.dtype == int)
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
    nt.assert_false((1, 2, 'xx') in model)

    # test with weights
    data = datacontainer.DataContainer({(0, 1, 'xx'): np.ones((5, 10), dtype=complex)})
    model = datacontainer.DataContainer({(0, 1, 'xx'): 2 * np.ones((5, 10), dtype=complex)})
    data_wgts = datacontainer.DataContainer({(0, 1, 'xx'): np.zeros((5, 10), dtype=float)})
    data_wgts[(0, 1, 'xx')][:, 0] = 1.0
    chisq, nObs, chisq_per_ant, nObs_per_ant = utils.chisq(data, model, data_wgts)
    nt.assert_equal(np.sum(chisq), 5.0)
    nt.assert_equal(np.sum(nObs), 5)

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
    nt.assert_almost_equal(np.sum(chisq), 0.0)
    nt.assert_equal(np.sum(nObs), 45)
    nt.assert_almost_equal(np.sum(chisq_per_ant[0, 'Jxx']), 0.0)
    nt.assert_almost_equal(np.sum(chisq_per_ant[1, 'Jxx']), 0.0)
    nt.assert_equal(np.sum(nObs_per_ant[1, 'Jxx']), 45)
    nt.assert_equal(np.sum(nObs_per_ant[1, 'Jxx']), 45)

    # test errors
    nt.assert_raises(ValueError, utils.chisq, data, model, data_wgts, chisq=chisq)
    nt.assert_raises(ValueError, utils.chisq, data, model, data_wgts, nObs=nObs)
    nt.assert_raises(AssertionError, utils.chisq, data, model, data_wgts, split_by_antpol=True, chisq={'Jxx': 1}, nObs={})
    nt.assert_raises(AssertionError, utils.chisq, data, model, data_wgts, split_by_antpol=True, nObs={'Jxx': 1}, chisq={})
    nt.assert_raises(ValueError, utils.chisq, data, model, data_wgts, chisq_per_ant=chisq_per_ant)
    nt.assert_raises(ValueError, utils.chisq, data, model, data_wgts, nObs_per_ant=nObs_per_ant)
    nt.assert_raises(AssertionError, utils.chisq, data, model, data_wgts, chisq_per_ant=chisq_per_ant, nObs_per_ant={})
    nt.assert_raises(AssertionError, utils.chisq, data, model, data_wgts, chisq_per_ant={}, nObs_per_ant=nObs_per_ant)
    nt.assert_raises(KeyError, utils.chisq, data, model, data_wgts, gains={(0, 'x'): np.ones((5, 10), dtype=complex)})
    data_wgts = datacontainer.DataContainer({(0, 1, 'xx'): 1.0j * np.ones((5, 10), dtype=float)})
    nt.assert_raises(AssertionError, utils.chisq, data, model, data_wgts)

    # test by_pol option
    data = datacontainer.DataContainer({(0, 1, 'xx'): np.ones((5, 10), dtype=complex)})
    model = datacontainer.DataContainer({(0, 1, 'xx'): 2 * np.ones((5, 10), dtype=complex)})
    data_wgts = datacontainer.DataContainer({(0, 1, 'xx'): np.ones((5, 10), dtype=float)})
    chisq, nObs, chisq_per_ant, nObs_per_ant = utils.chisq(data, model, data_wgts, split_by_antpol=True)
    nt.assert_true('Jxx' in chisq)
    nt.assert_true('Jxx' in nObs)
    nt.assert_true(chisq['Jxx'].shape, (5, 10))
    nt.assert_true(nObs['Jxx'].shape, (5, 10))
    np.testing.assert_array_equal(chisq['Jxx'], 1.0)
    np.testing.assert_array_equal(nObs['Jxx'], 1)
    data = datacontainer.DataContainer({(0, 1, 'xy'): np.ones((5, 10), dtype=complex)})
    model = datacontainer.DataContainer({(0, 1, 'xy'): 2 * np.ones((5, 10), dtype=complex)})
    data_wgts = datacontainer.DataContainer({(0, 1, 'xy'): np.ones((5, 10), dtype=float)})
    chisq, nObs, chisq_per_ant, nObs_per_ant = utils.chisq(data, model, data_wgts, split_by_antpol=True)
    nt.assert_true(len(chisq) == 0)
    nt.assert_true(len(nObs) == 0)
    nt.assert_true(len(chisq_per_ant) == 0)
    nt.assert_true(len(chisq_per_ant) == 0)


def test_echo():
    with captured_output() as (out, err):
        utils.echo('hi', verbose=True)
    output = out.getvalue().strip()
    nt.assert_equal(output, 'hi')

    with captured_output() as (out, err):
        utils.echo('hi', type=1, verbose=True)
    output = out.getvalue()
    print("output: ", output)
    nt.assert_equal(output[0], '\n')
    nt.assert_equal(output[1:4], 'hi\n')
    nt.assert_equal(output[4:], '-' * 40 + '\n')
