# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import nose.tools as nt
import os
import shutil
import json
import numpy as np
import aipy
import optparse
import sys
from collections import OrderedDict as odict
import copy
import glob
from six.moves import map, zip
from pyuvdata import UVCal, UVData
from pyuvdata import utils as uvutils
import pyuvdata.tests as uvtest
import warnings

from hera_cal import io, abscal, redcal, utils
from hera_cal.data import DATA_PATH
from hera_cal.datacontainer import DataContainer
from hera_cal.utils import split_pol
from hera_cal.apply_cal import calibrate_in_place


class Test_AbsCal_Funcs:

    def setUp(self):
        np.random.seed(0)

        # load into pyuvdata object
        self.data_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        self.uvd = UVData()
        self.uvd.read_miriad(self.data_file)
        self.freq_array = np.unique(self.uvd.freq_array)
        self.antpos, self.ants = self.uvd.get_ENU_antpos(center=True, pick_data_ants=True)
        self.antpos = odict(zip(self.ants, self.antpos))
        self.time_array = np.unique(self.uvd.time_array)

        # configure data into dictionaries
        data, flgs = io.load_vis(self.uvd, pop_autos=True)
        wgts = odict()
        for k in flgs.keys():
            wgts[k] = (~flgs[k]).astype(np.float)
        wgts = DataContainer(wgts)

        # configure baselines
        bls = odict([(x, self.antpos[x[0]] - self.antpos[x[1]]) for x in data.keys()])

        # make mock data
        abs_gain = 0.5
        TT_phi = np.array([-0.004, 0.006, 0])
        model = odict()
        for i, k in enumerate(data.keys()):
            model[k] = data[k] * np.exp(abs_gain + 1j * np.dot(TT_phi, bls[k]))

        # assign data
        self.data = data
        self.bls = bls
        self.model = model
        self.wgts = wgts

    def test_data_key_to_array_axis(self):
        m, pk = abscal.data_key_to_array_axis(self.model, 2)
        nt.assert_equal(m[(24, 25)].shape, (60, 64, 1))
        nt.assert_equal('xx' in pk, True)
        # test w/ avg_dict
        m, ad, pk = abscal.data_key_to_array_axis(self.model, 2, avg_dict=self.bls)
        nt.assert_equal(m[(24, 25)].shape, (60, 64, 1))
        nt.assert_equal(ad[(24, 25)].shape, (3,))
        nt.assert_equal('xx' in pk, True)

    def test_array_axis_to_data_key(self):
        m, pk = abscal.data_key_to_array_axis(self.model, 2)
        m2 = abscal.array_axis_to_data_key(m, 2, ['xx'])
        nt.assert_equal(m2[(24, 25, 'xx')].shape, (60, 64))
        # copy dict
        m, ad, pk = abscal.data_key_to_array_axis(self.model, 2, avg_dict=self.bls)
        m2, cd = abscal.array_axis_to_data_key(m, 2, ['xx'], copy_dict=ad)
        nt.assert_equal(m2[(24, 25, 'xx')].shape, (60, 64))
        nt.assert_equal(cd[(24, 25, 'xx')].shape, (3,))

    def test_interp2d(self):
        # test interpolation w/ warning
        m, mf = abscal.interp2d_vis(self.data, self.time_array, self.freq_array,
                                    self.time_array, self.freq_array, flags=self.wgts, medfilt_flagged=False)
        nt.assert_equal(m[(24, 25, 'xx')].shape, (60, 64))
        # downsampling w/ no flags
        m, mf = abscal.interp2d_vis(self.data, self.time_array, self.freq_array,
                                    self.time_array[::2], self.freq_array[::2])
        nt.assert_equal(m[(24, 25, 'xx')].shape, (30, 32))
        # test flag propagation
        m, mf = abscal.interp2d_vis(self.data, self.time_array, self.freq_array,
                                    self.time_array, self.freq_array, flags=self.wgts, medfilt_flagged=True)
        nt.assert_true(mf[(24, 25, 'xx')][10, 0])
        # test flag extrapolation
        m, mf = abscal.interp2d_vis(self.data, self.time_array, self.freq_array,
                                    self.time_array + .0001, self.freq_array, flags=self.wgts, flag_extrapolate=True)
        nt.assert_true(mf[(24, 25, 'xx')][-1].min())

    def test_wiener(self):
        # test smoothing
        d = abscal.wiener(self.data, window=(5, 15), noise=None, medfilt=True, medfilt_kernel=(1, 13))
        nt.assert_equal(d[(24, 37, 'xx')].shape, (60, 64))
        nt.assert_equal(d[(24, 37, 'xx')].dtype, np.complex)
        # test w/ noise
        d = abscal.wiener(self.data, window=(5, 15), noise=0.1, medfilt=True, medfilt_kernel=(1, 13))
        nt.assert_equal(d[(24, 37, 'xx')].shape, (60, 64))
        # test w/o medfilt
        d = abscal.wiener(self.data, window=(5, 15), medfilt=False)
        nt.assert_equal(d[(24, 37, 'xx')].shape, (60, 64))
        # test as array
        d = abscal.wiener(self.data[(24, 37, 'xx')], window=(5, 15), medfilt=False, array=True)
        nt.assert_equal(d.shape, (60, 64))
        nt.assert_equal(d.dtype, np.complex)

    def test_Baseline(self):
        # test basic execution
        keys = list(self.data.keys())
        k1 = (24, 25, 'xx')    # 14.6 m E-W
        i1 = keys.index(k1)
        k2 = (24, 37, 'xx')    # different
        i2 = keys.index(k2)
        k3 = (52, 53, 'xx')   # 14.6 m E-W
        i3 = keys.index(k3)
        bls = list(map(lambda k: abscal.Baseline(self.antpos[k[1]] - self.antpos[k[0]], tol=2.0), keys))
        bls_conj = list(map(lambda k: abscal.Baseline(self.antpos[k[0]] - self.antpos[k[1]], tol=2.0), keys))
        nt.assert_equal(bls[i1], bls[i1])
        nt.assert_false(bls[i1] == bls[i2])
        nt.assert_equal(bls[i1] == bls_conj[i1], 'conjugated')
        # test different yet redundant baselines still agree
        nt.assert_equal(bls[i1], bls[i3])
        # test tolerance works as expected
        bls = list(map(lambda k: abscal.Baseline(self.antpos[k[1]] - self.antpos[k[0]], tol=1e-4), keys))
        nt.assert_not_equal(bls[i1], bls[i3])

    def test_match_red_baselines(self):
        model = copy.deepcopy(self.data)
        model = DataContainer(odict([((k[0] + 1, k[1] + 1, k[2]), model[k]) for i, k in enumerate(model.keys())]))
        del model[(25, 54, 'xx')]
        model_antpos = odict([(k + 1, self.antpos[k]) for i, k in enumerate(self.antpos.keys())])
        new_model = abscal.match_red_baselines(model, model_antpos, self.data, self.antpos, tol=2.0, verbose=False)
        nt.assert_equal(len(new_model.keys()), 8)
        nt.assert_true((24, 37, 'xx') in new_model)
        nt.assert_false((24, 53, 'xx') in new_model)

    def test_mirror_data_to_red_bls(self):
        # make fake data
        reds = redcal.get_reds(self.antpos, pols=['xx'])
        data = DataContainer(odict(list(map(lambda k: (k[0], self.data[k[0]]), reds[:5]))))
        # test execuation
        d = abscal.mirror_data_to_red_bls(data, self.antpos)
        nt.assert_equal(len(d.keys()), 16)
        nt.assert_true((24, 25, 'xx') in d)
        # test correct value is propagated
        nt.assert_almost_equal(data[(24, 25, 'xx')][30, 30], d[(38, 39, 'xx')][30, 30])
        # test reweighting
        w = abscal.mirror_data_to_red_bls(self.wgts, self.antpos, weights=True)
        nt.assert_equal(w[(24, 25, 'xx')].dtype, np.float)
        nt.assert_almost_equal(w[(24, 25, 'xx')].max(), 16.0)

    def test_flatten(self):
        li = abscal.flatten([['hi']])
        nt.assert_equal(np.array(li).ndim, 1)

    def test_avg_data_across_red_bls(self):
        # test basic execution
        wgts = copy.deepcopy(self.wgts)
        wgts[(24, 25, 'xx')][45, 45] = 0.0
        data, flags, antpos, ants, freqs, times, lsts, pols = io.load_vis(self.data_file, return_meta=True)
        rd, rf, rk = abscal.avg_data_across_red_bls(data, antpos, wgts=wgts, tol=2.0, broadcast_wgts=False)
        nt.assert_equal(rd[(24, 25, 'xx')].shape, (60, 64))
        nt.assert_true(rf[(24, 25, 'xx')][45, 45] > 0.0)
        # test various kwargs
        wgts[(24, 25, 'xx')][45, 45] = 0.0
        rd, rf, rk = abscal.avg_data_across_red_bls(data, antpos, tol=2.0, wgts=wgts, broadcast_wgts=True)
        nt.assert_equal(len(rd.keys()), 9)
        nt.assert_equal(len(rf.keys()), 9)
        nt.assert_almost_equal(rf[(24, 25, 'xx')][45, 45], 0.0)
        # test averaging worked
        rd, rf, rk = abscal.avg_data_across_red_bls(data, antpos, tol=2.0, broadcast_wgts=False)
        v = np.mean([data[(52, 53, 'xx')], data[(37, 38, 'xx')], data[(24, 25, 'xx')], data[(38, 39, 'xx')]], axis=0)
        nt.assert_true(np.isclose(rd[(24, 25, 'xx')], v).min())
        # test mirror_red_data
        rd, rf, rk = abscal.avg_data_across_red_bls(data, antpos, wgts=self.wgts, tol=2.0, mirror_red_data=True)
        nt.assert_equal(len(rd.keys()), 21)
        nt.assert_equal(len(rf.keys()), 21)

    def test_match_times(self):
        dfiles = list(map(lambda f: os.path.join(DATA_PATH, f), ['zen.2458043.12552.xx.HH.uvORA',
                                                                 'zen.2458043.13298.xx.HH.uvORA']))
        mfiles = list(map(lambda f: os.path.join(DATA_PATH, f), ['zen.2458042.12552.xx.HH.uvXA',
                                                                 'zen.2458042.13298.xx.HH.uvXA']))
        # test basic execution
        relevant_mfiles = abscal.match_times(dfiles[0], mfiles, filetype='miriad')
        nt.assert_equal(len(relevant_mfiles), 2)
        # test basic execution
        relevant_mfiles = abscal.match_times(dfiles[1], mfiles, filetype='miriad')
        nt.assert_equal(len(relevant_mfiles), 1)
        # test exception
        mfiles = sorted(glob.glob(os.path.join(DATA_PATH, 'zen.2458045.*.xx.HH.uvXRAA')))
        relevant_mfiles = abscal.match_times(dfiles[0], mfiles, filetype='miriad')
        nt.assert_equal(len(relevant_mfiles), 0)

    def test_rephase_vis(self):
        dfile = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA')
        mfiles = list(map(lambda f: os.path.join(DATA_PATH, f), ['zen.2458042.12552.xx.HH.uvXA']))
        m, mf, mantp, mant, mfr, mt, ml, mp = io.load_vis(mfiles, return_meta=True)
        d, df, dantp, dant, dfr, dt, dl, dp = io.load_vis(dfile, return_meta=True)
        bls = odict(list(map(lambda k: (k, dantp[k[0]] - dantp[k[1]]), d.keys())))

        # basic execution
        new_m, new_f = abscal.rephase_vis(m, ml, dl, bls, dfr)

        k = list(new_m.keys())[0]
        nt.assert_equal(new_m[k].shape, d[k].shape)
        nt.assert_true(new_f[k][-1].min())
        nt.assert_false(new_f[k][0].max())

    def test_cut_bl(self):
        Nbls = len(self.data)
        _data = abscal.cut_bls(self.data, bls=self.bls, min_bl_cut=20.0, inplace=False)
        nt.assert_true(Nbls, 21)
        nt.assert_true(len(_data), 12)
        _data2 = copy.deepcopy(self.data)
        abscal.cut_bls(_data2, bls=self.bls, min_bl_cut=20.0, inplace=True)
        nt.assert_true(len(_data2), 12)
        _data = abscal.cut_bls(self.data, bls=self.bls, min_bl_cut=20.0, inplace=False)
        abscal.cut_bls(_data2, min_bl_cut=20.0, inplace=True)
        nt.assert_true(len(_data2), 12)


class Test_AbsCal:

    def setUp(self):
        np.random.seed(0)
        # load into pyuvdata object
        self.data_fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        self.model_fname = os.path.join(DATA_PATH, "zen.2458042.12552.xx.HH.uvXA")
        self.AC = abscal.AbsCal(self.data_fname, self.model_fname, refant=24)
        self.input_cal = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA.abs.calfits")

        # make custom gain keys
        d, fl, ap, a, f, t, l, p = io.load_vis(self.data_fname, return_meta=True, pick_data_ants=False)
        self.freq_array = f
        self.antpos = ap
        gain_pols = np.unique(list(map(split_pol, p)))
        self.ap = ap
        self.gk = abscal.flatten(list(map(lambda p: list(map(lambda k: (k, p), a)), gain_pols)))
        self.freqs = f

        # make custom error message lists
        messages = ['divide by zero encountered in true_divide',
                    'invalid value encountered in true_divide',
                    'divide by zero encountered in log']
        messages.extend(['divide by zero encountered in true_divide',
                         'invalid value encountered in true_divide'] * 20)
        messages.extend(['divide by zero encountered in log'])
        self.abscal_n44_messages = messages

        self.abscal_n42_messages = ['divide by zero encountered in true_divide',
                                    'invalid value encountered in true_divide'] * 21

    def test_init(self):
        # init with no meta
        AC = abscal.AbsCal(self.AC.model, self.AC.data)
        nt.assert_almost_equal(AC.bls, None)
        # init with meta
        AC = abscal.AbsCal(self.AC.model, self.AC.data, antpos=self.AC.antpos, freqs=self.AC.freqs)
        nt.assert_almost_equal(AC.bls[(24, 25, 'xx')][0], -14.607842046642745)
        # init with meta
        AC = abscal.AbsCal(self.AC.model, self.AC.data)
        # test feeding file and refant and bl_cut and bl_taper
        AC = abscal.AbsCal(self.model_fname, self.data_fname, refant=24, antpos=self.AC.antpos,
                           max_bl_cut=26.0, bl_taper_fwhm=15.0)
        # test ref ant
        nt.assert_equal(AC.refant, 24)
        nt.assert_almost_equal(np.linalg.norm(AC.antpos[24]), 0.0)
        # test bl cut
        nt.assert_false((np.array(list(map(lambda k: np.linalg.norm(AC.bls[k]), AC.bls.keys()))) > 26.0).any())
        # test bl taper
        nt.assert_true(np.median(AC.wgts[(24, 25, 'xx')]) > np.median(AC.wgts[(24, 39, 'xx')]))

        # test with input cal
        bl = (24, 25, 'xx')
        uvc = UVCal()
        uvc.read_calfits(self.input_cal)
        aa = uvc.ant_array.tolist()
        g = (uvc.gain_array[aa.index(bl[0])] * uvc.gain_array[aa.index(bl[1])].conj()).squeeze().T
        gf = (uvc.flag_array[aa.index(bl[0])] + uvc.flag_array[aa.index(bl[1])]).squeeze().T
        w = self.AC.wgts[bl] * ~gf
        AC2 = abscal.AbsCal(copy.deepcopy(self.AC.model), copy.deepcopy(self.AC.data), wgts=copy.deepcopy(self.AC.wgts), refant=24, input_cal=self.input_cal)
        np.testing.assert_array_almost_equal(self.AC.data[bl] / g * w, AC2.data[bl] * w)

    def test_abs_amp_logcal(self):
        # test execution and variable assignments
        uvtest.checkWarnings(self.AC.abs_amp_logcal, [], {'verbose': False},
                             nwarnings=44, category=RuntimeWarning,
                             message=self.abscal_n44_messages)
        nt.assert_equal(self.AC.abs_eta[(24, 'Jxx')].shape, (60, 64))
        nt.assert_equal(self.AC.abs_eta_gain[(24, 'Jxx')].shape, (60, 64))
        nt.assert_equal(self.AC.abs_eta_arr.shape, (7, 60, 64, 1))
        nt.assert_equal(self.AC.abs_eta_gain_arr.shape, (7, 60, 64, 1))
        # test Nones
        AC = abscal.AbsCal(self.AC.model, self.AC.data)
        nt.assert_equal(AC.abs_eta, None)
        nt.assert_equal(AC.abs_eta_arr, None)
        nt.assert_equal(AC.abs_eta_gain, None)
        nt.assert_equal(AC.abs_eta_gain_arr, None)
        # test propagation to gain_arr
        uvtest.checkWarnings(AC.abs_amp_logcal, [], {'verbose': False},
                             nwarnings=44, category=RuntimeWarning,
                             message=self.abscal_n44_messages)
        AC._abs_eta_arr *= 0
        nt.assert_almost_equal(np.abs(AC.abs_eta_gain_arr[0, 0, 0, 0]), 1.0)
        # test custom gain
        g = self.AC.custom_abs_eta_gain(self.gk)
        nt.assert_equal(len(g), 47)
        # test w/ no wgts
        AC.wgts = None
        uvtest.checkWarnings(AC.abs_amp_logcal, [], {'verbose': False},
                             nwarnings=44, category=RuntimeWarning,
                             message=self.abscal_n44_messages)

    def test_TT_phs_logcal(self):
        # test execution
        uvtest.checkWarnings(self.AC.TT_phs_logcal, [], {'verbose': False},
                             nwarnings=42, category=RuntimeWarning,
                             message=self.abscal_n42_messages)
        nt.assert_equal(self.AC.TT_Phi_arr.shape, (7, 2, 60, 64, 1))
        nt.assert_equal(self.AC.TT_Phi_gain_arr.shape, (7, 60, 64, 1))
        nt.assert_equal(self.AC.abs_psi_arr.shape, (7, 60, 64, 1))
        nt.assert_equal(self.AC.abs_psi_gain_arr.shape, (7, 60, 64, 1))
        nt.assert_equal(self.AC.abs_psi[(24, 'Jxx')].shape, (60, 64))
        nt.assert_equal(self.AC.abs_psi_gain[(24, 'Jxx')].shape, (60, 64))
        nt.assert_equal(self.AC.TT_Phi[(24, 'Jxx')].shape, (2, 60, 64))
        nt.assert_equal(self.AC.TT_Phi_gain[(24, 'Jxx')].shape, (60, 64))
        nt.assert_true(np.isclose(np.angle(self.AC.TT_Phi_gain[(24, 'Jxx')]), 0.0).all())
        # test merge pols
        uvtest.checkWarnings(self.AC.TT_phs_logcal, [], {'verbose': False, 'four_pol': True},
                             nwarnings=42, category=RuntimeWarning,
                             message=self.abscal_n42_messages)
        nt.assert_equal(self.AC.TT_Phi_arr.shape, (7, 2, 60, 64, 1))
        nt.assert_equal(self.AC.abs_psi_arr.shape, (7, 60, 64, 1))
        # test Nones
        AC = abscal.AbsCal(self.AC.model, self.AC.data, antpos=self.antpos)
        nt.assert_equal(AC.abs_psi_arr, None)
        nt.assert_equal(AC.abs_psi_gain_arr, None)
        nt.assert_equal(AC.TT_Phi_arr, None)
        nt.assert_equal(AC.TT_Phi_gain_arr, None)
        nt.assert_equal(AC.abs_psi, None)
        nt.assert_equal(AC.abs_psi_gain, None)
        nt.assert_equal(AC.TT_Phi, None)
        nt.assert_equal(AC.TT_Phi_gain, None)
        # test custom gain
        g = self.AC.custom_TT_Phi_gain(self.gk, self.ap)
        nt.assert_equal(len(g), 47)
        g = self.AC.custom_abs_psi_gain(self.gk)
        nt.assert_equal(g[(0, 'Jxx')].shape, (60, 64))
        # test w/ no wgts
        AC.wgts = None
        uvtest.checkWarnings(AC.TT_phs_logcal, [], {'verbose': False},
                             nwarnings=42, category=RuntimeWarning,
                             message=self.abscal_n42_messages)

    def test_amp_logcal(self):
        uvtest.checkWarnings(self.AC.amp_logcal, [], {'verbose': False},
                             nwarnings=44, category=RuntimeWarning,
                             message=self.abscal_n44_messages)
        nt.assert_equal(self.AC.ant_eta[(24, 'Jxx')].shape, (60, 64))
        nt.assert_equal(self.AC.ant_eta_gain[(24, 'Jxx')].shape, (60, 64))
        nt.assert_equal(self.AC.ant_eta_arr.shape, (7, 60, 64, 1))
        nt.assert_equal(self.AC.ant_eta_arr.dtype, np.float)
        nt.assert_equal(self.AC.ant_eta_gain_arr.shape, (7, 60, 64, 1))
        nt.assert_equal(self.AC.ant_eta_gain_arr.dtype, np.complex)
        # test Nones
        AC = abscal.AbsCal(self.AC.model, self.AC.data)
        nt.assert_equal(AC.ant_eta, None)
        nt.assert_equal(AC.ant_eta_gain, None)
        nt.assert_equal(AC.ant_eta_arr, None)
        nt.assert_equal(AC.ant_eta_gain_arr, None)
        # test w/ no wgts
        AC.wgts = None
        uvtest.checkWarnings(AC.amp_logcal, [], {'verbose': False},
                             nwarnings=44, category=RuntimeWarning,
                             message=self.abscal_n44_messages)

    def test_phs_logcal(self):
        self.AC.phs_logcal(verbose=False)
        nt.assert_equal(self.AC.ant_phi[(24, 'Jxx')].shape, (60, 64))
        nt.assert_equal(self.AC.ant_phi_gain[(24, 'Jxx')].shape, (60, 64))
        nt.assert_equal(self.AC.ant_phi_arr.shape, (7, 60, 64, 1))
        nt.assert_equal(self.AC.ant_phi_arr.dtype, np.float)
        nt.assert_equal(self.AC.ant_phi_gain_arr.shape, (7, 60, 64, 1))
        nt.assert_equal(self.AC.ant_phi_gain_arr.dtype, np.complex)
        nt.assert_true(np.isclose(np.angle(self.AC.ant_phi_gain[(24, 'Jxx')]), 0.0).all())
        self.AC.phs_logcal(verbose=False, avg=True)
        AC = abscal.AbsCal(self.AC.model, self.AC.data)
        nt.assert_equal(AC.ant_phi, None)
        nt.assert_equal(AC.ant_phi_gain, None)
        nt.assert_equal(AC.ant_phi_arr, None)
        nt.assert_equal(AC.ant_phi_gain_arr, None)
        # test w/ no wgts
        AC.wgts = None
        AC.phs_logcal(verbose=False)

    def test_delay_lincal(self):
        # test w/o offsets
        self.AC.delay_lincal(verbose=False, kernel=(1, 3), medfilt=False)
        nt.assert_equal(self.AC.ant_dly[(24, 'Jxx')].shape, (60, 1))
        nt.assert_equal(self.AC.ant_dly_gain[(24, 'Jxx')].shape, (60, 64))
        nt.assert_equal(self.AC.ant_dly_arr.shape, (7, 60, 1, 1))
        nt.assert_equal(self.AC.ant_dly_gain_arr.shape, (7, 60, 64, 1))
        # test w/ offsets
        self.AC.delay_lincal(verbose=False, kernel=(1, 3), medfilt=False)
        nt.assert_equal(self.AC.ant_dly_phi[(24, 'Jxx')].shape, (60, 1))
        nt.assert_equal(self.AC.ant_dly_phi_gain[(24, 'Jxx')].shape, (60, 64))
        nt.assert_equal(self.AC.ant_dly_phi_arr.shape, (7, 60, 1, 1))
        nt.assert_equal(self.AC.ant_dly_phi_gain_arr.shape, (7, 60, 64, 1))
        nt.assert_equal(self.AC.ant_dly_arr.shape, (7, 60, 1, 1))
        nt.assert_equal(self.AC.ant_dly_arr.dtype, np.float)
        nt.assert_equal(self.AC.ant_dly_gain_arr.shape, (7, 60, 64, 1))
        nt.assert_equal(self.AC.ant_dly_gain_arr.dtype, np.complex)
        nt.assert_true(np.isclose(np.angle(self.AC.ant_dly_gain[(24, 'Jxx')]), 0.0).all())
        nt.assert_true(np.isclose(np.angle(self.AC.ant_dly_phi_gain[(24, 'Jxx')]), 0.0).all())
        # test exception
        AC = abscal.AbsCal(self.AC.model, self.AC.data)
        nt.assert_raises(AttributeError, AC.delay_lincal)
        # test Nones
        AC = abscal.AbsCal(self.AC.model, self.AC.data, freqs=self.freq_array)
        nt.assert_equal(AC.ant_dly, None)
        nt.assert_equal(AC.ant_dly_gain, None)
        nt.assert_equal(AC.ant_dly_arr, None)
        nt.assert_equal(AC.ant_dly_gain_arr, None)
        nt.assert_equal(AC.ant_dly_phi, None)
        nt.assert_equal(AC.ant_dly_phi_gain, None)
        nt.assert_equal(AC.ant_dly_phi_arr, None)
        nt.assert_equal(AC.ant_dly_phi_gain_arr, None)
        # test flags handling
        AC = abscal.AbsCal(self.AC.model, self.AC.data, freqs=self.freqs)
        AC.wgts[(24, 25, 'xx')] *= 0
        AC.delay_lincal(verbose=False)
        # test medfilt
        self.AC.delay_lincal(verbose=False, medfilt=False)
        self.AC.delay_lincal(verbose=False, time_avg=True)
        # test w/ no wgts
        AC.wgts = None
        AC.delay_lincal(verbose=False)

    def test_delay_slope_lincal(self):
        # test w/o offsets
        self.AC.delay_slope_lincal(verbose=False, kernel=(1, 3), medfilt=False)
        nt.assert_equal(self.AC.dly_slope[(24, 'Jxx')].shape, (2, 60, 1))
        nt.assert_equal(self.AC.dly_slope_gain[(24, 'Jxx')].shape, (60, 64))
        nt.assert_equal(self.AC.dly_slope_arr.shape, (7, 2, 60, 1, 1))
        nt.assert_equal(self.AC.dly_slope_gain_arr.shape, (7, 60, 64, 1))
        nt.assert_equal(self.AC.dly_slope_ant_dly_arr.shape, (7, 60, 1, 1))
        nt.assert_true(np.isclose(np.angle(self.AC.dly_slope_gain[(24, 'Jxx')]), 0.0).all())
        g = self.AC.custom_dly_slope_gain(self.gk, self.ap)
        nt.assert_equal(g[(0, 'Jxx')].shape, (60, 64))
        # test exception
        AC = abscal.AbsCal(self.AC.model, self.AC.data)
        nt.assert_raises(AttributeError, AC.delay_slope_lincal)
        # test Nones
        AC = abscal.AbsCal(self.AC.model, self.AC.data, antpos=self.antpos, freqs=self.freq_array)
        nt.assert_equal(AC.dly_slope, None)
        nt.assert_equal(AC.dly_slope_gain, None)
        nt.assert_equal(AC.dly_slope_arr, None)
        nt.assert_equal(AC.dly_slope_gain_arr, None)
        nt.assert_equal(AC.dly_slope_ant_dly_arr, None)
        # test medfilt and time_avg
        self.AC.delay_slope_lincal(verbose=False, medfilt=False)
        self.AC.delay_slope_lincal(verbose=False, time_avg=True)
        # test four pol
        self.AC.delay_slope_lincal(verbose=False, four_pol=True)
        nt.assert_equal(self.AC.dly_slope[(24, 'Jxx')].shape, (2, 60, 1))
        nt.assert_equal(self.AC.dly_slope_gain[(24, 'Jxx')].shape, (60, 64))
        nt.assert_equal(self.AC.dly_slope_arr.shape, (7, 2, 60, 1, 1))
        nt.assert_equal(self.AC.dly_slope_gain_arr.shape, (7, 60, 64, 1))
        # test flags handling
        AC = abscal.AbsCal(self.AC.model, self.AC.data, antpos=self.ap, freqs=self.freqs)
        AC.wgts[(24, 25, 'xx')] *= 0
        AC.delay_slope_lincal(verbose=False)
        # test w/ no wgts
        AC.wgts = None
        AC.delay_slope_lincal(verbose=False)

    def test_global_phase_slope_logcal(self):
        for solver in ['dft', 'linfit']:
            # test w/o offsets
            self.AC.global_phase_slope_logcal(verbose=False, edge_cut=31, solver=solver)
            nt.assert_equal(self.AC.phs_slope[(24, 'Jxx')].shape, (2, 60, 1))
            nt.assert_equal(self.AC.phs_slope_gain[(24, 'Jxx')].shape, (60, 64))
            nt.assert_equal(self.AC.phs_slope_arr.shape, (7, 2, 60, 1, 1))
            nt.assert_equal(self.AC.phs_slope_gain_arr.shape, (7, 60, 64, 1))
            nt.assert_equal(self.AC.phs_slope_ant_phs_arr.shape, (7, 60, 1, 1))
            nt.assert_true(np.isclose(np.angle(self.AC.phs_slope_gain[(24, 'Jxx')]), 0.0).all())
            g = self.AC.custom_phs_slope_gain(self.gk, self.ap)
            nt.assert_equal(g[(0, 'Jxx')].shape, (60, 64))
            # test Nones
            AC = abscal.AbsCal(self.AC.model, self.AC.data, antpos=self.antpos, freqs=self.freq_array)
            nt.assert_equal(AC.phs_slope, None)
            nt.assert_equal(AC.phs_slope_gain, None)
            nt.assert_equal(AC.phs_slope_arr, None)
            nt.assert_equal(AC.phs_slope_gain_arr, None)
            nt.assert_equal(AC.phs_slope_ant_phs_arr, None)
            AC = abscal.AbsCal(self.AC.model, self.AC.data, antpos=self.ap, freqs=self.freqs)
            AC.wgts[(24, 25, 'xx')] *= 0
            AC.global_phase_slope_logcal(verbose=False, solver=solver)
            # test w/ no wgts
            AC.wgts = None
            AC.global_phase_slope_logcal(verbose=False, solver=solver)

    def test_merge_gains(self):
        self.AC.abs_amp_logcal(verbose=False)
        self.AC.TT_phs_logcal(verbose=False)
        self.AC.delay_lincal(verbose=False)
        self.AC.phs_logcal(verbose=False)
        self.AC.amp_logcal(verbose=False)
        gains = (self.AC.abs_eta_gain, self.AC.TT_Phi_gain, self.AC.abs_psi_gain,
                 self.AC.ant_dly_gain, self.AC.ant_eta_gain, self.AC.ant_phi_gain)
        gains = abscal.merge_gains(gains)
        k = (53, 'Jxx')
        nt.assert_equal(gains[k].shape, (60, 64))
        nt.assert_equal(gains[k].dtype, np.complex)
        nt.assert_almost_equal(np.abs(gains[k][0, 0]), np.abs(self.AC.abs_eta_gain[k] * self.AC.ant_eta_gain[k])[0, 0])
        nt.assert_almost_equal(np.angle(gains[k][0, 0]), np.angle(self.AC.TT_Phi_gain[k] * self.AC.abs_psi_gain[k]
                                                                  * self.AC.ant_dly_gain[k] * self.AC.ant_phi_gain[k])[0, 0])

    def test_fill_dict_nans(self):
        data = copy.deepcopy(self.AC.data)
        wgts = copy.deepcopy(self.AC.wgts)
        data[(25, 38, 'xx')][15, 20] *= np.nan
        data[(25, 38, 'xx')][20, 15] *= np.inf
        abscal.fill_dict_nans(data, wgts=wgts, nan_fill=-1, inf_fill=-2)
        nt.assert_equal(data[(25, 38, 'xx')][15, 20].real, -1)
        nt.assert_equal(data[(25, 38, 'xx')][20, 15].real, -2)
        nt.assert_almost_equal(wgts[(25, 38, 'xx')][15, 20], 0)
        nt.assert_almost_equal(wgts[(25, 38, 'xx')][20, 15], 0)
        data = copy.deepcopy(self.AC.data)
        wgts = copy.deepcopy(self.AC.wgts)
        data[(25, 38, 'xx')][15, 20] *= np.nan
        data[(25, 38, 'xx')][20, 15] *= np.inf
        abscal.fill_dict_nans(data[(25, 38, 'xx')], wgts=wgts[(25, 38, 'xx')], nan_fill=-1, inf_fill=-2, array=True)
        nt.assert_equal(data[(25, 38, 'xx')][15, 20].real, -1)
        nt.assert_equal(data[(25, 38, 'xx')][20, 15].real, -2)
        nt.assert_almost_equal(wgts[(25, 38, 'xx')][15, 20], 0)
        nt.assert_almost_equal(wgts[(25, 38, 'xx')][20, 15], 0)

    def test_mock_data(self):
        # load into pyuvdata object
        data_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        data, flgs, ap, a, f, t, l, p = io.load_vis(data_file, return_meta=True)
        wgts = odict()
        for k in flgs.keys():
            wgts[k] = (~flgs[k]).astype(np.float)
        wgts = DataContainer(wgts)
        # make mock data
        dly_slope = np.array([-1e-9, 2e-9, 0])
        model = odict()
        for i, k in enumerate(data.keys()):
            bl = np.around(ap[k[0]] - ap[k[1]], 0)
            model[k] = data[k] * np.exp(2j * np.pi * f * np.dot(dly_slope, bl))
        model = DataContainer(model)
        # setup AbsCal
        AC = abscal.AbsCal(model, data, antpos=ap, wgts=wgts, freqs=f)
        # run delay_slope_cal
        AC.delay_slope_lincal(time_avg=True, verbose=False)
        # test recovery: accuracy only checked at 10% level
        nt.assert_almost_equal(AC.dly_slope_arr[0, 0, 0, 0, 0], 1e-9, delta=1e-10)
        nt.assert_almost_equal(AC.dly_slope_arr[0, 1, 0, 0, 0], -2e-9, delta=1e-10)
        # make mock data
        abs_gain = 0.02
        TT_phi = np.array([1e-3, -1e-3, 0])
        model = odict()
        for i, k in enumerate(data.keys()):
            bl = np.around(ap[k[0]] - ap[k[1]], 0)
            model[k] = data[k] * np.exp(abs_gain + 1j * np.dot(TT_phi, bl))
        model = DataContainer(model)
        # setup AbsCal
        AC = abscal.AbsCal(model, data, antpos=ap, wgts=wgts, freqs=f)
        # run abs_amp cal
        AC.abs_amp_logcal(verbose=False)
        # run TT_phs_logcal
        AC.TT_phs_logcal(verbose=False)
        nt.assert_almost_equal(np.median(AC.abs_eta_arr[0, :, :, 0][AC.wgts[(24, 25, 'xx')].astype(np.bool)]),
                               -0.01, delta=1e-3)
        nt.assert_almost_equal(np.median(AC.TT_Phi_arr[0, 0, :, :, 0][AC.wgts[(24, 25, 'xx')].astype(np.bool)]),
                               -1e-3, delta=1e-4)
        nt.assert_almost_equal(np.median(AC.TT_Phi_arr[0, 1, :, :, 0][AC.wgts[(24, 25, 'xx')].astype(np.bool)]),
                               1e-3, delta=1e-4)


class Test_Post_Redcal_Abscal_Run:

    def setUp(self):
        self.data_file = os.path.join(DATA_PATH, 'test_input/zen.2458098.45361.HH.uvh5_downselected')
        self.redcal_file = os.path.join(DATA_PATH, 'test_input/zen.2458098.45361.HH.omni.calfits_downselected')
        self.model_files = [os.path.join(DATA_PATH, 'test_input/zen.2458042.60288.HH.uvRXLS.uvh5_downselected'),
                            os.path.join(DATA_PATH, 'test_input/zen.2458042.61034.HH.uvRXLS.uvh5_downselected')]

    def test_get_all_times_and_lsts(self):
        hd = io.HERAData(self.model_files)

        all_times, all_lsts = abscal.get_all_times_and_lsts(hd)
        nt.assert_equal(len(all_times), 120)
        nt.assert_equal(len(all_lsts), 120)
        np.testing.assert_array_equal(all_times, sorted(all_times))

        for f in hd.lsts.keys():
            hd.lsts[f] += 4.75
        all_times, all_lsts = abscal.get_all_times_and_lsts(hd, unwrap=True)
        nt.assert_true(all_lsts[-1] > 2 * np.pi)
        np.testing.assert_array_equal(all_lsts, sorted(all_lsts))
        c = abscal.get_all_times_and_lsts(hd)
        nt.assert_true(all_lsts[0] < all_lsts[-1])

        hd = io.HERAData(self.data_file)
        hd.times = hd.times[0:4] + .5
        hd.lsts = hd.lsts[0:4] + np.pi
        all_times, all_lsts = abscal.get_all_times_and_lsts(hd, solar_horizon=0.0)
        nt.assert_equal(len(all_times), 0)
        nt.assert_equal(len(all_lsts), 0)

    def test_get_d2m_time_map(self):
        hd = io.HERAData(self.data_file)
        hdm = io.HERAData(self.model_files)
        all_data_times, all_data_lsts = abscal.get_all_times_and_lsts(hd)
        all_model_times, all_model_lsts = abscal.get_all_times_and_lsts(hdm)
        d2m_time_map = abscal.get_d2m_time_map(all_data_times, all_data_lsts, all_model_times, all_model_lsts)
        for dtime, mtime in d2m_time_map.items():
            dlst = all_data_lsts[np.argwhere(all_data_times == dtime)[0][0]]
            mlst = all_model_lsts[np.argwhere(all_model_times == mtime)[0][0]]
            nt.assert_less(np.abs(dlst - mlst), np.median(np.ediff1d(all_data_lsts)))
            nt.assert_equal(np.min(np.abs(all_data_lsts - mlst)), np.abs(dlst - mlst))
            
        hd = io.HERAData(self.data_file)
        hdm = io.HERAData(self.model_files[0])
        all_data_times, all_data_lsts = abscal.get_all_times_and_lsts(hd)
        all_model_times, all_model_lsts = abscal.get_all_times_and_lsts(hdm)
        d2m_time_map = abscal.get_d2m_time_map(all_data_times, all_data_lsts, all_model_times, all_model_lsts)
        for dtime, mtime in d2m_time_map.items():
            dlst = all_data_lsts[np.argwhere(all_data_times == dtime)[0][0]]
            if mtime is None:
                for mlst in all_model_lsts:
                    nt.assert_less(np.min(np.abs(all_data_lsts - mlst)), np.abs(dlst - mlst))
            else:
                mlst = all_model_lsts[np.argwhere(all_model_times == mtime)[0][0]]
                nt.assert_less(np.abs(dlst - mlst), np.median(np.ediff1d(all_data_lsts)))
                nt.assert_equal(np.min(np.abs(all_data_lsts - mlst)), np.abs(dlst - mlst))

    def test_post_redcal_abscal(self):
        hd = io.HERAData(self.data_file)
        hdm = io.HERAData(self.model_files)
        hc = io.HERACal(self.redcal_file)
        rc_gains, rc_flags, rc_quals, rc_tot_qual = hc.read()
        all_data_times, all_data_lsts = abscal.get_all_times_and_lsts(hd)
        all_model_times, all_model_lsts = abscal.get_all_times_and_lsts(hdm)
        d2m_time_map = abscal.get_d2m_time_map(all_data_times, all_data_lsts, all_model_times, all_model_lsts)
        tinds = [0, 1, 2]
        data, flags, nsamples = hd.read(times=hd.times[tinds], polarizations=['xx'])
        model_times_to_load = [d2m_time_map[time] for time in hd.times[tinds]]
        model, model_flags, _ = io.partial_time_io(hdm, model_times_to_load, polarizations=['xx'])
        model_bls = {bl: model.antpos[bl[0]] - model.antpos[bl[1]] for bl in model.keys()}
        utils.lst_rephase(model, model_bls, model.freqs, data.lsts - model.lsts,
                          lat=hdm.telescope_location_lat_lon_alt_degrees[0], inplace=True)
        for k in flags.keys():
            if k in model_flags:
                flags[k] += model_flags[k]
        data_ants = set([ant for bl in data.keys() for ant in utils.split_bl(bl)])
        rc_gains_subset = {k: rc_gains[k][tinds, :] for k in data_ants}
        rc_flags_subset = {k: rc_flags[k][tinds, :] for k in data_ants}
        calibrate_in_place(data, rc_gains_subset, data_flags=flags, 
                           cal_flags=rc_flags_subset, gain_convention=hc.gain_convention)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            delta_gains, AC = abscal.post_redcal_abscal(model, copy.deepcopy(data), flags, rc_flags_subset, min_bl_cut=1, verbose=False)
        calibrate_in_place(data, delta_gains, data_flags=flags, 
                           cal_flags=rc_flags_subset, gain_convention=hc.gain_convention)

        for k in rc_gains.keys():
            if k[1] == 'Jxx':
                nt.assert_true(k in delta_gains)
                nt.assert_equal(delta_gains[k].shape, (3, rc_gains[k].shape[1]))
                nt.assert_true(delta_gains[k].dtype == np.complex)
        for k in AC.model.keys():
            np.testing.assert_array_equal(model[k], AC.model[k])
        for k in AC.data.keys():
            np.testing.assert_array_almost_equal(data[k][~flags[k]], AC.data[k][~flags[k]], 4)
        nt.assert_true(AC.ant_dly is None)
        nt.assert_true(AC.ant_dly_arr is None)
        nt.assert_true(AC.ant_dly_phi is None)
        nt.assert_true(AC.ant_dly_phi_arr is None)
        nt.assert_true(AC.dly_slope is not None)
        nt.assert_true(AC.dly_slope_arr is not None)
        nt.assert_true(AC.phs_slope is not None)
        nt.assert_true(AC.dly_slope_arr is not None)
        nt.assert_true(AC.abs_eta is not None)
        nt.assert_true(AC.abs_eta_arr is not None)
        nt.assert_true(AC.abs_psi is not None)
        nt.assert_true(AC.abs_psi_arr is not None)
        nt.assert_true(AC.TT_Phi is not None)
        nt.assert_true(AC.TT_Phi_arr is not None)

    def test_post_redcal_abscal_run(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hca = abscal.post_redcal_abscal_run(self.data_file, self.redcal_file, self.model_files, phs_conv_crit=1e-4, 
                                                nInt_to_load=30, verbose=False, add_to_history='testing')
        nt.assert_raises(IOError, abscal.post_redcal_abscal_run, self.data_file, self.redcal_file, self.model_files, clobber=False)
        nt.assert_true(os.path.exists(self.redcal_file.replace('.omni.', '.abs.')))
        os.remove(self.redcal_file.replace('.omni.', '.abs.'))
        ac_gains, ac_flags, ac_quals, ac_total_qual = hca.build_calcontainers()
        hcr = io.HERACal(self.redcal_file)
        rc_gains, rc_flags, rc_quals, rc_total_qual = hcr.read()

        nt.assert_true(hcr.history.replace('\n', '').replace(' ', '') in hca.history.replace('\n', '').replace(' ', ''))
        nt.assert_true('testing' in hca.history.replace('\n', '').replace(' ', ''))
        for k in rc_gains:
            nt.assert_true(k in ac_gains)
            nt.assert_equal(ac_gains[k].shape, rc_gains[k].shape)
            nt.assert_equal(ac_gains[k].dtype, complex)
        for k in rc_flags:
            nt.assert_true(k in ac_flags)
            nt.assert_equal(ac_flags[k].shape, rc_flags[k].shape)
            nt.assert_equal(ac_flags[k].dtype, bool)
            np.testing.assert_array_equal(ac_flags[k][rc_flags[k]], rc_flags[k][rc_flags[k]])
        for pol in ['Jxx', 'Jyy']:
            nt.assert_true(pol in ac_total_qual)
            nt.assert_equal(ac_total_qual[pol].shape, rc_total_qual[pol].shape)
            nt.assert_true(np.issubdtype(ac_total_qual[pol].dtype, np.floating))

        hd = io.HERAData(self.model_files[0])
        hd.read(return_data=False)
        hd.lst_array += 1
        temp_outfile = os.path.join(DATA_PATH, 'test_output/temp.uvh5')
        hd.write_uvh5(temp_outfile, clobber=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hca = abscal.post_redcal_abscal_run(self.data_file, self.redcal_file, [temp_outfile], phs_conv_crit=1e-4, 
                                                nInt_to_load=30, verbose=False, add_to_history='testing')
        nt.assert_true(os.path.exists(self.redcal_file.replace('.omni.', '.abs.')))
        np.testing.assert_array_equal(hca.total_quality_array, 0.0)
        np.testing.assert_array_equal(hca.gain_array, hcr.gain_array)
        np.testing.assert_array_equal(hca.flag_array, True)
        np.testing.assert_array_equal(hca.quality_array, 0.0)
        os.remove(self.redcal_file.replace('.omni.', '.abs.'))
        os.remove(temp_outfile)

    def test_post_redcal_abscal_argparser(self):
        sys.argv = [sys.argv[0], 'a', 'b', 'c', 'd', '--nInt_to_load', '6', '--verbose']
        a = abscal.post_redcal_abscal_argparser()
        nt.assert_equal(a.data_file, 'a')
        nt.assert_equal(a.redcal_file, 'b')
        nt.assert_equal(a.model_files[0], 'c')
        nt.assert_equal(a.model_files[1], 'd')
        nt.assert_equal(len(a.model_files), 2)
        nt.assert_equal(type(a.model_files), list)
        nt.assert_equal(a.nInt_to_load, 6)
        nt.assert_true(a.verbose)
