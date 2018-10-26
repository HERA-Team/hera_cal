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
from pyuvdata import UVCal, UVData
from pyuvdata import utils as uvutils
from collections import OrderedDict as odict
import copy
import glob

from hera_cal import io, abscal, redcal
from hera_cal.data import DATA_PATH
from hera_cal.datacontainer import DataContainer
from hera_cal.utils import split_pol


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
        keys = self.data.keys()
        k1 = (24, 25, 'xx')    # 14.6 m E-W
        i1 = keys.index(k1)
        k2 = (24, 37, 'xx')    # different
        i2 = keys.index(k2)
        k3 = (52, 53, 'xx')   # 14.6 m E-W
        i3 = keys.index(k3)
        bls = map(lambda k: abscal.Baseline(self.antpos[k[1]] - self.antpos[k[0]], tol=2.0), keys)
        bls_conj = map(lambda k: abscal.Baseline(self.antpos[k[0]] - self.antpos[k[1]], tol=2.0), keys)
        nt.assert_equal(bls[i1], bls[i1])
        nt.assert_false(bls[i1] == bls[i2])
        nt.assert_equal(bls[i1] == bls_conj[i1], 'conjugated')
        # test different yet redundant baselines still agree
        nt.assert_equal(bls[i1], bls[i3])
        # test tolerance works as expected
        bls = map(lambda k: abscal.Baseline(self.antpos[k[1]] - self.antpos[k[0]], tol=1e-4), keys)
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
        data = DataContainer(odict(map(lambda k: (k[0], self.data[k[0]]), reds[:5])))
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
        l = abscal.flatten([['hi']])
        nt.assert_equal(np.array(l).ndim, 1)

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

    def test_avg_file_across_red_bls(self):
        rd, rf, rk = abscal.avg_file_across_red_bls(self.data_file, write_miriad=False, output_data=True)
        nt.assert_raises(NotImplementedError, abscal.avg_file_across_red_bls, self.data_file, write_miriad=True)

    def test_match_times(self):
        dfiles = map(lambda f: os.path.join(DATA_PATH, f), ['zen.2458043.12552.xx.HH.uvORA',
                                                            'zen.2458043.13298.xx.HH.uvORA'])
        mfiles = map(lambda f: os.path.join(DATA_PATH, f), ['zen.2458042.12552.xx.HH.uvXA',
                                                            'zen.2458042.13298.xx.HH.uvXA'])
        # test basic execution
        relevant_mfiles = abscal.match_times(dfiles[0], mfiles)
        nt.assert_equal(len(relevant_mfiles), 2)
        # test basic execution
        relevant_mfiles = abscal.match_times(dfiles[1], mfiles)
        nt.assert_equal(len(relevant_mfiles), 1)
        # test exception
        mfiles = sorted(glob.glob(os.path.join(DATA_PATH, 'zen.2458045.*.xx.HH.uvXRAA')))
        relevant_mfiles = abscal.match_times(dfiles[0], mfiles)
        nt.assert_equal(len(relevant_mfiles), 0)

    def test_rephase_vis(self):
        dfile = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA')
        mfiles = map(lambda f: os.path.join(DATA_PATH, f), ['zen.2458042.12552.xx.HH.uvXA'])
        m, mf, mantp, mant, mfr, mt, ml, mp = io.load_vis(mfiles, return_meta=True)
        d, df, dantp, dant, dfr, dt, dl, dp = io.load_vis(dfile, return_meta=True)
        bls = odict(map(lambda k: (k, dantp[k[0]] - dantp[k[1]]), d.keys()))

        # basic execution
        new_m, new_f = abscal.rephase_vis(m, ml, dl, bls, dfr)

        k = new_m.keys()[0]
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

    def test_combine_calfits(self):
        test_file1 = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.abs.calfits')
        test_file2 = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.dly.calfits')
        # test basic execution
        if os.path.exists('ex.calfits'):
            os.remove('ex.calfits')
        abscal_funcs.combine_calfits([test_file1, test_file2], 'ex.calfits', outdir = './', overwrite = True, broadcast_flags = True)
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
        abscal_funcs.combine_calfits([test_file1, test_file2], 'ex.calfits', outdir = './', overwrite = True, broadcast_flags = False)
        nt.assert_true(os.path.exists('ex.calfits'))
        if os.path.exists('ex.calfits'):
            os.remove('ex.calfits')


class Test_AbsCal:

    def setUp(self):
        np.random.seed(0)
        # load into pyuvdata object
        self.data_fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        self.model_fname = os.path.join(DATA_PATH, "zen.2458042.12552.xx.HH.uvXA")
        self.AC = abscal.AbsCal(self.data_fname, self.model_fname, refant=24)

        # make custom gain keys
        d, fl, ap, a, f, t, l, p = io.load_vis(self.data_fname, return_meta=True, pick_data_ants=False)
        self.freq_array = f
        self.antpos = ap
        gain_pols = np.unique(map(split_pol, p))
        self.ap = ap
        self.gk = abscal.flatten(map(lambda p: map(lambda k: (k, p), a), gain_pols))
        self.freqs = f

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
        nt.assert_false((np.array(map(lambda k: np.linalg.norm(AC.bls[k]), AC.bls.keys())) > 26.0).any())
        # test bl taper
        nt.assert_true(np.median(AC.wgts[(24, 25, 'xx')]) > np.median(AC.wgts[(24, 39, 'xx')]))

    def test_abs_amp_logcal(self):
        # test execution and variable assignments
        self.AC.abs_amp_logcal(verbose=False)
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
        AC.abs_amp_logcal(verbose=False)
        AC._abs_eta_arr *= 0
        nt.assert_almost_equal(np.abs(AC.abs_eta_gain_arr[0, 0, 0, 0]), 1.0)
        # test custom gain
        g = self.AC.custom_abs_eta_gain(self.gk)
        nt.assert_equal(len(g), 47)
        # test w/ no wgts
        AC.wgts = None
        AC.abs_amp_logcal(verbose=False)

    def test_TT_phs_logcal(self):
        # test execution
        self.AC.TT_phs_logcal(verbose=False)
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
        self.AC.TT_phs_logcal(verbose=False, four_pol=True)
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
        AC.TT_phs_logcal(verbose=False)

    def test_amp_logcal(self):
        self.AC.amp_logcal(verbose=False)
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
        AC.amp_logcal(verbose=False)

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
        self.AC.delay_lincal(verbose=False, kernel=(1, 3), medfilt=False, solve_offsets=False)
        nt.assert_equal(self.AC.ant_dly[(24, 'Jxx')].shape, (60, 1))
        nt.assert_equal(self.AC.ant_dly_gain[(24, 'Jxx')].shape, (60, 64))
        nt.assert_equal(self.AC.ant_dly_arr.shape, (7, 60, 1, 1))
        nt.assert_equal(self.AC.ant_dly_gain_arr.shape, (7, 60, 64, 1))
        # test w/ offsets
        self.AC.delay_lincal(verbose=False, kernel=(1, 3), medfilt=False, solve_offsets=True)
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
        # test w/o offsets
        self.AC.global_phase_slope_logcal(verbose=False, edge_cut=31)
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
        AC.global_phase_slope_logcal(verbose=False)
        # test w/ no wgts
        AC.wgts = None
        AC.global_phase_slope_logcal(verbose=False)

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
        nt.assert_almost_equal(np.angle(gains[k][0, 0]), np.angle(self.AC.TT_Phi_gain[k] * self.AC.abs_psi_gain[k] *
                                                                  self.AC.ant_dly_gain[k] * self.AC.ant_phi_gain[k])[0, 0])

    def test_apply_gains(self):
        # test basic execution
        self.AC.abs_amp_logcal(verbose=False)
        self.AC.TT_phs_logcal(verbose=False)
        self.AC.delay_lincal(verbose=False)
        self.AC.phs_logcal(verbose=False)
        self.AC.amp_logcal(verbose=False)
        gains = (self.AC.abs_eta_gain, self.AC.TT_Phi_gain, self.AC.abs_psi_gain,
                 self.AC.ant_dly_gain, self.AC.ant_eta_gain, self.AC.ant_phi_gain)
        corr_data = abscal.apply_gains(self.AC.data, gains, gain_convention='multiply')
        nt.assert_equal(corr_data[(24, 25, 'xx')].shape, (60, 64))
        nt.assert_equal(corr_data[(24, 25, 'xx')].dtype, np.complex)
        nt.assert_almost_equal(corr_data[(24, 25, 'xx')][0, 0], (self.AC.data[(24, 25, 'xx')] *
                                                                 self.AC.abs_eta_gain[(24, 'Jxx')] * self.AC.abs_eta_gain[(25, 'Jxx')] * self.AC.ant_eta_gain[(24, 'Jxx')] *
                                                                 self.AC.ant_eta_gain[(25, 'Jxx')])[0, 0])
        corr_data = abscal.apply_gains(self.AC.data, gains, gain_convention='divide')
        nt.assert_equal(corr_data[(24, 25, 'xx')].shape, (60, 64))
        nt.assert_equal(corr_data[(24, 25, 'xx')].dtype, np.complex)
        nt.assert_almost_equal(corr_data[(24, 25, 'xx')][0, 0], (self.AC.data[(24, 25, 'xx')] /
                                                                 self.AC.abs_eta_gain[(24, 'Jxx')] / self.AC.abs_eta_gain[(25, 'Jxx')] / self.AC.ant_eta_gain[(24, 'Jxx')] /
                                                                 self.AC.ant_eta_gain[(25, 'Jxx')])[0, 0])
        # test for missing data
        gains = copy.deepcopy(self.AC.abs_eta_gain)
        del gains[(24, 'Jxx')]
        corr_data = abscal.apply_gains(self.AC.data, gains)
        nt.assert_true((24, 25, 'xx') not in corr_data)

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

    def test_fft_dly(self):
        # test basic execution
        k = (24, 25, 'xx')
        vis = self.AC.model[k] / self.AC.data[k]
        abscal.fill_dict_nans(vis, nan_fill=0.0, inf_fill=0.0, array=True)
        df = np.median(np.diff(self.AC.freqs))
        # basic execution
        dly, offset = abscal.fft_dly(vis, df, medfilt=False, solve_phase=False)
        nt.assert_equal(dly.shape, (60, 1))
        nt.assert_equal(offset, None)
        # median filtering
        dly, offset = abscal.fft_dly(vis, df, medfilt=True, solve_phase=False)
        nt.assert_equal(dly.shape, (60, 1))
        nt.assert_equal(offset, None)
        # solve phase
        dly, offset = abscal.fft_dly(vis, df, medfilt=True, solve_phase=True)
        nt.assert_equal(dly.shape, (60, 1))
        nt.assert_equal(offset.shape, (60, 1))
        # test windows and edgecut
        dly, offset = abscal.fft_dly(vis, df, medfilt=False, solve_phase=False, edge_cut=2, window='hann')
        dly, offset = abscal.fft_dly(vis, df, medfilt=False, solve_phase=False, window='blackmanharris')
        nt.assert_raises(ValueError, abscal.fft_dly, vis, df, window='foo')
        nt.assert_raises(AssertionError, abscal.fft_dly, vis, df, edge_cut=1000)
        # test mock data
        tau = np.array([1.5e-8]).reshape(1, -1)  # 15 nanoseconds
        f = np.linspace(0, 100e6, 1024)
        df = np.median(np.diff(f))
        r = np.exp(1j * 2 * np.pi * f * tau)
        dly, offset = abscal.fft_dly(r, df, medfilt=True, kernel=(1, 5), solve_phase=False)
        nt.assert_almost_equal(float(dly), 1.5e-8, delta=1e-9)

    def test_abscal_arg_parser(self):
        a = abscal.abscal_arg_parser()

    def test_omni_abscal_arg_parser(self):
        a = abscal.omni_abscal_arg_parser()

    def test_abscal_run(self):
        data_files = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        model_files = [os.path.join(DATA_PATH, "zen.2458042.12552.xx.HH.uvXA"),
                       os.path.join(DATA_PATH, "zen.2458042.13298.xx.HH.uvXA")]
        # blank run
        gains, flags = abscal.abscal_run(data_files, model_files, gen_amp_cal=True, write_calfits=False, return_gains=True, verbose=False)
        # assert shapes and types
        nt.assert_equal(gains[(24, 'Jxx')].dtype, np.complex)
        nt.assert_equal(gains[(24, 'Jxx')].shape, (60, 64))
        # first freq bin should be flagged due to complete flagging in model and data
        nt.assert_true(flags[(24, 'Jxx')][:, 0].all())
        # solar flag run
        gains, flags = abscal.abscal_run(data_files, model_files, solar_horizon=0.0, gen_amp_cal=True, write_calfits=False, return_gains=True, verbose=False)
        # all data should be flagged
        nt.assert_true(flags[(24, 'Jxx')].all())
        # write calfits
        outdir = "./"
        cf_name = "ex.calfits"
        if os.path.exists(os.path.join(outdir, cf_name)):
            os.remove(os.path.join(outdir, cf_name))
        gains, flags = abscal.abscal_run(data_files, model_files, gen_amp_cal=True, write_calfits=True, output_calfits_fname=cf_name, outdir=outdir,
                                            return_gains=True, verbose=False)
        nt.assert_true(os.path.exists(os.path.join(outdir, cf_name)))
        if os.path.exists(os.path.join(outdir, cf_name)):
            os.remove(os.path.join(outdir, cf_name))
        # check match_red_bls and reweight
        abscal.abscal_run(data_files, model_files, gen_amp_cal=True, write_calfits=False, verbose=False,
                             match_red_bls=True, reweight=True)
        # check all calibration routines
        gains, flags = abscal.abscal_run(data_files, model_files, write_calfits=False, verbose=False, return_gains=True, delay_slope_cal=True, phase_slope_cal=True,
                                            delay_cal=True, avg_phs_cal=True, abs_amp_cal=True, TT_phs_cal=True, gen_amp_cal=False, gen_phs_cal=False)
        nt.assert_equal(gains[(24, 'Jxx')].dtype, np.complex)
        nt.assert_equal(gains[(24, 'Jxx')].shape, (60, 64))
        if os.path.exists('./ex.calfits'):
            os.remove('./ex.calfits')
        # check exceptions
        nt.assert_raises(ValueError, abscal.abscal_run, data_files, model_files, all_antenna_gains=True, outdir='./',
                         output_calfits_fname='ex.calfits', abs_amp_cal=False, TT_phs_cal=False, delay_cal=True, verbose=False)
        nt.assert_raises(ValueError, abscal.abscal_run, data_files, model_files, all_antenna_gains=True, outdir='./',
                         output_calfits_fname='ex.calfits', abs_amp_cal=False, TT_phs_cal=False, gen_phs_cal=True, verbose=False)
        nt.assert_raises(ValueError, abscal.abscal_run, data_files, model_files, all_antenna_gains=True, outdir='./',
                         output_calfits_fname='ex.calfits', abs_amp_cal=False, TT_phs_cal=False, gen_amp_cal=True, verbose=False)
        if os.path.exists('./ex.calfits'):
            os.remove('./ex.calfits')
        # check all antenna gains run
        abscal.abscal_run(data_files, model_files, abs_amp_cal=True, all_antenna_gains=True, write_calfits=False)
        # test general bandpass solvers
        abscal.abscal_run(data_files, model_files, TT_phs_cal=False, abs_amp_cal=False, gen_amp_cal=True, gen_phs_cal=True, write_calfits=False)
        # test exception
        nt.assert_raises(ValueError, abscal.abscal_run, data_files, model_files, verbose=False, overwrite=True)
        # check blank & flagged calfits file written if no LST overlap
        bad_model_files = sorted(glob.glob(os.path.join(DATA_PATH, "zen.2458044.*.xx.HH.uvXRAA")))
        abscal.abscal_run(data_files, bad_model_files, write_calfits=True, overwrite=True, outdir='./',
                             output_calfits_fname='ex.calfits', verbose=False)
        uvc = UVCal()
        uvc.read_calfits('./ex.calfits')
        nt.assert_true(uvc.flag_array.min())
        nt.assert_almost_equal(uvc.gain_array.max(), 1.0)
        os.remove('./ex.calfits')
        # test w/ calfits files
        calfits_infile = os.path.join(DATA_PATH, 'zen.2458043.12552.HH.uvA.omni.calfits')
        abscal.abscal_run(data_files, model_files, calfits_infile=calfits_infile, delay_slope_cal=True, phase_slope_cal=True,
                             outdir='./', output_calfits_fname='ex.calfits', overwrite=True, verbose=False, refant=38)
        uvc = UVCal()
        uvc.read_calfits('./ex.calfits')
        nt.assert_true(uvc.total_quality_array is not None)
        nt.assert_almost_equal(uvc.quality_array[1, 0, 32, 0, 0], 12618138.92409363, places=3)
        nt.assert_true(uvc.flag_array[0].min())
        nt.assert_true(len(uvc.history) > 1000)
        # assert refant phase is zero
        nt.assert_true(np.isclose(np.angle(uvc.gain_array[uvc.ant_array.tolist().index(38)]), 0.0).all())
        os.remove('./ex.calfits')

    def test_mock_data(self):
        # load into pyuvdata object
        data_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        data, flgs, ap, a, f, t, l, p = hc.io.load_vis(data_file, return_meta=True)
        wgts = odict()
        for k in flgs.keys():
            wgts[k] = (~flgs[k]).astype(np.float)
        wgts = hc.datacontainer.DataContainer(wgts)
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
