# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import pytest
import os
from scipy import constants
import numpy as np
import sys
from collections import OrderedDict as odict
import copy
import glob
from pyuvdata import UVCal, UVData
import warnings
from hera_sim.antpos import hex_array, linear_array

from .. import io, abscal, redcal, utils
from ..data import DATA_PATH
from ..datacontainer import DataContainer
from ..utils import split_pol, reverse_bl, split_bl
from ..apply_cal import calibrate_in_place
from ..flag_utils import synthesize_ant_flags


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
@pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
class Test_AbsCal_Funcs(object):
    def setup_method(self):
        np.random.seed(0)

        # load into pyuvdata object
        self.data_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        self.uvd = UVData()
        self.uvd.read_miriad(self.data_file)
        self.uvd.use_future_array_shapes()
        self.freq_array = np.unique(self.uvd.freq_array)
        self.antpos, self.ants = self.uvd.get_ENU_antpos(center=True, pick_data_ants=True)
        self.antpos = odict(zip(self.ants, self.antpos))
        self.time_array = np.unique(self.uvd.time_array)

        # configure data into dictionaries
        data, flgs = io.load_vis(self.uvd, pop_autos=True)
        wgts = odict()
        for k in flgs.keys():
            wgts[k] = (~flgs[k]).astype(float)
        wgts = DataContainer(wgts)

        # configure baselines
        bls = odict([(x, self.antpos[x[0]] - self.antpos[x[1]]) for x in data.keys()])

        # make mock data
        abs_gain = 0.5
        TT_phi = np.array([-0.004, 0.006, 0])
        model = DataContainer({})
        for i, k in enumerate(data.keys()):
            model[k] = data[k] * np.exp(abs_gain + 1j * np.dot(TT_phi, bls[k]))

        # assign data
        self.data = data
        self.bls = bls
        self.model = model
        self.wgts = wgts

    @pytest.mark.parametrize("divide_gains", [True, False])
    def test_multiply_gains(self, tmpdir, divide_gains):
        tmp_path = tmpdir.strpath
        gain_1_path = os.path.join(tmp_path, 'gain_1.calfits')
        gain_2_path = os.path.join(tmp_path, 'gain_2.calfits')
        output_path = os.path.join(tmp_path, 'output.calfits')
        uvc1 = UVCal()
        uvc1 = uvc1.initialize_from_uvdata(self.uvd, gain_convention='divide', future_array_shapes=False,
                                           cal_style='redundant', metadata_only=False)
        uvc2 = UVCal()
        uvc2 = uvc2.initialize_from_uvdata(self.uvd, gain_convention='divide', future_array_shapes=False,
                                           cal_style='redundant', metadata_only=False)
        uvc1.gain_array[:] = np.random.rand(*uvc1.gain_array.shape) + 1j * np.random.rand(*uvc1.gain_array.shape)
        uvc2.gain_array[:] = np.random.rand(*uvc2.gain_array.shape) + 1j * np.random.rand(*uvc2.gain_array.shape)

        flag_times_1 = np.random.randint(low=0, high=self.uvd.Ntimes, size=self.uvd.Ntimes // 4)
        uvc1.flag_array[:, :, flag_times_1] = True
        flag_times_2 = np.random.randint(low=0, high=self.uvd.Ntimes, size=self.uvd.Ntimes // 4)
        uvc2.flag_array[:, :, flag_times_2] = True

        uvc1.quality_array = np.zeros_like(uvc1.gain_array, dtype=float) + 1.
        uvc2.quality_array = np.zeros_like(uvc1.quality_array) + 2.

        uvc1.write_calfits(gain_1_path, clobber=True)
        uvc2.write_calfits(gain_2_path, clobber=True)

        abscal.multiply_gains(gain_1_path, gain_2_path, output_path,
                              clobber=True, divide_gains=divide_gains)

        uvc3 = UVCal()
        uvc3.read_calfits(output_path)
        if divide_gains:
            np.testing.assert_array_almost_equal(uvc1.gain_array / uvc2.gain_array, uvc3.gain_array)
        else:
            np.testing.assert_array_almost_equal(uvc1.gain_array * uvc2.gain_array, uvc3.gain_array)
        np.testing.assert_array_almost_equal(uvc1.flag_array | uvc2.flag_array, uvc3.flag_array)
        assert np.all(np.isnan(uvc3.quality_array))
        assert uvc3.total_quality_array is None

    def test_data_key_to_array_axis(self):
        m, pk = abscal.data_key_to_array_axis(self.model, 2)
        assert m[(24, 25)].shape == (60, 64, 1)
        assert 'ee' in pk
        # test w/ avg_dict
        m, ad, pk = abscal.data_key_to_array_axis(self.model, 2, avg_dict=self.bls)
        assert m[(24, 25)].shape == (60, 64, 1)
        assert ad[(24, 25)].shape == (3,)
        assert 'ee' in pk

    def test_array_axis_to_data_key(self):
        m, pk = abscal.data_key_to_array_axis(self.model, 2)
        m2 = abscal.array_axis_to_data_key(m, 2, ['ee'])
        assert m2[(24, 25, 'ee')].shape == (60, 64)
        # copy dict
        m, ad, pk = abscal.data_key_to_array_axis(self.model, 2, avg_dict=self.bls)
        m2, cd = abscal.array_axis_to_data_key(m, 2, ['ee'], copy_dict=ad)
        assert m2[(24, 25, 'ee')].shape == (60, 64)
        assert cd[(24, 25, 'ee')].shape == (3,)

    def test_interp2d(self):
        # test interpolation w/ warning
        m, mf = abscal.interp2d_vis(self.data, self.time_array, self.freq_array,
                                    self.time_array, self.freq_array, flags=self.wgts, medfilt_flagged=False)
        assert m[(24, 25, 'ee')].shape == (60, 64)
        # downsampling w/ no flags
        m, mf = abscal.interp2d_vis(self.data, self.time_array, self.freq_array,
                                    self.time_array[::2], self.freq_array[::2])
        assert m[(24, 25, 'ee')].shape == (30, 32)
        # test flag propagation
        m, mf = abscal.interp2d_vis(self.data, self.time_array, self.freq_array,
                                    self.time_array, self.freq_array, flags=self.wgts, medfilt_flagged=True)
        assert np.all(mf[(24, 25, 'ee')][10, 0])
        # test flag extrapolation
        m, mf = abscal.interp2d_vis(self.data, self.time_array, self.freq_array,
                                    self.time_array + .0001, self.freq_array, flags=self.wgts, flag_extrapolate=True)
        assert np.all(mf[(24, 25, 'ee')][-1].min())

    def test_wiener(self):
        # test smoothing
        d = abscal.wiener(self.data, window=(5, 15), noise=None, medfilt=True, medfilt_kernel=(1, 13))
        assert d[(24, 37, 'ee')].shape == (60, 64)
        assert d[(24, 37, 'ee')].dtype == complex
        # test w/ noise
        d = abscal.wiener(self.data, window=(5, 15), noise=0.1, medfilt=True, medfilt_kernel=(1, 13))
        assert d[(24, 37, 'ee')].shape == (60, 64)
        # test w/o medfilt
        d = abscal.wiener(self.data, window=(5, 15), medfilt=False)
        assert d[(24, 37, 'ee')].shape == (60, 64)
        # test as array
        d = abscal.wiener(self.data[(24, 37, 'ee')], window=(5, 15), medfilt=False, array=True)
        assert d.shape == (60, 64)
        assert d.dtype == complex

    def test_Baseline(self):
        # test basic execution
        keys = list(self.data.keys())
        k1 = (24, 25, 'ee')    # 14.6 m E-W
        i1 = keys.index(k1)
        k2 = (24, 37, 'ee')    # different
        i2 = keys.index(k2)
        k3 = (52, 53, 'ee')   # 14.6 m E-W
        i3 = keys.index(k3)
        bls = [abscal.Baseline(self.antpos[k[1]] - self.antpos[k[0]], tol=2.0) for k in keys]
        bls_conj = [abscal.Baseline(self.antpos[k[0]] - self.antpos[k[1]], tol=2.0) for k in keys]
        assert bls[i1] == bls[i1]
        assert bls[i1] != bls[i2]
        assert (bls[i1] == bls_conj[i1]) == 'conjugated'
        # test different yet redundant baselines still agree
        assert bls[i1] == bls[i3]
        # test tolerance works as expected
        bls = [abscal.Baseline(self.antpos[k[1]] - self.antpos[k[0]], tol=1e-4) for k in keys]
        assert bls[i1] != bls[i3]

    def test_match_red_baselines(self):
        model = copy.deepcopy(self.data)
        model = DataContainer(odict([((k[0] + 1, k[1] + 1, k[2]), model[k]) for i, k in enumerate(model.keys())]))
        del model[(25, 54, 'ee')]
        model_antpos = odict([(k + 1, self.antpos[k]) for i, k in enumerate(self.antpos.keys())])
        new_model = abscal.match_red_baselines(model, model_antpos, self.data, self.antpos, tol=2.0, verbose=False)
        assert len(new_model.keys()) == 8
        assert (24, 37, 'ee') in new_model
        assert (24, 53, 'ee') not in new_model

    def test_mirror_data_to_red_bls(self):
        # make fake data
        reds = redcal.get_reds(self.antpos, pols=['ee'])
        data = DataContainer(odict([(k[0], self.data[k[0]]) for k in reds[:5]]))
        # test execuation
        d = abscal.mirror_data_to_red_bls(data, self.antpos)
        assert len(d.keys()) == 16
        assert (24, 25, 'ee') in d
        # test correct value is propagated
        assert np.allclose(data[(24, 25, 'ee')][30, 30], d[(38, 39, 'ee')][30, 30])
        # test reweighting
        w = abscal.mirror_data_to_red_bls(self.wgts, self.antpos, weights=True)
        assert w[(24, 25, 'ee')].dtype == float
        assert np.allclose(w[(24, 25, 'ee')].max(), 16.0)

    def test_flatten(self):
        li = abscal.flatten([['hi']])
        assert np.array(li).ndim == 1

    @pytest.mark.filterwarnings("ignore:Casting complex values to real discards the imaginary part")
    def test_avg_data_across_red_bls(self):
        # test basic execution
        wgts = copy.deepcopy(self.wgts)
        wgts[(24, 25, 'ee')][45, 45] = 0.0
        data, flags, antpos, ants, freqs, times, lsts, pols = io.load_vis(self.data_file, return_meta=True)
        rd, rf, rk = abscal.avg_data_across_red_bls(data, antpos, wgts=wgts, tol=2.0, broadcast_wgts=False)
        assert rd[(24, 25, 'ee')].shape == (60, 64)
        assert rf[(24, 25, 'ee')][45, 45] > 0.0
        # test various kwargs
        wgts[(24, 25, 'ee')][45, 45] = 0.0
        rd, rf, rk = abscal.avg_data_across_red_bls(data, antpos, tol=2.0, wgts=wgts, broadcast_wgts=True)
        assert len(rd.keys()) == 9
        assert len(rf.keys()) == 9
        assert np.allclose(rf[(24, 25, 'ee')][45, 45], 0.0)
        # test averaging worked
        rd, rf, rk = abscal.avg_data_across_red_bls(data, antpos, tol=2.0, broadcast_wgts=False)
        v = np.mean([data[(52, 53, 'ee')], data[(37, 38, 'ee')], data[(24, 25, 'ee')], data[(38, 39, 'ee')]], axis=0)
        assert np.allclose(rd[(24, 25, 'ee')], v)
        # test mirror_red_data
        rd, rf, rk = abscal.avg_data_across_red_bls(data, antpos, wgts=self.wgts, tol=2.0, mirror_red_data=True)
        assert len(rd.keys()) == 21
        assert len(rf.keys()) == 21

    def test_match_times(self):
        dfiles =[os.path.join(DATA_PATH, f'zen.2458043.{f}.xx.HH.uvORA') for f in (12552, 13298)]
        mfiles =[os.path.join(DATA_PATH, f'zen.2458042.{f}.xx.HH.uvXA') for f in (12552, 13298)]

        # test basic execution
        relevant_mfiles = abscal.match_times(dfiles[0], mfiles, filetype='miriad')
        assert len(relevant_mfiles) == 2
        # test basic execution
        relevant_mfiles = abscal.match_times(dfiles[1], mfiles, filetype='miriad')
        assert len(relevant_mfiles) == 1
        # test no overlap
        mfiles = sorted(glob.glob(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA')))
        relevant_mfiles = abscal.match_times(dfiles[0], mfiles, filetype='miriad')
        assert len(relevant_mfiles) == 0

    def test_rephase_vis(self):
        dfile = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA')
        mfiles = [os.path.join(DATA_PATH, 'zen.2458042.12552.xx.HH.uvXA')]
        m, mf, mantp, mant, mfr, mt, ml, mp = io.load_vis(mfiles, return_meta=True)
        d, df, dantp, dant, dfr, dt, dl, dp = io.load_vis(dfile, return_meta=True)
        bls = odict([(k, dantp[k[0]] - dantp[k[1]]) for k in d.keys()])

        # basic execution
        new_m, new_f = abscal.rephase_vis(m, ml, dl, bls, dfr)

        k = list(new_m.keys())[0]
        assert new_m[k].shape == d[k].shape
        assert np.all(new_f[k][-1])
        assert not np.any(new_f[k][0])

    def test_cut_bl(self):
        Nbls = len(self.data)
        _data = abscal.cut_bls(self.data, bls=self.bls, min_bl_cut=20.0, inplace=False)
        assert Nbls == 21
        assert len(_data) == 9
        _data2 = copy.deepcopy(self.data)
        abscal.cut_bls(_data2, bls=self.bls, min_bl_cut=20.0, inplace=True)
        assert len(_data2) == 9
        _data = abscal.cut_bls(self.data, bls=self.bls, min_bl_cut=20.0, inplace=False)
        abscal.cut_bls(_data2, min_bl_cut=20.0, inplace=True)
        assert len(_data2) == 9

    def test_dft_phase_slope_solver(self):
        np.random.seed(21)

        # build a perturbed grid
        xs = np.zeros(100)
        ys = np.zeros(100)
        i = 0
        for x in np.arange(0, 100, 10):
            for y in np.arange(0, 100, 10):
                xs[i] = x + 5 * (.5 - np.random.rand())
                ys[i] = y + 5 * (.5 - np.random.rand())
                i += 1

        phase_slopes_x = (.2 * np.random.rand(5, 2) - .1)  # not too many phase wraps over the array
        phase_slopes_y = (.2 * np.random.rand(5, 2) - .1)  # (i.e. avoid undersampling of very fast slopes)
        data = np.array([np.exp(1.0j * x * phase_slopes_x
                                + 1.0j * y * phase_slopes_y) for x, y in zip(xs, ys)])

        x_slope_est, y_slope_est = abscal.dft_phase_slope_solver(xs, ys, data)
        np.testing.assert_array_almost_equal(phase_slopes_x - x_slope_est, 0, decimal=7)
        np.testing.assert_array_almost_equal(phase_slopes_y - y_slope_est, 0, decimal=7)


@pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
@pytest.mark.filterwarnings("ignore:divide by zero encountered in true_divide")
@pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
class Test_Abscal_Solvers(object):
    def test_abs_amp_lincal_1pol(self):
        antpos = hex_array(2, split_core=False, outriggers=0)
        reds = redcal.get_reds(antpos, pols=['ee'], pol_mode='1pol')
        model = {bl: np.ones((10, 5)) for red in reds for bl in red}
        data = {bl: 4.0 * np.ones((10, 5)) for red in reds for bl in red}
        data[0, 1, 'ee'][0, 0] = np.nan
        data[0, 1, 'ee'][0, 1] = np.inf
        model[0, 1, 'ee'][0, 0] = np.nan
        model[0, 1, 'ee'][0, 1] = np.inf
        fit = abscal.abs_amp_lincal(model, data)
        np.testing.assert_array_equal(fit['A_Jee'], 2.0)
        ants = list(set([ant for bl in data for ant in utils.split_bl(bl)]))
        gains = abscal.abs_amp_lincal(model, data, return_gains=True, gain_ants=ants)
        for ant in ants:
            np.testing.assert_array_equal(gains[ant], 2.0)

    def test_abs_amp_lincal_4pol(self):
        antpos = hex_array(2, split_core=False, outriggers=0)
        reds = redcal.get_reds(antpos, pols=['ee', 'en', 'ne', 'nn'], pol_mode='4pol')
        model = {bl: np.ones((10, 5)) for red in reds for bl in red}
        gain_products = {'ee': 4.0, 'en': 6.0, 'ne': 6.0, 'nn': 9.0}
        data = {bl: gain_products[bl[2]] * np.ones((10, 5)) for red in reds for bl in red}
        data[0, 1, 'ee'][0, 0] = np.nan
        data[0, 1, 'ee'][0, 1] = np.inf
        model[0, 1, 'ee'][0, 0] = np.nan
        model[0, 1, 'ee'][0, 1] = np.inf
        fit = abscal.abs_amp_lincal(model, data)
        np.testing.assert_array_equal(fit['A_Jee'], 2.0)
        np.testing.assert_array_equal(fit['A_Jnn'], 3.0)
        ants = list(set([ant for bl in data for ant in utils.split_bl(bl)]))
        gains = abscal.abs_amp_lincal(model, data, return_gains=True, gain_ants=ants)
        for ant in ants:
            if ant[1] == 'Jee':
                np.testing.assert_array_equal(gains[ant], 2.0)
            elif ant[1] == 'Jnn':
                np.testing.assert_array_equal(gains[ant], 3.0)

    def test_TT_phs_logcal_1pol_assume_2D(self):
        antpos = hex_array(2, split_core=False, outriggers=0)
        reds = redcal.get_reds(antpos, pols=['ee'], pol_mode='1pol')
        model = {bl: np.ones((10, 5), dtype=complex) for red in reds for bl in red}
        data = {bl: np.ones((10, 5), dtype=complex) for red in reds for bl in red}
        bl_vecs = {bl: antpos[bl[0]] - antpos[bl[1]] for bl in data}
        for bl in data:
            data[bl] *= np.exp(1.0j * np.dot(bl_vecs[bl], [.01, .02, 0]))
        data[0, 1, 'ee'][0, 0] = np.nan
        data[0, 1, 'ee'][0, 1] = np.inf
        model[0, 1, 'ee'][0, 0] = np.nan
        model[0, 1, 'ee'][0, 1] = np.inf
        fit = abscal.TT_phs_logcal(model, data, antpos, assume_2D=True)
        np.testing.assert_array_almost_equal(fit['Phi_ew_Jee'], .01)
        np.testing.assert_array_almost_equal(fit['Phi_ns_Jee'], .02)

        ants = list(set([ant for bl in data for ant in utils.split_bl(bl)]))
        gains = abscal.TT_phs_logcal(model, data, antpos, assume_2D=True, return_gains=True, gain_ants=ants)
        rephased_gains = {ant: gains[ant] / gains[ants[0]] * np.abs(gains[ants[0]]) for ant in ants}
        true_gains = {ant: np.exp(1.0j * np.dot(antpos[ant[0]], [.01, .02, 0])) for ant in ants}
        rephased_true_gains = {ant: true_gains[ant] / true_gains[ants[0]] * np.abs(true_gains[ants[0]]) for ant in ants}
        for ant in ants:
            np.testing.assert_array_almost_equal(rephased_gains[ant], rephased_true_gains[ant])

    def test_TT_phs_logcal_4pol_assume_2D(self):
        antpos = hex_array(2, split_core=False, outriggers=0)
        reds = redcal.get_reds(antpos, pols=['ee', 'en', 'ne', 'nn'], pol_mode='4pol')
        model = {bl: np.ones((10, 5), dtype=complex) for red in reds for bl in red}
        data = {bl: np.ones((10, 5), dtype=complex) for red in reds for bl in red}
        bl_vecs = {bl: antpos[bl[0]] - antpos[bl[1]] for bl in data}
        for bl in data:
            data[bl] *= np.exp(1.0j * np.dot(bl_vecs[bl], [.01, .02, 0]))
        data[0, 1, 'ee'][0, 0] = np.nan
        data[0, 1, 'ee'][0, 1] = np.inf
        model[0, 1, 'ee'][0, 0] = np.nan
        model[0, 1, 'ee'][0, 1] = np.inf
        fit = abscal.TT_phs_logcal(model, data, antpos, assume_2D=True, four_pol=True)
        np.testing.assert_array_almost_equal(fit['Phi_ew'], .01)
        np.testing.assert_array_almost_equal(fit['Phi_ns'], .02)

        ants = list(set([ant for bl in data for ant in utils.split_bl(bl)]))
        gains = abscal.TT_phs_logcal(model, data, antpos, assume_2D=True, four_pol=True, return_gains=True, gain_ants=ants)
        rephased_gains = {ant: gains[ant] / gains[ants[0]] * np.abs(gains[ants[0]]) for ant in ants}
        true_gains = {ant: np.exp(1.0j * np.dot(antpos[ant[0]], [.01, .02, 0])) for ant in ants}
        rephased_true_gains = {ant: true_gains[ant] / true_gains[ants[0]] * np.abs(true_gains[ants[0]]) for ant in ants}
        for ant in ants:
            np.testing.assert_array_almost_equal(rephased_gains[ant], rephased_true_gains[ant])

    def test_TT_phs_logcal_1pol_nDim(self):
        # test assume_2D=False by introducing another 6 element hex 100 m away
        antpos = hex_array(2, split_core=False, outriggers=0)
        antpos2 = hex_array(2, split_core=False, outriggers=0)
        antpos.update({len(antpos) + ant: antpos2[ant] + np.array([100, 0, 0]) for ant in antpos2})
        reds = redcal.get_reds(antpos, pols=['ee'], pol_mode='1pol')
        antpos = redcal.reds_to_antpos(reds)
        reds = redcal.get_reds(antpos, pols=['ee'], pol_mode='1pol', bl_error_tol=1e-10)
        model = {bl: np.ones((10, 5), dtype=complex) for red in reds for bl in red}
        data = {bl: np.ones((10, 5), dtype=complex) for red in reds for bl in red}
        bl_vecs = {bl: antpos[bl[0]] - antpos[bl[1]] for bl in data}

        for bl in data:
            data[bl] *= np.exp(1.0j * np.dot(bl_vecs[bl], [.01, .02, .03]))
        data[0, 1, 'ee'][0, 0] = np.nan
        data[0, 1, 'ee'][0, 1] = np.inf
        model[0, 1, 'ee'][0, 0] = np.nan
        model[0, 1, 'ee'][0, 1] = np.inf
        fit = abscal.TT_phs_logcal(model, data, antpos, assume_2D=False)
        np.testing.assert_array_almost_equal(fit['Phi_0_Jee'], .01)
        np.testing.assert_array_almost_equal(fit['Phi_1_Jee'], .02)
        np.testing.assert_array_almost_equal(fit['Phi_2_Jee'], .03)

        ants = list(set([ant for bl in data for ant in utils.split_bl(bl)]))
        gains = abscal.TT_phs_logcal(model, data, antpos, assume_2D=False, return_gains=True, gain_ants=ants)
        rephased_gains = {ant: gains[ant] / gains[ants[0]] * np.abs(gains[ants[0]]) for ant in ants}
        true_gains = {ant: np.exp(1.0j * np.dot(antpos[ant[0]], [.01, .02, .03])) for ant in ants}
        rephased_true_gains = {ant: true_gains[ant] / true_gains[ants[0]] * np.abs(true_gains[ants[0]]) for ant in ants}
        for ant in ants:
            np.testing.assert_array_almost_equal(rephased_gains[ant], rephased_true_gains[ant])

    def test_delay_slope_lincal_1pol_assume_2D(self):
        antpos = hex_array(2, split_core=False, outriggers=0)
        reds = redcal.get_reds(antpos, pols=['ee'], pol_mode='1pol')
        model = {bl: np.ones((2, 1024), dtype=complex) for red in reds for bl in red}
        data = {bl: np.ones((2, 1024), dtype=complex) for red in reds for bl in red}
        freqs = np.linspace(100e6, 200e6, 1024)
        df = np.median(np.diff(freqs))

        ants = sorted(list(set([ant for bl in data for ant in utils.split_bl(bl)])))
        true_dlys = {ant: np.dot([1e-9, 2e-9, 0], antpos[ant[0]]) for ant in ants}
        true_gains = {ant: np.outer(np.ones(2), np.exp(2.0j * np.pi * true_dlys[ant] * (freqs))) for ant in ants}

        for bl in data:
            ant0, ant1 = utils.split_bl(bl)
            data[bl] *= true_gains[ant0] * np.conj(true_gains[ant1])

        fit = abscal.delay_slope_lincal(model, data, antpos, df=df, assume_2D=True, time_avg=True)
        np.testing.assert_array_almost_equal(1e9 * fit['T_ew_Jee'], 1.0, decimal=3)
        np.testing.assert_array_almost_equal(1e9 * fit['T_ns_Jee'], 2.0, decimal=3)

        gains = abscal.delay_slope_lincal(model, data, antpos, df=df, f0=freqs[0], assume_2D=True, time_avg=True, return_gains=True, gain_ants=ants)
        rephased_gains = {ant: gains[ant] / gains[ants[0]] * np.abs(gains[ants[0]]) for ant in ants}
        rephased_true_gains = {ant: true_gains[ant] / true_gains[ants[0]] * np.abs(true_gains[ants[0]]) for ant in ants}
        for ant in ants:
            np.testing.assert_array_almost_equal(rephased_gains[ant], rephased_true_gains[ant], decimal=3)

    def test_delay_slope_lincal_4pol_assume_2D(self):
        antpos = hex_array(2, split_core=False, outriggers=0)
        reds = redcal.get_reds(antpos, pols=['ee', 'en', 'ne', 'nn'], pol_mode='4pol')
        model = {bl: np.ones((2, 1024), dtype=complex) for red in reds for bl in red}
        data = {bl: np.ones((2, 1024), dtype=complex) for red in reds for bl in red}
        freqs = np.linspace(100e6, 200e6, 1024)
        df = np.median(np.diff(freqs))

        ants = sorted(list(set([ant for bl in data for ant in utils.split_bl(bl)])))
        true_dlys = {ant: np.dot([1e-9, 2e-9, 0], antpos[ant[0]]) for ant in ants}
        true_gains = {ant: np.outer(np.ones(2), np.exp(2.0j * np.pi * true_dlys[ant] * (freqs))) for ant in ants}

        for bl in data:
            ant0, ant1 = utils.split_bl(bl)
            data[bl] *= true_gains[ant0] * np.conj(true_gains[ant1])

        fit = abscal.delay_slope_lincal(model, data, antpos, df=df, assume_2D=True, four_pol=True)
        np.testing.assert_array_almost_equal(1e9 * fit['T_ew'], 1.0, decimal=3)
        np.testing.assert_array_almost_equal(1e9 * fit['T_ns'], 2.0, decimal=3)

        gains = abscal.delay_slope_lincal(model, data, antpos, df=df, f0=freqs[0], assume_2D=True, four_pol=True, return_gains=True, gain_ants=ants)
        rephased_gains = {ant: gains[ant] / gains[ants[0]] * np.abs(gains[ants[0]]) for ant in ants}
        rephased_true_gains = {ant: true_gains[ant] / true_gains[ants[0]] * np.abs(true_gains[ants[0]]) for ant in ants}
        for ant in ants:
            np.testing.assert_array_almost_equal(rephased_gains[ant], rephased_true_gains[ant], decimal=3)

    def test_delay_slope_lincal_1pol_nDim(self):
        antpos = hex_array(2, split_core=False, outriggers=0)
        antpos2 = hex_array(2, split_core=False, outriggers=0)
        antpos.update({len(antpos) + ant: antpos2[ant] + np.array([100, 0, 0]) for ant in antpos2})
        reds = redcal.get_reds(antpos, pols=['ee'], pol_mode='1pol')
        antpos = redcal.reds_to_antpos(reds)
        reds = redcal.get_reds(antpos, pols=['ee'], pol_mode='1pol', bl_error_tol=1e-10)
        model = {bl: np.ones((2, 1024), dtype=complex) for red in reds for bl in red}
        data = {bl: np.ones((2, 1024), dtype=complex) for red in reds for bl in red}
        freqs = np.linspace(100e6, 200e6, 1024)
        df = np.median(np.diff(freqs))

        ants = sorted(list(set([ant for bl in data for ant in utils.split_bl(bl)])))
        true_dlys = {ant: np.dot([1e-9, 2e-9, 3e-9], antpos[ant[0]]) for ant in ants}
        true_gains = {ant: np.outer(np.ones(2), np.exp(2.0j * np.pi * true_dlys[ant] * (freqs))) for ant in ants}

        for bl in data:
            ant0, ant1 = utils.split_bl(bl)
            data[bl] *= true_gains[ant0] * np.conj(true_gains[ant1])

        fit = abscal.delay_slope_lincal(model, data, antpos, df=df, assume_2D=False)
        np.testing.assert_array_almost_equal(1e9 * fit['T_0_Jee'], 1.0, decimal=3)
        np.testing.assert_array_almost_equal(1e9 * fit['T_1_Jee'], 2.0, decimal=3)
        np.testing.assert_array_almost_equal(1e9 * fit['T_2_Jee'], 3.0, decimal=3)

        gains = abscal.delay_slope_lincal(model, data, antpos, df=df, f0=freqs[0], assume_2D=False, return_gains=True, gain_ants=ants)
        rephased_gains = {ant: gains[ant] / gains[ants[0]] * np.abs(gains[ants[0]]) for ant in ants}
        rephased_true_gains = {ant: true_gains[ant] / true_gains[ants[0]] * np.abs(true_gains[ants[0]]) for ant in ants}
        for ant in ants:
            np.testing.assert_array_almost_equal(rephased_gains[ant], rephased_true_gains[ant], decimal=3)

    def test_RFI_delay_slope_cal(self):
        # build array
        antpos = hex_array(3, split_core=False, outriggers=0)
        antpos[19] = np.array([101, 102, 0])
        reds = redcal.get_reds(antpos, pols=['ee', 'nn'])
        red_data = DataContainer({red[0]: np.ones((5, 128), dtype=complex) for red in reds})
        freqs = np.linspace(50e6, 250e6, 128)
        unique_blvecs = {red[0]: np.mean([antpos[bl[1]] - antpos[bl[0]] for bl in red], axis=0) for red in reds}
        idealized_antpos = redcal.reds_to_antpos(reds)
        idealized_blvecs = {red[0]: idealized_antpos[red[0][1]] - idealized_antpos[red[0][0]] for red in reds}

        # Invent RFI stations and delay slopes
        rfi_chans = [7, 9, 12, 13, 22, 31, 33]
        rfi_angles = [0.7853981, 0.7853981, 0.7853981, 6.0632738, 6.0632738, 0.7853981, 6.0632738]
        rfi_headings = np.array([np.cos(rfi_angles), np.sin(rfi_angles), np.zeros_like(rfi_angles)])
        rfi_wgts = np.array([1, 2, 1, 3, 1, 5, 1])
        true_delay_slopes = {'T_ee_0': 1e-9, 'T_ee_1': -2e-9, 'T_ee_2': 1.5e-9,
                             'T_nn_0': 1.8e-9, 'T_nn_1': -5e-9, 'T_nn_2': 3.5e-9}

        # Add RFI and uncalibrate
        for bl in red_data:
            for chan, heading in zip(rfi_chans, rfi_headings.T):
                red_data[bl][:, chan] = 100 * np.exp(2j * np.pi * np.dot(unique_blvecs[bl], heading) * freqs[chan] / constants.c)
            for key, slope in true_delay_slopes.items():
                if key[2:4] == bl[2]:
                    red_data[bl] *= np.exp(-2j * np.pi * idealized_blvecs[bl][int(key[-1])] * slope * freqs)

        # Solve for delay slopes
        solved_dly_slopes = abscal.RFI_delay_slope_cal(reds, antpos, red_data, freqs, rfi_chans, rfi_headings, rfi_wgts=rfi_wgts)
        for key, slope in solved_dly_slopes.items():
            assert np.all(np.abs((slope - true_delay_slopes[key]) / true_delay_slopes[key]) < 1e-10)

        # test converting slopes to gains
        ants_in_reds = set([ant for red in reds for bl in red for ant in split_bl(bl)])
        gains = abscal.RFI_delay_slope_cal(reds, antpos, red_data, freqs, rfi_chans, rfi_headings, rfi_wgts=rfi_wgts,
                                           return_gains=True, gain_ants=ants_in_reds)
        # test showing that non-RFI contaminated channels have been returned to 1s
        calibrate_in_place(red_data, gains)
        not_rfi_chans = [i for i in range(128) if i not in rfi_chans]
        for bl in red_data:
            np.testing.assert_almost_equal(red_data[bl][:, not_rfi_chans], 1.0, decimal=10)

        with pytest.raises(NotImplementedError):
            reds = redcal.get_reds(antpos, pols=['ee', 'nn', 'en', 'ne'])
            solved_dly_slopes = abscal.RFI_delay_slope_cal(reds, antpos, red_data, freqs, rfi_chans, rfi_headings, rfi_wgts=rfi_wgts)

    def test_ndim_fft_phase_slope_solver_1D_ideal_antpos(self):
        antpos = linear_array(50)
        reds = redcal.get_reds(antpos, pols=['ee'], pol_mode='1pol')
        model = {red[0]: np.ones((2, 3), dtype=complex) for red in reds}
        data = {red[0]: np.ones((2, 3), dtype=complex) for red in reds}
        antpos = redcal.reds_to_antpos(reds)
        bl_vecs = {bl: (antpos[bl[0]] - antpos[bl[1]]) for bl in data}
        for bl in data:
            data[bl] *= np.exp(1.0j * np.dot(bl_vecs[bl], [-1.2]))

        phase_slopes = abscal.ndim_fft_phase_slope_solver(data, bl_vecs, assume_2D=False, zero_pad=3, bl_error_tol=1e-8)
        for ps, answer in zip(phase_slopes, [-1.2]):
            assert ps.shape == (2, 3)
            np.testing.assert_array_less(np.abs(ps - answer), .1)

    def test_ndim_fft_phase_slope_solver_2D_ideal_antpos(self):
        antpos = hex_array(6, split_core=False, outriggers=0)
        reds = redcal.get_reds(antpos, pols=['ee'], pol_mode='1pol')
        model = {red[0]: np.ones((2, 3), dtype=complex) for red in reds}
        data = {red[0]: np.ones((2, 3), dtype=complex) for red in reds}
        antpos = redcal.reds_to_antpos(reds)
        bl_vecs = {bl: (antpos[bl[0]] - antpos[bl[1]]) for bl in data}
        for bl in data:
            data[bl] *= np.exp(1.0j * np.dot(bl_vecs[bl], [-1, .1]))

        phase_slopes = abscal.ndim_fft_phase_slope_solver(data, bl_vecs, assume_2D=False, zero_pad=3, bl_error_tol=1e-8)
        for ps, answer in zip(phase_slopes, [-1, .1]):
            assert ps.shape == (2, 3)
            np.testing.assert_array_less(np.abs(ps - answer), .2)

    def test_ndim_fft_phase_slope_solver_3D_ideal_antpos(self):
        antpos = hex_array(4, split_core=False, outriggers=0)
        antpos2 = hex_array(4, split_core=False, outriggers=0)
        for d in [100.0, 200.0, 300.0, 400.0]:
            antpos.update({len(antpos) + ant: antpos2[ant] + np.array([d, 0, 0]) for ant in antpos2})
        reds = redcal.get_reds(antpos, pols=['ee'], pol_mode='1pol')
        model = {red[0]: np.ones((2, 3), dtype=complex) for red in reds}
        data = {red[0]: np.ones((2, 3), dtype=complex) for red in reds}
        antpos = redcal.reds_to_antpos(reds)
        bl_vecs = {bl: (antpos[bl[0]] - antpos[bl[1]]) for bl in data}
        for bl in data:
            data[bl] *= np.exp(1.0j * np.dot(bl_vecs[bl], [-1, -.1, 2.5]))

        phase_slopes = abscal.ndim_fft_phase_slope_solver(data, bl_vecs, assume_2D=False, zero_pad=3, bl_error_tol=1e-8)
        for ps, answer in zip(phase_slopes, [-1, -.1, 2.5]):
            assert ps.shape == (2, 3)
            np.testing.assert_array_less(np.abs(ps - answer), .2)

    def test_ndim_fft_phase_slope_solver_assume_2D_real_antpos(self):
        antpos = hex_array(8, split_core=False, outriggers=0)
        reds = redcal.get_reds(antpos, pols=['ee'], pol_mode='1pol')
        model = {red[0]: np.ones((2, 3), dtype=complex) for red in reds}
        data = {red[0]: np.ones((2, 3), dtype=complex) for red in reds}
        bl_vecs = {bl: (antpos[bl[0]] - antpos[bl[1]]) for bl in data}
        for bl in data:
            data[bl] *= np.exp(1.0j * np.dot(bl_vecs[bl], [-.02, .03, 0]))

        phase_slopes = abscal.ndim_fft_phase_slope_solver(data, bl_vecs, assume_2D=True, zero_pad=3, bl_error_tol=1)
        for ps, answer in zip(phase_slopes, [-.02, .03]):
            assert ps.shape == (2, 3)
            np.testing.assert_array_less(np.abs(ps - answer), .003)

    def test_global_phase_slope_logcal_2D(self):
        antpos = hex_array(5, split_core=False, outriggers=0)
        reds = redcal.get_reds(antpos, pols=['ee'], pol_mode='1pol')
        model = DataContainer({bl: np.ones((2, 3), dtype=complex) for red in reds for bl in red})
        uncal_data = DataContainer({bl: np.ones((2, 3), dtype=complex) for red in reds for bl in red})
        antpos = redcal.reds_to_antpos(reds)
        bl_vecs = {bl: (antpos[bl[0]] - antpos[bl[1]]) for bl in uncal_data}
        for bl in uncal_data:
            uncal_data[bl] *= np.exp(1.0j * np.dot(bl_vecs[bl], [-.2, 1]))

        # test results when fit is returned
        fit = abscal.global_phase_slope_logcal(model, uncal_data, antpos, solver='ndim_fft', assume_2D=False, verbose=False)
        fit2 = abscal.global_phase_slope_logcal(model, uncal_data, antpos, solver='ndim_fft', assume_2D=True, verbose=False)
        np.testing.assert_array_equal(fit['Phi_0_Jee'], fit2['Phi_ew_Jee'])
        np.testing.assert_array_equal(fit['Phi_1_Jee'], fit2['Phi_ns_Jee'])
        for f, answer in zip(fit.values(), [-.2, 1]):
            assert f.shape == (2, 1)
            np.testing.assert_array_less(np.abs(f - answer), .2)

        ants = sorted(list(set([ant for bl in uncal_data for ant in utils.split_bl(bl)])))
        # try doing the first iteration with either dft or ndim_fft
        for solver in ['dft', 'ndim_fft']:
            data = copy.deepcopy(uncal_data)
            for i in range(8):
                if i == 0:
                    gains = abscal.global_phase_slope_logcal(model, data, antpos, solver=solver, assume_2D=True,
                                                             time_avg=True, return_gains=True, gain_ants=ants, verbose=False)
                else:
                    gains = abscal.global_phase_slope_logcal(model, data, antpos, solver='linfit', assume_2D=False,
                                                             time_avg=True, return_gains=True, gain_ants=ants, verbose=False)
                calibrate_in_place(data, gains)
            np.testing.assert_array_almost_equal(np.linalg.norm([data[bl] - model[bl] for bl in data]), 0)

    def test_global_phase_slope_logcal_3D(self):
        antpos = hex_array(3, split_core=False, outriggers=0)
        antpos2 = hex_array(3, split_core=False, outriggers=0)
        for d in [100.0, 200.0, 300.0]:
            antpos.update({len(antpos) + ant: antpos2[ant] + np.array([d, 0, 0]) for ant in antpos2})

        reds = redcal.get_reds(antpos, pols=['ee'], pol_mode='1pol')
        model = DataContainer({bl: np.ones((2, 3), dtype=complex) for red in reds for bl in red})
        data = DataContainer({bl: np.ones((2, 3), dtype=complex) for red in reds for bl in red})
        antpos = redcal.reds_to_antpos(reds)
        bl_vecs = {bl: (antpos[bl[0]] - antpos[bl[1]]) for bl in data}
        for bl in data:
            data[bl] *= np.exp(1.0j * np.dot(bl_vecs[bl], [-.8, -.1, .5]))

        ants = sorted(list(set([ant for bl in data for ant in utils.split_bl(bl)])))
        for i in range(10):
            if i == 0:
                gains = abscal.global_phase_slope_logcal(model, data, antpos, solver='ndim_fft', assume_2D=False,
                                                         time_avg=True, return_gains=True, gain_ants=ants, verbose=False)
            else:
                gains = abscal.global_phase_slope_logcal(model, data, antpos, solver='linfit', assume_2D=False,
                                                         time_avg=True, return_gains=True, gain_ants=ants, verbose=False)
            calibrate_in_place(data, gains)
        np.testing.assert_array_almost_equal(np.linalg.norm([data[bl] - model[bl] for bl in data]), 0, 5)


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
@pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
@pytest.mark.filterwarnings("ignore:divide by zero encountered in true_divide")
@pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
class Test_AbsCal(object):
    def setup_method(self):
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
        gain_pols = np.unique([split_pol(pp) for pp in p])
        self.ap = ap
        self.gk = abscal.flatten([[(k, p) for k in a] for p in gain_pols])
        self.freqs = f

    def test_init(self):
        # init with no meta
        AC = abscal.AbsCal(self.AC.model, self.AC.data)
        assert AC.bls is None
        # init with meta
        AC = abscal.AbsCal(self.AC.model, self.AC.data, antpos=self.AC.antpos, freqs=self.AC.freqs)
        assert np.allclose(AC.bls[(24, 25, 'ee')][0], -14.607842046642745)
        # init with meta
        AC = abscal.AbsCal(self.AC.model, self.AC.data)
        # test feeding file and refant and bl_cut and bl_taper
        AC = abscal.AbsCal(self.model_fname, self.data_fname, refant=24, antpos=self.AC.antpos,
                           max_bl_cut=26.0, bl_taper_fwhm=15.0)
        # test ref ant
        assert AC.refant == 24
        assert np.allclose(np.linalg.norm(AC.antpos[24]), 0.0)
        # test bl cut
        assert not np.any(np.array([np.linalg.norm(AC.bls[k]) for k in AC.bls.keys()]) > 26.0)
        # test bl taper
        assert np.median(AC.wgts[(24, 25, 'ee')]) > np.median(AC.wgts[(24, 39, 'ee')])

        # test with input cal
        bl = (24, 25, 'ee')
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
        self.AC.abs_amp_logcal(verbose=False)
        assert self.AC.abs_eta[(24, 'Jee')].shape == (60, 64)
        assert self.AC.abs_eta_gain[(24, 'Jee')].shape == (60, 64)
        assert self.AC.abs_eta_arr.shape == (7, 60, 64, 1)
        assert self.AC.abs_eta_gain_arr.shape == (7, 60, 64, 1)
        # test Nones
        AC = abscal.AbsCal(self.AC.model, self.AC.data)
        assert AC.abs_eta is None
        assert AC.abs_eta_arr is None
        assert AC.abs_eta_gain is None
        assert AC.abs_eta_gain_arr is None
        # test propagation to gain_arr
        AC.abs_amp_logcal(verbose=False)
        AC._abs_eta_arr *= 0
        assert np.allclose(np.abs(AC.abs_eta_gain_arr[0, 0, 0, 0]), 1.0)
        # test custom gain
        g = self.AC.custom_abs_eta_gain(self.gk)
        assert len(g) == 47
        # test w/ no wgts
        AC.wgts = None
        AC.abs_amp_logcal(verbose=False)

    def test_TT_phs_logcal(self):
        # test execution
        self.AC.TT_phs_logcal(verbose=False)
        assert self.AC.TT_Phi_arr.shape == (7, 2, 60, 64, 1)
        assert self.AC.TT_Phi_gain_arr.shape == (7, 60, 64, 1)
        assert self.AC.abs_psi_arr.shape == (7, 60, 64, 1)
        assert self.AC.abs_psi_gain_arr.shape == (7, 60, 64, 1)
        assert self.AC.abs_psi[(24, 'Jee')].shape == (60, 64)
        assert self.AC.abs_psi_gain[(24, 'Jee')].shape == (60, 64)
        assert self.AC.TT_Phi[(24, 'Jee')].shape == (2, 60, 64)
        assert self.AC.TT_Phi_gain[(24, 'Jee')].shape == (60, 64)
        assert np.allclose(np.angle(self.AC.TT_Phi_gain[(24, 'Jee')]), 0.0)
        # test merge pols
        self.AC.TT_phs_logcal(verbose=False, four_pol=True)
        assert self.AC.TT_Phi_arr.shape == (7, 2, 60, 64, 1)
        assert self.AC.abs_psi_arr.shape == (7, 60, 64, 1)
        # test Nones
        AC = abscal.AbsCal(self.AC.model, self.AC.data, antpos=self.antpos)
        assert AC.abs_psi_arr is None
        assert AC.abs_psi_gain_arr is None
        assert AC.TT_Phi_arr is None
        assert AC.TT_Phi_gain_arr is None
        assert AC.abs_psi is None
        assert AC.abs_psi_gain is None
        assert AC.TT_Phi is None
        assert AC.TT_Phi_gain is None
        # test custom gain
        g = self.AC.custom_TT_Phi_gain(self.gk, self.ap)
        assert len(g) == 47
        g = self.AC.custom_abs_psi_gain(self.gk)
        assert g[(0, 'Jee')].shape == (60, 64)
        # test w/ no wgts
        AC.wgts = None
        AC.TT_phs_logcal(verbose=False)

    def test_amp_logcal(self):
        self.AC.amp_logcal(verbose=False)
        assert self.AC.ant_eta[(24, 'Jee')].shape == (60, 64)
        assert self.AC.ant_eta_gain[(24, 'Jee')].shape == (60, 64)
        assert self.AC.ant_eta_arr.shape == (7, 60, 64, 1)
        assert self.AC.ant_eta_arr.dtype == float
        assert self.AC.ant_eta_gain_arr.shape == (7, 60, 64, 1)
        assert self.AC.ant_eta_gain_arr.dtype == complex
        # test Nones
        AC = abscal.AbsCal(self.AC.model, self.AC.data)
        assert AC.ant_eta is None
        assert AC.ant_eta_gain is None
        assert AC.ant_eta_arr is None
        assert AC.ant_eta_gain_arr is None
        # test w/ no wgts
        AC.wgts = None
        AC.amp_logcal(verbose=False)

    def test_phs_logcal(self):
        self.AC.phs_logcal(verbose=False)
        assert self.AC.ant_phi[(24, 'Jee')].shape == (60, 64)
        assert self.AC.ant_phi_gain[(24, 'Jee')].shape == (60, 64)
        assert self.AC.ant_phi_arr.shape == (7, 60, 64, 1)
        assert self.AC.ant_phi_arr.dtype == float
        assert self.AC.ant_phi_gain_arr.shape == (7, 60, 64, 1)
        assert self.AC.ant_phi_gain_arr.dtype == complex
        assert np.allclose(np.angle(self.AC.ant_phi_gain[(24, 'Jee')]), 0.0)
        self.AC.phs_logcal(verbose=False, avg=True)
        AC = abscal.AbsCal(self.AC.model, self.AC.data)
        assert AC.ant_phi is None
        assert AC.ant_phi_gain is None
        assert AC.ant_phi_arr is None
        assert AC.ant_phi_gain_arr is None
        # test w/ no wgts
        AC.wgts = None
        AC.phs_logcal(verbose=False)

    def test_delay_lincal(self):
        # test w/o offsets
        self.AC.delay_lincal(verbose=False, kernel=(1, 3), medfilt=False)
        assert self.AC.ant_dly[(24, 'Jee')].shape == (60, 1)
        assert self.AC.ant_dly_gain[(24, 'Jee')].shape == (60, 64)
        assert self.AC.ant_dly_arr.shape == (7, 60, 1, 1)
        assert self.AC.ant_dly_gain_arr.shape == (7, 60, 64, 1)
        # test w/ offsets
        self.AC.delay_lincal(verbose=False, kernel=(1, 3), medfilt=False)
        assert self.AC.ant_dly_phi[(24, 'Jee')].shape == (60, 1)
        assert self.AC.ant_dly_phi_gain[(24, 'Jee')].shape == (60, 64)
        assert self.AC.ant_dly_phi_arr.shape == (7, 60, 1, 1)
        assert self.AC.ant_dly_phi_gain_arr.shape == (7, 60, 64, 1)
        assert self.AC.ant_dly_arr.shape == (7, 60, 1, 1)
        assert self.AC.ant_dly_arr.dtype == float
        assert self.AC.ant_dly_gain_arr.shape == (7, 60, 64, 1)
        assert self.AC.ant_dly_gain_arr.dtype == complex
        assert np.allclose(np.angle(self.AC.ant_dly_gain[(24, 'Jee')]), 0.0)
        assert np.allclose(np.angle(self.AC.ant_dly_phi_gain[(24, 'Jee')]), 0.0)
        # test exception
        AC = abscal.AbsCal(self.AC.model, self.AC.data)
        pytest.raises(AttributeError, AC.delay_lincal)
        # test Nones
        AC = abscal.AbsCal(self.AC.model, self.AC.data, freqs=self.freq_array)
        assert AC.ant_dly is None
        assert AC.ant_dly_gain is None
        assert AC.ant_dly_arr is None
        assert AC.ant_dly_gain_arr is None
        assert AC.ant_dly_phi is None
        assert AC.ant_dly_phi_gain is None
        assert AC.ant_dly_phi_arr is None
        assert AC.ant_dly_phi_gain_arr is None
        # test flags handling
        AC = abscal.AbsCal(self.AC.model, self.AC.data, freqs=self.freqs)
        AC.wgts[(24, 25, 'ee')] *= 0
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
        assert self.AC.dly_slope[(24, 'Jee')].shape == (2, 60, 1)
        assert self.AC.dly_slope_gain[(24, 'Jee')].shape == (60, 64)
        assert self.AC.dly_slope_arr.shape == (7, 2, 60, 1, 1)
        assert self.AC.dly_slope_gain_arr.shape == (7, 60, 64, 1)
        assert self.AC.dly_slope_ant_dly_arr.shape == (7, 60, 1, 1)
        assert np.allclose(np.angle(self.AC.dly_slope_gain[(24, 'Jee')]), 0.0)
        g = self.AC.custom_dly_slope_gain(self.gk, self.ap)
        assert g[(0, 'Jee')].shape == (60, 64)
        # test exception
        AC = abscal.AbsCal(self.AC.model, self.AC.data)
        pytest.raises(AttributeError, AC.delay_slope_lincal)
        # test Nones
        AC = abscal.AbsCal(self.AC.model, self.AC.data, antpos=self.antpos, freqs=self.freq_array)
        assert AC.dly_slope is None
        assert AC.dly_slope_gain is None
        assert AC.dly_slope_arr is None
        assert AC.dly_slope_gain_arr is None
        assert AC.dly_slope_ant_dly_arr is None
        # test medfilt and time_avg
        self.AC.delay_slope_lincal(verbose=False, medfilt=False)
        self.AC.delay_slope_lincal(verbose=False, time_avg=True)
        # test four pol
        self.AC.delay_slope_lincal(verbose=False, four_pol=True)
        assert self.AC.dly_slope[(24, 'Jee')].shape == (2, 60, 1)
        assert self.AC.dly_slope_gain[(24, 'Jee')].shape == (60, 64)
        assert self.AC.dly_slope_arr.shape == (7, 2, 60, 1, 1)
        assert self.AC.dly_slope_gain_arr.shape == (7, 60, 64, 1)
        # test flags handling
        AC = abscal.AbsCal(self.AC.model, self.AC.data, antpos=self.ap, freqs=self.freqs)
        AC.wgts[(24, 25, 'ee')] *= 0
        AC.delay_slope_lincal(verbose=False)
        # test w/ no wgts
        AC.wgts = None
        AC.delay_slope_lincal(verbose=False)

    def test_global_phase_slope_logcal(self):
        for solver in ['dft', 'linfit']:
            # test w/o offsets
            self.AC.global_phase_slope_logcal(verbose=False, edge_cut=31, solver=solver)
            assert self.AC.phs_slope[(24, 'Jee')].shape == (2, 60, 1)
            assert self.AC.phs_slope_gain[(24, 'Jee')].shape == (60, 64)
            assert self.AC.phs_slope_arr.shape == (7, 2, 60, 1, 1)
            assert self.AC.phs_slope_gain_arr.shape == (7, 60, 64, 1)
            assert self.AC.phs_slope_ant_phs_arr.shape == (7, 60, 1, 1)
            assert np.allclose(np.angle(self.AC.phs_slope_gain[(24, 'Jee')]), 0.0)
            g = self.AC.custom_phs_slope_gain(self.gk, self.ap)
            assert g[(0, 'Jee')].shape == (60, 64)
            # test Nones
            AC = abscal.AbsCal(self.AC.model, self.AC.data, antpos=self.antpos, freqs=self.freq_array)
            assert AC.phs_slope is None
            assert AC.phs_slope_gain is None
            assert AC.phs_slope_arr is None
            assert AC.phs_slope_gain_arr is None
            assert AC.phs_slope_ant_phs_arr is None
            AC = abscal.AbsCal(self.AC.model, self.AC.data, antpos=self.ap, freqs=self.freqs)
            AC.wgts[(24, 25, 'ee')] *= 0
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
        gains = [self.AC.abs_eta_gain, self.AC.TT_Phi_gain, self.AC.abs_psi_gain,
                 self.AC.ant_dly_gain, self.AC.ant_eta_gain, self.AC.ant_phi_gain]
        gains[0][(99, 'Jee')] = 1.0
        # merge shared keys
        mgains = abscal.merge_gains(gains, merge_shared=True)
        assert (99, 'Jee') not in mgains
        # merge all keys
        mgains = abscal.merge_gains(gains, merge_shared=False)
        assert (99, 'Jee') in mgains
        # test merge
        k = (53, 'Jee')
        assert mgains[k].shape == (60, 64)
        assert mgains[k].dtype == complex
        assert np.allclose(np.abs(mgains[k][0, 0]), np.abs(self.AC.abs_eta_gain[k] * self.AC.ant_eta_gain[k])[0, 0])
        assert np.allclose(np.angle(mgains[k][0, 0]), np.angle(self.AC.TT_Phi_gain[k] * self.AC.abs_psi_gain[k]
                                                               * self.AC.ant_dly_gain[k] * self.AC.ant_phi_gain[k])[0, 0])

        # test merge of flag dictionaries
        f1 = {(1, 'Jee'): np.zeros(5, bool)}
        f2 = {(1, 'Jee'): np.zeros(5, bool)}
        f3 = abscal.merge_gains([f1, f2])
        assert f3[(1, 'Jee')].dtype == np.bool_
        assert not np.any(f3[(1, 'Jee')])
        f2[(1, 'Jee')][:] = True
        f3 = abscal.merge_gains([f1, f2])
        assert np.all(f3[(1, 'Jee')])

    def test_fill_dict_nans(self):
        data = copy.deepcopy(self.AC.data)
        wgts = copy.deepcopy(self.AC.wgts)
        data[(25, 38, 'ee')][15, 20] *= np.nan
        data[(25, 38, 'ee')][20, 15] *= np.inf
        abscal.fill_dict_nans(data, wgts=wgts, nan_fill=-1, inf_fill=-2)
        assert data[(25, 38, 'ee')][15, 20].real == -1
        assert data[(25, 38, 'ee')][20, 15].real == -2
        assert np.allclose(wgts[(25, 38, 'ee')][15, 20], 0)
        assert np.allclose(wgts[(25, 38, 'ee')][20, 15], 0)
        data = copy.deepcopy(self.AC.data)
        wgts = copy.deepcopy(self.AC.wgts)
        data[(25, 38, 'ee')][15, 20] *= np.nan
        data[(25, 38, 'ee')][20, 15] *= np.inf
        abscal.fill_dict_nans(data[(25, 38, 'ee')], wgts=wgts[(25, 38, 'ee')], nan_fill=-1, inf_fill=-2, array=True)
        assert data[(25, 38, 'ee')][15, 20].real == -1
        assert data[(25, 38, 'ee')][20, 15].real == -2
        assert np.allclose(wgts[(25, 38, 'ee')][15, 20], 0)
        assert np.allclose(wgts[(25, 38, 'ee')][20, 15], 0)

    def test_mock_data(self):
        # load into pyuvdata object
        data_file = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
        data, flgs, ap, a, f, t, l, p = io.load_vis(data_file, return_meta=True)
        wgts = odict()
        for k in flgs.keys():
            wgts[k] = (~flgs[k]).astype(float)
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
        assert np.allclose(AC.dly_slope_arr[0, 0, 0, 0, 0], 1e-9, atol=1e-10)
        assert np.allclose(AC.dly_slope_arr[0, 1, 0, 0, 0], -2e-9, atol=1e-10)
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
        assert np.allclose(np.median(AC.abs_eta_arr[0, :, :, 0][AC.wgts[(24, 25, 'ee')].astype(bool)]),
                           -0.01, atol=1e-3)
        assert np.allclose(np.median(AC.TT_Phi_arr[0, 0, :, :, 0][AC.wgts[(24, 25, 'ee')].astype(bool)]),
                           -1e-3, atol=1e-4)
        assert np.allclose(np.median(AC.TT_Phi_arr[0, 1, :, :, 0][AC.wgts[(24, 25, 'ee')].astype(bool)]),
                           1e-3, atol=1e-4)

    def test_run_model_based_calibration(self, tmpdir):
        data_file = os.path.join(DATA_PATH, 'test_input/zen.2458098.45361.HH.uvh5_downselected')
        tmppath = tmpdir.strpath

        hd = io.HERAData(data_file)
        data, flags, nsamples = hd.read()
        antpairs = hd.get_antpairs()

        hdm = io.HERAData(data_file)
        model_data, model_flags, model_nsamples = hdm.read()

        precal_fname = os.path.join(tmppath, 'test_precal.calfits')

        # precalibration test gain (with unity gains).
        uvc_precal = UVCal()
        uvc_precal = uvc_precal.initialize_from_uvdata(uvdata=hd, gain_convention='divide', cal_style='sky',
                                                       ref_antenna_name='Amadeus', sky_catalog='The Library of Congress.',
                                                       metadata_only=False, sky_field='The Fields of Athenry', cal_type='gain')
        uvc_precal.gain_array[:] = 1. + 0j
        uvc_precal.write_calfits(precal_fname)

        # include a model random scale factor tiomes the amplitude of the data.
        scale_factor = np.random.rand() * 0.8 + 0.1
        hdm.data_array *= scale_factor
        # there are integrations and channels that need to be flagged.
        hdm.flag_array[np.isclose(hdm.data_array, 0.)] = True
        hd.flag_array[np.isclose(hd.data_array, 0.)] = True

        model_fname = os.path.join(tmppath, 'test_model.uvh5')
        data_fname = os.path.join(tmppath, 'test_data.uvh5')

        hdm.write_uvh5(model_fname)
        hd.write_uvh5(data_fname)

        # Now run abscal run
        cal_fname = os.path.join(tmppath, 'test_cal.calfits')
        abscal.run_model_based_calibration(data_file=data_fname, model_file=model_fname,
                                           output_filename=cal_fname, clobber=True, precalibration_gain_file=precal_fname)
        # check that gains equal to 1/sqrt(scale_factor)
        hc = io.HERACal(cal_fname)
        gains, gain_flags, _, _ = hc.read()
        for k in gains:
            np.testing.assert_array_almost_equal(gains[k][~gain_flags[k]], scale_factor ** -.5)

        # Now run abscal run with dly_lincal
        cal_fname = os.path.join(tmppath, 'test_cal.calfits')
        abscal.run_model_based_calibration(data_file=data_fname, model_file=model_fname, dly_lincal=True,
                                           output_filename=cal_fname, clobber=True, precalibration_gain_file=precal_fname)
        # check that gains equal to 1/sqrt(scale_factor)
        hc = io.HERACal(cal_fname)
        gains, gain_flags, _, _ = hc.read()
        for k in gains:
            np.testing.assert_array_almost_equal(gains[k][~gain_flags[k]], scale_factor ** -.5)

        # include auto_file and specify referance antenna.
        abscal.run_model_based_calibration(data_file=data_fname, model_file=model_fname, auto_file=data_fname,
                                           output_filename=cal_fname, clobber=True, refant=(0, 'Jnn'), precalibration_gain_file=precal_fname)
        # check that gains equal to1/sqrt(scale_factor)
        hc = io.HERACal(cal_fname)
        gains, gain_flags, _, _ = hc.read()
        for k in gains:
            np.testing.assert_array_almost_equal(gains[k][~gain_flags[k]], scale_factor ** -.5)

        hd = UVData()
        hdm = UVData()
        hd.read(data_fname)
        hd.use_future_array_shapes()
        hdm.read(model_fname)
        hdm.use_future_array_shapes()
        # test feeding UVData objects instead.
        abscal.run_model_based_calibration(data_file=hd, model_file=hdm, auto_file=hd,
                                           output_filename=cal_fname, clobber=True, refant=(0, 'Jnn'), precalibration_gain_file=precal_fname)

        # check that gains equal to1/sqrt(scale_factor)
        hc = io.HERACal(cal_fname)
        gains, gain_flags, _, _ = hc.read()
        for k in gains:
            np.testing.assert_array_almost_equal(gains[k][~gain_flags[k]], scale_factor ** -.5)

    def test_run_model_based_calibration_flagged_gains(self, tmpdir):
        """
        Test case when all gains are flagged.
        """
        data_file = os.path.join(DATA_PATH, 'test_input/zen.2458098.45361.HH.uvh5_downselected')
        tmppath = tmpdir.strpath

        hd = io.HERAData(data_file)
        data, flags, nsamples = hd.read()
        antpairs = hd.get_antpairs()

        hdm = io.HERAData(data_file)
        model_data, model_flags, model_nsamples = hdm.read()

        precal_fname = os.path.join(tmppath, 'test_precal.calfits')

        # include a model random scale factor tiomes the amplitude of the data.
        scale_factor = np.random.rand() * 0.8 + 0.1
        hdm.data_array *= scale_factor
        # there are integrations and channels that need to be flagged.
        hdm.flag_array[np.isclose(hdm.data_array, 0.)] = True
        hd.flag_array[np.isclose(hd.data_array, 0.)] = True

        model_fname = os.path.join(tmppath, 'test_model.uvh5')
        data_fname = os.path.join(tmppath, 'test_data.uvh5')
        hd.flag_array[:] = True

        hdm.write_uvh5(model_fname)
        hd.write_uvh5(data_fname)
        cal_fname = os.path.join(tmppath, 'test_cal.calfits')
        # test feeding UVData objects instead.
        abscal.run_model_based_calibration(data_file=data_fname, model_file=model_fname, auto_file=data_fname,
                                           output_filename=cal_fname, clobber=True,
                                           refant=(0, 'Jnn'),
                                           spoof_missing_channels=True)
        # assert all flags and gains equal 1.
        hc = io.HERACal(cal_fname)
        gains, gain_flags, _, _ = hc.read()
        for k in gains:
            np.testing.assert_array_almost_equal(gains[k][~gain_flags[k]], 1.)
            np.testing.assert_array_almost_equal(gain_flags[k], True)

    def test_run_model_based_calibration_nonuniform_channels(self, tmpdir):
        include_chans = np.hstack([np.arange(10), np.arange(12, 15), np.arange(64 - 10, 64)])

        data_file = os.path.join(DATA_PATH, 'test_input/zen.2458098.45361.HH.uvh5_downselected')
        tmppath = tmpdir.strpath

        hd = io.HERAData(data_file)
        data, flags, nsamples = hd.read(freq_chans=include_chans)
        antpairs = hd.get_antpairs()

        hdm = io.HERAData(data_file)
        model_data, model_flags, model_nsamples = hdm.read(freq_chans=include_chans)

        # include a model random scale factor tiomes the amplitude of the data.
        scale_factor = np.random.rand() * 0.8 + 0.1
        hdm.data_array *= scale_factor
        # there are integrations and channels that need to be flagged.
        hdm.flag_array[np.isclose(hdm.data_array, 0.)] = True
        hd.flag_array[np.isclose(hd.data_array, 0.)] = True

        model_fname = os.path.join(tmppath, 'test_model.uvh5')
        data_fname = os.path.join(tmppath, 'test_data.uvh5')

        hdm.write_uvh5(model_fname)
        hd.write_uvh5(data_fname)

        cal_fname = os.path.join(tmppath, 'test_cal.calfits')

        # test feeding UVData objects instead.
        abscal.run_model_based_calibration(data_file=data_fname, model_file=model_fname, auto_file=data_fname,
                                           output_filename=cal_fname, clobber=True,
                                           refant=(0, 'Jnn'),
                                           spoof_missing_channels=True)

        # check that gains equal to1/sqrt(scale_factor)
        hc = io.HERACal(cal_fname)
        gains, gain_flags, _, _ = hc.read()
        for k in gains:
            np.testing.assert_array_almost_equal(gains[k][~gain_flags[k]], scale_factor ** -.5)

    def test_run_model_based_calibration_redundant(self, tmpdir):

        data_file = os.path.join(DATA_PATH, 'test_input/zen.2458098.45361.HH.uvh5_downselected')
        tmppath = tmpdir.strpath

        hd = io.HERAData(data_file)
        data, flags, nsamples = hd.read()
        antpairs = hd.get_antpairs()

        hdm = io.HERAData(data_file)
        model_data, model_flags, model_nsamples = hdm.read()

        precal_fname = os.path.join(tmppath, 'test_precal.calfits')
        uvc_precal = UVCal()
        uvc_precal = uvc_precal.initialize_from_uvdata(uvdata=hd, gain_convention='divide', cal_style='sky',
                                                       ref_antenna_name='Amadeus', sky_catalog='The Library of Congress.',
                                                       metadata_only=False, sky_field='The Fields of Athenry', cal_type='gain')
        uvc_precal.gain_array[:] = 1. + 0j
        uvc_precal.write_calfits(precal_fname)

        # include a model random scale factor tiomes the amplitude of the data.
        scale_factor = np.random.rand() * 0.8 + 0.1
        hdm.data_array *= scale_factor
        # there are integrations and channels that need to be flagged.
        hdm.flag_array[np.isclose(hdm.data_array, 0.)] = True
        hd.flag_array[np.isclose(hd.data_array, 0.)] = True

        model_fname = os.path.join(tmppath, 'test_model.uvh5')
        data_fname = os.path.join(tmppath, 'test_data.uvh5')

        hdm.write_uvh5(model_fname)
        hd.write_uvh5(data_fname)

        cal_fname = os.path.join(tmppath, 'test_cal.calfits')

        # data file where all data in redundant group are equal to redundantly averaged data
        # (inflated by redundancy)
        red_data_fname = os.path.join(tmppath, 'test_data_red.uvh5')
        # model file that is redundantly averaged
        red_model_fname = os.path.join(tmppath, 'test_model_red.uvh5')

        # create a redundantly averaged model file.
        reds = redcal.get_pos_reds(hdm.antpos, include_autos=True)
        reds = [[bl for bl in redgrp if bl in antpairs or reverse_bl(bl) in antpairs] for redgrp in reds]
        reds = [redgrp for redgrp in reds if len(redgrp) > 0]

        utils.red_average(model_data, reds=reds, flags=model_flags,
                          nsamples=model_nsamples, inplace=True)
        hdm.select(bls=list(model_data.keys()))
        hdm.update(data=model_data, flags=model_flags, nsamples=model_nsamples)
        hdm.flag_array[np.isclose(hdm.data_array, 0.)] = True
        hdm.write_uvh5(red_model_fname)

        # generate a new data file that is inflated by redundancy from redundant odel file.
        hdm.select(antenna_nums=np.unique(np.hstack([hd.ant_1_array, hd.ant_2_array])),
                   keep_all_metadata=False)
        hdm.inflate_by_redundancy()
        hdm.select(bls=hd.bls)
        hdm.data_array /= scale_factor
        hdm.write_uvh5(red_data_fname)

        # use inflated redundant model.
        abscal.run_model_based_calibration(data_file=red_data_fname, model_file=red_model_fname,
                                           auto_file=red_data_fname,
                                           output_filename=cal_fname,
                                           clobber=True, refant=(0, 'Jnn'),
                                           constrain_model_to_data_ants=True, max_iter=1,
                                           inflate_model_by_redundancy=True, precalibration_gain_file=precal_fname)

        # check that gains equal to1/sqrt(scale_factor)
        hc = io.HERACal(cal_fname)
        gains, gain_flags, _, _ = hc.read()
        for k in gains:
            np.testing.assert_array_almost_equal(gains[k][~gain_flags[k]], scale_factor ** -.5)

    def test_model_calibration_argparser(self):
        sys.argv = [sys.argv[0], 'a', 'b', 'c', '--auto_file', 'd']
        ap = abscal.model_calibration_argparser()
        args = ap.parse_args()
        assert args.data_file == 'a'
        assert args.model_file == 'b'
        assert args.output_filename == 'c'
        assert args.auto_file == 'd'
        assert args.tol == 1e-6


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
class Test_Post_Redcal_Abscal_Run(object):
    def setup_method(self):
        self.data_file = os.path.join(DATA_PATH, 'test_input/zen.2458098.45361.HH.uvh5_downselected')
        self.redcal_file = os.path.join(DATA_PATH, 'test_input/zen.2458098.45361.HH.omni.calfits_downselected')
        self.model_files = [os.path.join(DATA_PATH, 'test_input/zen.2458042.60288.HH.uvRXLS.uvh5_downselected'),
                            os.path.join(DATA_PATH, 'test_input/zen.2458042.61034.HH.uvRXLS.uvh5_downselected')]
        self.model_files_missing_one_int = [os.path.join(DATA_PATH, 'test_input/zen.2458042.60288.HH.uvRXLS.uvh5_downselected'),
                                            os.path.join(DATA_PATH, 'test_input/zen.2458042.61034.HH.uvRXLS.uvh5_downselected_missing_first_integration')]
        self.red_data_file = os.path.join(DATA_PATH, 'test_input/zen.2458098.45361.HH.uvh5_downselected_redavg')
        self.red_model_files = [os.path.join(DATA_PATH, 'test_input/zen.2458042.60288.HH.uvRXLS.uvh5_downselected_redavg'),
                                os.path.join(DATA_PATH, 'test_input/zen.2458042.61034.HH.uvRXLS.uvh5_downselected_redavg')]

    def test_get_all_times_and_lsts(self):
        hd = io.HERAData(self.model_files)

        all_times, all_lsts = abscal.get_all_times_and_lsts(hd)
        assert len(all_times) == 120
        assert len(all_lsts) == 120
        np.testing.assert_array_equal(all_times, sorted(all_times))

        for f in hd.lsts.keys():
            hd.lsts[f] += 4.75
        all_times, all_lsts = abscal.get_all_times_and_lsts(hd, unwrap=True)
        assert all_lsts[-1] > 2 * np.pi
        np.testing.assert_array_equal(all_lsts, sorted(all_lsts))
        c = abscal.get_all_times_and_lsts(hd)
        assert all_lsts[0] < all_lsts[-1]

        hd = io.HERAData(self.data_file)
        hd.times = hd.times[0:4] + .5
        hd.lsts = hd.lsts[0:4] + np.pi
        all_times, all_lsts = abscal.get_all_times_and_lsts(hd, solar_horizon=0.0)
        assert len(all_times) == 0
        assert len(all_lsts) == 0

    def test_get_d2m_time_map(self):
        hd = io.HERAData(self.data_file)
        hdm = io.HERAData(self.model_files)
        all_data_times, all_data_lsts = abscal.get_all_times_and_lsts(hd)
        all_model_times, all_model_lsts = abscal.get_all_times_and_lsts(hdm)
        d2m_time_map = abscal.get_d2m_time_map(all_data_times, all_data_lsts, all_model_times, all_model_lsts)
        for dtime, mtime in d2m_time_map.items():
            dlst = all_data_lsts[np.argwhere(all_data_times == dtime)[0][0]]
            mlst = all_model_lsts[np.argwhere(all_model_times == mtime)[0][0]]
            assert np.abs(dlst - mlst) < np.median(np.ediff1d(all_data_lsts))
            assert np.min(np.abs(all_data_lsts - mlst)) == np.abs(dlst - mlst)

        hd = io.HERAData(self.data_file)
        hdm = io.HERAData(self.model_files[0])
        all_data_times, all_data_lsts = abscal.get_all_times_and_lsts(hd)
        all_model_times, all_model_lsts = abscal.get_all_times_and_lsts(hdm)
        d2m_time_map = abscal.get_d2m_time_map(all_data_times, all_data_lsts, all_model_times, all_model_lsts)
        for dtime, mtime in d2m_time_map.items():
            dlst = all_data_lsts[np.argwhere(all_data_times == dtime)[0][0]]
            if mtime is None:
                for mlst in all_model_lsts:
                    assert np.min(np.abs(all_data_lsts - mlst)) < np.abs(dlst - mlst)
            else:
                mlst = all_model_lsts[np.argwhere(all_model_times == mtime)[0][0]]
                assert np.abs(dlst - mlst) < np.median(np.ediff1d(all_data_lsts))
                assert np.min(np.abs(all_data_lsts - mlst)) == np.abs(dlst - mlst)

        # Test errors for when times/lsts don't match lengths
        with pytest.raises(ValueError):
            abscal.get_d2m_time_map(all_data_times[1:], all_data_lsts, all_model_times, all_model_lsts)
        with pytest.raises(ValueError):
            abscal.get_d2m_time_map(all_data_times, all_data_lsts, all_model_times[1:], all_model_lsts)

    def test_match_baselines(self):
        with pytest.raises(NotImplementedError):
            abscal.match_baselines(None, None, None, model_is_redundant=False, data_is_redsol=True)

        # try with data files:
        hd = io.HERAData(self.data_file)
        hdm = io.HERAData(self.model_files[0])
        data_bl_to_load, model_bl_to_load, data_to_model_bl_map = abscal.match_baselines(hd.bls, hdm.bls, hd.antpos)
        for bl in data_bl_to_load:
            assert bl in model_bl_to_load
            assert data_to_model_bl_map[bl] == bl
        for bl in model_bl_to_load:
            assert bl in data_bl_to_load

        # try with redundant model
        with pytest.raises(AssertionError):
            abscal.match_baselines(hd.bls, hdm.bls, hd.antpos, model_is_redundant=True)
        antpos = {0: np.array([0, 0, 0]), 1: np.array([10, 0, 0]), 2: np.array([20, 0, 0]), 3: np.array([100, 100, 0])}
        data_bls = [(0, 1, 'ee'), (0, 2, 'ee'), (1, 2, 'ee'), (0, 3, 'ee')]
        model_bls = [(0, 1, 'ee'), (0, 2, 'ee'), (1, 3, 'ee')]
        data_bl_to_load, model_bl_to_load, data_to_model_bl_map = abscal.match_baselines(data_bls, model_bls, antpos, model_is_redundant=True)
        assert len(data_bl_to_load) == 3
        assert len(model_bl_to_load) == 2
        assert data_to_model_bl_map[(0, 1, 'ee')] == (0, 1, 'ee')
        assert data_to_model_bl_map[(1, 2, 'ee')] == (0, 1, 'ee')
        assert data_to_model_bl_map[(0, 2, 'ee')] == (0, 2, 'ee')

        # try with cutting on baseline length
        with pytest.raises(AssertionError):
            abscal.match_baselines(hd.bls, hdm.bls, hd.antpos, model_is_redundant=True)
        antpos = {0: np.array([0, 0, 0]), 1: np.array([10, 0, 0]), 2: np.array([20, 0, 0]), 3: np.array([100, 100, 0])}
        data_bls = [(0, 1, 'ee'), (0, 2, 'ee'), (1, 2, 'ee'), (0, 3, 'ee')]
        model_bls = [(0, 1, 'ee'), (0, 2, 'ee'), (1, 3, 'ee'), (0, 3, 'ee')]
        data_bl_to_load, model_bl_to_load, data_to_model_bl_map = abscal.match_baselines(data_bls, model_bls, antpos, model_is_redundant=True, min_bl_cut=15, max_bl_cut=50)
        assert len(data_bl_to_load) == 1
        assert len(model_bl_to_load) == 1
        assert data_to_model_bl_map[(0, 2, 'ee')] == (0, 2, 'ee')

        # try with redundant model and some reversed baselines
        with pytest.raises(AssertionError):
            abscal.match_baselines(hd.bls, hdm.bls, hd.antpos, model_is_redundant=True)
        antpos = {0: np.array([0, 0, 0]), 1: np.array([10, 0, 0]), 2: np.array([20, 0, 0]), 3: np.array([100, 100, 0])}
        data_bls = [(0, 1, 'ee'), (0, 2, 'ee'), (2, 1, 'ee'), (0, 3, 'ee')]
        model_bls = [(0, 1, 'ee'), (2, 0, 'ee'), (1, 3, 'ee')]
        data_bl_to_load, model_bl_to_load, data_to_model_bl_map = abscal.match_baselines(data_bls, model_bls, antpos, model_is_redundant=True)
        assert len(data_bl_to_load) == 3
        assert len(model_bl_to_load) == 2
        assert data_to_model_bl_map[(0, 1, 'ee')] == (0, 1, 'ee')
        assert data_to_model_bl_map[(2, 1, 'ee')] == (1, 0, 'ee')
        assert data_to_model_bl_map[(0, 2, 'ee')] == (0, 2, 'ee')

        # try with different antenna numbering in model
        antpos = {0: np.array([0, 0, 0]), 1: np.array([10, 0, 0]), 2: np.array([20, 0, 0]), 3: np.array([100, 100, 0])}
        model_antpos = {100: np.array([0, 0, 0]), 101: np.array([10, 0, 0]), 102: np.array([20, 0, 0]), 103: np.array([100, 100, 0])}
        data_bls = [(0, 1, 'ee'), (0, 2, 'ee'), (1, 2, 'ee'), (0, 3, 'ee')]
        model_bls = [(100, 101, 'ee'), (100, 102, 'ee'), (101, 103, 'ee')]
        data_bl_to_load, model_bl_to_load, data_to_model_bl_map = abscal.match_baselines(data_bls, model_bls, antpos, model_antpos=model_antpos, model_is_redundant=True)
        assert len(data_bl_to_load) == 3
        assert len(model_bl_to_load) == 2
        assert data_to_model_bl_map[(0, 1, 'ee')] == (100, 101, 'ee')
        assert data_to_model_bl_map[(1, 2, 'ee')] == (100, 101, 'ee')
        assert data_to_model_bl_map[(0, 2, 'ee')] == (100, 102, 'ee')

        # try with both redundant
        with pytest.raises(AssertionError):
            abscal.match_baselines(data_bls, model_bls, antpos, model_antpos=model_antpos, model_is_redundant=True, data_is_redsol=True)
        antpos = {0: np.array([0, 0, 0]), 1: np.array([10, 0, 0]), 2: np.array([20, 0, 0]), 3: np.array([100, 100, 0])}
        model_antpos = {100: np.array([0, 0, 0]), 101: np.array([10, 0, 0]), 102: np.array([20, 0, 0]), 103: np.array([100, 100, 0])}
        data_bls = [(0, 2, 'ee'), (1, 2, 'ee'), (0, 3, 'ee')]
        model_bls = [(100, 101, 'ee'), (100, 102, 'ee'), (101, 103, 'ee')]
        data_bl_to_load, model_bl_to_load, data_to_model_bl_map = abscal.match_baselines(data_bls, model_bls, antpos, model_antpos=model_antpos,
                                                                                         data_is_redsol=True, model_is_redundant=True)
        assert len(data_bl_to_load) == 2
        assert len(model_bl_to_load) == 2
        assert data_to_model_bl_map[(1, 2, 'ee')] == (100, 101, 'ee')
        assert data_to_model_bl_map[(0, 2, 'ee')] == (100, 102, 'ee')

    def test_build_data_wgts(self):
        # test non-redundant version
        bls = [(0, 1, 'ee'), (0, 2, 'ee'), (1, 2, 'ee')]
        auto_bls = [(0, 0, 'ee'), (1, 1, 'ee'), (2, 2, 'ee')]
        data_flags = DataContainer({bl: np.zeros((3, 4), dtype=bool) for bl in bls})
        data_flags[(0, 1, 'ee')][0, 0] = True
        data_flags.times_by_bl = {bl[:2]: np.arange(3) / 86400 for bl in bls}
        data_flags.freqs = np.arange(4)
        data_flags.antpos = {0: np.array([0, 0, 0]), 1: np.array([10, 0, 0]), 2: np.array([20, 0, 0])}
        data_flags.data_antpos = {0: np.array([0, 0, 0]), 1: np.array([10, 0, 0]), 2: np.array([20, 0, 0])}
        data_nsamples = DataContainer({bl: np.ones((3, 4), dtype=float) for bl in bls})
        data_nsamples[(0, 1, 'ee')][1, 1] = 2
        model_flags = data_flags
        autocorrs = DataContainer({bl: np.ones((3, 4), dtype=complex) for bl in auto_bls})
        autocorrs[(1, 1, 'ee')][2, 2] = 3
        auto_flags = DataContainer({bl: np.zeros((3, 4), dtype=bool) for bl in auto_bls})

        wgts = abscal.build_data_wgts(data_flags, data_nsamples, model_flags, autocorrs, auto_flags)
        for bl in wgts:
            for t in range(3):
                for f in range(4):
                    if 1 in bl and t == 2 and f == 2:
                        assert wgts[bl][t, f] == 1 / 3
                    elif bl == (0, 1, 'ee'):
                        if t == 0 and f == 0:
                            assert wgts[bl][t, f] == 0
                        elif t == 1 and f == 1:
                            assert wgts[bl][t, f] == 2
                        else:
                            assert wgts[bl][t, f] == 1
                    else:
                        assert wgts[bl][t, f] == 1

        # test redundant verison
        bls = [(0, 1, 'ee'), (0, 2, 'ee')]
        data_flags = DataContainer({bl: np.zeros((3, 4), dtype=bool) for bl in bls})
        data_flags.times_by_bl = {bl[:2]: np.arange(3) / 86400 for bl in bls}
        data_flags.freqs = np.arange(4)
        data_flags.antpos = {0: np.array([0, 0, 0]), 1: np.array([10, 0, 0]), 2: np.array([20, 0, 0]), 3: np.array([30, 0, 0])}
        data_flags.data_antpos = {0: np.array([0, 0, 0]), 1: np.array([10, 0, 0]), 2: np.array([20, 0, 0]), 3: np.array([30, 0, 0])}
        data_nsamples = DataContainer({bl: np.ones((3, 4), dtype=float) for bl in bls})
        data_nsamples[(0, 1, 'ee')] *= 3
        data_nsamples[(0, 2, 'ee')] *= 2
        model_flags = data_flags
        autocorrs = DataContainer({bl: np.ones((3, 4), dtype=complex) for bl in auto_bls})
        autocorrs[(2, 2, 'ee')][2, 2] = 3
        auto_flags = DataContainer({bl: np.zeros((3, 4), dtype=bool) for bl in auto_bls})
        auto_flags[(0, 0, 'ee')][1, 1] = True

        gain_flags = {ant: np.zeros((3, 4), dtype=bool) for ant in [(0, 'Jee'), (1, 'Jee'), (2, 'Jee'), (-1, 'Jee')]}
        gain_flags[(0, 'Jee')] += True
        wgts = abscal.build_data_wgts(data_flags, data_nsamples, model_flags, autocorrs, auto_flags,
                                      data_is_redsol=True, gain_flags=gain_flags, tol=1.0)
        for bl in wgts:
            for t in range(3):
                for f in range(3):
                    if bl == (0, 1, 'ee'):
                        if t == 2 and f == 2:
                            assert wgts[bl][t, f] == 3 / (((1 / 3) + (1 / 1))**-1 * 2)
                        else:
                            assert wgts[bl][t, f] == 3
                    elif bl == (0, 2, 'ee'):
                        if t == 2 and f == 2:
                            assert wgts[bl][t, f] == 2 / (((1 / 3))**-1 * 1)
                        elif t == 1 and f == 1:
                            assert wgts[bl][t, f] == 0
                        else:
                            assert wgts[bl][t, f] == 2

    def test_get_idealized_antpos(self):
        # build 7 element hex with 1 outrigger. If all antennas are unflagged, the outrigger
        # is not redundant with the hex, so it introduces an extra degeneracy. That corresponds
        # to an extra dimension in an idealized antenna position.
        antpos = hex_array(2, split_core=False, outriggers=0)
        antpos[7] = np.array([100, 0, 0])
        reds = redcal.get_reds(antpos, pols=['ee'])

        # test with no flagged antennas
        cal_flags = {(ant, 'Jee'): np.array([False]) for ant in antpos}
        iap = abscal._get_idealized_antpos(cal_flags, antpos, ['ee'], keep_flagged_ants=True)
        assert len(iap) == 8  # all antennas are included
        assert len(iap[0]) == 3  # 3 degeneracies ==> 3 dimensions
        # check that the results are the same as in redcal.reds_to_antpos
        r2a = redcal.reds_to_antpos(reds)
        for ant in r2a:
            np.testing.assert_array_equal(iap[ant], r2a[ant])

        # test with flagged outrigger, which lowers the number of degeneracies
        cal_flags = {(ant, 'Jee'): np.array([False]) for ant in antpos}
        cal_flags[(7, 'Jee')] = True
        iap = abscal._get_idealized_antpos(cal_flags, antpos, ['ee'], keep_flagged_ants=True)
        # because keep_flagged_ants is True, the flagged antenna is still in the antpos dict
        assert len(iap) == 8
        # because the only antenna necessitating a 3rd tip-tilt degeneracy is flagged,
        # get_idealized_antpos enforces that all remaining antenna positions are expressed in 2D
        assert len(iap[0]) == 2
        r2a = redcal.reds_to_antpos(redcal.filter_reds(reds, ex_ants=[7]))
        for ant in r2a:
            np.testing.assert_array_equal(iap[ant], r2a[ant])
        # because there's no sensible way to describe the antenna's position in this basis, set it to 0
        assert np.all(iap[7] == 0)

        # test with flagged grid ant, which does not affect the number of degeneracies
        cal_flags = {(ant, 'Jee'): np.array([False]) for ant in antpos}
        cal_flags[(1, 'Jee')] = True
        iap = abscal._get_idealized_antpos(cal_flags, antpos, ['ee'], keep_flagged_ants=True)
        assert len(iap) == 8  # all antennas included
        # removing an on-grid antenna but keeping the outrigger doesn't change the number of degeneracies
        assert len(iap[0]) == 3
        # test that the flagged antenna has the position it would have had it if weren't flagged
        r2a = redcal.reds_to_antpos(reds)
        for ant in r2a:
            np.testing.assert_array_equal(iap[ant], r2a[ant])

        # test keep_flagged_ants=False
        cal_flags = {(ant, 'Jee'): np.array([False]) for ant in antpos}
        cal_flags[(1, 'Jee')] = True
        iap = abscal._get_idealized_antpos(cal_flags, antpos, ['ee'], keep_flagged_ants=False)
        assert 1 not in iap
        assert len(iap) == 7

        # test error when an antenna is somehow in the cal_flags (unflagged) but not antpos or the derived reds
        antpos2 = hex_array(2, split_core=False, outriggers=0)
        cal_flags = {(ant, 'Jee'): np.array([False]) for ant in antpos2}
        # remove antenna 0
        del antpos2[0]
        with pytest.raises(ValueError):
            iap = abscal._get_idealized_antpos(cal_flags, antpos2, ['ee'])

        # test error where an antenna has non-zero weight, but doesn't appear in cal_flags
        data_wgts = {bl: np.array([1]) for red in reds for bl in red}
        cal_flags = {(ant, 'Jee'): np.array([False]) for ant in antpos}
        cal_flags[(7, 'Jee')] = True
        with pytest.raises(ValueError):
            iap = abscal._get_idealized_antpos(cal_flags, antpos, ['ee'], data_wgts=data_wgts)

        # test error where antenna with non-zero weight is getting placed at position 0
        cal_flags = {(ant, 'Jee'): np.array([False]) for ant in antpos2}
        cal_flags[7, 'Jee'] = True
        with pytest.raises(ValueError):
            iap = abscal._get_idealized_antpos(cal_flags, antpos, ['ee'], data_wgts=data_wgts)

    def test_post_redcal_abscal(self):
        # setup
        hd = io.HERAData(self.data_file)
        hdm = io.HERAData(self.model_files)
        hc = io.HERACal(self.redcal_file)

        model_bls = list(set([bl for bls in list(hdm.bls.values()) for bl in bls]))
        model_antpos = {ant: pos for antpos in hdm.antpos.values() for ant, pos in antpos.items()}
        (data_bl_to_load,
         model_bl_to_load,
         data_to_model_bl_map) = abscal.match_baselines(hd.bls, model_bls, hd.antpos, model_antpos=model_antpos, pols=['ee', 'nn'], min_bl_cut=1.0)

        rc_gains, rc_flags, rc_quals, rc_tot_qual = hc.read()
        all_data_times, all_data_lsts = abscal.get_all_times_and_lsts(hd)
        all_model_times, all_model_lsts = abscal.get_all_times_and_lsts(hdm)
        d2m_time_map = abscal.get_d2m_time_map(all_data_times, all_data_lsts, all_model_times, all_model_lsts)
        tinds = [0, 1, 2]
        data, flags, nsamples = hd.read(times=hd.times[tinds], bls=data_bl_to_load)
        model_times_to_load = [d2m_time_map[time] for time in hd.times[tinds]]
        model, model_flags, _ = io.partial_time_io(hdm, model_times_to_load, bls=model_bl_to_load)
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
        wgts = DataContainer({k: (~flags[k]).astype(float) for k in flags.keys()})

        # run function
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            delta_gains = abscal.post_redcal_abscal(model, copy.deepcopy(data), wgts, rc_flags_subset, verbose=False)

        # use returned gains to calibrate data
        calibrate_in_place(data, delta_gains, data_flags=flags,
                           cal_flags=rc_flags_subset, gain_convention=hc.gain_convention)

        # basic shape & type checks
        for k in rc_gains.keys():
            assert k in delta_gains
            assert delta_gains[k].shape == (3, rc_gains[k].shape[1])
            assert delta_gains[k].dtype == complex

        # try running without amplitude solvers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            delta_gains = abscal.post_redcal_abscal(model, copy.deepcopy(data), wgts, rc_flags_subset, verbose=False,
                                                    use_abs_amp_logcal=False, use_abs_amp_lincal=False)
        for k in delta_gains:
            np.testing.assert_array_almost_equal(np.abs(delta_gains[k]), 1)

    def test_post_redcal_abscal_run_units_warning(self, tmpdir):
        tmp_path = tmpdir.strpath
        calfile_units = os.path.join(tmp_path, 'redcal_units.calfits')
        model_units = os.path.join(tmp_path, 'model_file_units.uvh5')
        hd = io.HERAData(self.model_files[0])
        hd.read()
        hd.vis_units = 'Jy'
        hd.write_uvh5(model_units)
        hcr = io.HERACal(self.redcal_file)
        hcr.read()
        hcr.gain_scale = 'k str'
        hcr.write_calfits(calfile_units)
        with pytest.warns(RuntimeWarning):
            hca = abscal.post_redcal_abscal_run(self.data_file, calfile_units, [model_units], phs_conv_crit=1e-4,
                                                nInt_to_load=30, verbose=False, add_to_history='testing')
        assert hca.gain_scale == 'Jy'

    def test_post_redcal_abscal_run(self, tmpdir):
        tmp_path = tmpdir.strpath
        output_file_delta = os.path.join(tmp_path, 'delta_gains.calfits')
        # test no model overlap
        hcr = io.HERACal(self.redcal_file)
        rc_gains, rc_flags, rc_quals, rc_total_qual = hcr.read()

        hd = io.HERAData(self.model_files[0])
        hd.read(return_data=False)
        hd.lst_array += 1
        temp_outfile = os.path.join(DATA_PATH, 'test_output/temp.uvh5')
        hd.write_uvh5(temp_outfile, clobber=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hca = abscal.post_redcal_abscal_run(self.data_file, self.redcal_file, [temp_outfile], phs_conv_crit=1e-4,
                                                nInt_to_load=30, verbose=False, add_to_history='testing')
        assert os.path.exists(self.redcal_file.replace('.omni.', '.abs.'))
        np.testing.assert_array_equal(hca.total_quality_array, 0.0)
        np.testing.assert_array_equal(hca.gain_array, hcr.gain_array)
        np.testing.assert_array_equal(hca.flag_array, True)
        np.testing.assert_array_equal(hca.quality_array, 0.0)
        os.remove(self.redcal_file.replace('.omni.', '.abs.'))
        os.remove(temp_outfile)

        # test normal operation of abscal (with one missing integration, to test assinging multiple data times to one model time and then rephasing)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hca = abscal.post_redcal_abscal_run(self.data_file, self.redcal_file, self.model_files_missing_one_int, extrap_limit=1.0,
                                                phs_conv_crit=1e-4, nInt_to_load=30, verbose=False, add_to_history='testing')
        pytest.raises(IOError, abscal.post_redcal_abscal_run, self.data_file, self.redcal_file, self.model_files, clobber=False)

        assert os.path.exists(self.redcal_file.replace('.omni.', '.abs.'))
        os.remove(self.redcal_file.replace('.omni.', '.abs.'))
        ac_gains, ac_flags, ac_quals, ac_total_qual = hca.build_calcontainers()
        hdm = io.HERAData(self.model_files_missing_one_int)
        assert hca.gain_scale == hdm.vis_units
        assert hcr.history.replace('\n', '').replace(' ', '') in hca.history.replace('\n', '').replace(' ', '')
        assert 'testing' in hca.history.replace('\n', '').replace(' ', '')
        for k in rc_gains:
            assert k in ac_gains
            assert ac_gains[k].shape == rc_gains[k].shape
            assert ac_gains[k].dtype == complex

        hd = io.HERAData(self.data_file)
        _, data_flags, _ = hd.read()
        ac_flags_expected = synthesize_ant_flags(data_flags)
        ac_flags_waterfall = np.all([f for f in ac_flags.values()], axis=0)
        for ant in ac_flags_expected:
            ac_flags_expected[ant] += rc_flags[ant]
            ac_flags_expected[ant] += ac_flags_waterfall
        for k in rc_flags:
            assert k in ac_flags
            assert ac_flags[k].shape == rc_flags[k].shape
            assert ac_flags[k].dtype == bool
            np.testing.assert_array_equal(ac_flags[k], ac_flags_expected[k])

        assert not np.all(list(ac_flags.values()))
        for pol in ['Jee', 'Jnn']:
            assert pol in ac_total_qual
            assert ac_total_qual[pol].shape == rc_total_qual[pol].shape
            assert np.issubdtype(ac_total_qual[pol].dtype, np.floating)

        # test redundant model and full data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hca_red = abscal.post_redcal_abscal_run(self.data_file, self.redcal_file, self.red_model_files, phs_conv_crit=1e-4,
                                                    nInt_to_load=10, verbose=False, add_to_history='testing2', model_is_redundant=True)
        assert os.path.exists(self.redcal_file.replace('.omni.', '.abs.'))
        os.remove(self.redcal_file.replace('.omni.', '.abs.'))
        ac_gains, ac_flags, ac_quals, ac_total_qual = hca_red.build_calcontainers()
        hcr = io.HERACal(self.redcal_file)
        rc_gains, rc_flags, rc_quals, rc_total_qual = hcr.read()

        assert hcr.history.replace('\n', '').replace(' ', '') in hca_red.history.replace('\n', '').replace(' ', '')
        assert 'testing2' in hca_red.history.replace('\n', '').replace(' ', '')
        for k in rc_gains:
            assert k in ac_gains
            assert ac_gains[k].shape == rc_gains[k].shape
            assert ac_gains[k].dtype == complex

        hd = io.HERAData(self.data_file)
        _, data_flags, _ = hd.read()
        ac_flags_expected = synthesize_ant_flags(data_flags)
        ac_flags_waterfall = np.all([f for f in ac_flags.values()], axis=0)
        for ant in ac_flags_expected:
            ac_flags_expected[ant] += rc_flags[ant]
            ac_flags_expected[ant] += ac_flags_waterfall
        for k in rc_flags:
            assert k in ac_flags
            assert ac_flags[k].shape == rc_flags[k].shape
            assert ac_flags[k].dtype == bool
            np.testing.assert_array_equal(ac_flags[k], ac_flags_expected[k])

        assert not np.all(list(ac_flags.values()))
        for pol in ['Jee', 'Jnn']:
            assert pol in ac_total_qual
            assert ac_total_qual[pol].shape == rc_total_qual[pol].shape
            assert np.issubdtype(ac_total_qual[pol].dtype, np.floating)

        # test redundant model and redundant data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hca_red_red = abscal.post_redcal_abscal_run(self.red_data_file, self.redcal_file, self.red_model_files, phs_conv_crit=1e-4,
                                                        nInt_to_load=10, verbose=False, add_to_history='testing3', model_is_redundant=True,
                                                        data_is_redsol=True, raw_auto_file=self.data_file,
                                                        write_delta_gains=True, output_file_delta=output_file_delta)
        hdm = io.HERAData(self.red_model_files)
        assert hca_red_red.gain_scale == hdm.vis_units
        assert os.path.exists(self.redcal_file.replace('.omni.', '.abs.'))
        hcat = io.HERACal(self.redcal_file.replace('.omni.', '.abs.'))
        hcat.read()
        os.remove(self.redcal_file.replace('.omni.', '.abs.'))
        ac_gains, ac_flags, ac_quals, ac_total_qual = hca_red_red.build_calcontainers()
        hcr = io.HERACal(self.redcal_file)
        rc_gains, rc_flags, rc_quals, rc_total_qual = hcr.read()
        assert os.path.exists(output_file_delta)
        hcg = io.HERACal(output_file_delta)
        hcg.read()
        # ensure that unflagged redundant gains times degenerate gains equal
        # abscal gains.
        assert np.allclose(hcat.gain_array[~hcat.flag_array],
                           hcr.gain_array[~hcat.flag_array] * hcg.gain_array[~hcat.flag_array])

        assert hcr.history.replace('\n', '').replace(' ', '') in hca_red_red.history.replace('\n', '').replace(' ', '')
        assert 'testing3' in hca_red_red.history.replace('\n', '').replace(' ', '')
        for k in rc_gains:
            assert k in ac_gains
            assert ac_gains[k].shape == rc_gains[k].shape
            assert ac_gains[k].dtype == complex

        hd = io.HERAData(self.data_file)
        _, data_flags, _ = hd.read()
        ac_flags_expected = synthesize_ant_flags(data_flags)
        ac_flags_waterfall = np.all([f for f in ac_flags.values()], axis=0)
        for ant in ac_flags_expected:
            ac_flags_expected[ant] += rc_flags[ant]
            ac_flags_expected[ant] += ac_flags_waterfall
        for k in rc_flags:
            assert k in ac_flags
            assert ac_flags[k].shape == rc_flags[k].shape
            assert ac_flags[k].dtype == bool
            np.testing.assert_array_equal(ac_flags[k], ac_flags_expected[k])

        assert not np.all(list(ac_flags.values()))
        for pol in ['Jee', 'Jnn']:
            assert pol in ac_total_qual
            assert ac_total_qual[pol].shape == rc_total_qual[pol].shape
            assert np.issubdtype(ac_total_qual[pol].dtype, np.floating)

        # compare all 3 versions
        g1, f1, q1, tq1 = hca.build_calcontainers()
        g2, f2, q2, tq2 = hca_red.build_calcontainers()
        g3, f3, q3, tq3 = hca_red_red.build_calcontainers()

        for ant in f1:
            np.testing.assert_array_equal(f1[ant], f2[ant])
            np.testing.assert_array_equal(f1[ant], f3[ant])

        for ant in g1:
            if not np.all(f1[ant]):
                assert np.abs(np.median(np.abs(g1[ant][~f1[ant]] / g2[ant][~f2[ant]])) - 1) < .1
                assert np.abs(np.median(np.abs(g1[ant][~f1[ant]] / g3[ant][~f3[ant]])) - 1) < .1

        for ant in q1:
            np.testing.assert_array_equal(q1[ant], 0.0)
            np.testing.assert_array_equal(q2[ant], 0.0)
            np.testing.assert_array_equal(q3[ant], 0.0)

    def test_post_redcal_abscal_argparser(self):
        sys.argv = [sys.argv[0], 'a', 'b', 'c', 'd', '--nInt_to_load', '6', '--verbose']
        a = abscal.post_redcal_abscal_argparser()
        assert a.data_file == 'a'
        assert a.redcal_file == 'b'
        assert a.model_files[0] == 'c'
        assert a.model_files[1] == 'd'
        assert len(a.model_files) == 2
        assert type(a.model_files) == list
        assert a.nInt_to_load == 6
        assert a.verbose is True

    def test_multiply_gains_argparser(self):
        sys.argv = [sys.argv[0], 'a', 'b', 'c', '--clobber']
        a = abscal.multiply_gains_argparser()
        a = a.parse_args()
        assert a.gain_file_1 == 'a'
        assert a.gain_file_2 == 'b'
        assert a.output_file == 'c'
        assert a.clobber
        assert a.divide_gains is False
