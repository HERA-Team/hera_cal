# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Unit tests for the hera_cal.apply_cal module."""

import pytest
import numpy as np
from copy import deepcopy
import os
import sys
import shutil
from scipy import constants
import warnings
from pyuvdata import UVCal, UVData

from .. import io
from .. import apply_cal as ac
from ..datacontainer import DataContainer
from ..data import DATA_PATH
from .. import utils
from .. import redcal
from hera_qm import metrics_io


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
@pytest.mark.filterwarnings("ignore:It seems that the latitude and longitude are in radians")
@pytest.mark.filterwarnings("ignore:Mean of empty slice")
class Test_Update_Cal(object):
    def test_check_polarization_consistency(self):
        gains = {(0, 'Jnn'): np.zeros((2, 2))}
        data = {(0, 1, 'nn'): np.zeros((2, 2))}
        ac._check_polarization_consistency(data, gains)

        gains = {(0, 'Jnn'): np.zeros((2, 2))}
        data = {(0, 1, 'xx'): np.zeros((2, 2))}
        with pytest.raises(KeyError):
            ac._check_polarization_consistency(data, gains)

        gains = {(0, 'Jxx'): np.zeros((2, 2))}
        data = {(0, 1, 'nn'): np.zeros((2, 2))}
        with pytest.raises(KeyError):
            ac._check_polarization_consistency(data, gains)

    def test_build_gains_by_cadences(self):
        # test upsampling
        data = {(0, 1, 'nn'): np.ones((8, 10), dtype=complex)}
        gains = {(0, 'Jnn'): np.array([np.arange(10)]).repeat(2, axis=0).astype(complex)}
        flags = {(0, 'Jnn'): np.zeros((2, 10), dtype=bool)}
        gains_by_Nt, cal_flags_by_Nt = ac.build_gains_by_cadences(data, gains, cal_flags=flags)
        for Nt in [2, 4, 8]:
            assert Nt in gains_by_Nt
            assert Nt in cal_flags_by_Nt
            assert gains_by_Nt[Nt][(0, 'Jnn')].shape[0] == Nt
            assert cal_flags_by_Nt[Nt][(0, 'Jnn')].shape[0] == Nt
            np.testing.assert_array_equal(gains_by_Nt[Nt][(0, 'Jnn')], np.outer(np.ones(Nt), np.arange(10).astype(complex)))
            assert not np.any(cal_flags_by_Nt[Nt][(0, 'Jnn')])

        # test downsampling without flags
        data = {(0, 1, 'nn'): np.ones((1, 3), dtype=complex)}
        gains = {(0, 'Jnn'): np.outer(np.arange(4), np.ones(3)).astype(complex)}
        gains_by_Nt, cal_flags_by_Nt = ac.build_gains_by_cadences(data, gains)
        assert cal_flags_by_Nt is None
        for Nt in [1, 2, 4]:
            assert Nt in gains_by_Nt
            assert gains_by_Nt[Nt][(0, 'Jnn')].shape[0] == Nt
            if Nt == 1:
                np.testing.assert_array_equal(gains_by_Nt[Nt][(0, 'Jnn')], 1.5)
            if Nt == 2:
                np.testing.assert_array_equal(gains_by_Nt[Nt][(0, 'Jnn')], np.outer([.5, 2.5], np.ones(3)))
            if Nt == 4:
                np.testing.assert_array_equal(gains_by_Nt[Nt][(0, 'Jnn')], gains[0, 'Jnn'])

        # test downsampling
        data = {(0, 1, 'nn'): np.ones((1, 3), dtype=complex)}
        gains = {(0, 'Jnn'): np.outer(np.arange(4), np.ones(3)).astype(complex)}
        flags = {(0, 'Jnn'): np.zeros((4, 3), dtype=bool)}
        flags[(0, 'Jnn')][::3, 0] = True
        gains_by_Nt, cal_flags_by_Nt = ac.build_gains_by_cadences(data, gains, cal_flags=flags)
        for Nt in [1, 2, 4]:
            assert Nt in gains_by_Nt
            assert Nt in cal_flags_by_Nt
            assert gains_by_Nt[Nt][(0, 'Jnn')].shape[0] == Nt
            assert cal_flags_by_Nt[Nt][(0, 'Jnn')].shape[0] == Nt
            assert not np.any(cal_flags_by_Nt[Nt][(0, 'Jnn')][:, 1:])
            if Nt < 4:
                assert np.all(cal_flags_by_Nt[Nt][(0, 'Jnn')][:, 0])
            if Nt == 1:
                np.testing.assert_array_equal(gains_by_Nt[Nt][(0, 'Jnn')][:, 1:], 1.5)
            if Nt == 2:
                np.testing.assert_array_equal(gains_by_Nt[Nt][(0, 'Jnn')][:, 1:], np.outer([.5, 2.5], np.ones(2)))
            if Nt == 4:
                np.testing.assert_array_equal(gains_by_Nt[Nt][(0, 'Jnn')], gains[0, 'Jnn'])

        # test downsampling with flags as weights
        data = {(0, 1, 'nn'): np.ones((1, 3), dtype=complex)}
        gains = {(0, 'Jnn'): np.outer(np.arange(4), np.ones(3)).astype(complex)}
        flags = {(0, 'Jnn'): np.ones((4, 3), dtype=float)}
        flags[(0, 'Jnn')][::3, 0] = 0
        gains_by_Nt, cal_flags_by_Nt = ac.build_gains_by_cadences(data, gains, cal_flags=flags, flags_are_wgts=True)
        for Nt in [1, 2, 4]:
            assert Nt in gains_by_Nt
            assert Nt in cal_flags_by_Nt
            assert gains_by_Nt[Nt][(0, 'Jnn')].shape[0] == Nt
            assert cal_flags_by_Nt[Nt][(0, 'Jnn')].shape[0] == Nt
            np.testing.assert_array_equal(cal_flags_by_Nt[Nt][(0, 'Jnn')][:, 1:], 1.0)
            if Nt < 4:
                assert np.all(cal_flags_by_Nt[Nt][(0, 'Jnn')][:, 0])
            if Nt == 1:
                np.testing.assert_array_equal(gains_by_Nt[Nt][(0, 'Jnn')][:, 1:], 1.5)
            if Nt == 2:
                np.testing.assert_array_equal(gains_by_Nt[Nt][(0, 'Jnn')][:, 1:], np.outer([.5, 2.5], np.ones(2)))
            if Nt == 4:
                np.testing.assert_array_equal(gains_by_Nt[Nt][(0, 'Jnn')], gains[0, 'Jnn'])

        # test empty dicts
        data = {(0, 1, 'nn'): np.ones((1, 3), dtype=complex),
                (0, 2, 'nn'): np.ones((2, 3), dtype=complex)}
        gains_by_Nt, cal_flags_by_Nt = ac.build_gains_by_cadences(data, {}, cal_flags={})
        assert gains_by_Nt == {1: {}, 2: {}}
        assert cal_flags_by_Nt == {1: {}, 2: {}}

        # test warnings
        with pytest.warns(UserWarning, match='is inconsistent with BDA by powers of 2'):
            data = {(0, 1, 'nn'): np.ones((1, 3), dtype=complex),
                    (0, 2, 'nn'): np.ones((3, 3), dtype=complex)}
            ac.build_gains_by_cadences(data, {})
        with pytest.warns(UserWarning, match='cannot be calibrated with any of gain cadences'):
            data = {(0, 1, 'nn'): np.ones((2, 3), dtype=complex),
                    (0, 2, 'nn'): np.ones((3, 3), dtype=complex)}
            gains = {(0, 'Jnn'): np.ones((2, 3), dtype=complex)}
            ac.build_gains_by_cadences(data, gains)

    def test_calibrate_avg_gains_in_place(self):
        np.random.seed(20)
        vis = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        f = np.random.randn(10, 10) > 0
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        dns = DataContainer({(0, 1, 'xx'): np.ones((10, 10))})
        g0_new = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        g1_new = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        g_new = {(0, 'Jxx'): g0_new, (1, 'Jxx'): g1_new}
        g0_old = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        g1_old = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        g_old = {(0, 'Jxx'): g0_old, (1, 'Jxx'): g1_old}
        f_old = {(0, 'Jxx'): np.random.randn(10, 10) > 0, (1, 'Jxx'): np.random.randn(10, 10) > 0}
        f_new = {(0, 'Jxx'): np.random.randn(10, 10) > 0, (1, 'Jxx'): np.random.randn(10, 10) > 0}
        all_reds = [[(0, 1, 'xx')]]

        # test average
        ac.calibrate_redundant_solution(dc, flags, g_new, f_new, all_reds,
                                        old_gains=g_old, old_flags=f_old, gain_convention='divide')
        gain_ratios = g_old[(0, 'Jxx')] * np.conj(g_old[(1, 'Jxx')]) / g_new[(0, 'Jxx')] / np.conj(g_new[(1, 'Jxx')])
        flagged = f_old[(0, 'Jxx')] | f_old[(1, 'Jxx')] | f_new[(0, 'Jxx')] | f_new[(1, 'Jxx')]
        gain_ratios[flagged] = np.nan
        avg_gains = np.nanmean(np.array([gain_ratios]), axis=0)
        avg_flags = ~np.isfinite(avg_gains)
        avg_gains[avg_flags] = 1. + 0.j

        for i in range(10):
            for j in range(10):
                if not np.isfinite(dc[(0, 1, 'xx')][i, j]):
                    assert np.allclose(dc[(0, 1, 'xx')][i, j], vis[i, j] * avg_gains[i, j])

    def test_apply_redundant_solutions(self, tmpdir):
        tmp_path = tmpdir.strpath
        miriad = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uvOCR_53x_54x_only")
        outname_uvh5 = os.path.join(tmp_path, "red_out.uvh5")
        old_cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        new_cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        ac.apply_cal(miriad, outname_uvh5, new_cal, old_calibration=old_cal, filetype_in='miriad', filetype_out='uvh5',
                     gain_convention='divide', redundant_solution=True, add_to_history='', clobber=True)
        # checking if file is created
        assert os.path.exists(outname_uvh5)

        # checking average
        inp_hc = io.HERAData(miriad, filetype='miriad')
        inp_data, inp_flags, _ = inp_hc.read()
        out_hc = io.HERAData(outname_uvh5)
        out_data, out_flags, _ = out_hc.read()
        np.testing.assert_almost_equal(inp_data[(54, 54, 'ee')], out_data[(54, 54, 'ee')])
        os.remove(outname_uvh5)

        # Now test with partial I/O
        uv = UVData()
        uv.read_miriad(miriad)
        uv.use_future_array_shapes()
        inname_uvh5 = os.path.join(tmp_path, "red_in.uvh5")
        uv.write_uvh5(inname_uvh5)
        ac.apply_cal(inname_uvh5, outname_uvh5, new_cal, old_calibration=old_cal, filetype_in='uvh5', filetype_out='uvh5',
                     gain_convention='divide', redundant_solution=True, nbl_per_load=1, add_to_history='', clobber=True)
        os.remove(inname_uvh5)
        # checking if file is created
        assert os.path.exists(outname_uvh5)

        # checking average
        inp_hc = io.HERAData(miriad, filetype='miriad')
        inp_data, inp_flags, _ = inp_hc.read()
        out_hc = io.HERAData(outname_uvh5)
        out_data, out_flags, _ = out_hc.read()
        np.testing.assert_almost_equal(inp_data[(54, 54, 'ee')], out_data[(54, 54, 'ee')])
        os.remove(outname_uvh5)

    def test_calibrate_in_place(self):
        np.random.seed(21)
        vis = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        f = np.random.randn(10, 10) > 0
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        g0_new = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        g1_new = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        g_new = {(0, 'Jxx'): g0_new, (1, 'Jxx'): g1_new}
        g0_old = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        g1_old = np.random.randn(10, 10) + 1.0j * np.random.randn(10, 10)
        g_old = {(0, 'Jxx'): g0_old, (1, 'Jxx'): g1_old}
        cal_flags = {(0, 'Jxx'): np.random.randn(10, 10) > 0, (1, 'Jxx'): np.random.randn(10, 10) > 0}
        # test standard operation
        ac.calibrate_in_place(dc, g_new, flags, cal_flags, old_gains=g_old, gain_convention='divide')
        for i in range(10):
            for j in range(10):
                assert np.allclose(dc[(0, 1, 'xx')][i, j], vis[i, j] * g0_old[i, j] * np.conj(g1_old[i, j]) / g0_new[i, j] / np.conj(g1_new[i, j]))
                if f[i, j] or cal_flags[(0, 'Jxx')][i, j] or cal_flags[(1, 'Jxx')][i, j]:
                    assert np.all(flags[(0, 1, 'xx')][i, j])
                else:
                    assert not np.any(flags[(0, 1, 'xx')][i, j])

        # test without old cal
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        ac.calibrate_in_place(dc, g_new, flags, cal_flags, gain_convention='divide')
        for i in range(10):
            for j in range(10):
                assert np.allclose(dc[(0, 1, 'xx')][i, j], vis[i, j] / g0_new[i, j] / np.conj(g1_new[i, j]))

        # test multiply
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        ac.calibrate_in_place(dc, g_new, flags, cal_flags, old_gains=g_old, gain_convention='multiply')
        for i in range(10):
            for j in range(10):
                assert np.allclose(dc[(0, 1, 'xx')][i, j], vis[i, j] / g0_old[i, j] / np.conj(g1_old[i, j]) * g0_new[i, j] * np.conj(g1_new[i, j]))

        # test flag propagation when missing antennas in gains
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        ac.calibrate_in_place(dc, {}, flags, cal_flags, gain_convention='divide')
        np.testing.assert_array_equal(flags[(0, 1, 'xx')], True)
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        ac.calibrate_in_place(dc, g_new, flags, cal_flags, old_gains={}, gain_convention='divide')
        np.testing.assert_array_equal(flags[(0, 1, 'xx')], True)

        # test error
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        with pytest.raises(KeyError):
            ac.calibrate_in_place(dc, g_new, flags, cal_flags, old_gains=g_old, gain_convention='blah')

        # test w/ data weights
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis)})
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f)})
        wgts = DataContainer({k: (~flags[k]).astype(float) for k in flags.keys()})
        del g_new[(0, 'Jxx')]
        ac.calibrate_in_place(dc, g_new, wgts, cal_flags, gain_convention='divide', flags_are_wgts=True)
        assert np.allclose(wgts[(0, 1, 'xx')].max(), 0.0)

        # test BDA runs without error
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis), (0, 2, 'xx'): deepcopy(vis[0:5, :])})
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f), (0, 2, 'xx'): deepcopy(f[0:5, :])})
        g_here = deepcopy(g_new)
        g_here[2, 'Jxx'] = deepcopy(g_here[1, 'Jxx'])
        ac.calibrate_in_place(dc, g_here, flags)

        # test BDA cadence errors
        dc = DataContainer({(0, 1, 'xx'): deepcopy(vis), (0, 2, 'xx'): deepcopy(vis[0:5, :])})
        flags = DataContainer({(0, 1, 'xx'): deepcopy(f), (0, 2, 'xx'): deepcopy(f[0:5, :])})
        g_here = {(0, 'Jxx'): g0_new[0:3, :], (1, 'Jxx'): g1_new[0:3, :]}
        with pytest.raises(ValueError, match='new_gains with'):
            ac.calibrate_in_place(dc, g_here, data_flags=flags, cal_flags=None, old_gains=None)
        g_here = {(0, 'Jxx'): g0_new[0:1, :], (1, 'Jxx'): g1_new[0:1, :]}
        cal_flags_here = {(0, 'Jxx'): cal_flags[(0, 'Jxx')][0:7, :], (1, 'Jxx'): cal_flags[(1, 'Jxx')][0:7, :]}
        with pytest.raises(ValueError, match='cal_flags with'):
            ac.calibrate_in_place(dc, g_here, data_flags=flags, cal_flags=cal_flags_here, old_gains=None)
        old_g_here = {(0, 'Jxx'): g0_old[0:8, :], (1, 'Jxx'): g1_old[0:8, :]}
        with pytest.raises(ValueError, match='old_gains with'):
            ac.calibrate_in_place(dc, g_here, data_flags=flags, cal_flags=None, old_gains=old_g_here)

    def test_apply_cal(self, tmpdir):
        tmp_path = tmpdir.strpath
        miriad = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uvOCR_53x_54x_only")
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outname_miriad = os.path.join(tmp_path, "out.uv")
        outname_uvh5 = os.path.join(tmp_path, "out.h5")
        calout = os.path.join(tmp_path, "out.cal")
        old_cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        new_cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        flags_npz = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uvOCR_53x_54x_only.flags.applied.npz")

        hd_old = io.HERAData(miriad, filetype='miriad')
        hd_old.read()
        hd_old.flag_array = np.logical_or(hd_old.flag_array, np.load(flags_npz)['flag_array'])
        data, data_flags, _ = hd_old.build_datacontainers()

        new_gains, new_flags = io.load_cal(new_cal)
        uvc_old = UVCal()
        uvc_old.read_calfits(old_cal)
        uvc_old.gain_array *= (3.0 + 4.0j)
        uvc_old.write_calfits(calout, clobber=True)

        ac.apply_cal(miriad, outname_miriad, new_cal, old_calibration=calout, gain_convention='divide',
                     flag_nchan_low=450, flag_nchan_high=400, flags_npz=flags_npz,
                     filetype_in='miriad', filetype_out='miriad', clobber=True, vis_units='Jy',
                     add_to_history='testing')
        hd = io.HERAData(outname_miriad, filetype='miriad')
        new_data, new_flags, _ = hd.read()
        assert 'testing' in hd.history.replace('\n', '').replace(' ', '')
        assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')
        assert hd.vis_units == 'Jy'
        for k in new_data.keys():
            for i in range(new_data[k].shape[0]):
                for j in range(new_data[k].shape[1]):
                    if not new_flags[k][i, j]:
                        assert np.allclose(new_data[k][i, j] / 25.0 / data[k][i, j], 1.0, atol=1e-4)
                    # from flag_nchan_low and flag_nchan_high above with 1024 total channels
                    if j < 450 or j > 623:
                        assert np.all(new_flags[k][i, j])

        # test partial load
        ac.apply_cal(uvh5, outname_uvh5, new_cal, old_calibration=calout, gain_convention='divide',
                     flag_nchan_low=450, flag_nchan_high=400, flags_npz=flags_npz, nbl_per_load=1,
                     filetype_in='uvh5', filetype_out='uvh5', clobber=True, vis_units='Jy')
        hd = io.HERAData(outname_uvh5, filetype='uvh5')
        new_data, new_flags, _ = hd.read()
        assert hd.vis_units == 'Jy'
        for k in new_data.keys():
            for i in range(new_data[k].shape[0]):
                for j in range(new_data[k].shape[1]):
                    if not new_flags[k][i, j]:
                        assert np.allclose(new_data[k][i, j] / 25.0 / data[k][i, j], 1.0, atol=1e-4)
                    # from flag_nchan_low and flag_nchan_high above with 1024 total channels
                    if j < 450 or j > 623:
                        assert np.all(new_flags[k][i, j])
        os.remove(outname_uvh5)

        # test errors
        with pytest.raises(ValueError):
            ac.apply_cal(miriad, outname_miriad, None)
        with pytest.raises(NotImplementedError):
            ac.apply_cal(miriad, outname_uvh5, new_cal, filetype_in='miriad', nbl_per_load=1)
        shutil.rmtree(outname_miriad)

        # test flagging yaml
        flag_yaml = os.path.join(DATA_PATH, 'test_input/a_priori_flags_sample_53_flagged.yaml')
        ac.apply_cal(uvh5, outname_uvh5, new_cal, old_calibration=calout, gain_convention='divide',
                     flags_npz=flags_npz,
                     filetype_in='uvh5', filetype_out='uvh5', clobber=True, vis_units='Jy', a_priori_flags_yaml=flag_yaml)
        hd = io.HERAData(outname_uvh5)
        new_data, new_flags, _ = hd.read()
        # check that all antennas, integrations, and frequencies from this yaml are flagged.
        flagged_ints = metrics_io.read_a_priori_int_flags(flag_yaml, times=hd.times, lsts=hd.lsts * 12 / np.pi)
        flagged_chans = metrics_io.read_a_priori_chan_flags(flag_yaml, freqs=hd.freqs)
        flagged_ants = metrics_io.read_a_priori_ant_flags(flag_yaml, ant_indices_only=True)
        for bl in new_flags:
            if bl[0] in flagged_ants or bl[1] in flagged_ants:
                assert np.all(new_flags[bl])
            assert np.all(new_flags[bl][flagged_ints])
            assert np.all(new_flags[bl][:, flagged_chans])

    def test_apply_cal_units(self, tmpdir):
        tmp_path = tmpdir.strpath
        # test that units are propagated from calibration gains to calibrated data.
        new_cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")

        uvd_with_units = UVData()
        uvd_with_units.read_uvh5(uvh5)
        uvd_with_units.use_future_array_shapes()
        uvd_with_units.vis_units = 'k str'
        uvh5_units = os.path.join(tmp_path, 'test_input_kstr.uvh5')
        uvd_with_units.write_uvh5(uvh5_units)

        hc = io.HERACal(new_cal)
        hc.read()
        # manually set gain-scale.
        hc.gain_scale = 'Jy'
        calfile = os.path.join(tmp_path, 'test_cal.calfits')
        output = os.path.join(tmp_path, 'test_calibrated_output.uvh5')
        hc.write_calfits(calfile)

        with pytest.warns(RuntimeWarning):
            ac.apply_cal(uvh5_units, output, calfile)
        hdc = io.HERAData(output)
        assert hdc.vis_units == 'Jy'
        ac.apply_cal(uvh5, output, calfile, vis_units='k str', clobber=True)
        hdc = io.HERAData(output)
        assert hdc.vis_units == 'k str'
        # test red_average mode.
        with pytest.warns(RuntimeWarning):
            ac.apply_cal(uvh5_units, output, calfile, clobber=True, redundant_average=True)
        hdc = io.HERAData(output)
        assert hdc.vis_units == 'Jy'
        ac.apply_cal(uvh5, output, calfile, clobber=True, redundant_average=True, vis_units='k str')
        hdc = io.HERAData(output)
        assert hdc.vis_units == 'k str'
        # do this with nbl_per_load set.
        with pytest.warns(RuntimeWarning):
            ac.apply_cal(uvh5_units, output, calfile, nbl_per_load=4, clobber=True)
        hdc = io.HERAData(output)
        assert hdc.vis_units == 'Jy'
        ac.apply_cal(uvh5, output, calfile, vis_units='k str', clobber=True, nbl_per_load=4)
        hdc = io.HERAData(output)
        assert hdc.vis_units == 'k str'
        # test red_average mode.
        with pytest.warns(RuntimeWarning):
            ac.apply_cal(uvh5_units, output, calfile, clobber=True, redundant_average=True, nbl_per_load=4)
        hdc = io.HERAData(output)
        assert hdc.vis_units == 'Jy'
        ac.apply_cal(uvh5, output, calfile, clobber=True, redundant_average=True, vis_units='k str', nbl_per_load=4)
        hdc = io.HERAData(output)
        assert hdc.vis_units == 'k str'
        # test red_average mode with partial i/o.
        with pytest.warns(RuntimeWarning):
            ac.apply_cal(uvh5_units, output, calfile, clobber=True, redundant_average=True, nbl_per_load=4)
        hdc = io.HERAData(output)
        assert hdc.vis_units == 'Jy'
        # test red_average mode with baseline groups.
        uncalibrated_file = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uncalibrated.uvh5")
        hdt = io.HERAData(uncalibrated_file)
        d, f, n = hdt.read()
        for bl in f:
            if not np.all(f[bl]):
                bl_not_flagged = bl
                break
        for bl in f:
            if not np.all(f[bl]):
                f[bl] = f[bl_not_flagged]
                n[bl] = n[bl_not_flagged]
        hdt.update(data=d, flags=f, nsamples=n)
        hdt.vis_units = 'k str'
        uncalibrated_file_homogenous_nsamples_flags = os.path.join(tmp_path, 'homogenous_nsamples_flags.uvh5')
        hdt.write_uvh5(uncalibrated_file_homogenous_nsamples_flags)
        with pytest.warns(RuntimeWarning):
            ac.apply_cal(uncalibrated_file_homogenous_nsamples_flags,
                         output, calfile, clobber=True, redundant_average=True, redundant_groups=3)
        for grpnum in range(3):
            hdc = io.HERAData(output.replace('.uvh5', f'.{grpnum}.uvh5'))
            assert hdc.vis_units == 'Jy'
        ac.apply_cal(uncalibrated_file_homogenous_nsamples_flags,
                     output, calfile, clobber=True, redundant_average=True, redundant_groups=3, vis_units='k str')
        for grpnum in range(3):
            hdc = io.HERAData(output.replace('.uvh5', f'.{grpnum}.uvh5'))
            assert hdc.vis_units == 'k str'

    def test_apply_cal_redundant_averaging(self, tmpdir):
        tmp_path = tmpdir.strpath
        # test redundant averaging functionality in apply_cal
        # we will do this by applying a calibration to a data set and then running red_average
        # on its output. We will then check that this gives the same results as activating the
        # red_average option in apply_cal.
        hd_calibrated = io.HERAData(os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5"))
        d, f, n = hd_calibrated.read()
        uncalibrated_file = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uncalibrated.uvh5")
        calibrated_redundant_averaged_file = os.path.join(tmp_path, "zen.2458043.40141.xx.HH.XRAA.redundantly_averaged.uvh5")
        calibrated_file = os.path.join(tmp_path, "zen.2458043.40141.xx.HH.XRAA.calibrated.uvh5")
        calfile = os.path.join(DATA_PATH, 'zen.2458043.40141.xx.HH.XRAA.abs.calfits')
        calfile_unity = os.path.join(DATA_PATH, 'zen.2458043.40141.xx.HH.XRAA.unity_gains.abs.calfits')
        # redundantly average the calibrated data file.
        reds = redcal.get_reds(hd_calibrated.antpos, bl_error_tol=1.0, include_autos=True)

        # apply_cal without redundant averaging and check that data arrays etc... are the same
        ac.apply_cal(uncalibrated_file, calibrated_file, calfile,
                     gain_convention='divide', redundant_average=False)
        hd_calibrated_with_apply_cal = io.HERAData(calibrated_file)
        hd_calibrated_with_apply_cal.read()
        hc_unity = io.HERACal(calfile_unity)
        g, gf, _, _ = hc_unity.read()
        ac.calibrate_in_place(data=d, new_gains=g, cal_flags=gf, data_flags=f)
        hd_calibrated.update(flags=f, data=d)
        assert np.all(np.isclose(hd_calibrated.data_array, hd_calibrated_with_apply_cal.data_array))
        assert np.all(np.isclose(hd_calibrated.nsample_array, hd_calibrated_with_apply_cal.nsample_array))
        assert np.all(np.isclose(hd_calibrated.flag_array, hd_calibrated_with_apply_cal.flag_array))

        # remove polarizations for red_average
        reds = [[bl[:2] for bl in redgrp] for redgrp in reds]
        wgts = deepcopy(n)
        for bl in wgts:
            if np.all(f[bl]):
                wgts[bl][:] = 0.
        hda_calibrated = utils.red_average(hd_calibrated, reds, inplace=False, wgts=wgts, propagate_flags=True)

        ac.apply_cal(uncalibrated_file, calibrated_redundant_averaged_file, calfile,
                     gain_convention='divide', redundant_average=True)

        # now load in the calibrated redundant data.
        hda_calibrated_with_apply_cal = io.HERAData(calibrated_redundant_averaged_file)
        hda_calibrated_with_apply_cal.read()

        # check that the data, flags, and nsamples arrays are close
        assert np.all(np.isclose(hda_calibrated.nsample_array, hda_calibrated_with_apply_cal.nsample_array))
        assert np.all(np.isclose(hda_calibrated.flag_array, hda_calibrated_with_apply_cal.flag_array))
        assert np.all(np.isclose(hda_calibrated.data_array, hda_calibrated_with_apply_cal.data_array))

        # now do chunked redundant groups.
        ac.apply_cal(uncalibrated_file, calibrated_redundant_averaged_file, calfile,
                     gain_convention='divide', redundant_average=True, nbl_per_load=4, clobber=True)
        hda_calibrated_with_apply_cal = io.HERAData(calibrated_redundant_averaged_file)
        hda_calibrated_with_apply_cal.read()
        # check that the data, flags, and nsamples arrays are close
        assert np.all(np.isclose(hda_calibrated.nsample_array, hda_calibrated_with_apply_cal.nsample_array))
        assert np.all(np.isclose(hda_calibrated.flag_array, hda_calibrated_with_apply_cal.flag_array))
        assert np.all(np.isclose(hda_calibrated.data_array, hda_calibrated_with_apply_cal.data_array))

        # now do chunked redundant groups with a large group size to catch a bug.
        ac.apply_cal(uncalibrated_file, calibrated_redundant_averaged_file, calfile,
                     gain_convention='divide', redundant_average=True, nbl_per_load=1000000, clobber=True)
        hda_calibrated_with_apply_cal = io.HERAData(calibrated_redundant_averaged_file)
        hda_calibrated_with_apply_cal.read()
        # check that the data, flags, and nsamples arrays are close
        assert np.all(np.isclose(hda_calibrated.nsample_array, hda_calibrated_with_apply_cal.nsample_array))
        assert np.all(np.isclose(hda_calibrated.flag_array, hda_calibrated_with_apply_cal.flag_array))
        assert np.all(np.isclose(hda_calibrated.data_array, hda_calibrated_with_apply_cal.data_array))
        dcal, fcal, ncal = hd_calibrated.build_datacontainers()

        with pytest.raises(NotImplementedError):
            ac.apply_cal(uncalibrated_file, calibrated_redundant_averaged_file, calfile, dont_red_average_flagged_data=True,
                         gain_convention='divide', redundant_average=True, nbl_per_load=2, clobber=True)

        # prepare calibrated file where all baselines have the same nsamples and the same flagging pattern if they are not all flagged.
        hdt = io.HERAData(uncalibrated_file)
        d, f, n = hdt.read()
        for bl in f:
            if not np.all(f[bl]):
                bl_not_flagged = bl
                break
        for bl in f:
            if not np.all(f[bl]):
                f[bl] = f[bl_not_flagged]
                n[bl] = n[bl_not_flagged]
        hdt.update(data=d, flags=f, nsamples=n)
        uncalibrated_file_homogenous_nsamples_flags = os.path.join(tmp_path, 'homogenous_nsamples_flags.uvh5')
        hdt.write_uvh5(uncalibrated_file_homogenous_nsamples_flags)

        # check not implemented error for partial i/o with redundant_groups > 1
        with pytest.raises(NotImplementedError):
            ac.apply_cal(uncalibrated_file, calibrated_redundant_averaged_file, calfile, dont_red_average_flagged_data=True,
                         gain_convention='divide', redundant_average=True, nbl_per_load=2, clobber=True, redundant_groups=2)

        # single redundant group for comparison.
        ac.apply_cal(uncalibrated_file_homogenous_nsamples_flags, calibrated_redundant_averaged_file, calfile, dont_red_average_flagged_data=True,
                     gain_convention='divide', redundant_average=True, nbl_per_load=None, clobber=True)
        hda_calibrated_with_apply_cal = io.HERAData(calibrated_redundant_averaged_file)
        hda_calibrated_with_apply_cal.read()
        d1, f1, n1 = hda_calibrated_with_apply_cal.build_datacontainers()

        for ngrps in range(3, 6):
            hda_calibrated_groups = []
            ac.apply_cal(uncalibrated_file_homogenous_nsamples_flags, calibrated_redundant_averaged_file, calfile, dont_red_average_flagged_data=True,
                         gain_convention='divide', redundant_average=True, nbl_per_load=None, clobber=True, redundant_groups=ngrps)
            for rc in range(ngrps):
                hda_calibrated_groups.append(io.HERAData(calibrated_redundant_averaged_file.replace('.uvh5', f'.{rc}.uvh5')))
                hda_calibrated_groups[-1].read()
                os.remove(calibrated_redundant_averaged_file.replace('.uvh5', f'.{rc}.uvh5'))
            # check that the sum of nsample arrays is equal to the nsamples in the redgroup in the original data.
            for m in range(len(hda_calibrated_groups)):
                _, _, nt = hda_calibrated_groups[m].build_datacontainers()
                if m == 0:
                    nsum = nt
                else:
                    nsum += nt
            for bl in nsum:
                assert np.all(np.isclose(n1[bl], nsum[bl]))

            equal_flags = []
            equal_times = []
            equal_baselines = []
            equal_data = []
            for m in range(ngrps - 1):
                equal_flags.append(np.all(np.isclose(hda_calibrated_groups[m].flag_array, hda_calibrated_groups[m + 1].flag_array)))
                equal_times.append(np.all(np.isclose(hda_calibrated_groups[m].time_array, hda_calibrated_groups[m + 1].time_array)))
                equal_data.append(np.all(np.isclose(hda_calibrated_groups[m].data_array, hda_calibrated_groups[m + 1].data_array)))
                equal_baselines.append(np.all(np.isclose(hda_calibrated_groups[m].baseline_array, hda_calibrated_groups[m + 1].baseline_array)))
            # check all flag arrays are equal
            assert np.all(equal_flags)
            # check that all baseline and time arrays are equal
            assert np.all(equal_baselines)
            assert np.all(equal_times)
            # check that data is not equal.
            assert not np.any(equal_data)

    def test_apply_cal_bda(self):
        upsampled_oc = os.path.join(DATA_PATH, 'zen.2459122.30030.sum.bda.downsampled.upsample_in_time.omni.calfits')
        downsampled_oc = os.path.join(DATA_PATH, 'zen.2459122.30030.sum.bda.downsampled.downsample_in_time.omni.calfits')

        # load input data file
        infile = os.path.join(DATA_PATH, 'zen.2459122.30030.sum.bda.downsampled.uvh5')
        hd_in = io.HERAData(infile)
        d_in, f_in, n_in = hd_in.read()

        # Try calibrating BDA data with omnical solution from downsampling
        outfile = os.path.join(DATA_PATH, 'zen.2459122.30030.sum.bda.downsampled.down_calibrated.uvh5')
        ac.apply_cal(infile, outfile, downsampled_oc, clobber=True)
        hd = io.HERAData(outfile)
        d, f, n = hd.read()
        for bl in d:
            assert d[bl].shape == d_in[bl].shape
        os.remove(outfile)

        # Try calibrating BDA data with omnical solution from upsampling
        outfile = os.path.join(DATA_PATH, 'zen.2459122.30030.sum.bda.downsampled.up_calibrated.uvh5')
        ac.apply_cal(infile, outfile, upsampled_oc, clobber=True)
        hd = io.HERAData(outfile)
        d, f, n = hd.read()
        for bl in d:
            assert d[bl].shape == d_in[bl].shape
        os.remove(outfile)

        # Try calibrating BDA and then downsampled data with omnical solution from downsampling
        outfile = os.path.join(DATA_PATH, 'zen.2459122.30030.sum.bda.downsampled.down_calibrated.uvh5')
        ac.apply_cal(infile, outfile, downsampled_oc, clobber=True, downsample=True)
        hd = io.HERAData(outfile)
        d, f, n = hd.read()
        for bl in d:
            assert d[bl].shape[0] == 1
        os.remove(outfile)

        # Try calibrating BDA and then upsampled data with omnical solution from upsampling
        outfile = os.path.join(DATA_PATH, 'zen.2459122.30030.sum.bda.downsampled.up_calibrated.uvh5')
        ac.apply_cal(infile, outfile, upsampled_oc, clobber=True, upsample=True)
        hd = io.HERAData(outfile)
        d, f, n = hd.read()
        for bl in d:
            assert d[bl].shape[0] == 8
        os.remove(outfile)

    def test_apply_cal_argparser(self):
        sys.argv = [sys.argv[0], 'a', 'b', '--new_cal', 'd']
        a = ac.apply_cal_argparser()
        args = a.parse_args()
        assert args.infilename == 'a'
        assert args.outfilename == 'b'
        assert args.new_cal == ['d']
