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
@pytest.mark.filterwarnings("ignore:telescope_location is not set")
@pytest.mark.filterwarnings("ignore:Fixing auto-correlations to be be real-only")
@pytest.mark.filterwarnings("ignore:antenna_positions are not set")
@pytest.mark.filterwarnings("ignore:Selected frequencies are not contiguous")
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
            with pytest.warns(UserWarning, match='is inconsistent with BDA by powers of 2'):

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
            with pytest.warns(UserWarning, match='integrations cannot be calibrated'):
                ac.calibrate_in_place(dc, g_here, data_flags=flags, cal_flags=None, old_gains=None)
        g_here = {(0, 'Jxx'): g0_new[0:1, :], (1, 'Jxx'): g1_new[0:1, :]}
        cal_flags_here = {(0, 'Jxx'): cal_flags[(0, 'Jxx')][0:7, :], (1, 'Jxx'): cal_flags[(1, 'Jxx')][0:7, :]}
        with pytest.raises(ValueError, match='cal_flags with'):
            ac.calibrate_in_place(dc, g_here, data_flags=flags, cal_flags=cal_flags_here, old_gains=None)
        old_g_here = {(0, 'Jxx'): g0_old[0:8, :], (1, 'Jxx'): g1_old[0:8, :]}
        with pytest.raises(ValueError, match='old_gains with'):
            with pytest.warns(UserWarning, match="integrations cannot be calibrated"):
                ac.calibrate_in_place(dc, g_here, data_flags=flags, cal_flags=None, old_gains=old_g_here)

    @pytest.mark.filterwarnings("ignore:writing default values for restfreq")
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Replacing original data vis_units of k str")
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="baseline group of length 7 encountered")

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

    @pytest.mark.filterwarnings("ignore:Fixing phases using antenna positions")
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


class TestCorrectSNAPDecoherence:
    '''Tests for ac.correct_SNAP_decoherence_in_place.'''

    def setup_method(self):
        np.random.seed(21)
        self.ntimes, self.nfreqs, self.ncpb = 3, 64, 16
        self.nblocks = 4
        self.SNAP_map = {0: 'A', 1: 'A', 2: 'B', 3: 'B'}
        self.pA = np.random.uniform(0.01, 0.1, (self.ntimes, self.nblocks))
        self.pB = np.random.uniform(0.01, 0.1, (self.ntimes, self.nblocks))
        self.deco = {'A': self.pA.copy(), 'B': self.pB.copy()}
        self.c2b = np.arange(self.nfreqs) // self.ncpb

    def _build_data(self):
        '''Synthetic data with the exact (1-p_i)(1-p_j) suppression injected
        on inter-SNAP crosses only; autos, cross-pol autos, and intra-SNAP
        baselines unsuppressed.'''
        bls = [(0, 0, 'ee'), (0, 0, 'en'), (0, 1, 'ee'),
               (0, 2, 'ee'), (1, 3, 'ee'), (0, 2, 'nn')]
        shape = (self.ntimes, self.nfreqs)
        true_vis = {bl: np.random.randn(*shape) + 1j * np.random.randn(*shape)
                    for bl in bls}
        coherence_factor = {s: 1 - self.deco[s][:, self.c2b] for s in self.deco}
        data = {}
        for bl in bls:
            i, j, pol = bl
            vis = true_vis[bl].copy()
            if self.SNAP_map[i] != self.SNAP_map[j]:
                vis *= (coherence_factor[self.SNAP_map[i]]
                        * coherence_factor[self.SNAP_map[j]])
            data[bl] = vis
        return DataContainer(data), true_vis

    def test_round_trip_and_exemptions(self):
        data, true_vis = self._build_data()
        untouched = {bl: data[bl].copy()
                     for bl in [(0, 0, 'ee'), (0, 0, 'en'), (0, 1, 'ee')]}
        ac.correct_SNAP_decoherence_in_place(data, self.deco, self.SNAP_map,
                                             nchans_per_block=self.ncpb)
        # inter-SNAP crosses recovered exactly, both pols
        for bl in [(0, 2, 'ee'), (1, 3, 'ee'), (0, 2, 'nn')]:
            np.testing.assert_allclose(data[bl], true_vis[bl], rtol=1e-12)
        # autos, cross-pol autos, and intra-SNAP baselines bit-identical
        for bl, vis in untouched.items():
            np.testing.assert_array_equal(data[bl], vis)

    def test_exact_product_form(self):
        # correction must be exp(+(ls_i + ls_j)), not 1/(1 - p_i - p_j)
        ls_A = np.random.uniform(0.02, 0.2, (self.ntimes, self.nblocks))
        ls_B = np.random.uniform(0.02, 0.2, (self.ntimes, self.nblocks))
        deco = {'A': 1 - np.exp(-ls_A), 'B': 1 - np.exp(-ls_B)}
        data = DataContainer({(0, 2, 'ee'): np.ones(
            (self.ntimes, self.nfreqs), dtype=complex)})
        ac.correct_SNAP_decoherence_in_place(data, deco, self.SNAP_map,
                                             nchans_per_block=self.ncpb)
        expected = np.exp((ls_A + ls_B)[:, self.c2b])
        np.testing.assert_allclose(data[(0, 2, 'ee')], expected, rtol=1e-12)

    def test_nan_over_flagged_ok(self):
        data, true_vis = self._build_data()
        suppressed = data[(0, 2, 'ee')].copy()
        self.deco['A'][1, 2] = np.nan
        flags = DataContainer({bl: np.zeros_like(data[bl], dtype=bool)
                               for bl in data})
        nan_chans = slice(2 * self.ncpb, 3 * self.ncpb)
        for bl in data:
            if self.SNAP_map[bl[0]] != self.SNAP_map[bl[1]]:
                flags[bl][1, nan_chans] = True
        flags_before = {bl: flags[bl].copy() for bl in flags}
        ac.correct_SNAP_decoherence_in_place(data, self.deco, self.SNAP_map,
                                             data_flags=flags,
                                             nchans_per_block=self.ncpb)
        # at unmeasured-A cells the A side contributes no correction but
        # the measured B side still applies (those cells are flagged)
        rB = (1 - self.pB[:, self.c2b])[1, nan_chans]
        np.testing.assert_allclose(data[(0, 2, 'ee')][1, nan_chans],
                                   suppressed[1, nan_chans] / rB, rtol=1e-12)
        np.testing.assert_allclose(data[(0, 2, 'ee')][0], true_vis[(0, 2, 'ee')][0],
                                   rtol=1e-12)
        # flags are never modified
        for bl in flags:
            np.testing.assert_array_equal(flags[bl], flags_before[bl])

    def test_nan_over_unflagged_raises(self):
        data, _ = self._build_data()
        self.deco['B'][0, 1] = np.nan
        # no flags at all: strictest reading, must raise
        with pytest.raises(ValueError, match='Unmeasured'):
            ac.correct_SNAP_decoherence_in_place(
                data, self.deco, self.SNAP_map, nchans_per_block=self.ncpb)
        # flags present but not covering the NaN cells: still raises
        flags = DataContainer({bl: np.zeros_like(data[bl], dtype=bool)
                               for bl in data})
        with pytest.raises(ValueError, match='Unmeasured'):
            ac.correct_SNAP_decoherence_in_place(
                data, self.deco, self.SNAP_map, data_flags=flags,
                nchans_per_block=self.ncpb)

    def test_validation_errors(self):
        data, _ = self._build_data()
        incomplete = {antnum: s for antnum, s in self.SNAP_map.items()
                      if antnum != 3}
        with pytest.raises(ValueError, match='missing antennas'):
            ac.correct_SNAP_decoherence_in_place(
                data, self.deco, incomplete, nchans_per_block=self.ncpb)
        SNAP_map = {**self.SNAP_map, 4: 'C'}
        data_with_c = DataContainer(
            {**{bl: data[bl] for bl in data.keys()},
             (0, 4, 'ee'): np.ones((self.ntimes, self.nfreqs), complex)})
        with pytest.raises(ValueError, match='missing SNAPs'):
            ac.correct_SNAP_decoherence_in_place(
                data_with_c, self.deco, SNAP_map,
                nchans_per_block=self.ncpb)
        bad_shape = {'A': np.zeros((self.ntimes, self.nblocks + 1)),
                     'B': np.zeros((self.ntimes, self.nblocks + 1))}
        with pytest.raises(ValueError, match='shape'):
            ac.correct_SNAP_decoherence_in_place(
                data, bad_shape, self.SNAP_map, nchans_per_block=self.ncpb)


def build_red_avg_sim(nfreqs=64, ntimes=2, nants=6, pols=['ee'], noise_amp=0.0, seed=0):
    '''Linear redundant array with known gains, per-group true visibilities, and autos.
    Returns a dict with the DataContainer (antpos attached), gains, truths, and dt/df.
    With noise_amp=1, injected noise has exactly the variance predicted from the autos.'''
    rng = np.random.default_rng(seed)
    antpos = {i: np.array([14.6 * i, 0.0, 0.0]) for i in range(nants)}
    freqs = 100e6 + np.arange(nfreqs) * 122070.3125
    dt, df = 9.66, 122070.3125
    reds = redcal.get_reds(antpos, pols=pols, include_autos=True)
    gains, true_autos = {}, {}
    for i in range(nants):
        for antpol in {utils.split_pol(pol)[sub] for pol in pols for sub in (0, 1)}:
            gains[(i, antpol)] = (1 + 0.1 * rng.standard_normal((ntimes, nfreqs))
                                  + 0.1j * rng.standard_normal((ntimes, nfreqs)))
            true_autos[(i, antpol)] = 10.0 + i + np.zeros((ntimes, nfreqs))
    true_vis, data = {}, {}
    for red in reds:
        if red[0][0] != red[0][1]:
            true_vis[red[0]] = (rng.standard_normal((ntimes, nfreqs))
                                + 1j * rng.standard_normal((ntimes, nfreqs)))
        for bl in red:
            ant_i, ant_j = utils.split_bl(bl)
            gi, gj = gains[ant_i], gains[ant_j]
            if bl[0] == bl[1]:
                if ant_i == ant_j:  # co-polarized autos only
                    data[bl] = (np.abs(gi)**2 * true_autos[ant_i]).astype(complex)
            else:
                vis = true_vis[red[0]].copy()
                if noise_amp > 0:
                    sigma = np.sqrt(true_autos[ant_i] * true_autos[ant_j] / (dt * df)) * noise_amp
                    vis += ((rng.standard_normal((ntimes, nfreqs))
                             + 1j * rng.standard_normal((ntimes, nfreqs))) * sigma / np.sqrt(2))
                data[bl] = gi * np.conj(gj) * vis
    dc = DataContainer(data)
    dc.antpos = antpos
    return dict(data=dc, gains=gains, true_vis=true_vis, true_autos=true_autos,
                reds=reds, antpos=antpos, freqs=freqs, dt=dt, df=df,
                ntimes=ntimes, nfreqs=nfreqs)


def build_test_SNAP_decoherence(sim, p_by_SNAP, ant_to_SNAP_dict, nchans_per_block=32):
    '''Wrap hand-built per-SNAP loss fractions (Ntimes, Nblocks) in a SNAPDecoherence.'''
    nblocks = sim['nfreqs'] // nchans_per_block
    times = 2459935.5 + np.arange(sim['ntimes']) * 9.66 / (24 * 3600)
    nan_like = {SNAP: np.full_like(p, np.nan) for SNAP, p in p_by_SNAP.items()}
    counts = {SNAP: np.full(sim['ntimes'], 4, dtype=int) for SNAP in p_by_SNAP}
    return io.SNAPDecoherence(
        decoherence=p_by_SNAP, decoherence_refit={S: p.copy() for S, p in p_by_SNAP.items()},
        log_suppression_sigma=nan_like, n_spectra_per_SNAP=counts,
        times=times, block_freqs=sim['freqs'].reshape(nblocks, nchans_per_block),
        ant_to_SNAP_dict=ant_to_SNAP_dict,
        covered_blocks=np.ones(nblocks, dtype=bool), edge_blocks=[],
        band_edges=np.array([[sim['freqs'][0], sim['freqs'][-1]]]))


class TestCalibrateAndRedAvg:
    def test_noiseless_recovery(self):
        sim = build_red_avg_sim()
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(
            sim['data'], sim['gains'], sim['reds'], dt=sim['dt'], df=sim['df'])
        for key, truth in sim['true_vis'].items():
            np.testing.assert_allclose(avg[key], truth, atol=1e-10)
            assert not np.any(flags[key])
        # the zero-length group averages the calibrated autos uniformly over antennas
        auto_key = next(bl for bl in avg if bl[0] == bl[1])
        expected = np.mean([sim['true_autos'][(i, 'Jee')] for i in range(6)], axis=0)
        np.testing.assert_allclose(avg[auto_key], expected, atol=1e-10)
        # noiseless residuals mean zero chi^2
        assert np.all(meta['chisq_per_ant'][(0, 'Jee')] < 1e-16)

    def test_noise_weights_and_nsamples(self):
        sim = build_red_avg_sim()
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(
            sim['data'], sim['gains'], sim['reds'], dt=sim['dt'], df=sim['df'], effective_nsamples=False)
        # weights go as 1 / (A_i * A_j): check a group against the hand-computed average
        red = [r for r in sim['reds'] if r[0][0] != r[0][1] and len(r) > 2][0]
        wgts = [1.0 / (sim['true_autos'][utils.split_bl(bl)[0]]
                       * sim['true_autos'][utils.split_bl(bl)[1]]) for bl in red]
        cal_vis = [sim['data'][bl] / (sim['gains'][utils.split_bl(bl)[0]]
                                      * np.conj(sim['gains'][utils.split_bl(bl)[1]])) for bl in red]
        expected = np.sum([w * v for w, v in zip(wgts, cal_vis)], axis=0) / np.sum(wgts, axis=0)
        np.testing.assert_allclose(avg[red[0]], expected, atol=1e-10)
        np.testing.assert_array_equal(nsamples[red[0]], len(red))

    def test_ant_flags(self):
        sim = build_red_avg_sim()
        wf = np.zeros((sim['ntimes'], sim['nfreqs']), dtype=bool)
        wf[0, :10] = True
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(
            sim['data'], sim['gains'], sim['reds'], ant_flags={(0, 'Jee'): wf}, dt=sim['dt'], df=sim['df'],
            effective_nsamples=False)
        # groups containing antenna 0 lose one sample where it is flagged, but stay correct
        red = [r for r in sim['reds'] if (0, 1, 'ee') in r][0]
        np.testing.assert_allclose(avg[red[0]], sim['true_vis'][red[0]], atol=1e-10)
        np.testing.assert_array_equal(nsamples[red[0]][0, :10], len(red) - 1)
        np.testing.assert_array_equal(nsamples[red[0]][0, 10:], len(red))

    def test_data_flags(self):
        sim = build_red_avg_sim()
        # flag (and corrupt) some cells of one baseline, and flag some cells of one auto
        data_flags = DataContainer({bl: np.zeros((sim['ntimes'], sim['nfreqs']), dtype=bool)
                                    for bl in sim['data']})
        data_flags[(0, 1, 'ee')][0, :10] = True
        data_flags[(2, 2, 'ee')][1, -5:] = True
        sim['data'][(0, 1, 'ee')][0, :10] = 100.0
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(
            sim['data'], sim['gains'], sim['reds'], data_flags=data_flags,
            dt=sim['dt'], df=sim['df'], effective_nsamples=False)
        # the corrupted flagged cells get zero weight: the group average stays exact, loses
        # exactly one sample there, and the corruption never reaches chi^2
        red = [r for r in sim['reds'] if (0, 1, 'ee') in r][0]
        np.testing.assert_allclose(avg[red[0]], sim['true_vis'][red[0]], atol=1e-10)
        np.testing.assert_array_equal(nsamples[red[0]][0, :10], len(red) - 1)
        np.testing.assert_array_equal(nsamples[red[0]][0, 10:], len(red))
        assert np.all(meta['chisq_per_ant'][(0, 'Jee')] < 1e-16)
        # the flagged auto cells drop out of the auto average, but do NOT affect the noise
        # weights of cross-correlations involving that antenna (that's ant_flags' job)
        auto_key = next(bl for bl in avg if bl[0] == bl[1])
        expected = np.mean([sim['true_autos'][(i, 'Jee')] for i in range(6) if i != 2], axis=0)
        np.testing.assert_allclose(avg[auto_key][1, -5:], expected[1, -5:], atol=1e-10)
        np.testing.assert_array_equal(nsamples[auto_key][1, -5:], 5)
        np.testing.assert_array_equal(nsamples[auto_key][0, :], 6)
        red2 = [r for r in sim['reds'] if (0, 2, 'ee') in r][0]
        np.testing.assert_allclose(avg[red2[0]], sim['true_vis'][red2[0]], atol=1e-10)
        np.testing.assert_array_equal(nsamples[red2[0]], len(red2))

    def test_decoherence_round_trip(self):
        sim = build_red_avg_sim()
        ant_to_SNAP = {i: ('S0' if i < 3 else 'S1') for i in range(6)}
        p_S1 = np.zeros((sim['ntimes'], 2))
        p_S1[:, 1] = [0.04, 0.02]
        sd = build_test_SNAP_decoherence(sim, {'S0': np.zeros((sim['ntimes'], 2)), 'S1': p_S1},
                                         ant_to_SNAP)
        # suppress inter-SNAP visibilities and put the staircase into the gains
        suppression = {SNAP: np.repeat(-np.log(1 - sd.decoherence[SNAP]), 32, axis=1)
                       for SNAP in sd.SNAPs}
        data = DataContainer({bl: sim['data'][bl].copy() for bl in sim['data']})
        data.antpos = sim['antpos']
        for bl in data:
            if bl[0] != bl[1] and ant_to_SNAP[bl[0]] != ant_to_SNAP[bl[1]]:
                data[bl] *= np.exp(-suppression[ant_to_SNAP[bl[0]]] - suppression[ant_to_SNAP[bl[1]]])
        # measured gains carry the staircase; autocorrelations are exempt and stay as built
        gains = {ant: g * np.exp(-suppression[ant_to_SNAP[ant[0]]]) for ant, g in sim['gains'].items()}
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(
            data, gains, sim['reds'], snap_decoherence=sd, dt=sim['dt'], df=sim['df'])
        for key, truth in sim['true_vis'].items():
            np.testing.assert_allclose(avg[key], truth, atol=1e-8)
        # without the decoherence handling, the staircase corrupts intra-SNAP members
        avg_wrong, _, _, _ = ac.calibrate_and_red_avg(data, gains, sim['reds'], dt=sim['dt'], df=sim['df'])
        assert any(not np.allclose(avg_wrong[key], truth, atol=1e-3)
                   for key, truth in sim['true_vis'].items())

    def test_unmeasured_blocks_flagged(self):
        sim = build_red_avg_sim()
        ant_to_SNAP = {i: ('S0' if i < 3 else 'S1') for i in range(6)}
        p_S1 = np.zeros((sim['ntimes'], 2))
        p_S1[0, 0] = np.nan  # unmeasured block for S1 at t = 0
        sd = build_test_SNAP_decoherence(sim, {'S0': np.zeros((sim['ntimes'], 2)), 'S1': p_S1},
                                         ant_to_SNAP)
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(
            sim['data'], sim['gains'], sim['reds'], snap_decoherence=sd, dt=sim['dt'], df=sim['df'],
            effective_nsamples=False)
        # the length-1 group has intra-SNAP members (0-1, 1-2, 3-4, 4-5) and one inter-SNAP
        # member (2-3): the inter-SNAP baseline is excluded where p is unmeasured
        red = [r for r in sim['reds'] if (0, 1, 'ee') in r][0]
        np.testing.assert_array_equal(nsamples[red[0]][0, :32], len(red) - 1)
        np.testing.assert_array_equal(nsamples[red[0]][0, 32:], len(red))
        np.testing.assert_array_equal(nsamples[red[0]][1], len(red))
        np.testing.assert_allclose(avg[red[0]], sim['true_vis'][red[0]], atol=1e-10)

    def test_chisq_noise_expectation(self):
        sim = build_red_avg_sim(nfreqs=128, noise_amp=1.0, seed=1)
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(
            sim['data'], sim['gains'], sim['reds'], dt=sim['dt'], df=sim['df'])
        for ant, cspa in meta['chisq_per_ant'].items():
            assert np.nanmean(cspa) == pytest.approx(1.0, abs=0.25)
        assert np.nanmean(meta['total_chisq']['Jee']) == pytest.approx(1.0, abs=0.1)

    def test_excluded_antennas(self):
        sim = build_red_avg_sim(nfreqs=128, noise_amp=1.0, seed=2)
        # corrupt antenna 5, as if miscalibrated
        corrupted = DataContainer({bl: (sim['data'][bl] * (3.0 if 5 in bl[:2] and bl[0] != bl[1] else 1.0))
                                   for bl in sim['data']})
        corrupted.antpos = sim['antpos']
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(
            corrupted, sim['gains'], sim['reds'], ex_ants=[(5, 'Jee')], dt=sim['dt'], df=sim['df'])
        # excluding antenna 5 makes the averages identical to simply not having it
        no5 = DataContainer({bl: corrupted[bl] for bl in corrupted if 5 not in bl[:2]})
        no5.antpos = sim['antpos']
        avg2, flags2, nsamples2, meta2 = ac.calibrate_and_red_avg(
            no5, sim['gains'], sim['reds'], dt=sim['dt'], df=sim['df'])
        for key in avg2:
            if 5 not in key[:2]:
                np.testing.assert_array_equal(avg[key], avg2[key])
        # excluded antennas' keys still resolve through the containers: their auto key
        # maps to the array-averaged auto, and their cross keys to the group averages
        np.testing.assert_array_equal(avg[(5, 5, 'ee')], avg[(0, 0, 'ee')])
        np.testing.assert_array_equal(avg[(4, 5, 'ee')], avg[(0, 1, 'ee')])
        # the corrupted antenna's chi^2 is large; its partners' chi^2 and the
        # per-polarization totals are exactly what they would be without it
        assert np.nanmean(meta['chisq_per_ant'][(5, 'Jee')]) > 10
        for ant in meta2['chisq_per_ant']:
            np.testing.assert_array_equal(meta['chisq_per_ant'][ant], meta2['chisq_per_ant'][ant])
        np.testing.assert_array_equal(meta['total_chisq']['Jee'], meta2['total_chisq']['Jee'])

    def test_cross_pols(self):
        sim = build_red_avg_sim(pols=['ee', 'nn', 'en'])
        # a cross-polarized "auto" is averaged too, though never used for noise weights
        sim['data'][(0, 0, 'en')] = np.ones((sim['ntimes'], sim['nfreqs']), dtype=complex)
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(
            sim['data'], sim['gains'], sim['reds'], dt=sim['dt'], df=sim['df'])
        en_keys = [key for key in sim['true_vis'] if key[2] == 'en']
        assert len(en_keys) > 0
        for key in en_keys:
            np.testing.assert_allclose(avg[key], sim['true_vis'][key], atol=1e-10)
        # the injected cross-polarized auto is calibrated and averaged (a group of one),
        # with finite effective nsamples anchored by both polarizations' averaged autos
        expected = 1.0 / (sim['gains'][(0, 'Jee')] * np.conj(sim['gains'][(0, 'Jnn')]))
        np.testing.assert_allclose(avg[(0, 0, 'en')], expected, atol=1e-10)
        assert np.all(nsamples[(0, 0, 'en')] > 0)
        # chi^2 stays co-polarized: totals keyed by Jee/Jnn only
        assert set(meta['total_chisq'].keys()) == {'Jee', 'Jnn'}
        # reds govern everything: co-polarized-only reds drop 'en' crosses and autos alike
        co_reds = [red for red in sim['reds']
                   if utils.split_pol(red[0][2])[0] == utils.split_pol(red[0][2])[1]]
        avg_co, _, _, _ = ac.calibrate_and_red_avg(
            sim['data'], sim['gains'], co_reds, dt=sim['dt'], df=sim['df'])
        assert not any(key[2] == 'en' for key in avg_co)

    def test_compute_chisq_false(self):
        sim = build_red_avg_sim()
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(
            sim['data'], sim['gains'], sim['reds'], compute_chisq=False, dt=sim['dt'], df=sim['df'])
        assert meta == {}

    def test_missing_gains_and_autos_omitted(self):
        sim = build_red_avg_sim()
        gains = {ant: g for ant, g in sim['gains'].items() if ant[0] != 0}
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(
            sim['data'], gains, sim['reds'], dt=sim['dt'], df=sim['df'], effective_nsamples=False)
        assert not any(0 in key[:2] for key in avg)
        red = [r for r in sim['reds'] if (0, 1, 'ee') in r][0]
        np.testing.assert_array_equal(nsamples[red[1]], len(red) - 1)

    def test_excluded_antenna_with_decoherence(self):
        sim = build_red_avg_sim()
        ant_to_SNAP = {i: ('S0' if i < 3 else 'S1') for i in range(6)}
        p_S1 = np.zeros((sim['ntimes'], 2))
        p_S1[:, 1] = [0.04, 0.02]
        sd = build_test_SNAP_decoherence(sim, {'S0': np.zeros((sim['ntimes'], 2)), 'S1': p_S1},
                                         ant_to_SNAP)
        suppression = {SNAP: np.repeat(-np.log(1 - sd.decoherence[SNAP]), 32, axis=1)
                       for SNAP in sd.SNAPs}
        data = DataContainer({bl: sim['data'][bl].copy() for bl in sim['data']})
        data.antpos = sim['antpos']
        for bl in data:
            if bl[0] != bl[1] and ant_to_SNAP[bl[0]] != ant_to_SNAP[bl[1]]:
                data[bl] *= np.exp(-suppression[ant_to_SNAP[bl[0]]] - suppression[ant_to_SNAP[bl[1]]])
        gains = {ant: g * np.exp(-suppression[ant_to_SNAP[ant[0]]]) for ant, g in sim['gains'].items()}
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(
            data, gains, sim['reds'], ex_ants=[(5, 'Jee')], snap_decoherence=sd, dt=sim['dt'], df=sim['df'])
        # the excluded antenna's data are noiseless and consistent with the good-antenna
        # averages once the inline decoherence correction is applied, so its chi^2 ~ 0
        assert np.all(meta['chisq_per_ant'][(5, 'Jee')] < 1e-16)

    def test_excluded_and_omitted_edge_cases(self):
        sim = build_red_avg_sim(nants=7)
        # inject a cross-polarized "auto", which is never averaged
        sim['data'][(0, 0, 'en')] = np.ones((sim['ntimes'], sim['nfreqs']), dtype=complex)
        # antenna 1 loses its autocorrelation; antenna 0 loses its gains
        del sim['data'][(1, 1, 'ee')]
        gains = {ant: g for ant, g in sim['gains'].items() if ant[0] != 0}
        # antenna 4 is entirely flagged; antenna 5 is excluded
        all_flagged = np.ones((sim['ntimes'], sim['nfreqs']), dtype=bool)
        # listed auto groups with no usable members (no 'nn' autos in data; no antenna has
        # both antpols' gains for 'en') are skipped without producing averages
        reds_plus = sim['reds'] + [[(i, i, 'nn') for i in range(7)], [(i, i, 'en') for i in range(7)]]
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(
            sim['data'], gains, reds_plus, ant_flags={(4, 'Jee'): all_flagged}, ex_ants=[(5, 'Jee')],
            dt=sim['dt'], df=sim['df'])
        # antennas 0 (no gains), 1 (no autos), and 4 (fully flagged) all drop out of the
        # averages, and none of the excluded-antenna chi^2 machinery crashes on their
        # baselines to antenna 5 (missing gains/autos, both-unexcluded, or fully flagged)
        assert not any(0 in key[:2] or 1 in key[:2] for key in avg)
        assert not any(key[2] == 'en' for key in avg)
        assert (0, 'Jee') not in meta['chisq_per_ant']
        assert (4, 'Jee') not in meta['chisq_per_ant']

    def test_dt_df_inference(self):
        sim = build_red_avg_sim()
        times = 2459935.5 + np.arange(sim['ntimes']) * sim['dt'] / (24 * 3600)
        sim['data'].times = times
        sim['data'].times_by_bl = {bl[:2]: times for bl in sim['data']}
        sim['data'].freqs = sim['freqs']
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(sim['data'], sim['gains'], sim['reds'])
        for key, truth in sim['true_vis'].items():
            np.testing.assert_allclose(avg[key], truth, atol=1e-10)

    def test_effective_nsamples_exact(self):
        sim = build_red_avg_sim(nfreqs=64, ntimes=400, noise_amp=1.0, seed=3)
        avg, flags, n_eff, meta = ac.calibrate_and_red_avg(
            sim['data'], sim['gains'], sim['reds'], dt=sim['dt'], df=sim['df'], compute_chisq=False)
        _, _, n_count, _ = ac.calibrate_and_red_avg(
            sim['data'], sim['gains'], sim['reds'], dt=sim['dt'], df=sim['df'], compute_chisq=False,
            effective_nsamples=False)
        auto_key = next(bl for bl in avg if bl[0] == bl[1])
        avg_auto = np.abs(avg[auto_key])
        key = sorted((bl for bl in avg if bl[0] != bl[1]), key=lambda bl: -np.max(n_count[bl]))[0]
        # effective nsamples exceeds the count for cross groups and makes the standard
        # predictor Abar^2 / (dt df nsamples) exact; the count over-predicts the noise
        assert np.all(n_eff[key] > n_count[key])
        # the sim's true visibilities vary with time, so subtract them to isolate the noise
        empirical_var = np.var(avg[key] - sim['true_vis'][key], axis=0)
        pred_eff = avg_auto[0]**2 / (sim['dt'] * sim['df'] * n_eff[key][0])
        pred_count = avg_auto[0]**2 / (sim['dt'] * sim['df'] * n_count[key][0])
        assert np.mean(empirical_var / pred_eff) == pytest.approx(1.0, abs=0.05)
        assert np.mean(empirical_var / pred_count) < 0.98  # ~3.5% over-prediction for this sim's auto spread
        # the averaged autos' own effective nsamples accounts for uniform weighting of
        # heteroscedastic autos, and so is at or below the antenna count
        assert np.all(n_eff[auto_key] <= n_count[auto_key] + 1e-10)

    def test_autos_required(self):
        sim = build_red_avg_sim()
        no_autos = DataContainer({bl: sim['data'][bl] for bl in sim['data'] if bl[0] != bl[1]})
        no_autos.antpos = sim['antpos']
        with pytest.raises(ValueError, match='autocorrelations'):
            ac.calibrate_and_red_avg(no_autos, sim['gains'], sim['reds'], dt=sim['dt'], df=sim['df'])

    def test_reds_must_include_auto_groups(self):
        sim = build_red_avg_sim()
        cross_reds = [red for red in sim['reds'] if red[0][0] != red[0][1]]
        with pytest.raises(ValueError, match='autocorrelation group'):
            ac.calibrate_and_red_avg(sim['data'], sim['gains'], cross_reds,
                                     dt=sim['dt'], df=sim['df'])
        # with auto groups listed, every antenna's auto key resolves to the average
        avg, flags, nsamples, meta = ac.calibrate_and_red_avg(
            sim['data'], sim['gains'], sim['reds'], dt=sim['dt'], df=sim['df'])
        expected = np.mean([sim['true_autos'][(i, 'Jee')] for i in range(6)], axis=0)
        np.testing.assert_allclose(avg[(3, 3, 'ee')], expected, atol=1e-10)
