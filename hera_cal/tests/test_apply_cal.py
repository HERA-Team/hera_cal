# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""Unit tests for the hera_cal.apply_cal module."""

from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from copy import deepcopy
import os
import sys
import shutil
from scipy import constants
import warnings
from pyuvdata.utils import check_histories
from pyuvdata import UVCal, UVData

from .. import io
from .. import apply_cal as ac
from ..datacontainer import DataContainer
from ..data import DATA_PATH
from .. import utils


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
@pytest.mark.filterwarnings("ignore:It seems that the latitude and longitude are in radians")
class Test_Update_Cal(object):
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
        avg_flags = ~np.isinf(avg_gains)
        avg_gains[avg_flags] = 1. + 0.j

        for i in range(10):
            for j in range(10):
                if not np.isinf(dc[(0, 1, 'xx')][i, j]):
                    assert np.allclose(dc[(0, 1, 'xx')][i, j], vis[i, j] * avg_gains[i, j])

    def test_apply_redundant_solutions(self):
        miriad = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uvOCR_53x_54x_only")
        outname_uvh5 = os.path.join(DATA_PATH, "test_output/red_out.h5")
        old_cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        new_cal = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only")
        all_reds = [[(54, 54, 'xx')]]
        ac.apply_redundant_solution(miriad, outname_uvh5, new_cal, all_reds, old_cal, filetype_in='miriad',
                                    filetype_out='uvh5', gain_convention='divide', add_to_history='',
                                    clobber=True)
        # checking if file is created
        assert os.path.exists(outname_uvh5)

        # checking average
        inp_hc = io.HERAData(miriad, filetype='miriad')
        inp_data, inp_flags, _ = inp_hc.read()
        out_hc = io.HERAData(outname_uvh5)
        out_data, out_flags, _ = out_hc.read()
        np.testing.assert_almost_equal(inp_data[(54, 54, 'xx')], out_data[(54, 54, 'xx')])

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
        wgts = DataContainer(dict(map(lambda k: (k, (~flags[k]).astype(np.float)), flags.keys())))
        del g_new[(0, 'Jxx')]
        ac.calibrate_in_place(dc, g_new, wgts, cal_flags, gain_convention='divide', flags_are_wgts=True)
        assert np.allclose(wgts[(0, 1, 'xx')].max(), 0.0)

    def test_apply_cal(self):
        miriad = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uvOCR_53x_54x_only")
        uvh5 = os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5")
        outname_miriad = os.path.join(DATA_PATH, "test_output/out.uv")
        outname_uvh5 = os.path.join(DATA_PATH, "test_output/out.h5")
        calout = os.path.join(DATA_PATH, "test_output/out.cal")
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

    def test_apply_cal_argparser(self):
        sys.argv = [sys.argv[0], 'a', 'b', '--new_cal', 'd']
        a = ac.apply_cal_argparser()
        args = a.parse_args()
        assert args.infilename == 'a'
        assert args.outfilename == 'b'
        assert args.new_cal == ['d']
