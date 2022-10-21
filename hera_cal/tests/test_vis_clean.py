# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License
import warnings
import pytest
import numpy as np
from copy import deepcopy
import os
import sys
import shutil
from scipy import constants, interpolate
from pyuvdata import UVCal, UVData
from hera_sim.interpolators import Beam
from hera_sim import DATA_PATH as HS_DATA_PATH
from hera_sim import noise
from hera_filters import dspec

from hera_cal import io, datacontainer
from hera_cal import vis_clean
from hera_cal.vis_clean import VisClean
from hera_cal.data import DATA_PATH
from hera_cal import frf
import glob
import copy


# test flagging utility funtions
def test_truncate_flagged_edges():
    Nfreqs = 64
    Ntimes = 60
    data_in = np.outer(np.arange(1, Ntimes + 1), np.arange(1, Nfreqs + 1))
    weights_in = np.abs(data_in).astype(float)
    data_in = data_in + .3j * data_in
    # flag channel 30
    weights_in[:, 30] = 0.
    # flag last channel
    weights_in[:, -1] = 0.
    # flag last two integrations
    weights_in[-2:, :] = 0.
    times = np.arange(60) * 10.
    freqs = np.arange(64) * 100e3
    # test freq truncation
    xout, dout, wout, edges, _ = vis_clean.truncate_flagged_edges(data_in, weights_in, freqs, ax='freq')
    assert np.all(np.isclose(xout, freqs[:-1]))
    assert np.all(np.isclose(dout, data_in[:, :-1]))
    assert np.all(np.isclose(wout, weights_in[:, :-1]))
    assert edges == [(0, 1)]
    # test time truncation
    xout, dout, wout, edges, _ = vis_clean.truncate_flagged_edges(data_in, weights_in, times, ax='time')
    assert np.all(np.isclose(xout, times[:-2]))
    assert np.all(np.isclose(dout, data_in[:-2, :]))
    assert np.all(np.isclose(wout, weights_in[:-2, :]))
    assert edges == [(0, 2)]
    # test truncating both.
    xout, dout, wout, edges, _ = vis_clean.truncate_flagged_edges(data_in, weights_in, (times, freqs), ax='both')
    assert np.all(np.isclose(xout[0], times[:-2]))
    assert np.all(np.isclose(xout[1], freqs[:-1]))
    assert np.all(np.isclose(dout, data_in[:-2, :-1]))
    assert np.all(np.isclose(wout, weights_in[:-2, :-1]))
    assert edges == [[(0, 2)], [(0, 1)]]


def test_restore_flagged_edges():
    Nfreqs = 64
    Ntimes = 60
    data_in = np.outer(np.arange(1, Ntimes + 1), np.arange(1, Nfreqs + 1))
    weights_in = np.abs(data_in).astype(float)
    data_in = data_in + .3j * data_in
    # flag channel 30
    weights_in[:, 30] = 0.
    # flag last channel
    weights_in[:, -1] = 0.
    # flag last two integrations
    weights_in[-2:, :] = 0.
    times = np.arange(60) * 10.
    freqs = np.arange(64) * 100e3
    # test freq truncation
    xout, dout, wout, edges, chunks = vis_clean.truncate_flagged_edges(data_in, weights_in, freqs, ax='freq')
    wrest = vis_clean.restore_flagged_edges(wout, chunks, edges)
    assert np.allclose(weights_in[:, :-1], wrest[:, :-1])
    assert np.allclose(wrest[:, -1], 0.0)
    xout, dout, wout, edges, chunks = vis_clean.truncate_flagged_edges(data_in, weights_in, times, ax='time')
    wrest = vis_clean.restore_flagged_edges(wout, chunks, edges, ax='time')
    assert np.allclose(wout, wrest[:-2, :])
    assert np.allclose(wrest[-2:, :], 0.0)
    xout, dout, wout, edges, chunks = vis_clean.truncate_flagged_edges(data_in, weights_in, (times, freqs), ax='both')
    wrest = vis_clean.restore_flagged_edges(wout, chunks, edges, ax='both')
    assert np.allclose(wrest[-2:, :], 0.0)
    assert np.allclose(wrest[:, -1], 0.0)
    assert np.allclose(wout, wrest[:-2, :-1])


def test_find_discontinuity_edges():
    assert vis_clean.find_discontinuity_edges([0, 1, 4, 9]) == [(0, 2), (2, 3), (3, 4)]
    assert vis_clean.find_discontinuity_edges([0, 1, 2, 4, 5, 6, 7, 9, 11, 12]) == [(0, 3), (3, 7), (7, 8), (8, 10)]


def test_flag_rows_with_flags_within_edge_distance():
    Nfreqs = 64
    Ntimes = 60
    weights_in = np.outer(np.arange(1, Ntimes + 1), np.arange(1, Nfreqs + 1))
    weights_in[32, 2] = 0.
    weights_in[33, 12] = 0.
    weights_in[2, 30] = 0.
    weights_in[-10, 20] = 0.
    freqs = np.arange(Nfreqs) * 100e3
    # under the above flagging pattern
    # freq flagging with min_flag_edge_distance=2 yields 32nd integration flagged only.
    wout = vis_clean.flag_rows_with_flags_within_edge_distance(freqs, weights_in, min_flag_edge_distance=3, ax='freq')
    for i in range(wout.shape[0]):
        if i == 32:
            assert np.all(np.isclose(wout[i], 0.0))
        else:
            assert np.all(np.isclose(wout[i], weights_in[i]))
    # extending edge_distance to 12 should yield 33rd integration being flagged as well.
    wout = vis_clean.flag_rows_with_flags_within_edge_distance(freqs, weights_in, min_flag_edge_distance=13, ax='freq')
    for i in range(wout.shape[0]):
        if i == 32 or i == 33:
            assert np.all(np.isclose(wout[i], 0.0))
        else:
            assert np.all(np.isclose(wout[i], weights_in[i]))
    # now do time axis. 30th channel should be flagged for this case.
    wout = vis_clean.flag_rows_with_flags_within_edge_distance(freqs, weights_in, min_flag_edge_distance=3, ax='time')
    for i in range(wout.shape[1]):
        if i == 30:
            assert np.all(np.isclose(wout[:, i], 0.0))
        else:
            assert np.all(np.isclose(wout[:, i], weights_in[:, i]))
    # 30th and 20th channels should end up flagged for this case.
    times = np.arange(Ntimes) * 10.
    wout = vis_clean.flag_rows_with_flags_within_edge_distance(times, weights_in, min_flag_edge_distance=11, ax='time')
    for i in range(wout.shape[1]):
        if i == 30 or i == 20:
            assert np.all(np.isclose(wout[:, i], 0.0))
        else:
            assert np.all(np.isclose(wout[:, i], weights_in[:, i]))
    # now do both
    wout = vis_clean.flag_rows_with_flags_within_edge_distance([times, freqs], weights_in, min_flag_edge_distance=(3, 3), ax='both')
    for i in range(wout.shape[1]):
        if i == 30:
            assert np.all(np.isclose(wout[:, i], 0.0))
    for i in range(wout.shape[0]):
        if i == 32:
            assert np.all(np.isclose(wout[i], 0.0))


def test_flag_rows_with_flags_within_edge_distance_with_breaks():
    Nfreqs = 64
    Ntimes = 60
    freqs = np.hstack([np.arange(23), 30 + np.arange(24), 58 + np.arange(17)]) * 100e3 + 150e6  # freq axis with discontinuities at 23 and 47 integrations.
    times = np.hstack([np.arange(20) * 11., 41 * 11. + np.arange(27) * 11., 200 * 11. + np.arange(13) * 11.])  # time axis with discontinuities at 29 abd 47 integrations
    weights_in = np.outer(np.arange(1, Ntimes + 1), np.arange(1, Nfreqs + 1))
    # frequency direction and time direction separately.
    weights_in[2, 30] = 0.  # time 2 should not get flagged
    weights_in[21, 48] = 0.  # time 21 should get flagged
    weights_in[55, 46] = 0.  # time 55 should get flagged
    weights_in[25, -2] = 0.  # time 25 should get flagged
    wout = vis_clean.flag_rows_with_flags_within_edge_distance(freqs, weights_in, min_flag_edge_distance=3, ax='freq')
    assert list(np.where(np.all(np.isclose(wout, 0.), axis=1))[0]) == [21, 25, 55]
    weights_in[22, 30] = 0.  # channel 30 should be flagged
    # channel 48 will also be flagged.
    wout = vis_clean.flag_rows_with_flags_within_edge_distance(times, weights_in, min_flag_edge_distance=3, ax='time')
    assert list(np.where(np.all(np.isclose(wout, 0.), axis=0))[0]) == [30, 48]
    weights_in = np.outer(np.arange(1, Ntimes + 1), np.arange(1, Nfreqs + 1))
    # both directions
    weights_in[22, 30] = 0.  # time 2 should not get flagged
    weights_in[55, 46] = 0.  # time 55 should get flagged
    weights_in[25, -2] = 0.  # time 25 should get flagged
    weights_in[22, 30] = 0.  # channel 30 should be flagged
    wout = vis_clean.flag_rows_with_flags_within_edge_distance([times, freqs], weights_in, min_flag_edge_distance=[2, 3], ax='both')
    assert list(np.where(np.all(np.isclose(wout, 0.), axis=0))[0]) == [30]
    assert list(np.where(np.all(np.isclose(wout, 0.), axis=1))[0]) == [25, 55]


def test_flag_rows_with_contiguous_flags():
    Nfreqs = 64
    Ntimes = 60
    weights_in = np.outer(np.arange(1, Ntimes + 1), np.arange(1, Nfreqs + 1))
    weights_in[32, 2:12] = 0.
    weights_in[35, 12:14] = 0.
    weights_in[2:12, 30] = 0.
    weights_in[-10:-8, 20] = 0.
    wout = vis_clean.flag_rows_with_contiguous_flags(weights_in, max_contiguous_flag=8, ax='freq')
    for i in range(wout.shape[0]):
        if i == 32:
            assert np.all(np.isclose(wout[i], 0.0))
        else:
            assert np.all(np.isclose(wout[i], weights_in[i]))
    # extending edge_distance to 12 should yield 33rd integration being flagged as well.
    wout = vis_clean.flag_rows_with_contiguous_flags(weights_in, max_contiguous_flag=2, ax='freq')
    for i in range(wout.shape[0]):
        if i == 32 or i == 35:
            assert np.all(np.isclose(wout[i], 0.0))
        else:
            assert np.all(np.isclose(wout[i], weights_in[i]))
    # now do time axis. 30th channel should be flagged for this case.
    wout = vis_clean.flag_rows_with_contiguous_flags(weights_in, max_contiguous_flag=8, ax='time')
    for i in range(wout.shape[1]):
        if i == 30:
            assert np.all(np.isclose(wout[:, i], 0.0))
        else:
            assert np.all(np.isclose(wout[:, i], weights_in[:, i]))
    # 30th and 20th channels should end up flagged for this case.
    wout = vis_clean.flag_rows_with_contiguous_flags(weights_in, max_contiguous_flag=2, ax='time')
    for i in range(wout.shape[1]):
        if i == 30 or i == 20:
            assert np.all(np.isclose(wout[:, i], 0.0))
        else:
            assert np.all(np.isclose(wout[:, i], weights_in[:, i]))
    # now do both
    wout = vis_clean.flag_rows_with_contiguous_flags(weights_in, max_contiguous_flag=(3, 3), ax='both')
    for i in range(wout.shape[1]):
        if i == 30:
            assert np.all(np.isclose(wout[:, i], 0.0))
    for i in range(wout.shape[0]):
        if i == 32:
            assert np.all(np.isclose(wout[i], 0.0))


def test_get_max_contiguous_flag_from_filter_periods():
    Nfreqs = 64
    Ntimes = 60
    times = np.arange(60) * 10.
    freqs = np.arange(64) * 100e3
    filter_centers = [[0.], [0.]]
    filter_half_widths = [[1 / (3. * 10)], [1 / (100e3 * 2)]]
    mcf = vis_clean.get_max_contiguous_flag_from_filter_periods(freqs, filter_centers[1], filter_half_widths[1])
    assert mcf == 2
    mcf = vis_clean.get_max_contiguous_flag_from_filter_periods(times, filter_centers[0], filter_half_widths[0])
    assert mcf == 3
    mcf = vis_clean.get_max_contiguous_flag_from_filter_periods((times, freqs), filter_centers, filter_half_widths)
    assert tuple(mcf) == (3, 2)
    # test assertion errors
    pytest.raises(ValueError, vis_clean.get_max_contiguous_flag_from_filter_periods, [1.], [0.], [.5])
    pytest.raises(ValueError, vis_clean.get_max_contiguous_flag_from_filter_periods, [[1.], [0.]], [[0.], [0.]], [[.5], [.5]])


def test_flag_model_rms():
    Nfreqs = 64
    Ntimes = 60
    times = np.arange(60) * 10.
    freqs = np.arange(64) * 100e3
    w = np.ones((Ntimes, Nfreqs), dtype=bool)
    d = np.random.randn(Ntimes, Nfreqs) * 1e-3 + 1j * np.random.randn(Ntimes, Nfreqs) * 1e-3
    d += np.ones_like(d) * 100
    d[30, 12] = 3.12315132e6
    w[30, 12] = 0.
    mdl = np.ones_like(d) * 100
    mdl[30, 24] = 1e6
    skipped = np.zeros_like(mdl, dtype=bool)
    skipped = vis_clean.flag_model_rms(skipped, d, w, mdl, ax='freq')
    for i in range(Ntimes):
        if i == 30:
            assert np.all(skipped[i])
        else:
            assert np.all(~skipped[i])
    skipped = np.zeros_like(mdl, dtype=bool)
    skipped = vis_clean.flag_model_rms(skipped, d, w, mdl, ax='time')
    for i in range(Ntimes):
        if i == 24:
            assert np.all(skipped[:, i])
        else:
            assert np.all(~skipped[:, i])
    skipped = np.zeros_like(mdl, dtype=bool)
    skipped = vis_clean.flag_model_rms(skipped, d, w, mdl, ax='both')
    for i in range(Nfreqs):
        if i == 24:
            assert np.all(skipped[:, i])
        else:
            assert ~np.all(skipped[:, i])
    for i in range(Ntimes):
        if i == 30:
            assert np.all(skipped[i])
        else:
            assert ~np.all(skipped[i])


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
@pytest.mark.filterwarnings("ignore:It seems that the latitude and longitude are in radians")
class Test_VisClean(object):

    def test_init(self):
        # test basic init
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        assert not hasattr(V, 'data')
        V.read(bls=[(24, 25, 'ee')])
        assert hasattr(V, 'data')
        assert hasattr(V, 'antpos')
        assert isinstance(V.hd, io.HERAData)
        assert isinstance(V.hd.data_array, np.ndarray)

        # test basic init w/ uvh5
        fname = os.path.join(DATA_PATH, 'zen.2458098.43124.subband.uvh5')
        V = VisClean(fname, filetype='uvh5')
        assert not hasattr(V, 'data')
        V.read(bls=[(13, 14, 'ee')])
        assert set(V.hd.ant_1_array) == set([13])
        assert isinstance(V.hd, io.HERAData)
        assert isinstance(V.hd.data_array, np.ndarray)

        # test input cal
        fname = os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA')
        uvc = io.HERACal(os.path.join(DATA_PATH, 'zen.2458043.12552.xx.HH.uvORA.abs.calfits'))
        gains, _, _, _ = uvc.read()
        V1 = VisClean(fname, filetype='miriad')
        bl = (52, 53, 'ee')
        V1.read(bls=[bl])
        V2 = VisClean(fname, filetype='miriad', input_cal=uvc)
        V2.read(bls=[bl])
        g = gains[(bl[0], 'Jee')] * gains[(bl[1], 'Jee')].conj()
        assert np.allclose((V1.data[bl] / g)[30, 30], V2.data[bl][30, 30])
        V2.apply_calibration(V2.hc, unapply=True)
        assert np.allclose(V1.data[bl][30, 30], V2.data[bl][30, 30], atol=1e-5)

        # test soft copy
        V1.hello = 'hi'
        V1.hello_there = 'bye'
        V1.foo = 'bar'
        V3 = V1.soft_copy(references=["hello*"])
        assert hex(id(V1.data[(52, 53, 'ee')])) == hex(id(V3.data[(52, 53, 'ee')]))
        assert hasattr(V3, 'hello')
        assert hasattr(V3, 'hello_there')
        assert not hasattr(V3, 'foo')
        assert V3.__class__ == VisClean

        # test clear
        V1.clear_containers()
        assert np.all([len(getattr(V1, c)) == 0 for c in ['data', 'flags', 'nsamples']])
        V2.clear_calibration()
        assert not hasattr(V2, 'hc')

    @pytest.mark.filterwarnings("ignore:Selected polarization values are not evenly spaced")
    def test_read_write(self):
        # test read data can be turned off for uvh5
        fname = os.path.join(DATA_PATH, 'zen.2458098.43124.subband.uvh5')
        V = VisClean(fname, filetype='uvh5')
        V.read(read_data=False)
        assert set(V.hd.ant_1_array) == set([1, 11, 12, 13, 14])

        # test read-write-read
        V.read()
        V.write_data(V.data, "./ex.uvh5", overwrite=True, filetype='uvh5', extra_attrs=dict(vis_units='Jy'))
        V2 = VisClean("./ex.uvh5", filetype='uvh5')
        V2.read()
        assert V2.hd.vis_units == 'Jy'
        assert 'Thisfilewasproducedbythefunction' in V2.hd.history.replace('\n', '').replace(' ', '')
        V.hd.history, V2.hd.history, V2.hd.vis_units = '', '', V.hd.vis_units
        if hasattr(V.hd, "filename"):
            # make sure filename attributes are what we're expecting
            assert V.hd.filename == ["zen.2458098.43124.subband.uvh5"]
            assert V2.hd.filename == ["ex.uvh5"]
            V.hd.filename = V2.hd.filename
        assert V.hd == V2.hd
        os.remove("./ex.uvh5")

        # exceptions
        pytest.raises(ValueError, V.write_data, V.data, 'foo', filetype='what')

        # test write on subset of data
        V.read(read_data=True)
        data = datacontainer.DataContainer(dict([(k, V.data[k]) for k in list(V.data.keys())[:2]]))
        V.write_data(data, "ex.uvh5", overwrite=True, filetype='uvh5')
        assert os.path.exists("ex.uvh5")
        os.remove('ex.uvh5')

    def test_fourier_filter(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        V.read()
        # test arg errors
        k = (24, 25, 'ee')
        fc = [0.]
        fw = [100e-9]
        ff = [1e-9]
        fwt = [1e-3]
        assert pytest.raises(ValueError, V.fourier_filter, keys=[k], overwrite=True,
                             filter_centers=fc, filter_half_widths=fw, suppression_factors=ff,
                             ax='height', mode='dayenu', fitting_options=None)
        V.fourier_filter(keys=[k], filter_centers=fc, filter_half_widths=fw, suppression_factors=ff,
                         ax='freq', mode='dayenu', output_prefix='clean', zeropad=10, overwrite=True, max_contiguous_edge_flags=20)
        # this line is repeated to cover the overwrite skip
        V.fourier_filter(keys=[k], filter_centers=fc, filter_half_widths=fw, suppression_factors=ff, max_contiguous_edge_flags=20,
                         ax='freq', mode='dayenu', zeropad=10, output_prefix='clean', overwrite=False)
        assert np.all([V.clean_info[k][(0, V.Nfreqs)]['status']['axis_1'][i] == 'success' for i in V.clean_info[k][(0, V.Nfreqs)]['status']['axis_1']])
        # now do a time filter
        V.fourier_filter(keys=[k], filter_centers=fc, filter_half_widths=fwt, suppression_factors=ff, overwrite=True,
                         ax='time', mode='dayenu', zeropad=10, max_contiguous_edge_flags=20)
        assert V.clean_info[k][(0, V.Nfreqs)]['status']['axis_0'][0] == 'skipped'
        assert V.clean_info[k][(0, V.Nfreqs)]['status']['axis_0'][3] == 'success'
        # check that clean resid is equal to zero in flagged channels
        assert np.all(V.clean_resid[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_resid[(24, 25, 'ee')][~(V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')])] != 0.)
        assert np.all(V.clean_model[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_model[(24, 25, 'ee')][~V.clean_flags[(24, 25, 'ee')]] != 0.)
        # check that filtered_data is the same in channels that were not flagged
        atol = 1e-6 * np.mean(np.abs(V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')]]) ** 2.) ** .5
        assert np.all(np.isclose(V.clean_data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]],
                      V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]], rtol=0., atol=atol))
        # raise errors.
        assert pytest.raises(ValueError, V.fourier_filter, filter_centers=[fc, fc], ax='both',
                             filter_half_widths=[fwt, fw], suppression_factors=[ff, ff],
                             mode='dayenu', zeropad=0, overwrite=True)
        assert pytest.raises(ValueError, V.fourier_filter, filter_centers=[fc, fc], ax='both',
                             filter_half_widths=[fwt, fw], suppression_factors=[ff, ff], overwrite=True,
                             mode='dayenu', zeropad=['Mathematical Universe', 'Crazy Universe'])
        # check 2d filter.
        V.fourier_filter(filter_centers=[fc, fc],
                         filter_half_widths=[fwt, fw],
                         suppression_factors=[ff, ff],
                         mode='dayenu', overwrite=True,
                         zeropad=[20, 10], ax='both', max_contiguous_edge_flags=100)
        assert V.clean_info[k][(0, V.Nfreqs)]['status']['axis_0'][0] == 'skipped'
        assert V.clean_info[k][(0, V.Nfreqs)]['status']['axis_0'][3] == 'success'
        # check that clean resid is equal to zero in flagged channels
        assert np.all(V.clean_resid[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_resid[(24, 25, 'ee')][~(V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')])] != 0.)
        assert np.all(V.clean_model[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_model[(24, 25, 'ee')][~V.clean_flags[(24, 25, 'ee')]] != 0.)
        # check that filtered_data is the same in channels that were not flagged
        atol = 1e-6 * np.mean(np.abs(V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')]]) ** 2.) ** .5
        assert np.allclose(V.clean_data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]],
                           V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]], rtol=0., atol=atol)

    @pytest.mark.filterwarnings("ignore:.*dspec.vis_filter will soon be deprecated")
    def test_vis_clean_dayenu(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        V.read()

        # most coverage is in dspec. Check that args go through here.
        # similar situation for test_vis_clean.
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='freq', overwrite=True, mode='dayenu')
        # check that clean resid is equal to zero in flagged channels
        assert np.all(V.clean_resid[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_resid[(24, 25, 'ee')][~(V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')])] != 0.)
        assert np.all(V.clean_model[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_model[(24, 25, 'ee')][~V.clean_flags[(24, 25, 'ee')]] != 0.)
        # check that filtered_data is the same in channels that were not flagged
        # had to set atol=1e-6 here so it won't fail on travis (it runs fine on my laptop). There are some funny
        # numpy issues.
        atol = 1e-6 * np.mean(np.abs(V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')]]) ** 2.) ** .5
        assert np.all(np.isclose(V.clean_data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]],
                                 V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]], atol=atol, rtol=0.))
        assert np.all([V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_1'][i] == 'success' for i in V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_1']])

        assert pytest.raises(AssertionError, V.vis_clean, keys=[(24, 25, 'ee')], ax='time', max_frate=None, mode='dayenu')
        assert pytest.raises(ValueError, V.vis_clean, keys=[(24, 25, 'ee')], ax='time', max_frate='arglebargle', mode='dayenu')

        # cover no overwrite = False skip lines.
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='freq', overwrite=False, mode='dayenu')

        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='time', overwrite=True, max_frate=1.0, mode='dayenu')
        assert V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_0'][0] == 'skipped'
        assert V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_0'][3] == 'success'
        # check that clean resid is equal to zero in flagged channels
        assert np.all(V.clean_resid[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_resid[(24, 25, 'ee')][~(V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')])] != 0.)
        assert np.all(V.clean_model[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_model[(24, 25, 'ee')][~V.clean_flags[(24, 25, 'ee')]] != 0.)
        # check that filtered_data is the same in channels that were not flagged
        atol = 1e-6 * np.mean(np.abs(V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')]]) ** 2.) ** .5
        assert np.allclose(V.clean_data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]],
                           V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]], atol=atol, rtol=0.)
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='both', overwrite=True, max_frate=1.0, mode='dayenu')
        assert np.all(['success' == V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_1'][i] for i in V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_1']])
        assert V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_0'][0] == 'skipped'
        assert V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_0'][3] == 'success'
        # check that clean resid is equal to zero in flagged channels
        assert np.all(V.clean_resid[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_resid[(24, 25, 'ee')][~(V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')])] != 0.)
        assert np.all(V.clean_model[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_model[(24, 25, 'ee')][~V.clean_flags[(24, 25, 'ee')]] != 0.)
        # check that filtered_data is the same in channels that were not flagged
        atol = 1e-6 * np.mean(np.abs(V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')]]) ** 2.) ** .5
        assert np.allclose(V.clean_data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]],
                           V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]], atol=atol, rtol=0.)
        # check whether dayenu filtering axis 1 and then axis 0 is the same as dayenu filtering axis 1 and then filtering the resid.
        # note that filtering axis orders do not commute, we filter axis 1 (foregrounds) before filtering cross-talk.
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='both', overwrite=True, max_frate=1.0, mode='dayenu')
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='freq', overwrite=True, max_frate=1.0, output_prefix='clean1', mode='dayenu')
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='time', overwrite=True, max_frate=1.0, data=V.clean1_resid, output_prefix='clean0', mode='dayenu')
        assert np.all(np.isclose(V.clean_resid[(24, 25, 'ee')], V.clean0_resid[(24, 25, 'ee')]))

    @pytest.mark.filterwarnings("ignore:.*dspec.vis_filter will soon be deprecated")
    def test_vis_clean_dpss(self):
        # Relax atol=1e-6 for clean_data and data equalities. there may be some numerical
        # issues going on. Notebook tests show that distributing minus signs has
        # consequences.
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        V.read()

        # most coverage is in dspec. Check that args go through here.
        # similar situation for test_vis_clean.
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='freq', overwrite=True, mode='dpss_leastsq')
        # check that clean resid is equal to zero in flagged channels
        assert np.all(V.clean_resid[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_resid[(24, 25, 'ee')][~(V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')])] != 0.)
        assert np.all(V.clean_model[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_model[(24, 25, 'ee')][~V.clean_flags[(24, 25, 'ee')]] != 0.)
        # check that filtered_data is the same in channels that were not flagged
        atol = 1e-6 * np.mean(np.abs(V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')]]) ** 2.) ** .5
        assert np.all(np.isclose(V.clean_data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]],
                                 V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]], atol=atol, rtol=0.))
        assert np.all([V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_1'][i] == 'success' for i in V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_1']])

        assert pytest.raises(AssertionError, V.vis_clean, keys=[(24, 25, 'ee')], ax='time', mode='dpss_leastsq')
        assert pytest.raises(ValueError, V.vis_clean, keys=[(24, 25, 'ee')], ax='time', max_frate='arglebargle', mode='dpss_leastsq')

        # cover no overwrite = False skip lines.
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='freq', overwrite=False, mode='dpss_leastsq')

        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='time', overwrite=True, max_frate=1.0, mode='dpss_leastsq')
        assert V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_0'][0] == 'skipped'
        assert V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_0'][3] == 'success'
        # check that clean resid is equal to zero in flagged channels
        assert np.all(V.clean_resid[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_resid[(24, 25, 'ee')][~(V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')])] != 0.)
        assert np.all(V.clean_model[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_model[(24, 25, 'ee')][~V.clean_flags[(24, 25, 'ee')]] != 0.)
        # check that filtered_data is the same in channels that were not flagged
        atol = 1e-6 * np.mean(np.abs(V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')]]) ** 2.) ** .5
        assert np.all(np.isclose(V.clean_data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]],
                                 V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]], atol=atol, rtol=0.))
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax='both', overwrite=True, max_frate=1.0, mode='dpss_leastsq')
        assert np.all(['success' == V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_1'][i] for i in V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_1']])
        # check that clean resid is equal to zero in flagged channels
        assert np.all(V.clean_resid[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_resid[(24, 25, 'ee')][~(V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')])] != 0.)
        assert np.all(V.clean_model[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_model[(24, 25, 'ee')][~V.clean_flags[(24, 25, 'ee')]] != 0.)
        # check that filtered_data is the same in channels that were not flagged
        atol = 1e-6 * np.mean(np.abs(V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')]]) ** 2.) ** .5
        assert np.all(np.isclose(V.clean_data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]],
                                 V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]], atol=atol, rtol=0.))
        # run with flag_model_rms_outliers
        for ax in ['freq', 'time', 'both']:
            for k in V.flags:
                V.flags[k][:] = False
                V.data[k][:] = np.random.randn(*V.data[k].shape) + 1j * np.random.randn(*V.data[k].shape)
            # run with rms threshold < 1 which should lead to everything being flagged.
            V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax=ax, overwrite=True,
                        max_frate=1.0, mode='dpss_leastsq', flag_model_rms_outliers=True, model_rms_threshold=0.1)
            for k in [(24, 25, 'ee'), (24, 25, 'ee')]:
                assert np.all(V.clean_flags[k])
            # now use a threshold which should not lead to any flags.
            V.vis_clean(keys=[(24, 25, 'ee'), (24, 25, 'ee')], ax=ax, overwrite=True,
                        max_frate=1.0, mode='dpss_leastsq', flag_model_rms_outliers=True, model_rms_threshold=1e6)
            for k in [(24, 25, 'ee'), (24, 25, 'ee')]:
                assert not np.any(V.clean_flags[k])

    def test_vis_clean_flag_options(self, tmpdir):
        # tests for time and frequency partial flagging.
        tmp_path = tmpdir.strpath
        template = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        # first run flagging channels and frequencies
        fname_edgeflags = os.path.join(tmp_path, "zen.2458043.40141.xx.HH.XRAA.edgeflags.uvh5")
        fname_flagged = os.path.join(tmp_path, "zen.2458043.40141.xx.HH.XRAA.allflags.uvh5")
        hdt = io.HERAData(template)
        d, f, n = hdt.read()
        for k in d:
            f[k][:] = False
            f[k][:, 0] = True
            f[k][0, :] = True
        hdt.update(flags=f)
        hdt.write_uvh5(fname_edgeflags)
        for k in d:
            f[k][:] = True
        hdt.update(flags=f)
        hdt.write_uvh5(fname_flagged)
        V = VisClean(fname_flagged, filetype='uvh5')
        V.read()
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='freq', overwrite=True,
                    skip_flagged_edges=True)
        # make sure if no unflagged channels exist, then the clean flags are all flagged.
        for k in V.clean_flags:
            assert np.all(V.clean_flags[k])
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='freq', overwrite=True,
                    skip_contiguous_flags=True)
        for k in V.clean_flags:
            assert np.all(V.clean_flags[k])
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='time', overwrite=True,
                    skip_contiguous_flags=True, max_frate=0.025)
        for k in V.clean_flags:
            assert np.all(V.clean_flags[k])
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='time', overwrite=True,
                    skip_flagged_edges=True, max_frate=0.025)
        for k in V.clean_flags:
            assert np.all(V.clean_flags[k])
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='both', overwrite=True,
                    skip_contiguous_flags=True, max_frate=0.025)
        for k in V.clean_flags:
            assert np.all(V.clean_flags[k])
        # now do file with some edge flags. Make sure the edge flags remain in clean_flags.
        V = VisClean(fname_edgeflags, filetype='uvh5')
        V.read()
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='freq', overwrite=True,
                    skip_flagged_edges=True)
        for k in V.clean_flags:
            if not np.all(V.flags[k]):
                assert not np.all(V.clean_flags[k])
            assert np.all(V.clean_flags[k][0])
            assert np.all(V.clean_flags[k][:, 0])
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='time', overwrite=True,
                    skip_flagged_edges=True, max_frate=0.025)
        for k in V.clean_flags:
            if not np.all(V.flags[k]):
                assert not np.all(V.clean_flags[k])
            assert np.all(V.clean_flags[k][0])
            assert np.all(V.clean_flags[k][:, 0])
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='both', overwrite=True,
                    skip_flagged_edges=True, max_frate=0.025)
        for k in V.clean_flags:
            if not np.all(V.flags[k]):
                assert not np.all(V.clean_flags[k])
            assert np.all(V.clean_flags[k][0])
            assert np.all(V.clean_flags[k][:, 0])
        # now try using skip_contiguous flag gaps.
        standoff = 1e9 / (np.median(np.diff(V.freqs)))
        max_frate = datacontainer.DataContainer({(24, 25, 'ee'): 2. / np.abs(np.median(np.diff(V.times)) * 3.6 * 24.),
                                                 (24, 24, 'ee'): 1. / np.abs(2 * np.median(np.diff(V.times)) * 3.6 * 24.)})
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='freq', overwrite=True,
                    skip_contiguous_flags=True, standoff=standoff)
        # with this standoff, all data should be skipped.
        assert np.all(V.clean_flags[(24, 25, 'ee')])
        assert np.all(V.clean_flags[(24, 24, 'ee')])
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='time', overwrite=True,
                    skip_contiguous_flags=True, max_frate=max_frate)
        # this time, should only skip (24, 25, 'ee')
        assert np.all(V.clean_flags[(24, 25, 'ee')])
        assert not np.all(V.clean_flags[(24, 24, 'ee')])
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='both', overwrite=True,
                    skip_contiguous_flags=True, max_frate=max_frate, standoff=standoff)

        # now test flagging integrations within edge distance.
        # these flags should cause channel 12 to be
        # completely flagged if flagging mode is "both".
        for k in [(24, 25, 'ee'), (24, 24, 'ee')]:
            V.flags[k][:] = False
            V.flags[k][12, 0] = True
            V.flags[k][-1, 32] = True
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='both', overwrite=True,
                    max_frate=0.025, standoff=0.0, min_dly=50.,
                    skip_if_flag_within_edge_distance=(2, 2), mode='dpss_leastsq')
        for k in [(24, 25, 'ee'), (24, 24, 'ee')]:
            for i in range(V.Ntimes):
                if i == 12:
                    assert V.clean_info[k][(0, V.Nfreqs)]['status']['axis_1'][i] == 'skipped'
                else:
                    assert not np.any(V.clean_flags[k][i])

        # if flagging mode is 'freq', then integration 12 should be flagged
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='freq', overwrite=True,
                    max_frate=0.025, standoff=0.0, min_dly=50.,
                    skip_if_flag_within_edge_distance=2, mode='dpss_leastsq')
        for k in [(24, 25, 'ee'), (24, 24, 'ee')]:
            for i in range(V.Ntimes):
                if i == 12:
                    assert np.all(V.clean_flags[k][i])
                else:
                    assert not np.any(V.clean_flags[k][i])

        # if flagging mode is 'time', then channel 32 should be flagged.
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='time', overwrite=True,
                    max_frate=0.025, standoff=0.0, min_dly=50.,
                    skip_if_flag_within_edge_distance=2, mode='dpss_leastsq')
        for k in [(24, 25, 'ee'), (24, 24, 'ee')]:
            for i in range(V.Nfreqs):
                if i == 32:
                    assert np.all(V.clean_flags[k][:, i])
                else:
                    assert not np.any(V.clean_flags[k][:, i])

        # test clean_flags in resid_flags
        for k in [(24, 25, 'ee'), (24, 24, 'ee')]:
            V.flags[k][:] = False
            V.flags[k][-1, :-2] = True
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='freq', overwrite=True,
                    clean_flags_in_resid_flags=True, mode='dpss_leastsq',
                    max_frate=max_frate, standoff=0.0, min_dly=50., skip_wgt=0.5)
        for k in [(24, 25, 'ee'), (24, 24, 'ee')]:
            assert np.all(V.clean_resid_flags[k][-1])

    def test_vis_clean_spws(self, tmpdir):
        # test selecting partial frequency chunks.
        tmp_path = tmpdir.strpath
        template = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        # first run flagging channels and frequencies
        fname_edgeflags = os.path.join(tmp_path, "zen.2458043.40141.xx.HH.XRAA.edgeflags.uvh5")
        fname_flagged = os.path.join(tmp_path, "zen.2458043.40141.xx.HH.XRAA.allflags.uvh5")
        hdt = io.HERAData(template)
        d, f, n = hdt.read(frequencies=np.hstack([hdt.freqs[:30], hdt.freqs[32:48], hdt.freqs[49:]]))
        for k in d:
            f[k][:] = False
            f[k][:, 0] = True
            for i in (28, 29, 30, 31, 32, 33):
                f[k][:, i] = True  # flags around first discont
            f[k][:, 12] = True  # should be inpainted
            f[k][:, 46] = True  # flag near second break
            f[k][:, 47] = True  # flag near second break
            f[k][:, 49] = True  # should be inpainted
            d[k] = np.random.randn(*d[k].shape) + np.random.randn(*d[k].shape) * 1j
            if k[0] == k[1]:
                d[k] = d[k].real + 0j

            n[k] = np.ones(d[k].shape)
        hdt.update(flags=f, data=d)
        hdt.write_uvh5(fname_edgeflags)
        V = VisClean(fname_edgeflags, filetype='uvh5')
        V.read()
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='freq', overwrite=True,
                    skip_flagged_edges=True, mode='dpss_leastsq')
        for k in V.clean_flags:
            for i in range(V.Nfreqs):
                if i in (0, 28, 29, 30, 31, 32, 33, 46, 47):
                    assert np.all(V.clean_flags[k][:, i])
                else:
                    assert not np.any(V.clean_flags[k][:, i])
        # test spw_range functionality.
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='freq', overwrite=True,
                    skip_flagged_edges=True, filter_spw_ranges=[(0, 30), (30, 46), (46, V.Nfreqs)])
        for k in V.clean_flags:
            for i in range(V.Nfreqs):
                if i in (0, 28, 29, 30, 31, 32, 33, 46, 47):
                    assert np.all(V.clean_flags[k][:, i])
                else:
                    assert not np.any(V.clean_flags[k][:, i])
        # test NotImplementedError
        pytest.raises(NotImplementedError, V.vis_clean, keys=[(24, 25, 'ee')], ax='freq', overwrite=True,
                      filter_spw_ranges=[(0, 30), (31, V.Nfreqs)])

    def test_vis_clean_spws_time(self, tmpdir):
        # test spw-ranges with time filtering.
        tmp_path = tmpdir.strpath
        template = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        # first run flagging channels and frequencies
        fname_edgeflags = os.path.join(tmp_path, "zen.2458043.40141.xx.HH.XRAA.edgeflags.uvh5")
        fname_flagged = os.path.join(tmp_path, "zen.2458043.40141.xx.HH.XRAA.allflags.uvh5")
        hdt = io.HERAData(template)
        d, f, n = hdt.read(times=np.hstack([hdt.times[:30], hdt.times[32:48], hdt.times[49:]]))
        for k in d:
            f[k][:] = False
            f[k][0, :] = True
            for i in (28, 29, 30, 31, 32, 33):
                f[k][i, :] = True  # flags around first discont
            f[k][12, :] = True  # should be inpainted
            f[k][46, :] = True  # flag near second break
            f[k][47, :] = True  # flag near second break
            f[k][49, :] = True  # should be inpainted
            d[k] = np.random.randn(*d[k].shape) + np.random.randn(*d[k].shape) * 1j
            n[k] = np.ones(d[k].shape)
            if k[0] == k[1]:
                d[k] = d[k].real + 0j
        hdt.update(flags=f, data=d)
        hdt.write_uvh5(fname_edgeflags)
        V = VisClean(fname_edgeflags, filetype='uvh5')
        V.read()
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='time', overwrite=True,
                    skip_flagged_edges=True, mode='dpss_leastsq', max_frate=0.025)
        for k in V.clean_flags:
            for i in range(V.Ntimes):
                if i in (0, 28, 29, 30, 31, 32, 33, 46, 47):
                    assert np.all(V.clean_flags[k][i])
                else:
                    assert not np.any(V.clean_flags[k][i])
        # test spw_range functionality in time axis clean.
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='time', overwrite=True,
                    skip_flagged_edges=True, filter_spw_ranges=[(0, 30), (30, 46), (46, V.Nfreqs)], max_frate=0.025, mode='dpss_leastsq')
        for k in V.clean_flags:
            for i in range(V.Ntimes):
                if i in (0, 28, 29, 30, 31, 32, 33, 46, 47):
                    assert np.all(V.clean_flags[k][i])
                else:
                    assert not np.any(V.clean_flags[k][i])

    def test_vis_clean_spws_both(self, tmpdir):
        # test spw-ranges with time filtering.
        tmp_path = tmpdir.strpath
        template = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        # first run flagging channels and frequencies
        fname_edgeflags = os.path.join(tmp_path, "zen.2458043.40141.xx.HH.XRAA.edgeflags.uvh5")
        fname_flagged = os.path.join(tmp_path, "zen.2458043.40141.xx.HH.XRAA.allflags.uvh5")
        hdt = io.HERAData(template)
        d, f, n = hdt.read(times=np.hstack([hdt.times[:30], hdt.times[32:48], hdt.times[49:]]),
                           frequencies=np.hstack([hdt.freqs[:21], hdt.freqs[23:45], hdt.freqs[52:]]))
        for k in d:
            f[k][:] = False
            f[k][0, :] = True
            for i in (12, 28, 29, 30, 31, 32, 33, 46, 49):
                f[k][i, :] = True  # flags around first discont
            for j in (4, 18, 21, 22, 23, 42, 43, 52):
                f[k][:, j] = True

            d[k] = np.random.randn(*d[k].shape) + np.random.randn(*d[k].shape) * 1j
            if k[0] == k[1]:
                d[k] = d[k].real + 0j
            n[k] = np.ones(d[k].shape)
        hdt.update(flags=f, data=d)
        hdt.write_uvh5(fname_edgeflags)
        V = VisClean(fname_edgeflags, filetype='uvh5')
        V.read()
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='both', overwrite=True,
                    skip_flagged_edges=True, mode='dpss_leastsq', max_frate=0.025)
        for k in V.clean_flags:
            for i in (range(V.Ntimes)):
                if i in (0, 28, 29, 30, 31, 32, 33, 46):
                    assert np.all(V.clean_flags[k][i])
                    for spw_range in V.clean_info[k]:
                        assert i not in V.clean_info[k][spw_range]['status']['axis_1']
                else:
                    assert np.count_nonzero(~V.clean_flags[k][i]) == V.Nfreqs - 5
            for j in range(V.Nfreqs):
                if j in (21, 22, 23, 42, 43):
                    assert np.all(V.clean_flags[k][:, j])
                else:
                    assert np.count_nonzero(~V.clean_flags[k][:, j]) == V.Ntimes - 8

        # test spw_range functionality in time axis clean.
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='both', overwrite=True,
                    skip_flagged_edges=True, filter_spw_ranges=[(0, 30), (30, 46), (46, V.Nfreqs)], max_frate=0.025, mode='dpss_leastsq')
        for k in V.clean_flags:
            for i in (range(V.Ntimes)):
                if i in (0, 28, 29, 30, 31, 32, 33, 46):
                    assert np.all(V.clean_flags[k][i])
                    for spw_range in V.clean_info[k]:
                        assert i not in V.clean_info[k][spw_range]['status']['axis_1']
                else:
                    assert np.count_nonzero(~V.clean_flags[k][i]) == V.Nfreqs - 5
                    for spw_range in V.clean_info[k]:
                        assert i in V.clean_info[k][spw_range]['status']['axis_1']
            for spw_range in V.clean_info[k]:
                for j in range(spw_range[0], spw_range[1]):
                    if j in (21, 22, 23, 42, 43):
                        assert np.all(V.clean_flags[k][:, j])
                        assert j not in V.clean_info[k][spw_range]['status']['axis_0']
                    else:
                        assert np.count_nonzero(~V.clean_flags[k][:, j]) == V.Ntimes - 8
                        assert j in V.clean_info[k][spw_range]['status']['axis_0']

    def test_apply_flags(self):
        # cover edge cases of apply_flags not covered in test_delay_filter and
        # test_xtalk_filter.
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        flag_yaml = os.path.join(DATA_PATH, 'test_input/a_priori_flags_sample.yaml')
        pytest.raises(ValueError, V.apply_flags, flag_yaml, filetype='invalid_type')
        # cover overwrite flags
        flag_yaml = os.path.join(DATA_PATH, 'test_input/a_priori_flags_sample_noflags.yaml')
        V.read()
        nk = 0
        for k in V.flags.keys():
            # flag every other baseline
            if nk % 2 == 0:
                V.flags[k][:] = False
        original_flags = copy.deepcopy(V.flags)
        # applying empty flag yaml should result in overwriting
        # all flags with False except on baselines where all flags were True.
        V.apply_flags(flag_yaml, filetype='yaml', overwrite_flags=True)
        # check that this is the case.
        for k in V.flags:
            if np.all(original_flags[k]):
                assert np.all(V.flags[k])
            else:
                assert not np.any(V.flags[k])

    @pytest.mark.filterwarnings("ignore:.*dspec.vis_filter will soon be deprecated")
    def test_vis_clean(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        V.read()

        # just need to make sure various kwargs run through
        # actual code unit-testing coverage has been done in hera_filters.dspec

        # basic freq clean
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='freq', overwrite=True)
        assert np.all([V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_1'][i] == 'success' for i in V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_1']])
        # check that clean resid is equal to zero in flagged channels
        assert np.all(V.clean_resid[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_resid[(24, 25, 'ee')][~(V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')])] != 0.)
        assert np.all(V.clean_model[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_model[(24, 25, 'ee')][~V.clean_flags[(24, 25, 'ee')]] != 0.)
        # check that filtered_data is the same in channels that were not flagged
        atol = 1e-6 * np.mean(np.abs(V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')]]) ** 2.) ** .5
        assert np.all(np.isclose(V.clean_data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]],
                                 V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]], atol=atol, rtol=0.))
        # basic time clean
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='time', max_frate=10., overwrite=True)
        assert 'skipped' == V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_0'][0]
        assert 'success' == V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_0'][3]
        # check that clean resid is equal to zero in flagged channels
        assert np.all(V.clean_resid[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_resid[(24, 25, 'ee')][~(V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')])] != 0.)
        assert np.all(V.clean_model[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_model[(24, 25, 'ee')][~V.clean_flags[(24, 25, 'ee')]] != 0.)
        # check that filtered_data is the same in channels that were not flagged
        assert np.all(np.isclose(V.clean_data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]],
                                 V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]]))
        # basic 2d clean
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='both', max_frate=10., overwrite=True,
                    filt2d_mode='plus')
        assert np.all(['success' == V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_0'][i] for i in V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_0']])
        assert np.all(['success' == V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_1'][i] for i in V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_1']])
        # check that clean resid is equal to zero in flagged channels
        assert np.all(V.clean_resid[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_resid[(24, 25, 'ee')][~(V.clean_flags[(24, 25, 'ee')] | V.flags[(24, 25, 'ee')])] != 0.)
        assert np.all(V.clean_model[(24, 25, 'ee')][V.clean_flags[(24, 25, 'ee')]] == 0.)
        assert np.any(V.clean_model[(24, 25, 'ee')][~V.clean_flags[(24, 25, 'ee')]] != 0.)
        # check that filtered_data is the same in channels that were not flagged
        atol = 1e-6 * np.mean(np.abs(V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')]]) ** 2.) ** .5
        assert np.all(np.isclose(V.clean_data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]],
                                 V.data[(24, 25, 'ee')][~V.flags[(24, 25, 'ee')] & ~V.clean_flags[(24, 25, 'ee')]], atol=atol, rtol=0.))

        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='both', flags=V.flags + True, max_frate=10.,
                    overwrite=True, filt2d_mode='plus')
        assert np.all([V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_1'][i] == 'skipped' for i in V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_1']])
        assert np.all([V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_0'][i] == 'skipped' for i in V.clean_info[(24, 25, 'ee')][(0, V.Nfreqs)]['status']['axis_0']])

        # test fft data
        V.vis_clean(keys=[(24, 25, 'ee'), (24, 24, 'ee')], ax='both', max_frate=10., overwrite=True,
                    filt2d_mode='rect')

        # assert foreground peak is at 0 delay bin
        V.fft_data(data=V.clean_model, keys=[(24, 25, 'ee')], ax='freq', window='hann', edgecut_low=10, edgecut_hi=10, overwrite=True)
        assert np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'ee')]), axis=0)) == 32

        # assert foreground peak is at 0 FR bin (just due to FR resolution)
        V.fft_data(data=V.clean_model, keys=[(24, 25, 'ee')], ax='time', window='hann', edgecut_low=10, edgecut_hi=10, overwrite=True)
        assert np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'ee')]), axis=1)) == 30

        # assert foreground peak is at both 0 FR and 0 delay bin
        V.fft_data(data=V.clean_model, keys=[(24, 25, 'ee')], ax='both', window='tukey', alpha=0.5, edgecut_low=10, edgecut_hi=10, overwrite=True)
        assert np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'ee')]), axis=0)) == 32
        assert np.argmax(np.mean(np.abs(V.dfft[(24, 25, 'ee')]), axis=1)) == 30

        # check various kwargs
        V.fft_data(keys=[(24, 25, 'ee')], assign='foo', ifft=True, fftshift=True)
        delays = V.delays
        assert hasattr(V, 'foo')
        V.fft_data(keys=[(24, 25, 'ee')], assign='foo', overwrite=True, ifft=False, fftshift=False)
        np.testing.assert_array_almost_equal(delays, np.fft.fftshift(V.delays))

        # test flag factorization
        flags = V.factorize_flags(inplace=False, time_thresh=0.05)
        assert np.all(flags[(24, 25, 'ee')][45, :])
        assert np.all(flags[(24, 25, 'ee')][:, 5])

    def test_fft_data(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        V.read()

        # fft
        V.fft_data(zeropad=30, ifft=False)
        assert V.dfft[(24, 25, 'ee')].shape == (60, 124)

        # exceptions
        pytest.raises(ValueError, V.fft_data, ax='foo')
        pytest.raises(ValueError, V.fft_data, keys=[])
        pytest.raises(ValueError, V.fft_data, keys=[('foo')])

    # THIS UNIT TEST IS BROKEN!!!
    # See https://github.com/HERA-Team/hera_cal/issues/603
    def test_trim_model(self):
        # load data
        V = VisClean(os.path.join(DATA_PATH, "PyGSM_Jy_downselect.uvh5"))
        V.read(bls=[(23, 23, 'ee'), (23, 24, 'ee')])

        # interpolate to 768 frequencies
        freqs = np.linspace(120e6, 180e6, 768)
        for k in V.data:
            V.data[k] = interpolate.interp1d(V.freqs, V.data[k], axis=1, fill_value='extrapolate', kind='cubic')(freqs)
            V.flags[k] = np.zeros_like(V.data[k], dtype=bool)
        V.freqs = freqs
        # the old unit test was using the wrong dnu (for the original frequencies) which means that it was actually cleaning
        # out to 1250 ns. I've fixed this dnu bug and used a larger min_dly below.
        V.Nfreqs = len(V.freqs)
        # dnu should have also been set here to be np.diff(np.median(freqs))
        # but it wasn't Because of this, the old version of vis_clean was cleaning with a
        # delay width = intended delay width x (manually set dnu / original dnu of the attached data)
        np.random.seed(0)
        k = (23, 24, 'ee')
        beam_interp = Beam(HS_DATA_PATH / 'HERA_H1C_BEAM_POLY.npy')
        Op = beam_interp(V.freqs / 1e9)
        # V.data[k] += noise.sky_noise_jy(autovis=V.data[(23, 23, 'ee')], freqs=V.freqs / 1e9, lsts=V.lsts, omega_p=Op)
        V.data[k] += noise.sky_noise_jy(V.lsts, V.freqs / 1e9, omega_p=Op, integration_time=50, autovis=V.data[(23, 23, 'ee')])

        # add lots of random flags
        f = np.zeros(V.Nfreqs, dtype=bool)[None, :]
        f[:, 127:156] = True
        f[:, 300:303] = True
        f[:, 450:455] = True
        f[:, 625:630] = True
        V.flags[k] += f
        # Note that the intended delay width of this unit test was 300 ns but because of the dnu bug, the delay width was
        # actuall 300 x V.dnu / np.median(np.diff(V.freqs))
        # the new vis_clean never explicitly references V.dnu so it doesn't have problems and uses the correct delay width.
        # however, using the correct delay width causes this unit test to fail.
        # so we need to fix it. SEP (Somebody Elses PR).
        V.vis_clean(data=V.data, flags=V.flags, keys=[k], tol=1e-6, min_dly=300. * (V.dnu / np.median(np.diff(V.freqs))), ax='freq', overwrite=True, window='tukey', alpha=0.2)
        V.fft_data(V.data, window='bh', overwrite=True, assign='dfft1')
        V.fft_data(V.clean_data, window='bh', overwrite=True, assign='dfft2')

        # trim model
        mdl, n = vis_clean.trim_model(V.clean_model, V.clean_resid, V.dnu, noise_thresh=3.0, delay_cut=500,
                                      kernel_size=21, polyfit_deg=None)
        clean_data2 = deepcopy(V.clean_data)
        clean_data2[k][V.flags[k]] = mdl[k][V.flags[k]]
        V.fft_data(clean_data2, window='bh', overwrite=True, assign='dfft3')

        # get averaged spectra
        n1 = vis_clean.noise_eq_bandwidth(dspec.gen_window('bh', V.Nfreqs))
        n2 = vis_clean.noise_eq_bandwidth(dspec.gen_window('bh', V.Nfreqs) * ~V.flags[k][0])
        d1 = np.mean(np.abs(V.dfft1[k]), axis=0) * n1
        d2 = np.mean(np.abs(V.dfft2[k]), axis=0) * n2
        d3 = np.mean(np.abs(V.dfft3[k]), axis=0) * n2

        # confirm that dfft3 and dfft1 match while dfft2 and dfft1 do not near CLEAN boundary
        select = (np.abs(V.delays) < 300) & (np.abs(V.delays) > 100)
        assert np.isclose(np.mean(np.abs(d1)[select]), np.mean(np.abs(d3)[select]), atol=.1)
        assert not np.isclose(np.mean(np.abs(d1)[select]), np.mean(np.abs(d2)[select]), atol=.1)

        # test that polynomial fitting is a good fit
        _, n1 = vis_clean.trim_model(V.clean_model, V.clean_resid, V.dnu, noise_thresh=3.0, delay_cut=500,
                                     kernel_size=None, polyfit_deg=None)
        _, n2 = vis_clean.trim_model(V.clean_model, V.clean_resid, V.dnu, noise_thresh=3.0, delay_cut=500,
                                     kernel_size=None, polyfit_deg=5)
        assert (np.std(n1[k] - n2[k]) / np.mean(n2[k])) < 0.1  # assert residual is below 10% of fit

        # test well-conditioned check takes effect
        V2 = deepcopy(V)
        V2.clean_resid[k][:-2] = 0.0  # zero all the data except last two integrations
        _, n2 = vis_clean.trim_model(V2.clean_model, V2.clean_resid, V2.dnu, noise_thresh=3.0, delay_cut=500,
                                     kernel_size=None, polyfit_deg=5)
        assert np.all(np.isclose(n2[k][-1], n1[k][-1]))  # assert non-zeroed output are same as n1 (no polyfit)

    def test_neb(self):
        n = vis_clean.noise_eq_bandwidth(dspec.gen_window('blackmanharris', 10000))
        assert np.isclose(n, 1.9689862471203075)

    def test_zeropad(self):
        fname = os.path.join(DATA_PATH, "zen.2458043.40141.xx.HH.XRAA.uvh5")
        V = VisClean(fname, filetype='uvh5')
        V.read()

        # test basic zeropad
        d, _ = vis_clean.zeropad_array(V.data[(24, 25, 'ee')], zeropad=30, axis=-1, undo=False)
        assert d.shape == (60, 124)
        assert np.allclose(d[:, :30], 0.0)
        assert np.allclose(d[:, -30:], 0.0)
        d, _ = vis_clean.zeropad_array(d, zeropad=30, axis=-1, undo=True)
        assert d.shape == (60, 64)

        # test zeropad with bool
        f, _ = vis_clean.zeropad_array(V.flags[(24, 25, 'ee')], zeropad=30, axis=-1, undo=False)
        assert f.shape == (60, 124)
        assert np.all(f[:, :30])
        assert np.all(f[:, -30:])

        # zeropad with binvals
        d, bval = vis_clean.zeropad_array(V.data[(24, 25, 'ee')], zeropad=30, axis=0, binvals=V.times)
        assert np.allclose(np.median(np.diff(V.times)), np.median(np.diff(bval)))
        assert len(bval) == 120

        # 2d zeropad
        d, bval = vis_clean.zeropad_array(V.data[(24, 25, 'ee')], zeropad=(30, 10), axis=(0, 1), binvals=[V.times, V.freqs])
        assert d.shape == (120, 84)
        assert (bval[0].size, bval[1].size) == (120, 84)

        # un-pad with bval
        d, bval = vis_clean.zeropad_array(d, zeropad=(30, 10), axis=(0, 1), binvals=bval, undo=True)
        assert d.shape == (60, 64)
        assert (bval[0].size, bval[1].size) == (60, 64)

        # test VisClean method
        V.zeropad_data(V.data, binvals=V.times, zeropad=10, axis=0, undo=False)
        assert V.data[(24, 25, 'ee')].shape == (80, 64)
        assert V.data.binvals.size == 80

        # exceptions
        pytest.raises(ValueError, vis_clean.zeropad_array, V.data[(24, 25, 'ee')], axis=(0, 1), zeropad=0)
        pytest.raises(ValueError, vis_clean.zeropad_array, V.data[(24, 25, 'ee')], axis=(0, 1), zeropad=(0,))

    def test_filter_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--spw_range', '0', '20', '--filter_spw_ranges', '0~10,12~20']
        parser = vis_clean._filter_argparser()
        a = parser.parse_args()
        assert a.datafilelist == ['a']
        assert a.clobber is True
        assert a.spw_range[0] == 0
        assert a.spw_range[1] == 20
        assert a.filter_spw_ranges == [(0, 10), (12, 20)]
        assert a.time_thresh == 0.05
        assert not a.factorize_flags
        # test alternative.filter_spw_ranges format.
        sys.argv = [sys.argv[0], 'a', '--clobber', '--spw_range', '0', '20', '--filter_spw_ranges', '0 10,12 20']
        parser = vis_clean._filter_argparser()
        a = parser.parse_args()
        assert a.datafilelist == ['a']
        assert a.clobber is True
        assert a.spw_range[0] == 0
        assert a.spw_range[1] == 20
        assert a.filter_spw_ranges == [(0, 10), (12, 20)]
        assert a.time_thresh == 0.05
        assert not a.factorize_flags

    def test_filter_argparser_multifile(self):
        # test multifile functionality of _filter_argparser
        sys.argv = [sys.argv[0], 'a', 'b', 'c', '--clobber', '--spw_range', '0', '20', '--calfilelist', 'cal1', 'cal2', 'cal3',
                    '--cornerturnfile', 'a']
        parser = vis_clean._filter_argparser()
        a = parser.parse_args()
        assert a.datafilelist == ['a', 'b', 'c']
        assert a.cornerturnfile == 'a'
        assert a.calfilelist == ['cal1', 'cal2', 'cal3']
        assert a.clobber is True
        assert a.spw_range[0] == 0
        assert a.spw_range[1] == 20
        assert a.time_thresh == 0.05
        assert not a.factorize_flags

    def test_time_chunk_from_baseline_chunks_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--clobber', '--baseline_chunk_files', 'a', 'b', 'c', 'd', '--outfilename', 'a.out']
        parser = vis_clean.time_chunk_from_baseline_chunks_argparser()
        a = parser.parse_args()
        assert a.clobber
        for char in ['a', 'b', 'c', 'd']:
            assert char in a.baseline_chunk_files
        assert a.time_chunk_template == 'a'
        assert a.outfilename == 'a.out'

    def test_time_chunk_from_baseline_chunks(self, tmp_path):
        # First, construct some cross-talk baseline files.
        datafiles = [os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.first.uvh5"),
                     os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.second.uvh5")]

        cals = [os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only.part1"),
                os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only.part2")]
        # make a cache directory
        cdir = tmp_path / "cache_temp"
        cdir.mkdir()
        # cross-talk filter chunked baselines
        for filenum, file in enumerate(datafiles):
            baselines = io.baselines_from_filelist_position(file, datafiles)
            fname = 'temp.fragment.part.%d.h5' % filenum
            fragment_filename = tmp_path / fname
            frf.load_tophat_frfilter_and_write(datafiles, baseline_list=baselines, calfile_list=cals,
                                               spw_range=[0, 20], cache_dir=cdir, read_cache=True, write_cache=True,
                                               res_outfilename=fragment_filename, clobber=True, case='sky')
            # load in fragment and make sure the number of baselines is equal to the length of the baseline list
            hd_fragment = io.HERAData(str(fragment_filename))
            assert len(hd_fragment.bls) == len(baselines)
            assert hd_fragment.Ntimes == 60
            assert hd_fragment.Nfreqs == 20

        fragments = glob.glob(DATA_PATH + '/test_output/temp.fragment.h5.part*')
        # reconstitute the filtered data
        for filenum, file in enumerate(datafiles):
            # reconstitute
            fname = 'temp.reconstituted.part.%d.h5' % filenum
            vis_clean.time_chunk_from_baseline_chunks(time_chunk_template=file,
                                                      baseline_chunk_files=glob.glob(str(tmp_path / 'temp.fragment.part.*.h5')), clobber=True,
                                                      outfilename=str(tmp_path / fname))
        # load in the reconstituted files.
        hd_reconstituted = io.HERAData(glob.glob(str(tmp_path / 'temp.reconstituted.part.*.h5')))
        hd_reconstituted.read()
        # compare to xtalk filtering the whole file.
        frf.load_tophat_frfilter_and_write(datafile_list=os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5"),
                                           calfile_list=os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only"),
                                           res_outfilename=str(tmp_path / 'temp.h5'), clobber=True, spw_range=[0, 20], case='sky')
        hd = io.HERAData(str(tmp_path / 'temp.h5'))
        hd.read()
        assert np.all(np.isclose(hd.data_array, hd_reconstituted.data_array))
        assert np.all(np.isclose(hd.flag_array, hd_reconstituted.flag_array))
        assert np.all(np.isclose(hd.nsample_array, hd_reconstituted.nsample_array))
        # Do the same thing with time-bounds mode.
        for filenum, file in enumerate(datafiles):
            # reconstitute
            fname = 'temp.reconstituted.part.%d.h5' % filenum
            vis_clean.time_chunk_from_baseline_chunks(time_chunk_template=file,
                                                      baseline_chunk_files=glob.glob(str(tmp_path / 'temp.fragment.part.*.h5')), clobber=True,
                                                      outfilename=str(tmp_path / fname), time_bounds=True)
        # load in the reconstituted files.
        hd_reconstituted = io.HERAData(glob.glob(str(tmp_path / 'temp.reconstituted.part.*.h5')))
        hd_reconstituted.read()
        # compare to xtalk filtering the whole file.
        frf.load_tophat_frfilter_and_write(datafile_list=os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.OCR_53x_54x_only.uvh5"),
                                           calfile_list=os.path.join(DATA_PATH, "test_input/zen.2458101.46106.xx.HH.uv.abs.calfits_54x_only"),
                                           res_outfilename=str(tmp_path / 'temp.h5'), clobber=True, spw_range=[0, 20], case='sky')
        hd = io.HERAData(str(tmp_path / 'temp.h5'))
        hd.read()
        assert np.all(np.isclose(hd.data_array, hd_reconstituted.data_array))
        assert np.all(np.isclose(hd.flag_array, hd_reconstituted.flag_array))
        assert np.all(np.isclose(hd.nsample_array, hd_reconstituted.nsample_array))
        # check warning.
        with pytest.warns(RuntimeWarning):
            vis_clean.time_chunk_from_baseline_chunks(datafiles[0], baseline_chunk_files=datafiles[1:], clobber=True, outfilename=str(tmp_path / fname), time_bounds=True)
