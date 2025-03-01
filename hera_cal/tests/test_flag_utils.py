# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License


import pytest
import numpy as np
import os
import copy
from pyuvdata import UVData

from .. import datacontainer, flag_utils, io, utils
from ..data import DATA_PATH


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
def test_solar_flag():
    data_fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
    uvd = UVData()
    uvd.read_miriad(data_fname)
    data, flags, antp, ant, f, t, _, p = io.load_vis(uvd, return_meta=True)
    # get solar altitude
    a = utils.get_sun_alt(2458043)
    assert isinstance(a, (float, np.floating, np.float64))
    a = utils.get_sun_alt([2458043, 2458043.5])
    assert isinstance(a, (np.ndarray))
    # test solar flag
    bl = (24, 25, 'ee')
    _flags = flag_utils.solar_flag(flags, times=t, flag_alt=20.0, inplace=False)
    assert _flags[bl][:41].all()
    assert not flags[bl][:41].all()
    # test ndarray
    flag_utils.solar_flag(flags[bl], times=t, flag_alt=20.0, inplace=True)
    assert flags[bl][:41].all()
    # test uvd
    flag_utils.solar_flag(uvd, flag_alt=20.0, inplace=True)
    assert uvd.get_flags(bl)[:41].all()
    # test exception
    pytest.raises(AssertionError, flag_utils.solar_flag, flags)


def test_synthesize_ant_flags():
    flags = datacontainer.DataContainer({(0, 0, 'xx'): np.ones((5, 5), bool),
                                         (0, 1, 'xx'): np.ones((5, 5), bool),
                                         (1, 2, 'xx'): np.zeros((5, 5), bool),
                                         (2, 3, 'xx'): np.zeros((5, 5), bool)})
    flags[(2, 3, 'xx')][:, 4] = True
    # aggressive flagging
    ant_flags = flag_utils.synthesize_ant_flags(flags, threshold=0.0)
    np.testing.assert_array_equal(ant_flags[(0, 'Jxx')], True)
    np.testing.assert_array_equal(ant_flags[(1, 'Jxx')], False)
    np.testing.assert_array_equal(ant_flags[(2, 'Jxx')][:, 0:4], False)
    np.testing.assert_array_equal(ant_flags[(2, 'Jxx')][:, 4], True)
    np.testing.assert_array_equal(ant_flags[(3, 'Jxx')][:, 0:4], False)
    np.testing.assert_array_equal(ant_flags[(3, 'Jxx')][:, 4], True)
    # conservative flagging
    ant_flags = flag_utils.synthesize_ant_flags(flags, threshold=0.75)
    np.testing.assert_array_equal(ant_flags[(2, 'Jxx')][:, 4], False)
    # very conservative flagging
    flags[(1, 2, 'xx')][:3, 4] = True
    ant_flags = flag_utils.synthesize_ant_flags(flags, threshold=1.0)
    np.testing.assert_array_equal(ant_flags[(2, 'Jxx')][:3, 4], True)
    np.testing.assert_array_equal(ant_flags[(2, 'Jxx')][3:, 4], False)


@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
def test_factorize_flags():
    # load data
    data_fname = os.path.join(DATA_PATH, "zen.2458043.12552.xx.HH.uvORA")
    hd = io.HERAData(data_fname, filetype='miriad')
    data, flags, _ = hd.read(bls=[(24, 25)])

    # run on ndarray
    f = flag_utils.factorize_flags(flags[(24, 25, 'ee')].copy(), time_thresh=1.5 / 60, inplace=False)
    assert f[52].all()
    assert f[:, 60:62].all()

    f = flag_utils.factorize_flags(flags[(24, 25, 'ee')].copy(), spw_ranges=[(45, 60)],
                                   time_thresh=0.5 / 60, inplace=False)
    assert f[:, 48].all()
    assert not np.min(f, axis=1).any()
    assert not f[:, 24].all()

    f = flags[(24, 25, 'ee')].copy()
    flag_utils.factorize_flags(f, time_thresh=0.5 / 60, inplace=True)
    assert f[:, 48].all()
    assert not np.min(f, axis=1).any()

    # run on datacontainer
    f2 = flag_utils.factorize_flags(copy.deepcopy(flags), time_thresh=0.5 / 60, inplace=False)
    np.testing.assert_array_equal(f2[(24, 25, 'ee')], f)

    # test exceptions
    pytest.raises(ValueError, flag_utils.factorize_flags, flags, spw_ranges=(0, 1))
    pytest.raises(ValueError, flag_utils.factorize_flags, 'hi')


def test_get_minimal_slices_all_true_empty_freq_cuts():
    # When flag_wf is all True and freq_cuts is empty,
    # freqs is set automatically (using np.arange) and the function
    # should return None for time_slice and [None] for band_slices.
    flag_wf = np.ones((5, 5), dtype=bool)
    time_slice, band_slices = flag_utils.get_minimal_slices(flag_wf, freqs=None, freq_cuts=[])
    assert time_slice is None
    assert band_slices == [None]


def test_get_minimal_slices_all_true_nonempty_freq_cuts():
    # When flag_wf is all True and freq_cuts is non-empty,
    # freqs must be provided; since no False pixels exist,
    # the function returns None for time_slice and [None, None] for band_slices.
    flag_wf = np.ones((3, 4), dtype=bool)
    freq_cuts = [2.5]
    freqs = np.array([1, 2, 3, 4])
    time_slice, band_slices = flag_utils.get_minimal_slices(flag_wf, freqs=freqs, freq_cuts=freq_cuts)
    assert time_slice is None
    assert band_slices == [None, None]


def test_get_minimal_slices_not_all_flagged_empty_freq_cuts():
    # Test with empty freq_cuts (default freqs will be used) and a single False pixel.
    flag_wf = np.array([
        [True, True, True, True, True],
        [True, False, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True]
    ], dtype=bool)
    # When freq_cuts is empty, the function sets freqs = np.arange(nfreqs)
    time_slice, band_slices = flag_utils.get_minimal_slices(flag_wf, freqs=None, freq_cuts=[])
    # The False pixel is in row index 1.
    assert time_slice.start == 1 and time_slice.stop == 2
    # Default freqs becomes [0, 1, 2, 3, 4]; the False pixel is at column index 1.
    assert band_slices[0] is not None
    assert band_slices[0].start == 1 and band_slices[0].stop == 2


def test_get_minimal_slices_not_all_flagged_nonempty_freq_cuts():
    # Test with non-empty freq_cuts and False pixels in different frequency bands.
    flag_wf = np.ones((4, 6), dtype=bool)
    flag_wf[1, 1] = False  # False pixel in band 0 (freq < 3.5)
    flag_wf[2, 4] = False  # False pixel in band 1 (freq > 3.5)
    freqs = np.array([1, 2, 3, 4, 5, 6])
    freq_cuts = [3.5]
    time_slice, band_slices = flag_utils.get_minimal_slices(flag_wf, freqs=freqs, freq_cuts=freq_cuts)
    # Time slice should cover rows 1 and 2.
    assert time_slice.start == 1 and time_slice.stop == 3
    # For band 0: interval (-inf, 3.5) covers freqs [1,2,3]; False pixel at column index 1.
    assert band_slices[0] is not None
    assert band_slices[0].start == 1 and band_slices[0].stop == 2
    # For band 1: interval (3.5, inf) covers freqs [4,5,6]; False pixel at column index 4.
    assert band_slices[1] is not None
    assert band_slices[1].start == 4 and band_slices[1].stop == 5


def test_get_minimal_slices_missing_freqs_error():
    # When freq_cuts is non-empty and freqs is None,
    # the function should raise a ValueError.
    flag_wf = np.array([[True, False],
                        [True, True]])
    freq_cuts = [1.5]
    with pytest.raises(ValueError):
        flag_utils.get_minimal_slices(flag_wf, freqs=None, freq_cuts=freq_cuts)


def test_get_minimal_slices_wrong_shape_freqs_error():
    # When freq_cuts is non-empty and the length of freqs does not match flag_wf's columns,
    # the function should raise a ValueError.
    flag_wf = np.array([[True, False, True],
                        [True, True, True]])
    freqs = np.array([1, 2])  # Incorrect length.
    freq_cuts = [1.5]
    with pytest.raises(ValueError):
        flag_utils.get_minimal_slices(flag_wf, freqs=freqs, freq_cuts=freq_cuts)


def test_get_minimal_slices_band_without_false():
    # Test a case where one frequency band (as defined by freq_cuts) does not contain any False pixels.
    flag_wf = np.ones((3, 6), dtype=bool)
    flag_wf[1, 1] = False  # False pixel only in band 0.
    freqs = np.array([1, 2, 3, 4, 5, 6])
    freq_cuts = [3.5]
    time_slice, band_slices = flag_utils.get_minimal_slices(flag_wf, freqs=freqs, freq_cuts=freq_cuts)
    # Time slice should cover row 1.
    assert time_slice.start == 1 and time_slice.stop == 2
    # Band 0 should capture the False pixel at column 1.
    assert band_slices[0] is not None
    assert band_slices[0].start == 1 and band_slices[0].stop == 2
    # Band 1 should remain None because there are no False pixels in that band.
    assert band_slices[1] is None
