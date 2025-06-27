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
    # freqs is set automatically and the function
    # should return [None] for time_slices and [None] for band_slices.
    flag_wf = np.ones((5, 5), dtype=bool)
    time_slices, band_slices = flag_utils.get_minimal_slices(flag_wf, freqs=None, freq_cuts=[])
    assert time_slices == [None]
    assert band_slices == [None]


def test_get_minimal_slices_all_true_nonempty_freq_cuts():
    # When flag_wf is all True and freq_cuts is non-empty,
    # freqs must be provided; since no False pixels exist,
    # the function returns [None, None] for both time_slices and band_slices.
    flag_wf = np.ones((3, 4), dtype=bool)
    freq_cuts = [2.5]
    freqs = np.array([1, 2, 3, 4])
    time_slices, band_slices = flag_utils.get_minimal_slices(flag_wf, freqs=freqs, freq_cuts=freq_cuts)
    assert time_slices == [None, None]
    assert band_slices == [None, None]


def test_get_minimal_slices_not_all_flagged_empty_freq_cuts():
    # Single-band case (freq_cuts=[]). One False pixel at row 1, col 1.
    flag_wf = np.array([
        [True, True, True, True, True],
        [True, False, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True]
    ], dtype=bool)
    time_slices, band_slices = flag_utils.get_minimal_slices(flag_wf, freqs=None, freq_cuts=[])
    # Only one band, so inspect index 0
    assert time_slices[0].start == 1 and time_slices[0].stop == 2
    assert band_slices[0] is not None
    assert band_slices[0].start == 1 and band_slices[0].stop == 2


def test_get_minimal_slices_not_all_flagged_nonempty_freq_cuts():
    # Two-band case (freq_cuts=[3.5]). False at (1,1) in band0 and (2,4) in band1.
    flag_wf = np.ones((4, 6), dtype=bool)
    flag_wf[1, 1] = False
    flag_wf[2, 4] = False
    freqs = np.array([1, 2, 3, 4, 5, 6])
    freq_cuts = [3.5]
    time_slices, band_slices = flag_utils.get_minimal_slices(flag_wf, freqs=freqs, freq_cuts=freq_cuts)

    # Per‐band time slices
    assert time_slices[0].start == 1 and time_slices[0].stop == 2
    assert time_slices[1].start == 2 and time_slices[1].stop == 3

    # Per‐band frequency slices
    assert band_slices[0] is not None
    assert band_slices[0].start == 1 and band_slices[0].stop == 2
    assert band_slices[1] is not None
    assert band_slices[1].start == 4 and band_slices[1].stop == 5


def test_get_minimal_slices_missing_freqs_error():
    # freq_cuts non-empty but freqs=None should still raise ValueError
    flag_wf = np.array([[True, False],
                        [True, True]])
    freq_cuts = [1.5]
    with pytest.raises(ValueError):
        flag_utils.get_minimal_slices(flag_wf, freqs=None, freq_cuts=freq_cuts)


def test_get_minimal_slices_wrong_shape_freqs_error():
    # freq_cuts non-empty with mismatched freqs length should raise ValueError
    flag_wf = np.array([[True, False, True],
                        [True, True, True]])
    freqs = np.array([1, 2])
    freq_cuts = [1.5]
    with pytest.raises(ValueError):
        flag_utils.get_minimal_slices(flag_wf, freqs=freqs, freq_cuts=freq_cuts)


def test_get_minimal_slices_band_without_false():
    # Two-band case where band1 has no False pixels
    flag_wf = np.ones((3, 6), dtype=bool)
    flag_wf[1, 1] = False  # only in band0
    freqs = np.array([1, 2, 3, 4, 5, 6])
    freq_cuts = [3.5]
    time_slices, band_slices = flag_utils.get_minimal_slices(flag_wf, freqs=freqs, freq_cuts=freq_cuts)

    # band0 populated, band1 stays None
    assert time_slices[0].start == 1 and time_slices[0].stop == 2
    assert band_slices[0] is not None
    assert band_slices[0].start == 1 and band_slices[0].stop == 2

    assert time_slices[1] is None
    assert band_slices[1] is None


@pytest.mark.parametrize(
    "arr, expected",
    [
        # 1-D – interior non-zeros
        (
            np.array([0, 0, 3, 0, 0, 5, 0]),
            np.array([2, 1, 0, 1, 1, 0, 1], dtype=float),
        ),
        # 1-D – non-zero only at start
        (
            np.array([4, 0, 0, 0, 0]),
            np.array([0, 1, 2, 3, 4], dtype=float),
        ),
        # 1-D – non-zero only at end
        (
            np.array([0, 0, 0, 0, 7]),
            np.array([4, 3, 2, 1, 0], dtype=float),
        ),
        # 2-D – verify broadcasting across leading axes
        (
            np.array([[0, 1, 0],
                      [2, 0, 0]]),
            np.array([[1, 0, 1],
                      [0, 1, 2]], dtype=float),
        ),
        # 3-D case
        (
            np.array([[[0, 1, 0],      # first row, first “plane”
                       [0, 0, 2]],
                      [[3, 0, 0],      # second “plane”
                       [0, 0, 0]]]),
            np.array([[[1, 0, 1],
                       [2, 1, 0]],
                      [[0, 1, 2],
                       [np.inf, np.inf, np.inf]]], dtype=float),
        ),
    ],
)
def test_expected_values(arr, expected):
    """Exact values for simple cases."""
    result = flag_utils.distance_to_nearest_nonzero(arr)
    np.testing.assert_array_equal(result, expected)


def test_all_zeros_returns_inf():
    """If there are no non-zeros at all, the distance should be ∞ everywhere."""
    arr = np.zeros((3, 4, 5))
    out = flag_utils.distance_to_nearest_nonzero(arr)
    assert out.shape == arr.shape
    assert np.all(np.isinf(out))


def test_random_nd_matches_bruteforce():
    """
    For a random 3-D array, compare against a slow but simple
    brute-force implementation to ensure correctness on N-D input.
    """
    rng = np.random.default_rng(0)
    arr = rng.integers(-1, 2, size=(4, 3, 7))  # -1, 0, or 1

    def brute_force(a):
        out = np.empty_like(a, dtype=float)
        *lead_axes, L = a.shape
        for index in np.ndindex(*lead_axes):
            line = a[index]                      # shape (L,)
            nz = np.flatnonzero(line)
            if nz.size == 0:
                out[index] = np.inf
                continue
            for i in range(L):
                out[index + (i,)] = np.abs(i - nz).min()
        return out

    np.testing.assert_array_equal(
        flag_utils.distance_to_nearest_nonzero(arr),
        brute_force(arr),
    )
