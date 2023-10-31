"""
An attempt at a simpler LST binner that makes more assumptions but runs faster.

In particular, we assume that all baselines with a particular observation have the same
time and frequency arrays, and use this to vectorize many of the calculations.

The main entry-point is the :func:`lst_bin_files` function, which takes a specific
configuration file, produced by :func:`make_lst_bin_config_file`, that specifies which
data files to bin into which LST bins, and outputs files containing the binned data.
This is similar to the older :func:`~lstbin.lst_bin_files` function (though
the interface is slightly different, as the new function assumes a pre-configured
configuration file).

"""
from __future__ import annotations

import numpy as np
from . import utils
import warnings
from pathlib import Path
from .lstbin import sigma_clip, make_lst_grid
from . import abscal
import os
from . import io
import logging
from hera_qm.metrics_io import read_a_priori_ant_flags
from . import apply_cal
from typing import Sequence, Any
import argparse
from pyuvdata.uvdata.uvh5 import FastUVH5Meta
from pyuvdata import utils as uvutils
from .red_groups import RedundantGroups
import h5py
from functools import partial
import yaml
from .types import Antpair
from .datacontainer import DataContainer

try:
    profile
except NameError:

    def profile(fnc):
        return fnc


logger = logging.getLogger(__name__)


@profile
def lst_align(
    data: np.ndarray,
    data_lsts: np.ndarray,
    antpairs: list[Antpair],
    lst_bin_edges: np.ndarray,
    freq_array: np.ndarray,
    flags: np.ndarray | None = None,
    nsamples: np.ndarray | None = None,
    where_inpainted: np.ndarray | None = None,
    rephase: bool = True,
    antpos: dict[int, np.ndarray] | None = None,
    lat: float = -30.72152,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Split input data into a list of LST bins.

    This function simply splits a data array with multiple time stamps into a list of
    arrays, each containing a single LST bin. Each of the data arrays in each bin
    may also be rephased onto a common LST grid, taken to be the center of each bin.

    The data is binned via a simple histogram, i.e. the data represented at each LST
    is essentially assumed to be a delta function in LST, and is fully assigned to one
    particular LST bin. Due to this, it is irrelevant whether the ``data_lsts``
    represent the start, end, or centre of each integration -- either choice will
    incur similar errors.

    Parameters
    ----------
    data
        The complex visibility data. Must be shape ``(ntimes, nbls, nfreqs, npols)``,
        where the times may be sourced from multiple days.
    data_lsts
        The LSTs corresponding to each of the time stamps in the data. Must have
        length ``data.shape[0]``. As noted above, these may be the start, end, or
        centre of each integration, as long as it is consistent for all the data.
    antpairs
        The list of antenna pairs in the data, in the order they appear in ``data``.
        Each element is a tuple of two antenna numbers, e.g. ``(0, 1)``.
    lst_bin_edges
        A sequence of floats specifying the *edges* of the LST bins to use, with length
        ``N_lstbins + 1``. Bins are thus assumed to be contiguous, but not necessarily
        of equal size.
    freq_array
        An array of frequencies in the data, in Hz. Size must be ``data.shape[2]``.
    flags
        An array of boolean flags, indicating data NOT to use. Same shape as ``data``.
    nsamples
        An float array of sample counts, same shape as ``data``.
    rephase
        Whether to apply re-phasing to the data, to bring it to a common LST grid.
        The LSTs to which the data are rephased are the centres of the LST bins (i.e.
        the mid-point of each pair of ``lst_bin_edges``).
    antpos
        3D Antenna positions for each antenna in the data. Only required if rephasing.
        Keys are antenna numbers, values are 3-element arrays of ENU coordinates.
        Units are metres.
    lat
        The latitude (in degrees) of the telescope. Only required if rephasing.

    Returns
    -------
    lst_bin_centers
        The centres of the LST bins, in radians. Shape is ``(N_lstbins,)``, which is
        one less than the length of ``lst_bin_edges``.
    data
        A list of length ``N_lstbins`` of arrays, each of shape
        ``(nintegrations_in_lst, nbls, nfreq, npol)``, where LST bins without data
        simply have a first-axis of size zero.
    flags
        Same as ``data``, but boolean flags.
    nsamples
        Same as ``data``, but sample counts.

    See Also
    --------
    :func:`reduce_lst_bins`
        Function that takes outputs from this function and computes reduced values (e.g.
        mean, std) from them.
    """
    npols = data.shape[-1]
    required_shape = (len(data_lsts), len(antpairs), len(freq_array), npols)

    if npols > 4:
        raise ValueError(f"data has more than 4 pols! Got {npols} (last axis of data)")

    if data.shape != required_shape:
        raise ValueError(
            f"data should have shape {required_shape} but got {data.shape}"
        )

    if flags is None:
        flags = np.zeros(data.shape, dtype=bool)

    if flags.shape != data.shape:
        raise ValueError(f"flags should have shape {data.shape} but got {flags.shape}")

    if nsamples is None:
        nsamples = np.ones(data.shape, dtype=float)

    if nsamples.shape != data.shape:
        raise ValueError(
            f"nsamples should have shape {data.shape} but got {nsamples.shape}"
        )

    if len(lst_bin_edges) < 2:
        raise ValueError("lst_bin_edges must have at least 2 elements")

    # Ensure the lst bin edges start within (0, 2pi)
    adjust_lst_bin_edges(lst_bin_edges)

    if not np.all(np.diff(lst_bin_edges) > 0):
        raise ValueError("lst_bin_edges must be monotonically increasing.")

    # Now ensure that all the observed LSTs are wrapped so they start above the first bin edges
    grid_indices, data_lsts, lst_mask = get_lst_bins(data_lsts, lst_bin_edges)
    lst_bin_centres = (lst_bin_edges[1:] + lst_bin_edges[:-1]) / 2

    logger.info(f"Data Shape: {data.shape}")

    # Now, a the data to the lst bin centres.
    if rephase:
        logger.info("Rephasing data")

        # lst_mask is a boolean mask that masks out LSTs that are not in any bin
        # we don't want to spend time rephasing data outside our LST range completely,
        # so we just mask them out here. Indexing by a boolean mask makes a *copy*
        # of the data, so we can rephase in-place without worrying about overwriting
        # the original input data.
        data = data[lst_mask]
        flags = flags[lst_mask]
        nsamples = nsamples[lst_mask]
        data_lsts = data_lsts[lst_mask]
        grid_indices = grid_indices[lst_mask]

        if freq_array is None or antpos is None:
            raise ValueError("freq_array and antpos is needed for rephase")

        bls = np.array([antpos[k[0]] - antpos[k[1]] for k in antpairs])

        # get appropriate lst_shift for each integration, then rephase
        lst_shift = lst_bin_centres[grid_indices] - data_lsts

        # this makes a copy of the data in d
        utils.lst_rephase(data, bls, freq_array, lst_shift, lat=lat, inplace=True)

    # In case we don't rephase, the data/flags/nsamples arrays are still the original
    # input arrays. We don't mask out the data outside the LST range, because we're
    # just going to omit it from our bins naturally anyway. We also don't care if its
    # not a copy here, because we're not going to modify it, and this saves memory.

    # We anyway end up with a ~full copy of the data in the output arrays, because
    # we do a fancy-index of the input arrays to get the relevant data for each bin.

    # TODO: we should think a little more carefully about how we might reduce the
    #       number of copies made in this function. When rephasing, we essentially
    #       get three full copies while inside the function (though one is only local
    #       in scope, and is therefore removed when the function returns).

    # shortcut -- just return all the data, re-organized.
    _data, _flags, _nsamples, _where_inpainted = [], [], [], []
    empty_shape = (0, len(antpairs), len(freq_array), npols)
    for lstbin in range(len(lst_bin_centres)):
        mask = grid_indices == lstbin
        if np.any(mask):
            _data.append(data[mask])
            _flags.append(flags[mask])
            _nsamples.append(nsamples[mask])
            if where_inpainted is not None:
                _where_inpainted.append(where_inpainted[mask])
        else:
            _data.append(np.zeros(empty_shape, complex))
            _flags.append(np.zeros(empty_shape, bool))
            _nsamples.append(np.zeros(empty_shape, int))
            if where_inpainted is not None:
                _where_inpainted.append(np.zeros(empty_shape, bool))

    return lst_bin_centres, _data, _flags, _nsamples, _where_inpainted or None


def get_lst_bins(
    lsts: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the LST bin indices for a set of LSTs.

    Parameters
    ----------
    lsts
        The LSTs to bin, in radians.
    edges
        The edges of the LST bins, in radians.

    Returns
    -------
    bins
        The bin indices for each LST.
    lsts
        The LSTs, wrapped so that the minimum is at the lowest edge, and all are within
        2pi of that minimum.
    mask
        A boolean mask indicating which LSTs are within the range of the LST bins.
    """
    lsts = np.array(lsts).copy()

    # Now ensure that all the observed LSTs are wrapped so they start above the first bin edges
    lsts %= 2 * np.pi
    lsts[lsts < edges[0]] += 2 * np.pi
    bins = np.digitize(lsts, edges, right=True) - 1
    mask = (bins >= 0) & (bins < (len(edges) - 1))
    return bins, lsts, mask


def reduce_lst_bins(
    data: list[np.ndarray],
    flags: list[np.ndarray],
    nsamples: list[np.ndarray],
    where_inpainted: list[np.ndarray] | None = None,
    inpainted_mode: bool = False,
    mutable: bool = False,
    sigma_clip_thresh: float | None = None,
    sigma_clip_min_N: int = 4,
    flag_below_min_N: bool = False,
    flag_thresh: float = 0.7,
    get_mad: bool = False,
) -> dict[str, np.ndarray]:
    """
    From a list of LST-binned data, produce reduced statistics.

    Use this function to reduce a list of `nlst_bins` arrays, each with multiple time
    integrations in them (i.e. the output of :func:`lst_align`) to arrays of shape
    ``(nbl, nlst_bins, nfreq, npol)``, each representing different statistics of the
    data in each LST bin (eg. mean, std, etc.).

    Parameters
    ----------
    data
        The data to perform the reduction over. The length of the list is the number
        of LST bins. Each array in the list should have shape
        ``(nbl, nintegrations_in_lst, nfreq, npol)``.
    flags
        A list, the same length/shape as ``data``, containing the flags.
    nsamples
        A list, the same length/shape as ``data``, containing the number of samples
        for each measurement.
    where_inpainted
        A list, the same length/shape as ``data``, containing a boolean mask indicating
        which samples have been inpainted.
    inpainted_mode
        Whether to use the inpainted samples when calculating the statistics. If False,
        the inpainted samples are ignored. If True, the inpainted samples are used, and
        the statistics are calculated over the un-inpainted samples.
    mutable
        Whether the input data (and flags and nsamples) can be modified in place within
        the algorithm. Setting to true saves memory, and is safe for a one-shot script.
    sigma_clip_thresh
        The number of standard deviations to use as a threshold for sigma clipping.
        If None (default), no sigma clipping is performed. Note that sigma-clipping is performed
        per baseline, frequency, and polarization.
    sigma_clip_min_N
        The minimum number of unflagged samples required to perform sigma clipping.
    flag_below_min_N
        Whether to flag data that has fewer than ``sigma_clip_min_N`` unflagged samples.
    flag_thresh
        The fraction of integrations for a particular (antpair, pol, channel) combination
        within an LST-bin that can be flagged before that combination is flagged
        in the LST-average.
    get_mad
        Whether to compute the median and median absolute deviation of the data in each
        LST bin, in addition to the mean and standard deviation.

    Returns
    -------
    dict
        The reduced data in a dictionary. Keys are 'data' (the lst-binned mean),
        'nsamples', 'flags', 'days_binned' (the number of days that went into each bin),
        'std' (standard deviation) and *otionally* 'median' and 'mad' (if `get_mad` is
        True). All values are arrays of the same shape: ``(nbl, nlst_bins, nfreq, npol)``.
    """
    nlst_bins = len(data)
    (_, nbl, nfreq, npol) = data[0].shape

    for d, f, n in zip(data, flags, nsamples):
        assert d.shape == f.shape == n.shape

    # Do this just so that we can save memory if the call to this function already
    # has allocated memory.

    out_data = np.zeros((nbl, nlst_bins, nfreq, npol), dtype=complex)
    out_flags = np.zeros(out_data.shape, dtype=bool)
    out_std = np.ones(out_data.shape, dtype=complex)
    out_nsamples = np.zeros(out_data.shape, dtype=float)
    days_binned = np.zeros(out_nsamples.shape, dtype=int)

    if get_mad:
        mad = np.ones(out_data.shape, dtype=complex)
        med = np.ones(out_data.shape, dtype=complex)

    if where_inpainted is None:
        where_inpainted = [None] * nlst_bins

    for lstbin, (d, n, f, inpf) in enumerate(
        zip(data, nsamples, flags, where_inpainted)
    ):
        logger.info(f"Computing LST bin {lstbin+1} / {nlst_bins}")

        # TODO: check that this doesn't make yet another copy...
        # This is just the data in this particular lst-bin.

        if d.size:
            d, f = get_masked_data(
                d, n, f, inpainted=inpf, inpainted_mode=inpainted_mode
            )
            f = threshold_flags(f, inplace=True, flag_thresh=flag_thresh)
            d.mask |= f

            (
                out_data[:, lstbin],
                out_flags[:, lstbin],
                out_std[:, lstbin],
                out_nsamples[:, lstbin],
                days_binned[:, lstbin],
            ) = lst_average(
                d,
                n,
                f,
                inpainted_mode=inpainted_mode,
                sigma_clip_thresh=sigma_clip_thresh,
                sigma_clip_min_N=sigma_clip_min_N,
                flag_below_min_N=flag_below_min_N,
            )

            if get_mad:
                med[:, lstbin], mad[:, lstbin] = get_lst_median_and_mad(d)
        else:
            out_data[:, lstbin] *= np.nan
            out_flags[:, lstbin] = True
            out_std[:, lstbin] *= np.inf
            out_nsamples[:, lstbin] = 0.0

            if get_mad:
                mad[:, lstbin] *= np.inf
                med[:, lstbin] *= np.nan

    out = {
        "data": out_data,
        "flags": out_flags,
        "std": out_std,
        "nsamples": out_nsamples,
        "days_binned": days_binned,
    }
    if get_mad:
        out["mad"] = mad
        out["median"] = med

    return out


def _allocate_dfn(shape: tuple[int], d=0.0, f=0, n=0):
    data = np.full(shape, d, dtype=complex)
    flags = np.full(shape, f, dtype=bool)
    nsamples = np.full(shape, n, dtype=float)
    return data, flags, nsamples


def get_masked_data(
    data: np.ndarray,
    nsamples: np.ndarray,
    flags: np.ndarray,
    inpainted: np.ndarray | None = None,
    inpainted_mode: bool = False,
) -> np.ma.MaskedArray:
    if not inpainted_mode:
        # Act like nothing is inpainted.
        inpainted = np.zeros(flags.shape, dtype=bool)
    elif inpainted is None:
        # Flag if a whole blt is flagged:
        allf = np.all(flags, axis=2)[:, :, None, :]

        # Assume everything else that's flagged is inpainted.
        inpainted = flags.copy() * (~allf)

    flags = flags | np.isnan(data) | np.isinf(data) | (nsamples == 0)
    data = np.ma.masked_array(data, mask=(flags & ~inpainted))
    return data, flags


def get_lst_median_and_mad(
    data: np.ndarray | np.ma.MaskedArray,
):
    """Compute the median absolute deviation of a set of data over its zeroth axis.

    Flagged data will be ignored in flagged mode, but included in inpainted mode.
    Nsamples is not taken into account at all, unless Nsamples=0.
    """
    fnc = np.ma.median if isinstance(data, np.ma.MaskedArray) else np.median
    med = fnc(data, axis=0)
    madrl = fnc(np.abs(data.real - med.real), axis=0) * 1.482579
    madim = fnc(np.abs(data.imag - med.imag), axis=0) * 1.482579
    return med, madrl + madim * 1j


def threshold_flags(
    flags: np.ndarray,
    inplace: bool = False,
    flag_thresh: float = 0.7,
):
    if not inplace:
        flags = flags.copy()

    # Flag entire LST bins if there are too many flags over time
    flag_frac = np.sum(flags, axis=0) / flags.shape[0]
    nflags = np.sum(flags)
    logger.info(
        f"Percent of data flagged before thresholding: {100*nflags/flags.size:.2f}%"
    )
    flags |= flag_frac > flag_thresh
    logger.info(
        f"Flagged a further {100*(np.sum(flags) - nflags)/flags.size:.2f}% of visibilities due to flag_frac > {flag_thresh}"
    )

    return flags


@profile
def lst_average(
    data: np.ndarray | np.ma.MaskedArray,
    nsamples: np.ndarray,
    flags: np.ndarray,
    inpainted_mode: bool = False,
    sigma_clip_thresh: float | None = None,
    sigma_clip_min_N: int = 4,
    flag_below_min_N: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute statistics of a set of data over its zeroth axis.

    The idea here is that the data's zeroth axis is "nights", and that each night is
    at the same LST. However, this function is agnostic to the meaning of the first
    axis. It just computes the mean, std, and nsamples over the first axis.

    This function is meant to be used on a single element of a list returned by
    :func:`simple_lst_bin`.

    Parameters
    ----------
    data
        The data to compute the statistics over. Shape ``(nnights, nbl, nfreq, npol)``.
    nsamples
        The number of samples for each measurement. Same shape as ``data``.
    flags
        The flags for each measurement. Same shape as ``data``.
    flag_thresh
        The fraction of integrations for a particular (antpair, pol, channel) combination
        within an LST-bin that can be flagged before that combination is flagged
        in the LST-average.
    sigma_clip_thresh
        The number of standard deviations to use as a threshold for sigma clipping.
        If None (default), no sigma clipping is performed. Note that sigma-clipping is performed
        per baseline, frequency, and polarization.
    sigma_clip_min_N
        The minimum number of unflagged samples required to perform sigma clipping.
    flag_below_min_N
        Whether to flag data that has fewer than ``sigma_clip_min_N`` unflagged samples.

    Returns
    -------
    out_data, out_flags, out_std, out_nsamples
        The reduced data, flags, standard deviation (across nights) and nsamples.
        Shape ``(nbl, nfreq, npol)``.
    """
    # data has shape (nnights, nbl, npols, nfreqs)
    # all data is assumed to be in the same LST bin.

    if not isinstance(data, np.ma.MaskedArray):
        # Generally, we want a MaskedArray for 'data', where the mask is *either*
        # the flags or the 'non-inpainted flags', as obtained by `threshold_flags`.
        # However, if this hasn't been called, and we just have an array, apply flags
        # appropriately here.
        data, flags = get_masked_data(
            data, nsamples, flags, inpainted_mode=inpainted_mode
        )

    # Now do sigma-clipping.
    if sigma_clip_thresh is not None:
        if inpainted_mode:
            warnings.warn(
                "Sigma-clipping in in-painted mode is a bad idea, because it creates "
                "non-uniform flags over frequency, which can cause artificial spectral "
                "structure. In-painted mode specifically attempts to avoid this."
            )

        nflags = np.sum(flags)
        clip_flags = sigma_clip(
            data.real, sigma=sigma_clip_thresh, min_N=sigma_clip_min_N
        )
        clip_flags |= sigma_clip(
            data.imag, sigma=sigma_clip_thresh, min_N=sigma_clip_min_N
        )

        # Need to restore min_N condition properly here because it's not done properly in sigma_clip
        sc_min_N = np.sum(~flags, axis=0) < sigma_clip_min_N
        clip_flags[:, sc_min_N] = False

        flags |= clip_flags

        data.mask |= clip_flags

        logger.info(
            f"Flagged a further {100*(np.sum(flags) - nflags)/flags.size:.2f}% of visibilities due to sigma clipping"
        )

    # Here we do a check to make sure Nsamples is uniform across frequency.
    # Do this before setting non_inpainted to zero nsamples.
    ndiff = np.diff(nsamples, axis=2)
    if np.any(ndiff != 0):
        warnings.warn(
            "Nsamples is not uniform across frequency. This will result in spectral structure."
        )

    nsamples = np.ma.masked_array(nsamples, mask=data.mask)

    norm = np.sum(nsamples, axis=0)
    ndays_binned = np.sum((~flags).astype(int), axis=0)

    logger.info("Calculating mean")
    # np.sum works the same for both masked and non-masked arrays.
    meandata = np.sum(data * nsamples, axis=0)

    lstbin_flagged = np.all(flags, axis=0)
    if flag_below_min_N:
        lstbin_flagged[ndays_binned < sigma_clip_min_N] = True

    normalizable = norm > 0
    meandata[normalizable] /= norm[normalizable]
    # Multiply by nan instead of just setting as nan, so both real and imag parts are nan
    meandata[~normalizable] *= np.nan

    # get other stats
    logger.info("Calculating std")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")
        std = np.square(data.real - meandata.real) + 1j * np.square(
            data.imag - meandata.imag
        )
        std = np.sum(std * nsamples, axis=0)
        std[normalizable] /= norm[normalizable]
        std = np.sqrt(std.real) + 1j * np.sqrt(std.imag)

    std[~normalizable] = np.inf

    # While the previous norm is correct for normalizing the mean, we now
    # calculate nsamples as the unflagged samples in each LST bin.
    nsamples.mask = flags
    nsamples = np.sum(nsamples, axis=0)

    return meandata.data, lstbin_flagged, std.data, nsamples.data, ndays_binned


def adjust_lst_bin_edges(lst_bin_edges: np.ndarray) -> np.ndarray:
    """
    Adjust the LST bin edges so that they start in the range [0, 2pi) and increase.

    Performs the adjustment in-place.
    """
    while lst_bin_edges[0] < 0:
        lst_bin_edges += 2 * np.pi
    while lst_bin_edges[0] >= 2 * np.pi:
        lst_bin_edges -= 2 * np.pi


def lst_bin_files_for_baselines(
    data_files: list[Path | FastUVH5Meta],
    lst_bin_edges: np.ndarray,
    antpairs: Sequence[tuple[int, int]],
    freqs: np.ndarray | None = None,
    pols: np.ndarray | None = None,
    cal_files: list[Path | None] | None = None,
    time_arrays: list[np.ndarray] | None = None,
    time_idx: list[np.ndarray] | None = None,
    ignore_flags: bool = False,
    rephase: bool = True,
    antpos: dict[int, np.ndarray] | None = None,
    lsts: np.ndarray | None = None,
    redundantly_averaged: bool = False,
    reds: RedundantGroups | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
    where_inpainted_files: list[list[str | Path | None]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Produce a set of LST-binned (but not averaged) data for a set of baselines.

    This function takes a set of input data files, and reads any data in them that
    falls within the LST bins specified by ``lst_bin_edges`` (optionally calibrating
    the data as it is read). The data is sorted into the LST-bins provided and returned
    as a list of arrays, one for each LST bin. The data is not averaged within LST bins.

    Only the list of baselines given will be read from each file, which makes it
    possible to iterate over baseline chunks and call this function on each chunk,
    to reduce maximum memory usage.

    The data is binned via a simple histogram, i.e. the data represented at each LST
    is essentially assumed to be a delta function in LST, and is fully assigned to one
    particular LST bin. See :func:`lst_align` for details.

    Parameters
    ----------
    data_files
        A list of paths to data files to read. Instead of paths, you can also pass
        FastUVH5Meta objects, which will be used to read the data.
    lst_bin_edges
        A sequence of floats specifying the *edges* of the LST bins to use, with length
        ``N_lstbins + 1``. Bins are thus assumed to be contiguous, but not necessarily
        of equal size.
    antpairs
        A list of antenna pairs to read from each file. Each pair should be a tuple
        of antenna numbers. Note that having pairs in this list that are not present
        in a particular file will not cause an error -- that file will simply not
        contribute for that antpair.
    freqs
        Frequencies contained in the files. If not provided, will be read from the
        first file in ``data_files``.
    pols
        Polarizations to read. If not provided, will be read from the first file in
        ``data_files``.
    cal_files
        A list of paths to calibration files to apply to the data. If not provided,
        no calibration will be applied. If provided, must be the same length as
        ``data_files``. If a particular element is None, no calibration will be
        applied to that file.
    time_arrays
        A list of time arrays for each file. If not provided, will be read from the
        files. If provided, must be the same length as ``data_files``.
    time_idx
        A list of arrays, one for each file, where the array is the same length as
        the time array for that file, and is boolean, indicating whether each time
        is required to be read (i.e. if it appears in any LST bin). If not provided,
        will be calculated from the LST bin edges and the time arrays.
    ignore_flags
        If True, ignore flags in the data files and bin all data.
    rephase
        If True, rephase the data in each LST bin to the LST bin center.
    antpos
        A dictionary mapping antenna numbers to antenna positions. Only required
        if ``rephase`` is True. If not provided, and required, will be determined
        by reading as many of the files as required to obtain all antenna positions
        in antpairs.
    lsts
        A list of LST arrays for each file. If not provided, will be read from the
        files. If provided, must be the same length as ``data_files``.
    freq_min, freq_max
        Minimum and maximum frequencies to include in the data. If not provided,
        all frequencies will be included.
    where_inpainted_files
        A list of lists of strings, one for each file, where each file is a UVFlag file
        specifying which data are in-painted. If not provided, no inpainting will be
        assumed.

    Returns
    -------
    bin_lst
        The bin centres for each of the LST bins.
    data
        A nlst-length list of arrays, each of shape
        ``(ntimes_in_lst, nbls, nfreq, npol)``, where LST bins without data simply have
        a first-axis of size zero.
    flags
        Same as ``data``, but boolean flags.
    nsamples
        Same as ``data``, but sample counts.
    where_inpainted
        Same as ``data``, but boolean flags indicating where inpainting has been done.
    times_in_bins
        The JDs that are in each LST bin -- a list of arrays.
    """
    metas = [
        fl
        if isinstance(fl, FastUVH5Meta)
        else FastUVH5Meta(fl, blts_are_rectangular=True)
        for fl in data_files
    ]

    lst_bin_edges = np.array(lst_bin_edges)

    if freqs is None:
        freqs = np.squeeze(metas[0].freq_array)

    if freq_min is None and freq_max is None:
        freq_chans = None
    else:
        freq_chans = np.arange(len(freqs))

    if freq_min is not None:
        mask = freqs >= freq_min
        freqs = freqs[mask]
        freq_chans = freq_chans[mask]
    if freq_max is not None:
        mask = freqs <= freq_max
        freqs = freqs[mask]
        freq_chans = freq_chans[mask]

    if pols is None:
        pols = metas[0].pols
    else:
        if not all(isinstance(p, str) for p in pols):
            pols = uvutils.polnum2str(pols, x_orientation=metas[0].x_orientation)

    if antpos is None and rephase:
        warnings.warn(
            "Getting antpos from the first file only. This is almost always correct, "
            "but will be wrong if different files have different antenna_position arrays."
        )
        antpos = dict(zip(metas[0].antenna_numbers, metas[0].antpos_enu))

    if time_idx is None:
        adjust_lst_bin_edges(lst_bin_edges)
        lst_bin_edges %= 2 * np.pi
        op = np.logical_and if lst_bin_edges[0] < lst_bin_edges[-1] else np.logical_or
        time_idx = []
        for meta in metas:
            _lsts = meta.get_transactional("lsts")
            time_idx.append(op(_lsts >= lst_bin_edges[0], _lsts < lst_bin_edges[-1]))

    if time_arrays is None:
        time_arrays = [
            meta.get_transactional("times")[idx] for meta, idx in zip(metas, time_idx)
        ]

    if lsts is None:
        lsts = np.concatenate(
            [meta.get_transactional("lsts")[idx] for meta, idx in zip(metas, time_idx)]
        )

    # Now we can set up our master arrays of data.
    data, flags, nsamples = _allocate_dfn(
        (len(lsts), len(antpairs), len(freqs), len(pols)),
        d=np.nan + np.nan * 1j,
        f=True,
    )

    if where_inpainted_files is None or all(w is None for w in where_inpainted_files):
        where_inpainted_files = [None] * len(metas)
        where_inpainted = None
    else:
        where_inpainted = np.zeros_like(flags)

    if cal_files is None:
        cal_files = [None] * len(metas)

    if redundantly_averaged and reds is None:
        raise ValueError("reds must be provided if redundantly_averaged is True")
    if redundantly_averaged and any(c is not None for c in cal_files):
        raise ValueError("Cannot apply calibration if redundantly_averaged is True")

    # This loop actually reads the associated data in this LST bin.
    ntimes_so_far = 0
    for meta, calfl, tind, tarr, inpfile in zip(
        metas, cal_files, time_idx, time_arrays, where_inpainted_files
    ):
        logger.info(f"Reading {meta.path}")
        slc = slice(ntimes_so_far, ntimes_so_far + len(tarr))
        ntimes_so_far += len(tarr)

        # hd = io.HERAData(str(fl.path), filetype='uvh5')
        data_antpairs = meta.get_transactional("antpairs")

        if redundantly_averaged:
            bls_to_load = [
                bl
                for bl in data_antpairs
                if reds.get_ubl_key(bl) in antpairs
                or reds.get_ubl_key(bl[::-1]) in antpairs
            ]
        else:
            bls_to_load = [
                bl
                for bl in antpairs
                if bl in data_antpairs or bl[::-1] in data_antpairs
            ]

        if not bls_to_load or not np.any(tind):
            # If none of the requested baselines are in this file, then just
            # set stuff as nan and go to next file.
            logger.warning(f"None of the baseline-times are in {meta.path}. Skipping.")
            data[slc] = np.nan
            flags[slc] = True
            nsamples[slc] = 0
            continue

        # TODO: use Fast readers here instead.
        _data, _flags, _nsamples = io.HERAData(meta.path).read(
            bls=bls_to_load,
            times=tarr,
            freq_chans=freq_chans,
            polarizations=pols,
        )
        if inpfile is not None:
            # This returns a DataContainer (unless something went wrong) since it should
            # always be a 'baseline' type of UVFlag.
            inpainted = io.load_flags(inpfile)
            if not isinstance(inpainted, DataContainer):
                raise ValueError(f"Expected {inpfile} to be a DataContainer")

            # We need to down-selecton times/freqs (bls and pols will be sub-selected
            # based on those in the data through the next loop)
            inpainted.select_or_expand_times(new_times=tarr, skip_bda_check=True)
            inpainted.select_freqs(channels=freq_chans)
        else:
            inpainted = None

        if redundantly_averaged:
            keyed = reds.keyed_on_bls(_data.antpairs())

        # load calibration
        if calfl is not None:
            logger.info(f"Opening and applying {calfl}")
            uvc = io.to_HERACal(calfl)
            gains, cal_flags, _, _ = uvc.read(freq_chans=freq_chans)
            # down select times if necessary
            if False in tind and uvc.Ntimes > 1:
                # If uvc has Ntimes == 1, then broadcast across time will work automatically
                uvc.select(times=uvc.time_array[tind])
                gains, cal_flags, _, _ = uvc.build_calcontainers()

            apply_cal.calibrate_in_place(
                _data,
                gains,
                data_flags=_flags,
                cal_flags=cal_flags,
                gain_convention=uvc.gain_convention,
            )

        for i, bl in enumerate(antpairs):
            if redundantly_averaged:
                bl = keyed.get_ubl_key(bl)

            for j, pol in enumerate(pols):
                blpol = bl + (pol,)

                if blpol in _data:  # DataContainer takes care of conjugates.
                    data[slc, i, :, j] = _data[blpol]
                    flags[slc, i, :, j] = _flags[blpol]
                    nsamples[slc, i, :, j] = _nsamples[blpol]

                    if inpainted is not None:
                        # Get the representative baseline key from this bl group that
                        # exists in the where_inpainted data.
                        if redundantly_averaged:
                            for inpbl in reds[bl]:
                                if inpbl + (pol,) in inpainted:
                                    blpol = inpbl + (pol,)
                                    break
                            else:
                                raise ValueError(
                                    f"Could not find any baseline from group {bl} in "
                                    "inpainted file"
                                )

                        where_inpainted[slc, i, :, j] = inpainted[blpol]
                else:
                    # This baseline+pol doesn't exist in this file. That's
                    # OK, we don't assume all baselines are in every file.
                    data[slc, i, :, j] = np.nan
                    flags[slc, i, :, j] = True
                    nsamples[slc, i, :, j] = 0

    logger.info("About to run LST binning...")
    # LST bin edges are the actual edges of the bins, so should have length
    # +1 of the LST centres. We use +dlst instead of +dlst/2 on the top edge
    # so that np.arange definitely gets the last edge.
    bin_lst, data, flags, nsamples, where_inpainted = lst_align(
        data=data,
        flags=None if ignore_flags else flags,
        nsamples=nsamples,
        data_lsts=lsts,
        where_inpainted=where_inpainted,
        antpairs=antpairs,
        lst_bin_edges=lst_bin_edges,
        freq_array=freqs,
        rephase=rephase,
        antpos=antpos,
    )

    bins = get_lst_bins(lsts, lst_bin_edges)[0]
    times = np.concatenate(time_arrays)
    times_in_bins = []
    for i in range(len(bin_lst)):
        mask = bins == i
        times_in_bins.append(times[mask])

    return bin_lst, data, flags, nsamples, where_inpainted, times_in_bins


def apply_calfile_rules(
    data_files: list[list[str]],
    calfile_rules: list[tuple[str, str]],
    ignore_missing: bool,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Apply a set of rules to convert data file names to calibration file names.

    Parameters
    ----------
    data_files : list of list of str
        List of lists of data file names. Each inner list is a night of data.
    calfile_rules : list of tuple of str
        List of rules to apply. Each rule is a tuple of two strings, the first
        is the string to replace, the second is the string to replace it with.
        Each 2-tuple constitutes one "rule" and they are applied in order to each file.
        All rules are applied to all files.
    ignore_missing : bool
        If True, ignore missing calibration files. If False, raise an error.

    Returns
    -------
    data_files : list of list of str
        List of lists of data file names. Each inner list is a night of data.
        Files that were removed due to missing calibration files are removed.
    input_cals : list of list of str
        List of lists of calibration file names. Each inner list is a night of data.
        Any calibration files that were missing are not included.
    """
    input_cals = []
    for night, dflist in enumerate(data_files):
        this = []
        input_cals.append(this)
        missing = []
        for df in dflist:
            cf = df
            for rule in calfile_rules:
                cf = cf.replace(rule[0], rule[1])

            if os.path.exists(cf):
                this.append(cf)
            elif ignore_missing:
                warnings.warn(f"Calibration file {cf} does not exist")
                missing.append(df)
            else:
                raise IOError(f"Calibration file {cf} does not exist")
        data_files[night] = [df for df in dflist if df not in missing]
    return data_files, input_cals


def filter_required_files_by_times(
    lst_range: tuple[float, float],
    data_metas: list[list[FastUVH5Meta]],
    cal_files: list[list[str]] | None = None,
    where_inpainted_files: list[list[str]] | None = None,
) -> tuple[list, list, list, list, list, list]:
    lstmin, lstmax = lst_range
    lstmin %= 2 * np.pi
    lstmax %= 2 * np.pi
    if lstmin > lstmax:
        lstmax += 2 * np.pi

    if not cal_files:
        cal_files = [[None for _ in dm] for dm in data_metas]

    if not where_inpainted_files:
        where_inpainted_files = [[None for _ in dm] for dm in data_metas]

    tinds = []
    all_lsts = []
    file_list = []
    time_arrays = []
    cals = []
    where_inpainted = []

    # This loop just gets the number of times that we'll be reading.
    # Even though the input files should already have been configured to be those
    # that fall within the output LST range, we still need to read them in to
    # check exactly which time integrations we require.

    for night, callist, inplist in zip(data_metas, cal_files, where_inpainted_files):
        for meta, cal, inp in zip(night, callist, inplist):
            tarr = meta.times
            lsts = meta.lsts % (2 * np.pi)
            lsts[lsts < lstmin] += 2 * np.pi

            tind = (lsts > lstmin) & (lsts < lstmax)

            if np.any(tind):
                tinds.append(tind)
                time_arrays.append(tarr[tind])
                all_lsts.append(lsts[tind])
                file_list.append(meta)
                cals.append(cal)
                where_inpainted.append(inp)

    return tinds, time_arrays, all_lsts, file_list, cals, where_inpainted


def _get_where_inpainted_files(
    data_files: list[list[str | Path]],
    where_inpainted_file_rules: list[tuple[str, str]] | None,
) -> list[list[str | Path]] | None:
    if where_inpainted_file_rules is None:
        return None

    where_inpainted_files = []
    for dflist in data_files:
        this = []
        where_inpainted_files.append(this)
        for df in dflist:
            wif = str(df)
            for rule in where_inpainted_file_rules:
                wif = wif.replace(rule[0], rule[1])
            if os.path.exists(wif):
                this.append(wif)
            else:
                raise IOError(f"Where inpainted file {wif} does not exist")

    return where_inpainted_files


def _configure_inpainted_mode(output_flagged, output_inpainted, where_inpainted_files):
    # Sort out defaults for inpaint/flagging mode
    if output_inpainted is None:
        output_inpainted = bool(where_inpainted_files)

    if not output_inpainted and not output_flagged:
        raise ValueError(
            "Both output_inpainted and output_flagged are False. One must be True."
        )

    return output_flagged, output_inpainted


def lst_bin_files_single_outfile(
    config_opts: dict[str, Any],
    metadata: dict[str, Any],
    lst_bins: np.ndarray,
    data_files: list[list[str]],
    calfile_rules: list[tuple[str, str]] | None = None,
    ignore_missing_calfiles: bool = False,
    outdir: str | Path | None = None,
    reds: RedundantGroups | None = None,
    redundantly_averaged: bool | None = None,
    only_last_file_per_night: bool = False,
    history: str = "",
    fname_format: str = "zen.{kind}.{lst:7.5f}.uvh5",
    overwrite: bool = False,
    rephase: bool = False,
    Nbls_to_load: int | None = None,
    ignore_flags: bool = False,
    include_autos: bool = True,
    ex_ant_yaml_files=None,
    ignore_ants: tuple[int] = (),
    write_kwargs: dict | None = None,
    save_channels: list[int] = (),
    golden_lsts: tuple[float] = (),
    sigma_clip_thresh: float | None = None,
    sigma_clip_min_N: int = 4,
    flag_below_min_N: bool = False,
    flag_thresh: float = 0.7,
    freq_min: float | None = None,
    freq_max: float | None = None,
    output_inpainted: bool | None = None,
    output_flagged: bool = True,
    where_inpainted_file_rules: list[tuple[str, str]] | None = None,
    sigma_clip_in_inpainted_mode: bool = False,
    write_med_mad: bool = False,
) -> dict[str, Path]:
    """
    Bin data files into LST bins, and write all bins to disk in a single file.

    Note that this is generally not meant to be called directly, but rather through
    the :func:`lst_bin_files` function.

    The mode(s) in which the function does the averaging can be specified via the
    `output_inpainted`, `output_flagged` and `where_inpainted_file_rules` options.
    Algorithmically, there are two modes of averaging: either flagged data is *ignored*
    in the average (and in Nsamples) or the flagged data is included in the average but
    ignored in Nsamples. These are called "flagged" and "inpainted" modes respectively.
    The latter only makes sense if the data in the flagged regions has been inpainted,
    rather than left as raw data. For delay-filtered data, either mode is equivalent,
    since the flagged data itself is set to zero. By default, this function *only*
    uses the "flagged" mode, *unless* the `where_inpainted_file_rules` option is set,
    which indicates that the files are definitely in-painted, and in this case it will
    use *both* modes by default.

    .. note:: Both "flagged" and "inpainted" mode as implemented in this function are
        *not* spectrally smooth if the input Nsamples are not spectrally uniform.

    Parameters
    ----------
    config_opts
        A dictionary of LST-bin configuration options. Exactly the "config_params"
        section of the configuration file produced by :func:`make_lst_bin_config`.
    metadata
        A dictionary of metadata for the LST binning. Exactly the "metadata" section
        of the configuration file produced by :func:`make_lst_bin_config`.
    lst_bins
        An array of LST bin *centres* in radians. These should be *one* of the entries
        of the "lst_bins" section of the configuration file produced by
        :func:`make_lst_bin_config` (which is a list of arrays of LST bin centres, one
        for each output file).
    data_files
        A list of lists of data files to LST bin. Each list of files is treated as coming
        from a single night. These should be *one* of the entries of the "matched_files"
        section of the configuration file produced by :func:`make_lst_bin_config` (which
        is a list of lists of lists of data files, one for each output file).
    calfile_rules
        A list of tuples of strings. Each tuple is a pair of strings that are used to
        replace the first string with the second string in the data file name to get
        the calibration file name. For example, providing [(".uvh5", ".calfits")] will
        generate a list of calfiles that have the same basename as the data files, but
        with the extension ".calfits" instead of ".uvh5". Multiple entries to the list
        are allowed, and the replacements are applied in order. If the resulting calfile
        name does not exist, the data file is ignored.
    ignore_missing_calfiles
        If True, ignore missing calibration files (i.e. just drop the corresponding
        data file from the binning). If False, raise an error if a calfile is missing.
    outdir
        The output directory. If not provided, this is set to the lowest-level common
        directory for all data files.
    reds
        A :class:`RedundantGroups` object describing the redundant groups of the array.
        If not provided, this is calculated from the first data file on the first night.
    redundantly_averaged
        If True, the data are assumed to have been redundantly averaged. If not provided
        this is set to True if the first data file on the first night has been redundantly
        averaged, and False otherwise.
    only_last_file_per_night
        If True, only the last file from each night is used to infer the observed
        antpairs. Setting to False can be very slow for large data sets, and is almost
        never necessary, as the antpairs observed are generally set per-night.
    history
        Custom history string to insert into the output file.
    fname_format
        A formatting string to use to write the output file. This can have the following
        fields: "kind" (which will evaluate to one of 'LST', 'STD', 'GOLDEN' or 'REDUCEDCHAN'),
        "lst" (which will evaluate to the LST of the bin), and "pol" (which will evaluate
        to the polarization of the data). Example: "zen.{kind}.{lst:7.5f}.uvh5"
    overwrite
        If True, overwrite output files.
    rephase
        If True, rephase data points in LST bin to center of bin.
    Nbls_to_load
        The number of baselines to load at a time. If None, load all baselines at once.
    ignore_flags
        If True, ignore flags when binning data.
    include_autos
        If True, include autocorrelations when binning data.
    ex_ant_yaml_files
        A list of yaml files that specify which antennas to exclude from each
        input data file.
    ignore_ants
        A list of antennas to ignore when binning data.
    write_kwargs
        Arguments to pass to :func:`create_lstbin_output_file`.
    save_channels
        A list of channels for which to save the a full file of LST-gridded data.
        One REDUCEDCHAN file is saved for each output file, corresponding to the
        first LST-bin in that file. The data in that file will have the shape
        ``(Nbls*Ndays, Nsave_chans, Npols)``. This can be helpful for debugging.
    golden_lsts
        A list of LSTs for which to save a full file of LST-aligned (but not
        averaged) data. One GOLDEN file is saved for each ``golden_lst``, with shape
        ``(Nbls*Ndays, Nfreqs, Npols)`` -- that is, the normal "time" axis of a
        UVData array is replaced by a "night" axis. This is an easy way to load up
        the full data that goes into a particular LST-bin after the fact.
    sigma_clip_thresh
        If provided, this is the threshold for sigma clipping. If this is provided,
        then the data is sigma clipped before being averaged. This is done for each
        (antpair, pol, channel) combination.
    sigma_clip_min_N
        The minimum number of integrations for a particular (antpair, pol, channel)
        within an LST-bin required to perform sigma clipping. If `flag_below_min_N`
        is False, these (antpair,pol,channel) combinations are not flagged by
        sigma-clipping (otherwise they are).
    flag_below_min_N
        If True, flag all (antpair, pol,channel) combinations  for an LST-bin that
        contiain fewer than `flag_below_min_N` unflagged integrations within the bin.
    flag_thresh
        The fraction of integrations for a particular (antpair, pol, channel) combination
        within an LST-bin that can be flagged before that combination is flagged
        in the LST-average.
    freq_min
        The minimum frequency to include in the output files. If not provided, this
        is set to the minimum frequency in the first data file on the first night.
    freq_max
        The maximum frequency to include in the output files. If not provided, this
        is set to the maximum frequency in the first data file on the first night.
    output_inpainted
        If True, output data LST-binned in in-painted mode. This mode does *not* flag
        data for the averaging, assuming that data that has flags has been in-painted
        to improve spectral smoothness. It does take the flags into account for the
        LST-binned Nsamples, however.
    output_flagged
        If True, output data LST-binned in flagged mode. This mode *does* apply flags
        to the data before averaging. It will yield the same Nsamples as the in-painted
        mode, but simply ignores flagged data for the average, which can yield less
        spectrally-smooth LST-binned results.
    where_inpainted_file_rules
        Rules to transform the input data file names into the corresponding "where
        inpainted" files (which should be in UVFlag format). If provided, this indicates
        that the data itself is in-painted, and the `output_inpainted` mode will be
        switched on by default. These files should specify which data is in-painted
        in the associated data file (which may be different than the in-situ flags
        of the data object). If not provided, but `output_inpainted` is set to True,
        all data-flags will be considered in-painted except for baseline-times that are
        fully flagged, which will be completely ignored.
    sigma_clip_in_inpainted_mode
        If True, sigma clip the data in inpainted mode (if sigma-clipping is turned on).
        This is generally not a good idea, since the point of inpainting is to get
        smoother spectra, and sigma-clipping creates non-uniform Nsamples, which can
        lead to less smooth spectra. This option is only here to enable sigma-clipping
        to be turned on for flagged mode, and off for inpainted mode.
    write_med_mad
        If True, write out the median and MAD of the data in each LST bin.

    Returns
    -------
    out_files
        A dict of output files, keyed by the type of file (e.g. 'LST', 'STD', 'GOLDEN',
        'REDUCEDCHAN').

    Notes
    -----
    It is worth describing in a bit more detail what is actually _in_ the output files.
    The "LST" files contain the LST-binned data, with the data averaged over each LST
    bin. There are two possible modes for this: inpainted and flagged. In inpainted mode,
    the data in flagged bl-channel-pols is used for the averaging, as it is considered
    to be in-painted. This gives the most spectrally-smooth results. In order to ignore
    a particular bl-channel-pol while still using inpaint mode, supply a "where inpainted"
    file, which should be a UVFlag object that specifies which bl-channel-pols are
    inpainted in the associated data file. Anything that's flagged but not inpainted is
    ignored for the averaging. In this inpainted mode, the Nsamples are the number of
    un-flagged integrations (whether they were in-painted or not). The LST-binned flags
    are only set to True if ALL of the nights for a given bl-channel-pol are flagged
    (again, whether they were in-painted or not). In flagged mode, both the Nsamples
    and Flags are the same as inpainted mode. The averaged data, however, ignores any
    flagged data. This can lead to less spectrally-smooth results. The "STD" files
    contain LST-binned "data" that is the standard deviation of the data in each LST-bin.
    This differs between inpainted and flagged modes in the same way as the "LST" files:
    in inpainted mode, the flagged and inpainted data is used for calculating the sample
    variance, while in flagged mode, only the unflagged data is used. The final flags
    in the "STD" files is equivalent to that in the "LST" files. The Nsamples in the
    "STD" files is actually the number of unflagged nights in the LST bin (so, not the
    sum of Nsamples), where "unflagged" really does mean unflagged -- whether inpainted
    or not.

    One note here about what is considered "flagged" vs. "flagged and inpainted" vs
    "flagged and not inpainted". In flagged mode, there are input flags that exist in the
    input files. These are potentially *augmented* by sigma clipping within the LST
    binner, and also by flagging whole LST bins if they have too few unflagged integrations.
    In inpainted mode, input flags are considered as normal flags. However, only
    "non-inpainted" flags are *ignored* for the averaging. By default, all flagged data
    is considered to to be in-painted UNLESS it is a blt-pol that is fully flagged (i.e.
    all channels are flagged for an integration for a single bl and pol). However, you
    can tell the routine that other data is NOT in-painted by supplying a "where inpainted"
    file. Now, integrations in LST-bins that end up having "too few" unflagged
    integrations will be flagged inside the binner, however in inpainted mode, if these
    are considered "inpainted", they will still be used in averaging (this means they
    will have "valid" data for the average, but their average will be flagged).
    On the other hand, flags that are applied by sigma-clipping will be considered
    NOT inpainted, i.e. those data will be ignored in the averaged, just like flagging
    mode. In this case, either choice is bad: to include them in the average is bad
    because even though they may have been actually in-painted, whatever value they have
    is clearly triggering the sigma-clipper and is therefore an outlier. On the other
    hand, to ignore them is bad because the point of in-painting mode is to get
    smoother spectra, and this negates that. So, it's best just to not do sigma-clipping
    in inpainted mode.
    """
    write_kwargs = write_kwargs or {}

    # Check that that there are the same number of input data files and
    # calibration files each night.
    input_cals = []
    if calfile_rules:
        data_files, input_cals = apply_calfile_rules(
            data_files, calfile_rules, ignore_missing=ignore_missing_calfiles
        )

    where_inpainted_files = _get_where_inpainted_files(
        data_files, where_inpainted_file_rules
    )

    output_flagged, output_inpainted = _configure_inpainted_mode(
        output_flagged, output_inpainted, where_inpainted_files
    )

    # Prune empty nights (some nights start with files, but have files removed because
    # they have no associated calibration)
    data_files = [df for df in data_files if df]
    input_cals = [cf for cf in input_cals if cf]
    if where_inpainted_files is not None:
        where_inpainted_files = [wif for wif in where_inpainted_files if wif]

    logger.info("Got the following numbers of data files per night:")
    for dflist in data_files:
        logger.info(f"{dflist[0].split('/')[-1]}: {len(dflist)}")

    data_metas = [
        [
            FastUVH5Meta(
                df,
                blts_are_rectangular=metadata["blts_are_rectangular"],
                time_axis_faster_than_bls=metadata["time_axis_faster_than_bls"],
            )
            for df in dflist
        ]
        for dflist in data_files
    ]

    # get outdir
    if outdir is None:
        outdir = os.path.dirname(os.path.commonprefix(abscal.flatten(data_files)))

    start_jd = metadata["start_jd"]

    # get metadata
    logger.info("Getting metadata from first file...")
    meta = data_metas[0][0]

    freq_array = np.squeeze(meta.freq_array)

    # reds will contain all of the redundant groups for the whole array, because
    # all the antenna positions are included in every file.
    antpos = dict(zip(meta.antenna_numbers, meta.antpos_enu))
    if reds is None:
        reds = RedundantGroups.from_antpos(antpos=antpos, include_autos=include_autos)

    if redundantly_averaged is None:
        # Try to work out if the files are redundantly averaged.
        # just look at the middle file from each night.
        for fl_list in data_metas:
            meta = fl_list[len(fl_list) // 2]
            antpairs = meta.get_transactional("antpairs")
            ubls = {reds.get_ubl_key(ap) for ap in antpairs}
            if len(ubls) != len(antpairs):
                # At least two of the antpairs are in the same redundant group.
                redundantly_averaged = False
                logger.info("Inferred that files are not redundantly averaged.")
                break
        else:
            redundantly_averaged = True
            logger.info("Inferred that files are redundantly averaged.")

    logger.info("Compiling all unflagged baselines...")
    all_baselines, all_pols = get_all_unflagged_baselines(
        data_metas,
        ex_ant_yaml_files,
        include_autos=include_autos,
        ignore_ants=ignore_ants,
        redundantly_averaged=redundantly_averaged,
        reds=reds,
        only_last_file_per_night=only_last_file_per_night,
    )
    nants0 = meta.header["antenna_numbers"].size

    # Do a quick check to make sure all nights at least have the same number of Nants
    for dflist in data_metas:
        _nants = dflist[0].header["antenna_numbers"].size
        dflist[0].close()
        if _nants != nants0:
            raise ValueError(
                f"Not all nights have the same number of antennas! Got {_nants} for "
                f"{dflist[0].path} and {nants0} for {meta.path} for {meta.path}"
            )

    # Split up the baselines into chunks that will be LST-binned together.
    # This is just to save on RAM.
    if Nbls_to_load is None:
        Nbls_to_load = len(all_baselines) + 1
    n_bl_chunks = len(all_baselines) // Nbls_to_load + 1
    bl_chunks = [
        all_baselines[i * Nbls_to_load: (i + 1) * Nbls_to_load]
        for i in range(n_bl_chunks)
    ]
    bl_chunks = [blg for blg in bl_chunks if len(blg) > 0]

    dlst = config_opts["dlst"]
    lst_bin_edges = np.array(
        [x - dlst / 2 for x in lst_bins] + [lst_bins[-1] + dlst / 2]
    )

    (
        tinds,
        time_arrays,
        all_lsts,
        file_list,
        cals,
        where_inpainted_files,
    ) = filter_required_files_by_times(
        (lst_bin_edges[0], lst_bin_edges[-1]),
        data_metas,
        input_cals,
        where_inpainted_files,
    )

    # If we have no times at all for this file, just return
    if len(all_lsts) == 0:
        return {}

    all_lsts = np.concatenate(all_lsts)

    # The "golden" data is the full data over all days for a small subset of LST
    # bins. This works best if the LST bins are small (similar to the size of the
    # raw integrations). Usually, the length of "bins" will be zero.
    # NOTE: we work under the assumption that the LST bins are small, so that
    # each night only gets one integration in each LST bin. If there are *more*
    # than one integration in the bin, we take the first one only.
    golden_bins, _, mask = get_lst_bins(golden_lsts, lst_bin_edges)
    golden_bins = golden_bins[mask]
    logger.info(
        f"golden_lsts bins in this output file: {golden_bins}, "
        f"lst_bin_edges={lst_bin_edges}"
    )

    # make it a bit easier to create the outfiles
    create_outfile = partial(
        create_lstbin_output_file,
        outdir=outdir,
        pols=all_pols,
        file_list=file_list,
        history=history,
        fname_format=fname_format,
        overwrite=overwrite,
        antpairs=all_baselines,
        start_jd=start_jd,
        freq_min=freq_min,
        freq_max=freq_max,
        lst_branch_cut=metadata["lst_branch_cut"],
        **write_kwargs,
    )
    out_files = {}
    for inpaint_mode in [True, False]:
        if inpaint_mode and not output_inpainted:
            continue
        if not inpaint_mode and not output_flagged:
            continue

        kinds = ["LST", "STD"]
        if write_med_mad:
            kinds += ["MED", "MAD"]
        for kind in kinds:
            # Create the files we'll write to
            out_files[(kind, inpaint_mode)] = create_outfile(
                kind=kind,
                lst=lst_bin_edges[0],
                lsts=lst_bins,
                inpaint_mode=inpaint_mode,
            )

    nbls_so_far = 0
    for bi, bl_chunk in enumerate(bl_chunks):
        logger.info(f"Baseline Chunk {bi+1} / {len(bl_chunks)}")
        # data/flags/nsamples are *lists*, with nlst_bins entries, each being an
        # array, with shape (times, bls, freqs, npols)
        (
            bin_lst,
            data,
            flags,
            nsamples,
            where_inpainted,
            binned_times,
        ) = lst_bin_files_for_baselines(
            data_files=file_list,
            lst_bin_edges=lst_bin_edges,
            antpairs=bl_chunk,
            freqs=freq_array,
            pols=all_pols,
            cal_files=cals,
            time_arrays=time_arrays,
            time_idx=tinds,
            ignore_flags=ignore_flags,
            rephase=rephase,
            antpos=antpos,
            lsts=all_lsts,
            redundantly_averaged=redundantly_averaged,
            reds=reds,
            freq_min=freq_min,
            freq_max=freq_max,
            where_inpainted_files=where_inpainted_files,
        )

        slc = slice(nbls_so_far, nbls_so_far + len(bl_chunk))

        if bi == 0:
            # On the first baseline chunk, create the output file
            # TODO: we're not writing out the where_inpainted data for the GOLDEN
            #       or REDUCEDCHAN files yet -- it looks like we'd have to write out
            #       completely new UVFlag files for this.
            out_files["GOLDEN"] = []
            for nbin in golden_bins:
                out_files["GOLDEN"].append(
                    create_outfile(
                        kind="GOLDEN",
                        lst=lst_bin_edges[nbin],
                        times=binned_times[nbin],
                    )
                )

            if save_channels and len(binned_times[0]) > 0:
                out_files["REDUCEDCHAN"] = create_outfile(
                    kind="REDUCEDCHAN",
                    lst=lst_bin_edges[0],
                    times=binned_times[0],
                    channels=list(save_channels),
                )

        if len(golden_bins) > 0:
            for fl, nbin in zip(out_files["GOLDEN"], golden_bins):
                write_baseline_slc_to_file(
                    fl=fl,
                    slc=slc,
                    data=data[nbin].transpose((1, 0, 2, 3)),
                    flags=flags[nbin].transpose((1, 0, 2, 3)),
                    nsamples=nsamples[nbin].transpose((1, 0, 2, 3)),
                )

        if "REDUCEDCHAN" in out_files:
            write_baseline_slc_to_file(
                fl=out_files["REDUCEDCHAN"],
                slc=slc,
                data=data[0][:, :, save_channels].transpose((1, 0, 2, 3)),
                flags=flags[0][:, :, save_channels].transpose((1, 0, 2, 3)),
                nsamples=nsamples[0][:, :, save_channels].transpose((1, 0, 2, 3)),
            )

        for inpainted in [True, False]:
            if inpainted and not output_inpainted:
                continue
            if not inpainted and not output_flagged:
                continue

            rdc = reduce_lst_bins(
                data,
                flags,
                nsamples,
                where_inpainted=where_inpainted,
                inpainted_mode=inpainted,
                flag_thresh=flag_thresh,
                sigma_clip_thresh=(
                    None
                    if inpainted and not sigma_clip_in_inpainted_mode
                    else sigma_clip_thresh
                ),
                sigma_clip_min_N=sigma_clip_min_N,
                flag_below_min_N=flag_below_min_N,
                get_mad=write_med_mad,
            )

            write_baseline_slc_to_file(
                fl=out_files[("LST", inpainted)],
                slc=slc,
                data=rdc["data"],
                flags=rdc["flags"],
                nsamples=rdc["nsamples"],
            )

            write_baseline_slc_to_file(
                fl=out_files[("STD", inpainted)],
                slc=slc,
                data=rdc["std"],
                flags=rdc["flags"],
                nsamples=rdc["days_binned"],
            )

            if write_med_mad:
                write_baseline_slc_to_file(
                    fl=out_files[("MED", inpainted)],
                    slc=slc,
                    data=rdc["median"],
                    flags=rdc["flags"],
                    nsamples=rdc["nsamples"],
                )
                write_baseline_slc_to_file(
                    fl=out_files[("MAD", inpainted)],
                    slc=slc,
                    data=rdc["mad"],
                    flags=rdc["flags"],
                    nsamples=rdc["days_binned"],
                )

        nbls_so_far += len(bl_chunk)

    return out_files


@profile
def lst_bin_files(
    config_file: str | Path,
    output_file_select: int | Sequence[int] | None = None,
    include_autos: bool = True,
    **kwargs,
) -> list[dict[str, Path]]:
    """
    LST bin a series of UVH5 files.

    This takes a series of UVH5 files where each file has the same frequency bins and
    pols, grids them onto a common LST grid, and then averages all integrations
    that appear in that LST bin. It writes a series of UVH5 files, as configured in the
    `config_file`, including the LST-averaged data, the standard deviation of the data
    in each LST bin, optional full data across nights for each LST-bin with a reduced
    number of frequency channels, and optionally the full data across nights (and all
    channels) for a 'GOLDEN' subset of LST bins.

    Parameters
    ----------
    config_files
        A configuration file to use. This should be a YAML file constructed by
        :func:`~make_lst_bin_config_file`, encoding the configuration of the LST
        grid, and the matching of input data files to LST bins.
    output_file_select
        If provided, this is a list of integers that select which output files to
        write. For example, if this is [0, 2], then only the first and third output
        files will be written. This is useful for parallelizing the LST binning.
    include_autos
        If True, include autocorrelations in the LST binning. If False, ignore them.
    **kwargs
        Additional keyword arguments are passed to :func:`~lstbin.lst_bin_files_single_outfile`.


    Returns
    -------
    list of dicts
        list of dicts -- one for each output file.
        Each dict contains keys that indicate the type of output file (e.g. 'LST', 'STD',
        'REDUCEDCHAN', 'GOLDEN') and values that are the path to that file.
    """
    with open(config_file, "r") as fl:
        configuration = yaml.safe_load(fl)

    config_opts = configuration["config_params"]
    lst_grid = configuration["lst_grid"]
    matched_files = configuration["matched_files"]
    metadata = configuration["metadata"]

    if output_file_select is None:
        output_file_select = list(range(len(matched_files)))
    elif isinstance(output_file_select, int):
        output_file_select = [output_file_select]
    output_file_select = [int(i) for i in output_file_select]

    if max(output_file_select) >= len(matched_files):
        raise ValueError(
            "output_file_select must be less than the number of output files"
        )

    meta = FastUVH5Meta(
        matched_files[0][0][0],
        blts_are_rectangular=metadata["blts_are_rectangular"],
        time_axis_faster_than_bls=metadata["time_axis_faster_than_bls"],
    )

    antpos = dict(zip(meta.antenna_numbers, meta.antpos_enu))
    reds = RedundantGroups.from_antpos(antpos=antpos, include_autos=include_autos)

    output_files = []
    for outfile_index in output_file_select:
        data_files = matched_files[outfile_index]
        output_files.append(
            lst_bin_files_single_outfile(
                config_opts=config_opts,
                metadata=metadata,
                lst_bins=lst_grid[outfile_index],
                data_files=data_files,
                reds=reds,
                include_autos=include_autos,
                **kwargs,
            )
        )

    return output_files


@profile
def get_all_unflagged_baselines(
    data_files: list[list[str | Path | FastUVH5Meta]],
    ex_ant_yaml_files: list[str] | None = None,
    include_autos: bool = True,
    ignore_ants: tuple[int] = (),
    only_last_file_per_night: bool = False,
    redundantly_averaged: bool | None = None,
    reds: RedundantGroups | None = None,
    blts_are_rectangular: bool | None = None,
    time_axis_faster_than_bls: bool | None = None,
) -> tuple[list[tuple[int, int]], list[str]]:
    """Generate a set of all antpairs that have at least one un-flagged entry.

    This is performed over a list of nights, each of which consists of a list of
    individual uvh5 files. Each UVH5 file is *assumed* to have the same set of times
    for each baseline internally (different nights obviously have different times).

    If ``reds`` is provided, then any baseline found is mapped back to the first
    baseline in the redundant group it appears in. This *must* be set if

    Returns
    -------
    all_baselines
        The set of all antpairs in all files in the given list.
    all_pols
        A list of all polarizations in the files in the given list, as strings like
        'ee' and 'nn' (i.e. with x_orientation information).
    """
    if blts_are_rectangular is None and not isinstance(data_files[0][0], FastUVH5Meta):
        meta0 = FastUVH5Meta(data_files[0][0])
        blts_are_rectangular = meta0.blts_are_rectangular
        time_axis_faster_than_bls = meta0.time_axis_faster_than_bls

    data_files = [
        [
            fl
            if isinstance(fl, FastUVH5Meta)
            else FastUVH5Meta(
                fl,
                blts_are_rectangular=blts_are_rectangular,
                time_axis_faster_than_bls=time_axis_faster_than_bls,
            )
            for fl in fl_list
        ]
        for fl_list in data_files
    ]

    all_baselines = set()
    all_pols = set()

    meta0 = data_files[0][0]
    x_orientation = meta0.get_transactional("x_orientation")

    # reds will contain all of the redundant groups for the whole array, because
    # all the antenna positions are included in every file.
    if reds is None:
        reds = RedundantGroups.from_antpos(
            antpos=dict(zip(meta0.antenna_numbers, meta0.antpos_enu)),
            include_autos=True,
        )

    if redundantly_averaged is None:
        # Try to work out if the files are redundantly averaged.
        # just look at the middle file from each night.
        for fl_list in data_files:
            meta = fl_list[len(fl_list) // 2]
            antpairs = meta.get_transactional("antpairs")
            ubls = set(reds.get_ubl_key(ap) for ap in antpairs)
            if len(ubls) != len(antpairs):
                # At least two of the antpairs are in the same redundant group.
                redundantly_averaged = False
                logger.info("Inferred that files are not redundantly averaged.")
                break
        else:
            redundantly_averaged = True
            logger.info("Inferred that files are redundantly averaged.")

    if redundantly_averaged:
        if ignore_ants:
            raise ValueError(
                "Cannot ignore antennas if the files are redundantly averaged."
            )
        if ex_ant_yaml_files:
            raise ValueError(
                "Cannot exclude antennas if the files are redundantly averaged."
            )

    for night, fl_list in enumerate(data_files):
        if ex_ant_yaml_files:
            a_priori_antenna_flags = read_a_priori_ant_flags(
                ex_ant_yaml_files[night], ant_indices_only=True
            )
        else:
            a_priori_antenna_flags = set()

        if only_last_file_per_night:
            # Actually, use first AND last, just to be cautious
            fl_list = [fl_list[0], fl_list[-1]]

        for meta in fl_list:
            antpairs = meta.antpairs
            all_pols.update(set(meta.pols))
            this_xorient = meta.x_orientation

            # Clear the cache to save memory.
            meta.close()
            del meta.antpairs

            if this_xorient != x_orientation:
                raise ValueError(
                    f"Not all files have the same xorientation! The x_orientation in {meta.path} "
                    f"is {this_xorient}, but in {meta0.path} it is {x_orientation}."
                )

            for a1, a2 in antpairs:
                if redundantly_averaged:
                    a1, a2 = reds.get_ubl_key((a1, a2))

                if (
                    (a1, a2) not in all_baselines
                    and (a2, a1) not in all_baselines  # Do this first because after the
                    and a1 not in ignore_ants  # first file it often triggers.
                    and a2 not in ignore_ants
                    and (include_autos or a1 != a2)
                    and a1 not in a_priori_antenna_flags
                    and a2 not in a_priori_antenna_flags
                ):
                    all_baselines.add((a1, a2))

    return sorted(all_baselines), sorted(all_pols)


def create_lstbin_output_file(
    outdir: Path,
    kind: str,
    lst: float,
    pols: list[str],
    file_list: list[FastUVH5Meta],
    start_jd: float,
    times: np.ndarray | None = None,
    lsts: np.ndarray | None = None,
    history: str = "",
    fname_format: str = "zen.{kind}.{lst:7.5f}.{inpaint_mode}.uvh5",
    overwrite: bool = False,
    antpairs: list[tuple[int, int]] | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
    channels: np.ndarray | list[int] | None = None,
    vis_units: str = "Jy",
    inpaint_mode: bool | None = None,
    lst_branch_cut: float = 0.0,
) -> Path:
    outdir = Path(outdir)

    # update history
    file_list_str = "-".join(ff.path.name for ff in file_list)
    file_history = f"{history} Input files: {file_list_str}"
    _history = file_history + utils.history_string()

    if lst < lst_branch_cut:
        lst += 2 * np.pi

    fname = fname_format.format(
        kind=kind,
        lst=lst,
        pol="".join(pols),
        inpaint_mode="inpaint"
        if inpaint_mode
        else ("flagged" if inpaint_mode is False else ""),
    )
    # There's a weird gotcha with pathlib where if you do path / "/file.name"
    # You get just "/file.name" which is in root.
    if fname.startswith('/'):
        fname = fname[1:]
    fname = outdir / fname

    logger.info(f"Initializing {fname}")

    # check for overwrite
    if fname.exists() and not overwrite:
        raise FileExistsError(f"{fname} exists, not overwriting")

    freqs = np.squeeze(file_list[0].freq_array)
    if freq_min:
        freqs = freqs[freqs >= freq_min]
    if freq_max:
        freqs = freqs[freqs <= freq_max]
    if channels:
        channels = list(channels)
        freqs = freqs[channels]

    uvd_template = io.uvdata_from_fastuvh5(
        meta=file_list[0],
        antpairs=antpairs,
        lsts=lsts,
        times=times,
        history=_history,
        start_jd=start_jd,
        time_axis_faster_than_bls=True,
        vis_units=vis_units,
        lst_branch_cut=lst_branch_cut,
    )
    uvd_template.select(frequencies=freqs, polarizations=pols, inplace=True)
    # Need to set the polarization array manually because even though the select
    # operation does the down-select, it doesn't re-order the pols.
    uvd_template.polarization_array = np.array(
        uvutils.polstr2num(pols, x_orientation=uvd_template.x_orientation)
    )
    uvd_template.initialize_uvh5_file(str(fname.absolute()), clobber=overwrite)

    return fname


def write_baseline_slc_to_file(
    fl: Path, slc: slice, data: np.ndarray, flags: np.ndarray, nsamples: np.ndarray
):
    """Write a baseline slice to a file."""
    with h5py.File(fl, "r+") as f:
        ntimes = int(f["Header"]["Ntimes"][()])
        timefirst = bool(f["Header"]["time_axis_faster_than_bls"][()])
        if not timefirst and ntimes > 1:
            raise NotImplementedError("Can only do time-first files for now.")

        slc = slice(slc.start * ntimes, slc.stop * ntimes, 1)
        f["Data"]["visdata"][slc] = data.reshape((-1, data.shape[2], data.shape[3]))
        f["Data"]["flags"][slc] = flags.reshape((-1, data.shape[2], data.shape[3]))
        f["Data"]["nsamples"][slc] = nsamples.reshape(
            (-1, data.shape[2], data.shape[3])
        )


def config_lst_bin_files(
    data_files: list[list[str | FastUVH5Meta]],
    dlst: float | None = None,
    atol: float = 1e-10,
    lst_start: float | None = None,
    lst_width: float = 2 * np.pi,
    blts_are_rectangular: bool | None = None,
    time_axis_faster_than_bls: bool | None = None,
    ntimes_per_file: int | None = None,
    jd_regex: str = r"zen\.(\d+\.\d+)\.",
):
    """
    Configure data for LST binning.

    Make a 24 hour lst grid, starting LST and output files given
    input data files and LSTbin params.

    Parameters
    ----------
    data_files : list of lists
        nested set of lists, with each nested list containing paths to
        data files from a particular night. Frequency axis of each file must be identical.
    dlst : float
        LST bin width. If None, will get this from the first file in data_files.
    atol : float
        absolute tolerance for LST bin float comparison
    lst_start : float
        starting LST for binner as it sweeps from lst_start to lst_start + 2pi.
        Default is first LST of the first file of the first night.
    lst_width : float
        How much LST to bin.
    ntimes_per_file : int
        number of LST bins in a single output file

    Returns
    -------
    lst_grid : float ndarray holding LST bin centers.
    matched_files : list of lists of files, one list for each output file.
    """
    logger.info("Configuring lst_grid")

    data_files = [sorted(df) for df in data_files]

    df0 = data_files[0][0]

    # get dlst from first data file if None
    if dlst is None:
        dlst = io.get_file_times(df0, filetype="uvh5")[0]

    # Get rectangularity of blts from first file if None
    meta = FastUVH5Meta(
        df0,
        blts_are_rectangular=blts_are_rectangular,
        time_axis_faster_than_bls=time_axis_faster_than_bls,
    )

    if blts_are_rectangular is None:
        blts_are_rectangular = meta.blts_are_rectangular
    if time_axis_faster_than_bls is None:
        time_axis_faster_than_bls = meta.time_axis_faster_than_bls

    if ntimes_per_file is None:
        ntimes_per_file = meta.Ntimes

    # Get the initial LST as the lowest LST in any of the first files on each night
    first_files = [
        FastUVH5Meta(
            df[0],
            blts_are_rectangular=blts_are_rectangular,
            time_axis_faster_than_bls=time_axis_faster_than_bls,
        )
        for df in data_files
    ]

    # get begin_lst from lst_start or from the first JD in the data_files
    if lst_start is None:
        lst_start = np.min([ff.lsts[0] for ff in first_files])
    begin_lst = lst_start

    # make LST grid that divides into 2pi
    lst_grid = make_lst_grid(dlst, begin_lst=begin_lst, lst_width=lst_width)
    dlst = lst_grid[1] - lst_grid[0]

    lst_edges = np.concatenate([lst_grid - dlst / 2, [lst_grid[-1] + dlst / 2]])

    # Now, what we need here is actually the lst_edges of the *files*, not the actual
    # bins.
    nfiles = int(np.ceil(len(lst_grid) / ntimes_per_file))
    last_edge = lst_edges[-1]
    lst_edges = lst_edges[::ntimes_per_file]
    if len(lst_edges) < nfiles + 1:
        lst_edges = np.concatenate([lst_edges, [last_edge]])

    matched_files = [[] for _ in lst_grid]
    for fllist in data_files:
        matched = utils.match_files_to_lst_bins(
            lst_edges=lst_edges,
            file_list=fllist,
            files_sorted=True,
            jd_regex=jd_regex,
            blts_are_rectangular=blts_are_rectangular,
            time_axis_faster_than_bls=time_axis_faster_than_bls,
            atol=atol,
        )
        for i, m in enumerate(matched):
            matched_files[i].append(m)

    nfiles = int(np.ceil(len(lst_grid) / ntimes_per_file))
    lst_grid = [
        lst_grid[ntimes_per_file * i: ntimes_per_file * (i + 1)] for i in range(nfiles)
    ]

    # Only keep output files that have data associated
    lst_grid = [
        lg for lg, mf in zip(lst_grid, matched_files) if any(len(mff) > 0 for mff in mf)
    ]
    matched_files = [mf for mf in matched_files if any(len(mff) > 0 for mff in mf)]
    return lst_grid, matched_files


def make_lst_bin_config_file(
    config_file: str | Path,
    data_files: list[list[str | FastUVH5Meta]],
    clobber: bool = False,
    dlst: float | None = None,
    atol: float = 1e-10,
    lst_start: float | None = None,
    lst_width: float = 2 * np.pi,
    ntimes_per_file: int = 60,
    blts_are_rectangular: bool | None = None,
    time_axis_faster_than_bls: bool | None = None,
    jd_regex: str = r"zen\.(\d+\.\d+)\.",
    lst_branch_cut: float | None = None,
) -> dict[str, Any]:
    """Construct and write a YAML configuration file for lst-binning.

    This determines an LST-grid, and then determines which files should be
    included in each LST-bin. The output is a YAML file that can be used to
    quickly read in raw files that correspond to a particular LST-bin.

    The idea of this function is for it to be run as a separate step (e.g. by hera_opm)
    that needs to run to setup a full LST-binning run on multiple parallel tasks. Each
    task will read the YAML file, and select out the portion appropriate for that
    LST bin file.

    The algorithm for matching files to LST bins is only approximate, but is conservative
    (i.e. it includes *at least* all files that should be included in the LST bin).

    Parameters
    ----------
    config_file : str or Path
        Path to write the YAML configuration file to.
    data_files : list of lists of str or FastUVH5Meta
        List of lists of data files to consider for LST-binning. The outer list
        is a list of nights, and the inner list is a list of files for that night.
    clobber : bool, optional
        If True, overwrite the config_file if it exists. If False, raise an error
        if the config_file exists.
    dlst : float, optional
        The approximate width of each LST bin in radians. Default is integration time of
        the first file on the first night. This is approximate because the final bin
        width is always equally divided into 2pi.
    atol : float, optional
        Absolute tolerance for matching LSTs to LST bins. Default is 1e-10.
    lst_start : float, optional
        The starting LST for the LST grid. Default is the lowest LST in the first file
        on the first night.
    lst_width : float, optional
        The width of the LST grid. Default is 2pi. Note that this is not the width of
        each LST bin, which is given by dlst. Further note that the LST grid is always
        equally divided into 2pi, regardless of `lst_width`.
    ntimes_per_file : int, optional
        The number of LST bins to include in each output file. Default is 60.
    blts_are_rectangular : bool, optional
        If True, assume that the data-layout in the input files is rectangular in
        baseline-times. This will be determined if not given.
    time_axis_faster_than_bls : bool, optional
        If True, assume that the time axis moves faster than the baseline axis in the
        input files. This will be determined if not given.
    jd_regex : str, optional
        Regex to use to extract the JD from the file name. Set to None or empty
        to force the LST-matching to use the LSTs in the metadata within the file.
    lst_branch_cut
        The LST at which to branch cut the LST grid for file writing. The JDs in the
        output LST-binned files will be *lowest* at the lst_branch_cut, and all file
        names will have LSTs that are higher than lst_branch_cut. If None, this will
        be determined automatically by finding the largest gap in LSTs and starting
        AFTER it.

    Returns
    -------
    config : dict
        The configuration dictionary that was written to the YAML file.
    """
    config_file = Path(config_file)
    if config_file.exists() and not clobber:
        raise IOError(f"{config_file} exists and clobber is False")

    lst_grid, matched_files = config_lst_bin_files(
        data_files=data_files,
        dlst=dlst,
        atol=atol,
        lst_start=lst_start,
        lst_width=lst_width,
        blts_are_rectangular=blts_are_rectangular,
        time_axis_faster_than_bls=time_axis_faster_than_bls,
        jd_regex=jd_regex,
        ntimes_per_file=ntimes_per_file,
    )

    # Get the best lst_branch_cut by finding the largest gap in LSTs and starting
    # AFTER it
    if lst_branch_cut is None:
        lst_branch_cut = float(utils.get_best_lst_branch_cut(np.concatenate(lst_grid)))

    dlst = lst_grid[0][1] - lst_grid[0][0]
    # Make it a real list of floats to make the YAML easier to read
    lst_grid = [[float(lst) for lst in lsts] for lsts in lst_grid]

    # now matched files is a list of output files, each containing a list of nights,
    # each containing a list of files
    logger.info("Getting metadata from first file...")

    def get_meta():
        for outfile in matched_files:
            for night in outfile:
                for i, fl in enumerate(night):
                    return fl

    meta = get_meta()

    tint = np.median(meta.integration_time)
    if not np.all(np.abs(np.diff(np.diff(meta.times))) < 1e-6):
        raise ValueError(
            "All integrations must be of equal length (BDA not supported), got diffs: "
            f"{np.diff(meta.times)}"
        )

    matched_files = [
        [[str(m.path) for m in night] for night in outfiles]
        for outfiles in matched_files
    ]

    output = {
        "config_params": {
            "dlst": float(dlst),
            "atol": atol,
            "lst_start": lst_start,
            "lst_width": lst_width,
            "jd_regex": jd_regex,
        },
        "lst_grid": lst_grid,
        "matched_files": matched_files,
        "metadata": {
            "x_orientation": meta.x_orientation,
            "blts_are_rectangular": meta.blts_are_rectangular,
            "time_axis_faster_than_bls": meta.time_axis_faster_than_bls,
            "start_jd": int(meta.times[0]),
            "integration_time": float(tint),
            "lst_branch_cut": lst_branch_cut,
        },
    }

    with open(config_file, "w") as fl:
        yaml.safe_dump(output, fl)

    return output


def lst_bin_arg_parser():
    """
    arg parser for lst_bin_files() function. data_files argument must be quotation-bounded
    glob-parsable search strings to nightly data. For example:

    '2458042/zen.2458042.*.xx.HH.uv' '2458043/zen.2458043.*.xx.HH.uv'
    """
    a = argparse.ArgumentParser(
        description=(
            "drive script for lstbin.lst_bin_files(). "
            "data_files argument must be quotation-bounded "
            "glob-parsable search strings to nightly data. For example: \n"
            "'2458042/zen.2458042.*.xx.HH.uv' '2458043/zen.2458043.*.xx.HH.uv' \n"
            "Consult lstbin.lst_bin_files() for further details on functionality."
        )
    )
    a.add_argument(
        "configfile",
        type=str,
        help="config file produced by lstbin.make_lst_bin_config_file",
    )
    a.add_argument(
        "--calfile-rules",
        nargs="*",
        type=str,
        help="rules to convert datafile names to calfile names. A series of two strings where the first will be replaced by the latter",
    )
    a.add_argument(
        "--fname-format",
        type=str,
        default="zen.{kind}.{lst:7.5f}.uvh5",
        help="filename format for output files. See docstring for details.",
    )
    a.add_argument(
        "--outdir", default=None, type=str, help="directory for writing output"
    )
    a.add_argument(
        "--overwrite", default=False, action="store_true", help="overwrite output files"
    )
    a.add_argument(
        "--rephase",
        default=False,
        action="store_true",
        help="rephase data to center of LST bin before binning",
    )
    a.add_argument(
        "--history", default=" ", type=str, help="history to insert into output files"
    )
    a.add_argument(
        "--output_file_select",
        default=None,
        nargs="*",
        type=int,
        help="list of output file integers to run on. Default is all output files.",
    )
    a.add_argument(
        "--vis_units", default="Jy", type=str, help="visibility units of output files."
    )
    a.add_argument(
        "--ignore_flags",
        default=False,
        action="store_true",
        help="Ignore flags in data files, such that all input data is included in binning.",
    )
    a.add_argument(
        "--Nbls_to_load",
        default=None,
        type=int,
        help="Number of baselines to load and bin simultaneously. Default is all.",
    )
    a.add_argument(
        "--ex_ant_yaml_files",
        default=None,
        type=str,
        nargs="+",
        help="list of paths to yamls with lists of antennas from each night to exclude lstbinned data files.",
    )
    a.add_argument(
        "--ignore-ants", default=(), type=int, nargs="+", help="ants to ignore"
    )
    a.add_argument(
        "--ignore-missing-calfiles",
        default=False,
        action="store_true",
        help="if true, any datafile with missing calfile will just be removed from lstbinning.",
    )
    a.add_argument(
        "--write_kwargs",
        default="{}",
        type=str,
        help="json dictionary of arguments to the uvh5 writer",
    )
    a.add_argument(
        "--golden-lsts",
        type=str,
        help="LSTS (rad) to save longitudinal data for, separated by commas",
    )
    a.add_argument(
        "--save-channels",
        type=str,
        help="integer channels separated by commas to save longitudinal data for",
    )
    a.add_argument(
        "--sigma-clip-thresh",
        type=float,
        help="sigma clip threshold for flagging data in an LST bin over time. Zero means no clipping.",
        default=None,
    )
    a.add_argument(
        "--sigma-clip-min-N",
        type=int,
        help="number of unflagged data points over time to require before considering sigma clipping",
        default=4,
    )
    a.add_argument(
        "--flag-below-min-N",
        action="store_true",
        help="if true, flag all data in an LST bin if there are fewer than --sigma-clip-min-N unflagged data points over time",
    )
    a.add_argument(
        "--flag-thresh",
        type=float,
        help="fraction of integrations in an LST bin for a particular (antpair, pol, channel) that must be flagged for the entire bin to be flagged",
        default=0.7,
    )
    a.add_argument(
        "--redundantly-averaged",
        action="store_true",
        default=None,
        help="if true, assume input files are redundantly averaged",
    )
    a.add_argument(
        "--only-last-file-per-night",
        action="store_true",
        default=False,
        help="if true, only use the first and last file every night to obtain antpairs",
    )
    a.add_argument(
        "--freq-min",
        type=float,
        default=None,
        help="minimum frequency to include in lstbinning",
    )
    a.add_argument(
        "--freq-max",
        type=float,
        default=None,
        help="maximum frequency to include in lstbinning",
    )
    a.add_argument(
        "--no-flagged-mode",
        action="store_true",
        help="turn off output of flagged mode LST-binning",
    )
    a.add_argument(
        "--do-inpaint-mode",
        action="store_true",
        default=None,
        help="turn on inpainting mode LST-binning",
    )
    a.add_argument(
        "--where-inpainted-file-rules",
        nargs="*",
        type=str,
        help="rules to convert datafile names to where-inpainted-file names. A series of two strings where the first will be replaced by the latter",
    )
    a.add_argument(
        "--sigma-clip-in-inpainted-mode",
        action="store_true",
        default=False,
        help="allow sigma-clipping in inpainted mode",
    )
    a.add_argument(
        "--write-med-mad",
        action="store_true",
        default=False,
        help="option to write out MED/MAD files in addition to LST/STD files",
    )
    return a
