"""
An attempt at a simpler LST binner that makes more assumptions but runs faster.

In particular, we assume that all baselines have the same time array and frequency array,
and that each is present throughout the data array. This allows a vectorization.
"""
from __future__ import annotations

import numpy as np
from . import utils
import warnings
from pathlib import Path
from .lstbin import config_lst_bin_files, sigma_clip
from . import abscal
import os
from . import io
import logging
from hera_qm.metrics_io import read_a_priori_ant_flags
from . import apply_cal
from typing import Sequence
import argparse
from pyuvdata.uvdata.uvh5 import FastUVH5Meta
from pyuvdata import utils as uvutils
from .red_groups import RedundantGroups
import h5py
from functools import partial
import yaml

try:
    profile
except NameError:
    def profile(fnc):
        return fnc

logger = logging.getLogger(__name__)

@profile
def simple_lst_bin(
    data: np.ndarray,
    data_lsts: np.ndarray,
    baselines: list[tuple[int, int]],
    lst_bin_edges: np.ndarray,
    freq_array: np.ndarray,
    flags: np.ndarray | None = None,
    nsamples: np.ndarray | None = None,
    rephase: bool = True,
    antpos: np.ndarray | None = None,
    lat: float = -30.72152,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Split input data into a list of LST bins.

    This function simply splits a data array with multiple time stamps into a list of
    arrays, each containing a single LST bin. Each of the data arrays in each bin
    are also rephased onto a common LST grid.

    Parameters
    ----------
    data
        The visibility data. Must be shape (ntimes, nbls, nfreqs, npols)
    data_lsts
        The LSTs corresponding to each of the time stamps in the data. Must have
        length ``data.shape[0]``
    baselines
        The list 2-tuples of baselines in the data array.
    lst_bin_edges
        A sequence of floats specifying the *edges* of the LST bins to use.
    freq_array
        An array of frequencies in the data, in Hz.
    flags
        An array of boolean flags, indicating bins NOT to use. Same shape as data.
    nsamples
        An array of sample counts, same shape as data.
    rephase
        Whether to apply re-phasing to the data, to bring it to a common LST grid.
    antpos
        3D Antenna positions for each antenna in the data.
    lat
        The latitude (in degrees) of the telescope.

    Returns
    -------
    data
        A nlst-length list of arrays, each of shape 
        ``(ntimes_in_lst, nbls, nfreq, npol)``, where LST bins without data simply have
        a first-axis of size zero.
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
    required_shape = (len(data_lsts), len(baselines), len(freq_array), npols)
    
    if npols > 4:
        raise ValueError(f"data has more than 4 pols! Got {npols} (last axis of data)")

    if data.shape != required_shape:
        raise ValueError(
            f"data should have shape {required_shape} but got {data.shape}"
        )

    if flags is None:
        flags = np.zeros(data.shape, dtype=bool)

    if flags.shape != data.shape:
        raise ValueError(
            f"flags should have shape {data.shape} but got {flags.shape}"
        )

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
        raise ValueError(
            "lst_bin_edges must be monotonically increasing."
        )

    # Now ensure that all the observed LSTs are wrapped so they start above the first bin edges
    grid_indices, data_lsts, lst_mask = get_lst_bins(data_lsts, lst_bin_edges)
    lst_bin_centres = (lst_bin_edges[1:] + lst_bin_edges[:-1])/2

    # TODO: check whether this creates a data copy. Don't want the extra RAM...
    data = data[lst_mask]  # actually good if this is copied, because we do LST rephase in-place
    flags = flags[lst_mask]
    nsamples = nsamples[lst_mask]
    data_lsts = data_lsts[lst_mask]
    grid_indices = grid_indices[lst_mask]

    logger.info(f"Data Shape: {data.shape}")

    # Now, rephase the data to the lst bin centres.
    if rephase:
        logger.info("Rephasing data")
        if freq_array is None or antpos is None:
            raise ValueError("freq_array and antpos is needed for rephase")

        bls = np.array([antpos[k[0]] - antpos[k[1]] for k in baselines])

        # get appropriate lst_shift for each integration, then rephase
        lst_shift = lst_bin_centres[grid_indices] - data_lsts

        # this makes a copy of the data in d
        utils.lst_rephase_vectorized(data, bls, freq_array, lst_shift, lat=lat, inplace=True)

    # shortcut -- just return all the data, re-organized.
    _data, _flags, _nsamples = [], [], []
    empty_shape = (0, len(baselines), len(freq_array), npols)
    for lstbin in range(len(lst_bin_centres)):
        mask = grid_indices == lstbin
        if np.any(mask):
            _data.append(data[mask])
            _flags.append(flags[mask])
            _nsamples.append(nsamples[mask])
        else:
            _data.append(np.zeros(empty_shape, complex))
            _flags.append(np.zeros(empty_shape, bool))
            _nsamples.append(np.zeros(empty_shape, int))

    return lst_bin_centres, _data, _flags, _nsamples

def get_lst_bins(lsts: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    lsts %= 2*np.pi
    lsts[lsts < edges[0]] += 2* np.pi
    bins = np.digitize(lsts, edges, right=True) - 1
    mask = (bins >= 0) & (bins < (len(edges)-1))
    return bins, lsts, mask

def reduce_lst_bins(
    data: list[np.ndarray], flags: list[np.ndarray], nsamples: list[np.ndarray],
    out_data: np.ndarray | None = None, 
    out_flags: np.ndarray | None = None,
    out_std: np.ndarray | None = None,
    out_nsamples: np.ndarray | None = None,
    mutable: bool = False,
    sigma_clip_thresh: float = 0.0,
    sigma_clip_min_N: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    From a list of LST-binned data, produce reduced statistics.

    Use this function to reduce lists of arrays with multiple time integrations per bin
    (i.e. the output of :func:`simple_lst_bin`) to arrays of shape 
    ``(nbl, nlst_bins, nfreq, npol)``. For example, compute the mean/std.

    Parameters
    ----------
    data
        The data to perform the reduction over. The length of the list is the number
        of LST bins. Each array in the list should have shape 
        ``(nbl, ntimes_per_lst, nfreq, npol)``.
    flags
        A list, the same length/shape as ``data``, containing the flags.
    nsamples
        A list, the same length/shape as ``data``, containing the number of samples
        for each measurement.
    out_data, out_flags, out_std, out_nsamples
        Optional Arrays into which the output can be placed. Useful to provide if 
        iterating over a set of input files, for example. 
        Shape ``(nbl, nlst_bins, nfreq, npol)``.
    mutable
        Whether the input data (and flags and nsamples) can be modified in place within
        the algorithm. Setting to true saves memory, and is safe for a one-shot script.

    Returns
    -------
    out_data, out_flags, out_std, out_nsamples
        The reduced data, flags, standard deviation (across days) and nsamples. 
        Shape ``(nbl, nlst_bins, nfreq, npol)``.
    """
    nlst_bins = len(data)
    (_, nbl, nfreq, npol) = data[0].shape

    for d, f, n in zip(data, flags, nsamples):
        assert d.shape == f.shape == n.shape

    # Do this just so that we can save memory if the call to this function already
    # has allocated memory.
    if out_data is None:
        out_data = np.zeros((nbl, nlst_bins, nfreq, npol), dtype=complex)
    if out_flags is None:
        out_flags = np.zeros(out_data.shape, dtype=bool)
    if out_std is None:
        out_std = np.ones(out_data.shape, dtype=complex)
    if out_nsamples is None:
        out_nsamples = np.zeros(out_data.shape, dtype=float)

    assert out_data.shape == out_flags.shape == out_std.shape == out_nsamples.shape
    assert out_data.shape == (nbl, nlst_bins, nfreq, npol)

    for lstbin, (d,n,f) in enumerate(zip(data, nsamples, flags)):
        logger.info(f"Computing LST bin {lstbin+1} / {nlst_bins}")
        
        # TODO: check that this doesn't make yet another copy...
        # This is just the data in this particular lst-bin.
        if d.size:
            (
                out_data[:, lstbin], 
                out_flags[:, lstbin], 
                out_std[:, lstbin], 
                out_nsamples[:, lstbin]
            ) = lst_average(
                d, n, f, mutable=mutable, 
                sigma_clip_thresh=sigma_clip_thresh, 
                sigma_clip_min_N=sigma_clip_min_N
            )
        else:
            out_data[:, lstbin] = 1.0
            out_flags[:, lstbin] = True
            out_std[:, lstbin] = 1.0
            out_nsamples[:, lstbin] = 0.0

        
    return out_data, out_flags, out_std, out_nsamples

def _allocate_dnf(shape: tuple[int], d=0.0, f=0, n=0):
    data = np.full(shape, d, dtype=complex)
    flags = np.full(shape, f, dtype=bool)
    nsamples = np.full(shape, n, dtype=float)
    return data, flags, nsamples

@profile
def lst_average(
    data: np.ndarray, nsamples: np.ndarray, flags: np.ndarray, 
    flag_thresh: float = 0.7, median: bool = False,
    mutable: bool = False,
    sigma_clip_thresh: float = 0.0,
    sigma_clip_min_N: int=4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute statistics of a set of data over its first axis.

    The idea here is that the data's first axis is "nights", and that each night is
    at the same LST. However, this function is agnostic to the meaning of the first
    axis. It just computes the mean, std, and nsamples over the first axis.

    This function is meant to be used on a single element of a list returned by
    :func:`simple_lst_bin`.

    Parameters
    ----------
    data
        The data to compute the statistics over. Shape ``(ntimes, nbl, nfreq, npol)``.
    nsamples
        The number of samples for each measurement. Shape ``(ntimes, nbl, nfreq, npol)``.
    flags
        The flags for each measurement. Shape ``(ntimes, nbl, nfreq, npol)``.
    flag_thresh
        The fraction of times a baseline/frequency/pol must be flagged in order to
        flag the baseline/frequency/pol for all nights.
    sigma_clip_thresh
        The number of standard deviations to use as a threshold for sigma clipping.
        If 0, no sigma clipping is performed. Note that sigma-clipping is performed
        per baseline, frequency, and polarization.
    sigma_clip_min_N
        The minimum number of unflagged samples required to perform sigma clipping.

    Returns
    -------
    out_data, out_flags, out_std, out_nsamples
        The reduced data, flags, standard deviation (across days) and nsamples.
        Shape ``(nbl, nfreq, npol)``.
    """
    # data has shape (ntimes, nbl, npols, nfreqs)
    # all data is assumed to be in the same LST bin.

    assert data.shape == nsamples.shape == flags.shape
    
    if not mutable:
        flags = flags.copy()
        nsamples = nsamples.copy()
        data = data.copy()

    flags[np.isnan(data) | np.isinf(data) | (nsamples == 0)] = True

    # Flag entire LST bins if there are too many flags over time
    flag_frac = np.sum(flags, axis=0) / flags.shape[0]
    nflags = np.sum(flags)
    logger.info(f"Percent of data flagged before thresholding: {100*np.sum(flags)/flags.size:.2f}%")
    flags |= flag_frac > flag_thresh
    data[flags] *= np.nan  # do this so that we can do nansum later. multiply to get both real/imag as nan
    logger.info(f"Flagged a further {100*(np.sum(flags) - nflags)/flags.size:.2f}% of visibilities due to flag_frac > {flag_thresh}")

    # Now do sigma-clipping.
    if sigma_clip_thresh > 0:
        nflags = np.sum(flags)
        flags |= sigma_clip(data, sigma=sigma_clip_thresh, min_N = sigma_clip_min_N)
        data[flags] *= np.nan
        logger.info(f"Flagged a further {100*(np.sum(flags) - nflags)/flags.size:.2f}% of visibilities due to sigma clipping")

    # get other stats
    logger.info("Calculating std")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")
        std = np.nanstd(data.real, axis=0) + 1j*np.nanstd(data.imag, axis=0)
        
    nsamples[flags] = 0
    norm = np.sum(nsamples, axis=0)  # missing a "clip" between 1e-99 and inf here...
    
    logger.info("Calculating mean")
    data = np.nansum(data * nsamples, axis=0)
    data[norm>0] /= norm[norm>0]
    data[norm<=0] = 1  # any value, it's flagged anyway
        
    f_min = np.all(flags, axis=0)
    std[f_min] = 1.0
    norm[f_min] = 0  # This is probably redundant.

    return data, f_min, std, norm

def adjust_lst_bin_edges(lst_bin_edges: np.ndarray) -> np.ndarray:
    """
    Adjust the LST bin edges so that they start in the range [0, 2pi) and increase.
    
    Performs the adjustment in-place.
    """
    while lst_bin_edges[0] < 0:
        lst_bin_edges += 2*np.pi
    while lst_bin_edges[0] >= 2*np.pi:
        lst_bin_edges -= 2*np.pi

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Produce a set of LST-binned (but not averaged) data for a set of baselines.

    This function takes a set of input data files, and reads any data in them that 
    falls within the LST bins specified by ``lst_bin_edges`` (optionally calibrating
    the data as it is read). The data is sorted into the LST-bins provided and returned
    as a list of arrays, one for each LST bin. The data is not averaged over LST bins.

    Only the list of baselines given will be read from each file, which makes it
    possible to iterate over baseline chunks and call this function on each chunk,
    to reduce maximum memory usage.
    
    Parameters
    ----------
    data_files
        A list of paths to data files to read. Instead of paths, you can also pass
        FastUVH5Meta objects, which will be used to read the data.
    lst_bin_edges
        A list of LST bin edges, in radians.
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
    times_in_bins
        The JDs that are in each LST bin -- a list of arrays.
    """
    metas = [fl if isinstance(fl, FastUVH5Meta) else FastUVH5Meta(fl, blts_are_rectangular=True) for fl in data_files]
    
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

    if any(isinstance(p, (int, np.int_)) for p in pols):
        pols = uvutils.polnum2str(pols, x_orientation=metas[0].x_orientation)

    if any(not isinstance(p, str) for p in pols):
        raise ValueError("pols must be a sequence of strings, e.g. ('xx', 'yy', 'xy', 'yx')")
    
    if antpos is None and rephase:
        warnings.warn(
            "Getting antpos from the first file only. This is almost always correct, "
            "but will be wrong if different files have different antenna_position arrays."
        )
        antpos = dict(zip(metas[0].antenna_numbers, metas[0].antpos_enu))

    if time_idx is None:
        adjust_lst_bin_edges(lst_bin_edges)        
        lst_bin_edges %= 2*np.pi
        op = np.logical_and if lst_bin_edges[0] < lst_bin_edges[-1] else np.logical_or
        time_idx = []
        for meta in metas:
            _lsts = meta.get_transactional('lsts')
            time_idx.append(op(_lsts >= lst_bin_edges[0], _lsts < lst_bin_edges[-1]))

    if time_arrays is None:
        time_arrays = [meta.get_transactional('times')[idx] for meta, idx in zip(metas, time_idx)]

    
    if lsts is None:
        lsts = np.concatenate(
            [meta.get_transactional('lsts')[idx] for meta, idx in zip(metas, time_idx)]
        )            

    # Now we can set up our master arrays of data. 
    data, flags, nsamples = _allocate_dnf(
        (len(lsts), len(antpairs), len(freqs), len(pols)),
        d=np.nan + np.nan*1j,
        f=True
    )

    if cal_files is None:
        cal_files = [None] * len(metas)

    if redundantly_averaged and reds is None:
        raise ValueError("reds must be provided if redundantly_averaged is True")
    if redundantly_averaged and any(c is not None for c in cal_files):
        raise ValueError("Cannot apply calibration if redundantly_averaged is True")
    
    # This loop actually reads the associated data in this LST bin.
    ntimes_so_far = 0
    for meta, calfl, tind, tarr in zip(metas, cal_files, time_idx, time_arrays):
        logger.info(f"Reading {meta.path}")
        slc = slice(ntimes_so_far,ntimes_so_far+len(tarr))
        ntimes_so_far += len(tarr)

        #hd = io.HERAData(str(fl.path), filetype='uvh5')
        data_antpairs = meta.get_transactional('antpairs')

        if redundantly_averaged:
            bls_to_load = [bl for bl in data_antpairs if reds.get_ubl_key(bl) in antpairs]
        else:
            bls_to_load = [bl for bl in antpairs if bl in data_antpairs or bl[::-1] in data_antpairs]

        if not bls_to_load:
            # If none of the requested baselines are in this file, then just 
            # set stuff as nan and go to next file. 
            logger.warning(f"None of the baselines are in {meta.path}. Skipping.")
            data[slc] = np.nan
            flags[slc] = True
            nsamples[slc] = 0
            continue

        # TODO: use Fast readers here instead.
        _data, _flags, _nsamples = io.HERAData(meta.path).read(
            bls=bls_to_load, times=tarr, freq_chans=freq_chans, polarizations=pols,
        )
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
                _data, gains, data_flags=_flags, cal_flags=cal_flags,
                gain_convention=uvc.gain_convention
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
    # lst_edges = np.arange(outfile_lsts[0] - dlst/2, outfile_lsts[-1] + dlst, dlst)
    bin_lst, data, flags, nsamples = simple_lst_bin(
        data=data, 
        flags=None if ignore_flags else flags,
        nsamples=nsamples,
        data_lsts=lsts,
        baselines=antpairs,
        lst_bin_edges=lst_bin_edges,
        freq_array = freqs,
        rephase = rephase,
        antpos=antpos,
    )

    bins = get_lst_bins(lsts, lst_bin_edges)[0]
    times = np.concatenate(time_arrays)
    times_in_bins = []
    for i in range(len(bin_lst)):
        mask = bins == i
        times_in_bins.append(times[mask])

    return bin_lst, data, flags, nsamples, times_in_bins

def apply_calfile_rules(
        data_files: list[list[str]], 
        calfile_rules: list[tuple[str, str]],
        ignore_missing: bool
) -> tuple[list[list[str]], list[list[str]]]:
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
) -> tuple[list, list, list, list, list]:

    lstmin, lstmax = lst_range
    lstmin %= (2 * np.pi)
    lstmax %= (2 * np.pi)
    if lstmin > lstmax:
        lstmax += 2 * np.pi

    if not cal_files:
        cal_files = [[None for d in dm] for dm in data_metas]

    tinds = []
    all_lsts = []
    file_list = []
    time_arrays = []
    cals = []

    # This loop just gets the number of times that we'll be reading.
    # Even though the input files should already have been configured to be those
    # that fall within the output LST range, we still need to read them in to
    # check exactly which time integrations we require.
    
    for night, callist in zip(data_metas, cal_files):
        for meta, cal in zip(night, callist):
            tarr = meta.times
            lsts = meta.lsts % (2*np.pi)
            lsts[lsts < lstmin] += 2 * np.pi
            
            tind = (lsts > lstmin) & (lsts < lstmax)

            if np.any(tind):
                tinds.append(tind)
                time_arrays.append(tarr[tind])
                all_lsts.append(lsts[tind])
                file_list.append(meta)
                cals.append(cal)
    return tinds, time_arrays, all_lsts, file_list, cals


def make_lst_grid(
    dlst: float, 
    begin_lst: float | None = None, 
    lst_width: float = 2*np.pi, 
) -> np.ndarray:
    """
    Make a uniform grid in local sidereal time.

    By default, this grid will span 2pi radians, starting at zero radians. Even if
    the ``lst_width`` is not 2pi, we enforce that the grid equally divides 2pi, so
    that it can wrap around if later the width is increased and the same dlst is used.

    Parameters:
    -----------
    dlst : 
        The width of a single LST bin in radians. 2pi must be equally divisible
        by dlst. If not, will default to the closest dlst that satisfies this criterion that
        is also greater than the input dlst. There is a minimum allowed dlst of 6.283e-6 radians,
        or .0864 seconds.
    begin_lst
        Beginning point for lst_grid. ``begin_lst`` must fall exactly on an LST bin 
        given a dlst, within 0-2pi. If not, it is replaced with the closest bin. 
        Default is zero radians.

    Output:
    -------
    lst_grid
        Uniform LST grid marking the center of each LST bin
    """
    assert dlst >= 6.283e-6, "dlst must be greater than 6.283e-6 radians, or .0864 seconds."
    assert dlst < 2 * np.pi, "dlst must be less than 2pi radians, or 24 hours."

    # check 2pi is equally divisible by dlst
    if not (
        np.isclose((2 * np.pi / dlst) % 1, 0.0, atol=1e-5) 
        or np.isclose((2 * np.pi / dlst) % 1, 1.0, atol=1e-5)
    ):
        # generate array of appropriate dlsts
        dlsts = 2 * np.pi / np.arange(1, 1000000)

        # get dlsts closest to dlst, but also greater than dlst
        dlst_diff = dlsts - dlst
        dlst_diff[dlst_diff < 0] = 10
        new_dlst = dlsts[np.argmin(dlst_diff)]
        logger.warning(
            f"2pi is not equally divisible by input dlst ({dlst:.16f}) at 1 part in 1e7.\n"
            f"Using {new_dlst:.16f} instead."
        )
        dlst = new_dlst

    # make an lst grid from [0, 2pi), with the first bin having a left-edge at 0 radians.
    lst_grid = np.arange(0, 2*np.pi - 1e-7, dlst) + dlst / 2

    # shift grid by begin_lst
    if begin_lst is not None:
        # enforce begin_lst to be within 0-2pi
        if begin_lst < 0 or begin_lst >= 2 * np.pi:
            logger.warning("begin_lst was < 0 or >= 2pi, taking modulus with (2pi)")
            begin_lst = begin_lst % (2 * np.pi)
        begin_lst = lst_grid[np.argmin(np.abs(lst_grid - begin_lst))] - dlst / 2
        lst_grid += begin_lst
    else:
        begin_lst = 0.0

    lst_grid = lst_grid[lst_grid < (begin_lst + lst_width)]
    
    return lst_grid


def lst_bin_files_single_outfile(
    config_opts: dict[str, Any],
    metadata: dict[str, Any],
    lst_bins: np.ndarray,
    data_files: list[list[str]],
    calfile_rules: list[tuple[str, str]] | None= None,
    ignore_missing_calfiles: bool = False,
    outdir: str | Path | None = None,
    meta: FastUVH5Meta | None = None,
    reds: RedundantGroups | None = None,
    redundantly_averaged: bool | None = None,
    only_last_file_per_night: bool = False,
    history: str = '',
    fname_format: str="zen.{kind}.{lst:7.5f}.uvh5", 
    overwrite: bool=False, 
    rephase: bool=False,
    Nbls_to_load: int | None=None, 
    ignore_flags: bool=False, 
    include_autos: bool=True, 
    ex_ant_yaml_files=None, 
    ignore_ants: tuple[int]=(),
    write_kwargs: dict | None = None,
    save_channels: tuple[int] = (),
    golden_lsts: tuple[float] = (),
    sigma_clip_thresh: float = 0.0,
    sigma_clip_min_N: int = 4,
    freq_min: float | None = None,
    freq_max: float | None = None,
) -> dict[str, Path]:
    # Check that that there are the same number of input data files and 
    # calibration files each night.
    input_cals = []
    if calfile_rules:
        data_files, input_cals = apply_calfile_rules(
            data_files, calfile_rules, ignore_missing=ignore_missing_calfiles
        )    

    # Prune empty nights (some nights start with files, but have files removed because
    # they have no associated calibration)
    data_files = [df for df in data_files if df]
    input_cals = [cf for cf in input_cals if cf]

    logger.info("Got the following numbers of data files per night:")
    for dflist in data_files:
        logger.info(f"{dflist[0].split('/')[-1]}: {len(dflist)}")

    data_metas = [[
        FastUVH5Meta(
            df, 
            blts_are_rectangular=metadata['blts_are_rectangular'], 
            time_axis_faster_than_bls=metadata['time_axis_faster_than_bls']
        ) for df in dflist
        ] 
        for dflist in data_files
    ]

    # get outdir
    if outdir is None:
        outdir = os.path.dirname(os.path.commonprefix(abscal.flatten(data_files)))

    x_orientation = metadata['x_orientation']
    start_jd = metadata['start_jd']
    integration_time = metadata['integration_time']

    # get metadata
    logger.info("Getting metadata from first file...")
    if meta is None:
        meta = data_metas[0][0]
    
    freq_array = np.squeeze(meta.freq_array)
    
    # reds will contain all of the redundant groups for the whole array, because
    # all the antenna positions are included in every file.
    antpos = dict(zip(meta.antenna_numbers, meta.antpos_enu))
    if reds is None:
        reds = RedundantGroups.from_antpos(
            antpos=antpos,
            include_autos=include_autos
        )

    if redundantly_averaged is None:
        # Try to work out if the files are redundantly averaged.
        # just look at the middle file from each night.
        for fl_list in data_metas:
            meta = fl_list[len(fl_list)//2]
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
    all_baselines = sorted(all_baselines)

    nants0 = meta.header['antenna_numbers'].size

    # Do a quick check to make sure all nights at least have the same number of Nants
    for dflist in data_metas:
        _nants = dflist[0].header['antenna_numbers'].size
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
    bl_chunks = [all_baselines[i * Nbls_to_load:(i + 1) * Nbls_to_load] for i in range(n_bl_chunks)]
    bl_chunks = [blg for blg in bl_chunks if len(blg) > 0]

    dlst = config_opts['dlst']
    lst_bin_edges = np.array(
        [x - dlst/2 for x in lst_bins] + [lst_bins[-1] + dlst/2]
    )

    tinds, time_arrays, all_lsts, file_list, cals = filter_required_files_by_times(
        (lst_bin_edges[0], lst_bin_edges[-1]),
        data_metas,
        input_cals,
    )

    all_lsts = np.concatenate(all_lsts)

    # If we have no times at all for this file, just return
    if len(all_lsts) == 0:
        return {}

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
        outdir = outdir, 
        pols=all_pols,
        file_list=file_list,
        history=history,
        fname_format=fname_format,
        overwrite=overwrite,
        antpairs=all_baselines,
        start_jd=start_jd,
        freq_min=freq_min,
        freq_max=freq_max,
    )
    out_files = {}
    for kind in ['LST', 'STD']:
        # Create the files we'll write to
        out_files[kind] = create_outfile(
            kind = kind,
            lst=lst_bin_edges[0],
            lsts=lst_bins,
        )
        
    nbls_so_far = 0
    for bi, bl_chunk in enumerate(bl_chunks):
        logger.info(f"Baseline Chunk {bi+1} / {len(bl_chunks)}")
        # data/flags/nsamples are *lists*, with nlst_bins entries, each being an
        # array, with shape (times, bls, freqs, npols)
        bin_lst, data, flags, nsamples, binned_times = lst_bin_files_for_baselines(
            data_files = file_list, 
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
        )

        slc = slice(nbls_so_far, nbls_so_far + len(bl_chunk))
        out_data, out_flags, out_std, out_nsamples = reduce_lst_bins(
            data, flags, nsamples,
            sigma_clip_thresh = sigma_clip_thresh,
            sigma_clip_min_N = sigma_clip_min_N,
        )

        write_baseline_slc_to_file(
            fl=out_files['LST'],
            slc=slc,
            data=out_data,
            flags=out_flags,
            nsamples=out_nsamples,
        )
        write_baseline_slc_to_file(
            fl=out_files['STD'],
            slc=slc,
            data=out_std,
            flags=out_flags,
            nsamples=out_nsamples,
        )

        if bi == 0:
            # On the first baseline chunk, create the output file
            out_files['GOLDEN'] = []
            for ibin, nbin in enumerate(golden_bins):
                out_files["GOLDEN"].append(
                    create_outfile(
                        kind = 'GOLDEN',
                        lst=lst_bin_edges[nbin],
                        times=binned_times[ibin],
                    )
                )

            if save_channels:
                out_files['REDUCEDCHAN'] = create_outfile(
                    kind = 'REDUCEDCHAN',
                    lst=lst_bin_edges[0],
                    times=binned_times[0],
                    channels=save_channels
                )

        if len(golden_bins)>0:
            for nbin, b in enumerate(golden_bins):
                write_baseline_slc_to_file(
                    fl = out_files['GOLDEN'][nbin],
                    slc=slc,
                    data=data[b].transpose((1, 0, 2, 3)),
                    flags=flags[b].transpose((1, 0, 2, 3)),
                    nsamples=nsamples[b].transpose((1, 0, 2, 3)),
                )

        if len(save_channels):
            write_baseline_slc_to_file(
                fl = out_files['REDUCEDCHAN'],
                slc=slc,
                data=data[0][:, :, save_channels].transpose((1, 0, 2, 3)),
                flags=flags[0][:, :, save_channels].transpose((1, 0, 2, 3)),
                nsamples=nsamples[0][:, :, save_channels].transpose((1, 0, 2, 3)),
            )

        nbls_so_far += len(bl_chunk)

    return out_files


@profile
def lst_bin_files(
    config_file: str | Path,
    output_file_select: int | Sequence[int] | None=None, 
    include_autos: bool=True,
    **kwargs
) -> list[dict[str, Path]]:
    """
    LST bin a series of UVH5 files.
    
    This takes a series of UVH5 files where each file has the same frequency bins and 
    pols, grids them onto a common LST grid, and then averages all integrations
    that appear in that LST bin.

    Parameters
    ----------
    data_files
        A list of lists of data files to LST bin. Each list of files is treated as coming
        from a single night.
    calfile_rules
        A list of tuples of strings. Each tuple is a pair of strings that are used to
        replace the first string with the second string in the data file name to get
        the calibration file name. For example, providing [(".uvh5", ".calfits")] will
        generate a list of calfiles that have the same basename as the data files, but
        with the extension ".calfits" instead of ".uvh5". Multiple entries to the list
        are allowed, and the replacements are applied in order. If the resulting calfile
        name does not exist, the data file is ignored.
    input_cals
        A list of lists of calibration files to use. If this is provided, it overrides
        the calfile_rules. If this is provided, it must have the same structure as
        data_files.
    dlst
        The width of the LST bins in radians. If not provided, this is set to the 
        interval between the first two LSTs in the first data file on the first night.
    n_lstbins_per_outfile
        The number of LST bins to put in each output file.
    fname_format
        A formatting string to use to write the output file. This can have the following
        fields: "kind" (which will evaluate to one of 'LST', 'STD', 'GOLDEN' or 'REDUCEDCHAN'),
        "lst" (which will evaluate to the LST of the bin), and "pol" (which will evaluate
        to the polarization of the data). Example: "zen.{kind}.{lst:7.5f}.uvh5"
    outdir
        The output directory. If not provided, this is set to the lowest-level common
        directory for all data files.
    overwrite
        If True, overwrite output files.
    history
        History to insert into output files.
    lst_start
        Starting LST for binner as it sweeps from lst_start to lst_start + lst_width.
        By default, use the LST associated with the earliest time in any of the 
        provided files.
    lst_width
        The total width of the LST grid in radians. By default, this is 2pi.
    atol
        Absolute tolerance for comparing LSTs.
    rephase
        If True, rephase data points in LST bin to center of bin.
    output_file_select
        If provided, this is a list of integers that select which output files to
        write. For example, if this is [0, 2], then only the first and third output
        files will be written. This is useful for parallelizing the LST binning.
    Nbls_to_load
        The number of baselines to load at a time. If None, load all baselines at once.
    ignore_flags
        If True, ignore flags when binning data.
    include_autos
        If True, include autocorrelations when binning data.
    ex_ant_yaml_files
        A list of yaml files that specify which antennas to exclude from each data
        file
    ignore_ants
        A list of antennas to ignore when binning data.
    write_kwargs
        A dictionary of keyword arguments to pass to the write function.
    ignore_missing_calfiles
        If True, ignore missing calibration files. If False, raise an error if a 
        calfile is missing.
    save_channels
        A list of channels for which to save the a full file of LST-gridded data. 
        One REDUCEDCHAN file is saved for each output file, corresponding to the
        first LST-bin in that file. The data in that file will have the shape 
        ``(Nbls*Ndays, Nsave_chans, Npols)``. This can be helpful for debugging.
    golden_lsts
        A list of LSTs for which to save the a full file of LST-gridded data.
        One GOLDEN file is saved for each ``golden_lst``, with shape ``(Nbls*Ndays, Nfreqs, Npols)``.
        This can be helpful for debugging.
    sigma_clip_thresh
        If provided, this is the threshold for sigma clipping. If this is provided,
        then the data is sigma clipped before being averaged. This is done separately
        for each baseline, frequency and polarization.
    sigma_clip_min_N
        The minimum number of points required to perform sigma clipping.
    redundantly_averaged
        If True, the input data is assumed to be redundantly averaged. By default
        the value is inferred by looking at metadata from the central file of each 
        night.
    blts_are_rectangular
        If True, the input data is assumed to be rectangular. By default the value
        is inferred by looking at metadata from the first file of the first night.
    time_axis_faster_than_bls
        If True, the input data is assumed to have the time axis faster than the
        baseline axis. By default the value is inferred by looking at metadata from
        the first file of the first night.
    freq_min, freq_max
        The min/max frequency to include in the output files. If None, use all 
        frequencies.

    Result
    ------
    zen.{pol}.LST.{file_lst}.uv : holds LST bin avg (data_array) and bin count (nsample_array)
    zen.{pol}.STD.{file_lst}.uv : holds LST bin stand dev along real and imag (data_array)
    
    Returns
    -------
    list of str
        list of output file paths for the LST binned data only (not the standard 
        deviation files or REDUCEDCHAN or GOLDENLST files).

    """
    with open(config_file, "r") as fl:
        configuration = yaml.safe_load(fl)
    
    config_opts = configuration['config_params']
    lst_grid = configuration['lst_grid']
    matched_files = configuration['matched_files']
    metadata = configuration['metadata']

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
        blts_are_rectangular=metadata['blts_are_rectangular'],
        time_axis_faster_than_bls=metadata['time_axis_faster_than_bls'],
    )

    antpos = dict(zip(meta.antenna_numbers, meta.antpos_enu))
    reds = RedundantGroups.from_antpos(
        antpos=antpos,
        include_autos=include_autos
    )

    output_files = []
    for outfile_index in output_file_select:
        data_files = matched_files[outfile_index]
        output_files.append(
            lst_bin_files_single_outfile(
                config_opts = config_opts,
                metadata = metadata,
                lst_bins = lst_grid[outfile_index],
                data_files = data_files,
                meta = meta,
                reds = reds,
                include_autos = include_autos,
                **kwargs
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
) -> tuple[set[tuple[int, int]], list[str]]:
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
        fl if isinstance(fl, FastUVH5Meta) else 
        FastUVH5Meta(
            fl, blts_are_rectangular=blts_are_rectangular, 
            time_axis_faster_than_bls=time_axis_faster_than_bls
        )  for fl in fl_list
        ] for fl_list in data_files
    ]
    

    all_baselines = set()
    all_pols = set()

    meta0 = data_files[0][0]
    x_orientation = meta0.get_transactional('x_orientation')

    # reds will contain all of the redundant groups for the whole array, because
    # all the antenna positions are included in every file.
    if reds is None:
        reds = RedundantGroups.from_antpos(
            antpos=dict(zip(meta0.antenna_numbers, meta0.antpos_enu)), include_autos=True
        )

    if redundantly_averaged is None:
        # Try to work out if the files are redundantly averaged.
        # just look at the middle file from each night.
        for fl_list in data_files:
            meta = fl_list[len(fl_list)//2]
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
                    (a1, a2) not in all_baselines and # Do this first because after the
                    (a2, a1) not in all_baselines and # first file it often triggers.
                    a1 not in ignore_ants and 
                    a2 not in ignore_ants and 
                    (include_autos or a1 != a2) and 
                    a1 not in a_priori_antenna_flags and 
                    a2 not in a_priori_antenna_flags
                ):
                    all_baselines.add((a1, a2))
                    

    return all_baselines, all_pols

def create_lstbin_output_file(
    outdir: Path,
    kind: str, 
    lst: float,
    pols: list[str],
    file_list: list[FastUVH5Meta],
    start_jd: float,
    times: np.ndarray | None = None,
    lsts: np.ndarray | None = None,
    history: str  = "",
    fname_format: str="zen.{kind}.{lst:7.5f}.uvh5", 
    overwrite: bool = False,
    antpairs: list[tuple[int, int]] | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
    channels: np.ndarray | list[int] | None = None
    
) -> Path:
    outdir = Path(outdir)
    # pols = set(pols)
    # update history
    file_list_str = "-".join(ff.path.name for ff in file_list)
    file_history = f"{history} Input files: {file_list_str}"
    _history = file_history + utils.history_string()

    fname = outdir / fname_format.format(kind=kind, lst=lst, pol=''.join(pols))

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
        freqs = freqs[channels]

    uvd_template = io.uvdata_from_fastuvh5(
        meta=file_list[0],
        antpairs=antpairs,
        pols=pols,
        lsts=lsts,
        times=times,
        history=_history,
        start_jd=start_jd,
        time_axis_faster_than_bls = True,
        vis_units="Jy",
    )
    uvd_template.select(frequencies=freqs, inplace=True)

    uvd_template.initialize_uvh5_file(str(fname.absolute()), clobber=overwrite)
    return fname

def write_baseline_slc_to_file(
    fl: Path, 
    slc: slice, 
    data: np.ndarray, 
    flags: np.ndarray, 
    nsamples: np.ndarray
):
    """Write a baseline slice to a file."""
    
    with h5py.File(fl, 'r+') as f:
        ntimes = int(f['Header']['Ntimes'][()])
        timefirst = bool(f['Header']['time_axis_faster_than_bls'][()])
        if not timefirst:
            raise NotImplementedError("Can only do time-first files for now.")
        
        slc = slice(slc.start*ntimes, slc.stop*ntimes, 1)
        f['Data']['visdata'][slc] = data.reshape((-1, data.shape[2], data.shape[3]))
        f['Data']['flags'][slc] = flags.reshape((-1, data.shape[2], data.shape[3]))
        f['Data']['nsamples'][slc] = nsamples.reshape((-1, data.shape[2], data.shape[3]))

def config_lst_bin_files(
    data_files: list[list[str | FastUVH5Meta]], 
    dlst: float | None=None, 
    atol: float=1e-10, 
    lst_start: float | None=None, 
    lst_width: float = 2*np.pi,
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
    dlst : float, LST bin width of output lst_grid
    file_lsts : list, contains the lst grid of each output file. Empty files are dropped.
    begin_lst : float, starting lst for LST binner. If lst_start is not None, this equals lst_start.
    lst_arrays : list, list of lst arrays for each file. These will have 2 pis added or subtracted
                 to match the range of lst_grid given lst_start
    time_arrays : list, list of time arrays for each file
    """
    logger.info("Configuring lst_grid")
    
    data_files=[sorted(df) for df in data_files]

    df0 = data_files[0][0]

    # get dlst from first data file if None
    if dlst is None:
        dlst = io.get_file_times(df0, filetype='uvh5')[0]

    # Get rectangularity of blts from first file if None
    meta = FastUVH5Meta(df0, blts_are_rectangular=blts_are_rectangular, time_axis_faster_than_bls=time_axis_faster_than_bls)
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
        ) for df in data_files
    ]

    # get begin_lst from lst_start or from the first JD in the data_files
    if lst_start is None:
        lst_start = np.min([ff.lsts[0] for ff in first_files])
    begin_lst = lst_start

    # make LST grid that divides into 2pi
    lst_grid = make_lst_grid(dlst, begin_lst=begin_lst, lst_width=lst_width)
    dlst = lst_grid[1] - lst_grid[0]

    lst_edges = np.concatenate([lst_grid - dlst/2, [lst_grid[-1] + dlst/2]])

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
            file_list = fllist,
            files_sorted=True,
            jd_regex=jd_regex,
            blts_are_rectangular=blts_are_rectangular,
            time_axis_faster_than_bls=time_axis_faster_than_bls,
            atol=atol
        )
        for i, m in enumerate(matched):
            matched_files[i].append(m)

    # Only keep output files that have data associated
    matched_files = [mf for mf in matched_files if any(len(mff)>0 for mff in mf)]
    return lst_grid, matched_files

def make_lst_bin_config_file(
    config_file: str | Path,
    data_files: list[list[str | FastUVH5Meta]],
    clobber: bool = False,
    dlst: float | None=None, 
    atol: float=1e-10, 
    lst_start: float | None=None, 
    lst_width: float = 2*np.pi,
    ntimes_per_file: int=60,
    blts_are_rectangular: bool | None = None,
    time_axis_faster_than_bls: bool | None = None,
    jd_regex: str = r"zen\.(\d+\.\d+)\.",
    lst_branch_cut: float | None=None,
):
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
    )

    nfiles = int(np.ceil(len(lst_grid) / ntimes_per_file))
    dlst = lst_grid[1] - lst_grid[0]
    lst_grid = [lst_grid[ntimes_per_file*i:ntimes_per_file*(i+1)] for i in range(nfiles)]
    # Make it a real list of floats to make the YAML easier to read
    lst_grid = [[float(l) for l in lsts] for lsts in lst_grid]

    # Set branch cut before trimming files -- want it to be the same for all files
    if lst_branch_cut is None:
        if lst_start is not None:
            lst_branch_cut = 0.0
        else:
            lst_branch_cut = lst_grid[0][0]

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
            'All integrations must be of equal length (BDA not supported), got diffs: '
            f'{np.diff(meta.times)}'
        )

    matched_files = [[[str(m.path) for m in night] for night in outfiles] for outfiles in matched_files]
    output = {
        'config_params': {
            'dlst': float(dlst),
            'atol': atol,
            'lst_start': lst_start,
            'lst_width': lst_width,
            'jd_regex': jd_regex,
            'lst_branch_cut': lst_branch_cut,
        },
        'lst_grid': lst_grid,
        'matched_files': matched_files,
        'metadata': {
            'x_orientation': meta.x_orientation,
            'blts_are_rectangular': meta.blts_are_rectangular,
            'time_axis_faster_than_bls': meta.time_axis_faster_than_bls,
            'start_jd': int(meta.times[0]),
            'integration_time': float(tint),
        }
    }

    with open(config_file, 'w') as fl:
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
    a.add_argument('configfile', type=str, help="config file produced by lstbin.make_lst_bin_config_file")
    a.add_argument(
        "--calfile-rules", nargs='*', type=str, 
        help="rules to convert datafile names to calfile names. A series of two strings where the first will be replaced by the latter"
    )
    a.add_argument("--fname-format", type=str, default="zen.{kind}.{lst:7.5f}.uvh5", help="filename format for output files. See docstring for details.")
    a.add_argument("--outdir", default=None, type=str, help="directory for writing output")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output files")
    a.add_argument("--rephase", default=False, action='store_true', help="rephase data to center of LST bin before binning")
    a.add_argument("--history", default=' ', type=str, help="history to insert into output files")
    a.add_argument("--output_file_select", default=None, nargs='*', type=int, help="list of output file integers to run on. Default is all output files.")
    a.add_argument("--vis_units", default='Jy', type=str, help="visibility units of output files.")
    a.add_argument("--ignore_flags", default=False, action='store_true', help="Ignore flags in data files, such that all input data is included in binning.")
    a.add_argument("--Nbls_to_load", default=None, type=int, help="Number of baselines to load and bin simultaneously. Default is all.")
    a.add_argument("--ex_ant_yaml_files", default=None, type=str, nargs='+', help="list of paths to yamls with lists of antennas from each night to exclude lstbinned data files.")
    a.add_argument("--ignore-ants", default=(), type=int, nargs='+', help='ants to ignore')
    a.add_argument("--ignore-missing-calfiles", default=False,action='store_true', help='if true, any datafile with missing calfile will just be removed from lstbinning.')
    a.add_argument("--write_kwargs", default='{}', type=str, help="json dictionary of arguments to the uvh5 writer")
    a.add_argument("--golden-lsts", type=str, help="LSTS (rad) to save longitudinal data for, separated by commas")
    a.add_argument("--save-channels", type=str, help="integer channels separated by commas to save longitudinal data for")
    a.add_argument("--sigma-clip-thresh", type=float, help="sigma clip threshold for flagging data in an LST bin over time. Zero means no clipping.", default=0.0)
    a.add_argument("--sigma-clip-min-N", type=int, help="number of unflagged data points over time to require before considering sigma clipping", default=4)
    a.add_argument("--redundantly-averaged", action='store_true', default=None, help="if true, assume input files are redundantly averaged")
    a.add_argument("--only-last-file-per-night", action='store_true', default=False, help="if true, only use the first and last file every night to obtain antpairs")
    a.add_argument("--freq-min", type=float, default=None, help="minimum frequency to include in lstbinning")
    a.add_argument("--freq-max", type=float, default=None, help="maximum frequency to include in lstbinning")
    return a