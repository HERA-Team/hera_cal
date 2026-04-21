from __future__ import annotations
import numpy as np
from pathlib import Path
import warnings
import logging
from ..types import Antpair
from .. import utils
from typing import Sequence
from ..red_groups import RedundantGroups
from pyuvdata.uvdata import FastUVH5Meta
from pyuvdata import UVData, UVFlag
from functools import cached_property, reduce
from astropy import units

from pyuvdata import utils as uvutils
from pyuvdata.telescopes import Telescope
from .. import io
from ..datacontainer import DataContainer
from .. import apply_cal
from .config import LSTConfigSingle, LSTBinConfiguratorSingleBaseline
logger = logging.getLogger(__name__)
from astropy.coordinates import EarthLocation
from ..utils import _comply_vispol
from hera_qm.time_series_metrics import true_stretches

from concurrent.futures import ThreadPoolExecutor, as_completed


def adjust_lst_bin_edges(lst_bin_edges: np.ndarray) -> np.ndarray:
    """
    Adjust the LST bin edges so that they start in the range [0, 2pi) and increase.

    Performs the adjustment in-place.
    """
    if lst_bin_edges.ndim != 1:
        raise ValueError("lst_bin_edges must be a 1D array")

    if np.any(np.diff(lst_bin_edges) < 0):
        raise ValueError("lst_bin_edges must be monotonically increasing.")

    while lst_bin_edges[0] < 0:
        lst_bin_edges += 2 * np.pi
    while lst_bin_edges[0] >= 2 * np.pi:
        lst_bin_edges -= 2 * np.pi


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

    .. warning::
        When ``rephase=True``, the input ``data`` array is modified **in place** to
        save memory. Rows whose LSTs fall inside the LST range are
        overwritten with their rephased values; rows outside the LST range are left
        unchanged (their rephase shift is forced to 0). If the caller still needs
        the original unrephased data afterward, it must pass in a ``data.copy()``.
        The per-bin arrays returned in the output list are fancy-index copies of
        this (now rephased) master, so they are independent of it.
        ``flags``, ``nsamples``, and ``where_inpainted`` are **not** mutated.

    Parameters
    ----------
    data
        The complex visibility data. Must be shape ``(ntimes, nbls, nfreqs, npols)``,
        where the times may be sourced from multiple days. **Modified in place when
        ``rephase=True``** — see warning above.
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

        if freq_array is None or antpos is None:
            raise ValueError("freq_array and antpos is needed for rephase")

        bls = np.array([antpos[k[0]] - antpos[k[1]] for k in antpairs])

        # Rephase in-place on the full data array. This is slightly more CPU-intensive
        # than using data[lst_mask], but far less memory intensitive. Instead,
        # compute a full-length lst_shift with 0 for out-of-range rows so they're
        # untouched, and rephase the master in place. grid_indices from
        # get_lst_bins can be -1 or Nbins for out-of-range rows, so clip before
        # indexing lst_bin_centres (the shift value for those rows is overwritten
        # to 0 below, so the clipped lookup value is discarded).
        _clipped = np.clip(grid_indices, 0, len(lst_bin_centres) - 1)  # avoid index errors
        lst_shift = lst_bin_centres[_clipped] - data_lsts
        lst_shift[~lst_mask] = 0.0  # doesn't do any rephasing
        utils.lst_rephase(data, bls, freq_array, lst_shift, lat=lat, inplace=True)

    # In case we don't rephase, the data/flags/nsamples arrays are still the original
    # input arrays. We don't mask out the data outside the LST range, because we're
    # just going to omit it from our bins naturally anyway. We also don't care if its
    # not a copy here, because we're not going to modify it, and this saves memory.

    # We anyway end up with a ~full copy of the data in the output arrays, because
    # we do a fancy-index of the input arrays to get the relevant data for each bin.

    # shortcut -- just return all the data, re-organized.
    _data, _flags, _nsamples, _where_inpainted = [], [], [], []
    empty_shape = (0, len(antpairs), len(freq_array), npols)
    for lstbin in range(len(lst_bin_centres)):
        mask = (grid_indices == lstbin)
        if np.any(mask):
            _data.append(data[mask])
            _flags.append(flags[mask])
            _nsamples.append(nsamples[mask])
            if where_inpainted is not None:
                _where_inpainted.append(where_inpainted[mask])
            else:
                _where_inpainted.append(None)
        else:
            _data.append(np.zeros(empty_shape, complex))
            _flags.append(np.zeros(empty_shape, bool))
            _nsamples.append(np.zeros(empty_shape, int))
            if where_inpainted is not None:
                _where_inpainted.append(np.zeros(empty_shape, bool))
            else:
                _where_inpainted.append(None)

    return lst_bin_centres, _data, _flags, _nsamples, _where_inpainted


def _allocate_dfn(shape: tuple[int], d=0.0, f=0, n=0):
    data = np.full(shape, d, dtype=complex)
    flags = np.full(shape, f, dtype=bool)
    nsamples = np.full(shape, n, dtype=float)
    return data, flags, nsamples


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


def _get_freqs_chans(freqs, freq_min: float | None = None, freq_max: float | None = None):

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

    return freqs, freq_chans


def _read_one_file(
    meta_path: str,
    calfl: str | None,
    tind: np.ndarray,
    inpfile: str | None,
    antpairs: list[tuple],
    pols: list[str],
    freq_chans: np.ndarray | None,
    redundantly_averaged: bool,
    reds: RedundantGroups | None,
    cal_file_loader: callable | None,
    cal_file_loader_kwargs: dict | None,
    blts_are_rectangular: bool = True,
) -> dict:
    """Read visibility data from a single file and return it as a plain dict.

    Designed to be called from either a thread or a process pool.  All
    arguments must therefore be picklable (paths as strings, arrays as numpy,
    etc.) when used with ProcessPoolExecutor.

    Parameters
    ----------
    meta_path
        Path to the UVH5 file to read, as a plain string (not a Path object,
        for picklability with ProcessPoolExecutor).
    calfl
        Path to a calibration file to apply to the data, or ``None`` to skip
        calibration.
    tind
        Indices into the time axis of the file that fall within at least one
        LST bin.  Only these rows are read into memory.
    inpfile
        Path to a UVFlag file recording where the data have
        been inpainted, or ``None`` if no inpainting information is available.
    antpairs
        The antenna pairs (baselines) to load.  Pairs absent from this file
        are silently skipped; conjugate pairs are handled automatically.
    pols
        Polarization strings to read.
    freq_chans
        Channel indices to read, or ``None`` to read all channels.
    redundantly_averaged
        If ``True``, the file stores one row per unique-baseline group rather
        than one row per physical baseline.  The ``antpairs`` argument is then
        interpreted as unique-baseline keys, and ``reds`` is used to map them
        to whatever physical baseline is stored in the file.
    reds
        Redundant-baseline group information.  Required when
        ``redundantly_averaged`` is ``True``; ignored otherwise.
    cal_file_loader
        Optional callable for reading calibration solutions in a non-standard
        format.  Must accept ``(calfl, antpairs=..., polarizations=..., **kwargs)``
        and return ``(gains, cal_flags)``.  If ``None``, the default
        HERAData/HERACal readers are used.
    cal_file_loader_kwargs
        Extra keyword arguments forwarded to ``cal_file_loader``.  Pass
        ``None`` (or omit) when using the default loader.
    blts_are_rectangular
        Passed through to :class:`~pyuvdata.uvdata.FastUVH5Meta`; set to
        ``False`` only for files where baselines and times are not on a
        regular grid.

    Returns
    -------
    dict with keys:
        skip        bool: True when there is nothing useful in this file
        ntimes      int: number of time integrations in tind
        data        DataContainer or None
        flags       DataContainer or None
        nsamples    DataContainer or None
        inpainted   DataContainer or None
        bls_loaded  list of antpairs that were actually loaded
    """
    # Inspect file metadata (cheap; no I/O on the data arrays)
    meta = FastUVH5Meta(meta_path, blts_are_rectangular=blts_are_rectangular)
    data_antpairs = meta.get_transactional("antpairs")
    ntimes = len(tind)

    # Determine which baselines to actually read
    # For redundantly-averaged files we must map the requested unique-baseline
    # keys back to whichever physical baseline is stored in this file.
    if redundantly_averaged:
        bls_to_load = [
            bl for bl in data_antpairs
            if reds.get_ubl_key(bl) in antpairs
            or reds.get_ubl_key(bl[::-1]) in antpairs
        ]
    else:
        bls_to_load = [
            bl for bl in antpairs
            if bl in data_antpairs or bl[::-1] in data_antpairs
        ]

    if not bls_to_load or ntimes == 0:
        # If none of the requested baselines are in this file, then just
        # set stuff as nan and go to next file.
        logger.warning(
            f"None of the baseline-times are in {meta_path}. Skipping."
        )
        return {"skip": True, "ntimes": ntimes,
                "data": None, "flags": None, "nsamples": None,
                "inpainted": None, "bls_loaded": []}

    # Read visibility data
    logger.info(f"Reading {meta_path}")

    # TODO: use Fast readers here instead, and select times directly on read.
    _data, _flags, _nsamples = io.HERAData(meta_path).read(
        bls=bls_to_load,
        freq_chans=freq_chans,
        polarizations=pols,
    )

    # Trim to only the time indices that fall within an LST bin.
    _data.select_or_expand_times(indices=tind, skip_bda_check=True)
    _flags.select_or_expand_times(indices=tind, skip_bda_check=True)
    _nsamples.select_or_expand_times(indices=tind, skip_bda_check=True)

    # Load inpainting flags (optional)
    inpainted = None
    if inpfile is not None:
        # This returns a DataContainer (unless something went wrong) since it should
        # always be a 'baseline' type of UVFlag.
        inpainted = io.load_flags(inpfile)
        if not isinstance(inpainted, DataContainer):
            raise ValueError(f"Expected {inpfile} to be a DataContainer")

        # We need to down-select on times/freqs (bls and pols will be sub-selected
        # based on those in the data through the next loop)
        inpainted.select_or_expand_times(indices=tind, skip_bda_check=True)
        inpainted.select_freqs(channels=freq_chans)

    # Load and apply calibration (optional)
    if calfl is not None:
        logger.info(f"Opening and applying {calfl}")
        if cal_file_loader is not None:
            # Use the custom loader to read the calibration solutions. This is useful if the
            # calibration files are in a different format than HERACal files, or if the user wants
            # to apply some custom pre-processing to the calibration solutions as they are read in.
            gains, cal_flags = cal_file_loader(
                calfl,
                antpairs=bls_to_load,
                polarizations=pols,
                **(cal_file_loader_kwargs or {}),
            )
            gain_convention = "divide"
        else:
            uvc = io.to_HERACal(calfl)
            gains, cal_flags, _, _ = uvc.read(freq_chans=freq_chans)
            if len(tind) < uvc.Ntimes and uvc.Ntimes > 1:
                uvc.select(times=uvc.time_array[tind])
                gains, cal_flags, _, _ = uvc.build_calcontainers()
            gain_convention = uvc.gain_convention

        apply_cal.calibrate_in_place(
            _data,
            gains,
            data_flags=_flags,
            cal_flags=cal_flags,
            gain_convention=gain_convention,
        )

    return {
        "skip": False,
        "ntimes": ntimes,
        "data": _data,
        "flags": _flags,
        "nsamples": _nsamples,
        "inpainted": inpainted,
        "bls_loaded": bls_to_load,
    }


def lst_bin_files_for_baselines(
    data_files: list[Path | FastUVH5Meta],
    lst_bin_edges: np.ndarray,
    antpairs: Sequence[tuple[int, int]],
    freqs: np.ndarray | None = None,
    pols: np.ndarray | None = None,
    cal_files: list[Path | None] | None = None,
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
    cal_file_loader: callable | None = None,
    cal_file_loader_kwargs: dict | None = None,
    blts_are_rectangular: bool = True,
    n_workers: int = 1,
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
    redundantly_averaged
        If True, the data files are assumed to have already been averaged in time and
        redundant baseline groups, so that each row in the data corresponds to a unique
        redundant baseline group, rather than a unique baseline. In this case, the
        ``antpairs`` argument is interpreted as a list of redundant baseline groups to
        read, rather than a list of actual antenna pairs. If True, the ``reds`` argument
        must be provided, and the function will automatically map the redundant baseline
        groups in ``antpairs`` to the actual baselines in the data files using the
        redundant groups information in ``reds``.
    freq_min, freq_max
        Minimum and maximum frequencies to include in the data. If not provided,
        all frequencies will be included.
    where_inpainted_files
        A list of lists of strings, one for each file, where each file is a UVFlag file
        specifying which data are in-painted. If not provided, no inpainting will be
        assumed.
    cal_file_loader
        A callable that takes a calibration file path, a list of baselines, and a list
        of polarizations, and returns the corresponding calibration solutions. If
        not provided, will use the default HERAData/HERACal readers to read the
        calibration solutions. Useful if the calibration files are in a different
        format than HERACal files.
    cal_file_loader_kwargs
        A dictionary of keyword arguments to pass to ``cal_file_loader``.
    blts_are_rectangular: bool
        Whether to assume that the blt axis of the input files is rectangular (i.e. that
        all baselines have the same time samples).
    n_workers : int, optional
        Number of parallel workers to use when reading files. ``1`` (the default)
        reproduces the original serial behaviour exactly. ``n_workers`` must be a
        positive integer (``>= 1``); passing ``0`` or a negative value is invalid and
        will result in a ``ValueError``. Values greater than 1 submit each file read
        to a thread pool so that multiple nights can be read concurrently.

        A sensible effective upper bound is ``min(n_workers, len(data_files))``,
        which is applied internally.

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
    lsts_in_bins
    """
    metas = [
        (
            fl
            if isinstance(fl, FastUVH5Meta)
            else FastUVH5Meta(fl, blts_are_rectangular=blts_are_rectangular)
        )
        for fl in data_files
    ]

    # Make sure n_workers is a positive integer
    if n_workers < 1 or not isinstance(n_workers, int):
        raise ValueError(f"n_workers must be a positive integer, got {n_workers}")

    lst_bin_edges = np.array(lst_bin_edges)

    if freqs is None:
        freqs = np.squeeze(metas[0].freq_array)

    freqs, freq_chans = _get_freqs_chans(freqs, freq_min, freq_max)

    if pols is None:
        pols = metas[0].pols
    elif not all(isinstance(p, str) for p in pols):
        pols = uvutils.polnum2str(pols, x_orientation=metas[0].x_orientation)

    if antpos is None and rephase:
        warnings.warn(
            "Getting antpos from the first file only. This is almost always correct, "
            "but will be wrong if different files have different antenna_position arrays."
        )
        antpos = dict(zip(metas[0].antenna_numbers, metas[0].antpos_enu))

    # Add LST filtering info to cal_file_loader_kwargs if using custom loader
    if cal_file_loader is not None:
        if cal_file_loader_kwargs is None:
            cal_file_loader_kwargs = {}

        # Add LST bin edges if not already present
        if 'lst_bin_edges' not in cal_file_loader_kwargs:
            cal_file_loader_kwargs['lst_bin_edges'] = lst_bin_edges

        # Add telescope location if not already present
        if 'telescope_location_lat_lon_alt_degrees' not in cal_file_loader_kwargs:
            cal_file_loader_kwargs['telescope_location_lat_lon_alt_degrees'] = metas[0].telescope_location_lat_lon_alt_degrees

    if time_idx is None:
        adjust_lst_bin_edges(lst_bin_edges)
        lst_bin_edges %= 2 * np.pi
        op = np.logical_and if lst_bin_edges[0] < lst_bin_edges[-1] else np.logical_or
        time_idx = []
        for meta in metas:
            _lsts = meta.get_transactional("lsts")
            time_idx.append(
                np.argwhere(
                    op(_lsts >= lst_bin_edges[0], _lsts < lst_bin_edges[-1])
                ).flatten()
            )

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

    # ── build per-file argument tuples (shared by serial and parallel paths) ─
    file_args = []
    for meta, calfl, tind, inpfile in zip(metas, cal_files, time_idx, where_inpainted_files):
        file_args.append(
            (
                str(meta.path),
                calfl,
                tind,
                inpfile,
                list(antpairs),       # plain list for picklability
                list(pols),
                freq_chans,
                redundantly_averaged,
                reds,
                cal_file_loader,
                # make per-task copy so worker threads don't share mutable kwargs
                dict(cal_file_loader_kwargs) if cal_file_loader_kwargs is not None else None,
                blts_are_rectangular,
            )
        )

    # ── helper: write one file's result into the master arrays ───────────────
    def _fill_master_arrays(res: dict, slc: slice) -> None:
        """Copy a single file's result dict into the pre-allocated master arrays.

        ``slc`` is the row slice in the time axis of the master arrays that
        corresponds to this file.  Results can therefore be written in any
        order, which is what allows the parallel path to fill as each future
        completes rather than waiting for all files to finish first.

        Parameters
        ----------
        res
            The dict returned by :func:`_read_one_file`.
        slc
            The slice into the time axis of the master arrays for this file,
            pre-computed from each file's ``time_idx`` length.
        """
        if res["skip"]:
            # File had no usable data; fill its time rows with flagged nans.
            data[slc] = np.nan
            flags[slc] = True
            nsamples[slc] = 0
            return

        _data = res["data"]
        _flags = res["flags"]
        _nsamples = res["nsamples"]
        inpainted = res["inpainted"]

        for i, bl in enumerate(antpairs):
            _bl = bl  # may be remapped below for redundantly_averaged

            # For redundantly-averaged data, find which stored physical baseline
            # represents this unique-baseline group in the current file.
            if redundantly_averaged:
                bls = reds.get_reds_in_bl_set(bl, _data.antpairs(), include_conj=True)
                if len(bls) > 1:
                    raise ValueError(
                        f"Expected only one baseline in group for {bl}, got {bls}"
                    )
                if bls:
                    # if there are no bls, just keep bl the same, and it won't be found,
                    # triggering the data to be filled with nans anyway
                    _bl = next(iter(bls))  # use next(iter) since bls is a set

            for j, pol in enumerate(pols):
                blpol = _bl + (pol,)

                if blpol in _data:
                    # Baseline found: copy data, flags, nsamples into master arrays.
                    data[slc, i, :, j] = _data[blpol]
                    flags[slc, i, :, j] = _flags[blpol]
                    nsamples[slc, i, :, j] = _nsamples[blpol]

                    if inpainted is not None and where_inpainted is not None:
                        # Get the representative baseline key from this bl group that
                        # exists in the where_inpainted data.
                        inpblpol = blpol
                        if redundantly_averaged:
                            for inpbl in reds[bl]:
                                if inpbl + (pol,) in inpainted:
                                    inpblpol = inpbl + (pol,)
                                    break
                            else:
                                raise ValueError(
                                    f"Could not find any baseline from group {bl} "
                                    "in inpainted file"
                                )
                        where_inpainted[slc, i, :, j] = inpainted[inpblpol]
                else:
                    # This baseline+pol doesn't exist in this file. That's
                    # OK, we don't assume all baselines are in every file.
                    data[slc, i, :, j] = np.nan
                    flags[slc, i, :, j] = True
                    nsamples[slc, i, :, j] = 0

    # Precompute each file's time slice so results can be filled in any order.
    ntimes_offsets = np.cumsum([0] + [len(t) for t in time_idx])
    file_slices = [
        slice(int(ntimes_offsets[i]), int(ntimes_offsets[i + 1]))
        for i in range(len(file_args))
    ]

    # dispatch: serial (n_workers==1) or parallel
    if n_workers == 1:
        # Call worker directly and fill immediately -- no results list needed.
        for i, args in enumerate(file_args):
            _fill_master_arrays(_read_one_file(*args), file_slices[i])
    else:
        n_workers = min(n_workers, len(metas))   # never spin up more than needed
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {
                pool.submit(_read_one_file, *args): i
                for i, args in enumerate(file_args)
            }
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                # Fill immediately and let res go out of scope so the GC can
                # reclaim each file's DataContainers as soon as they are consumed,
                # rather than holding all results in memory until the loop ends.
                _fill_master_arrays(fut.result(), file_slices[i])

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
    times = np.concatenate([
        meta.get_transactional("times")[idx] for meta, idx in zip(metas, time_idx)
    ])

    times_in_bins = []
    lsts_in_bins = []
    for i in range(len(bin_lst)):
        mask = bins == i
        times_in_bins.append(times[mask])
        lsts_in_bins.append(lsts[mask])

    return bin_lst, data, flags, nsamples, where_inpainted, times_in_bins, lsts_in_bins


class SingleBaselineStacker:
    """Class to hold multi-night single-baseline data.

    Wraps around ``lst_stack.binning.lst_bin_files_for_baselines()`` and loads
    single baseline data from multiple nights and stores it internally.

    It also provides a method average_over_nights() to average the data across nights, accounting for
    flags, nsamples, and where the data was inpainted. In general, where_inpainted data is not flagged, but rather
    it *was* flagged at some point previously and has since been inpainted, necessitating special bookkeeping.
    """

    # lists of numpy arrays whose length is the number of LST bins and whose 0th dimension is the number of nights
    _list_objects = ('data', 'flags', 'nsamples', 'where_inpainted', 'times_in_bins', 'lsts_in_bins')

    def __init__(self,
            *,
            bin_lst: np.ndarray,
            data: list[np.ndarray],
            flags: list[np.ndarray],
            nsamples: list[np.ndarray],
            where_inpainted: list[np.ndarray | None],
            times_in_bins: list[np.ndarray],
            lsts_in_bins: list[np.ndarray],
            bl_str: str | None = None,
            configurator: LSTBinConfiguratorSingleBaseline | None = None,
            lst_branch_cut: float | None = None,
            slice_kept: slice | None = None,
            hd: io.HERAData | None = None,
    ):
        """Creates a SingleBaselineStacker object with all of the parameters saved to self.[parameter_name].
        Typically, one uses the class method from_configurator() to create an instance of this class."""
        # store exactly what we're given
        self.bin_lst = bin_lst
        self.data = data
        self.flags = flags
        self.nsamples = nsamples
        self.where_inpainted = where_inpainted
        self.times_in_bins = times_in_bins
        self.lsts_in_bins = lsts_in_bins
        self.bl_str = bl_str
        self.configurator = configurator
        self.lst_branch_cut = lst_branch_cut
        self.slice_kept = slice_kept
        self.hd = hd

    @classmethod
    def from_configurator(cls,
            configurator: LSTBinConfiguratorSingleBaseline,
            bl_str: str,
            lst_bin_edges: np.ndarray,
            lst_branch_cut: float | None = None,
            where_inpainted_file_rules: list[list[str]] | None = None,
            to_keep_slice: slice | None = None,
            cal_file_loader: callable | None = None,
            cal_file_loader_kwargs: dict | None = None,
            n_workers: int = 1,
            pols: list[str] | None = None,
        ) -> SingleBaselineStacker:
        """Creates a SingleBaselineStacker object that loads data for a single baseline, optionally rolls to start after a branch cut,
        and removes any times at the beginning or end of the data set that have no data.

        Data are stored internally as lists of numpy arrays, just as lst_stack.binning.lst_bin_files_for_baselines() returns.
        - self.data: list of length Nlst, each element is complex and shape (Nnights, Nfreqs, Npols)
        - self.flags: list of length Nlst, each element is bool and shape (Nnights, Nfreqs, Npols)
        - self.nsamples: list of length Nlst, each element is float and shape (Nnights, Nfreqs, Npols)
        - self.where_inpainted: list of length Nlst, each element is bool and shape (Nnights, Nfreqs, Npols) or None
        - self.times_in_bins: list of length Nlst, each element is float and shape (Nnights,)
        - self.lsts_in_bins: list of length Nlst, each element is float and shape (Nnights,)
        - self.bin_lst: array of shape (Nlst,) containing the LST bin centres

        Additionally, the following parameters are stored:
        - self.bl_str: the baseline string used to load the data, e.g., '0_4'
        - self.configurator: the configurator object that contains the mapping from baseline strings to data files
        - self.lst_branch_cut: the LST branch cut in radians, if provided
        - self.hd: the HERAData object for the last file loaded, which contains metadata like antenna positions, frequencies, etc.

        Parameters
        ----------
        configurator : LSTBinConfiguratorSingleBaseline
            The configurator object that contains the mapping from baseline strings to data files.
        bl_str : str
            The baseline string in the filenames of data to load, e.g., '0_4'.
        lst_bin_edges : np.ndarray
            Array of LST bin edges (should be one more than the number of bins).
        lst_branch_cut : float | None
            If provided, the LSTs will be rolled to start after this branch cut (in radians).
        where_inpainted_file_rules : list[list[str]] | None
            If provided, a list of pairs of strings that will be used to replace parts of the filenames to find
            the "where inpainted" files, UVFlag files that record where that data was previously inpainted.
        to_keep_slice : slice | None
            For advanced users only. Typically, times are removed at the beginning and end of the data set if they have no data.
            This option allows that behavior to be overridden with an explicit slice into an array of length len(lst_bin_edges) - 1.
        cal_file_loader : callable | None
            A callable that takes a calibration file path, a list of baselines, and a list
            of polarizations, and returns the corresponding calibration solutions. If
            not provided, will use the default HERAData/HERACal readers to read the
            calibration solutions. Useful if the calibration files are in a different
            format than HERACal files.
        cal_file_loader_kwargs : dict | None
            A dictionary of keyword arguments to pass to ``cal_file_loader``.
        n_workers : int
            Number of parallel workers to use when reading files. ``1`` (the default) reproduces the original serial behaviour exactly. ``n_workers`` must be a
            positive integer (``>= 1``); passing ``0`` or a negative value is invalid and will result in a ``ValueError``. Values greater than 1 submit each file read to a thread pool so
            that multiple nights can be read concurrently.
        pols : list[str] | None
            If provided, only these polarizations will be loaded (e.g., ``['ee']`` to load a single pol).
            If ``None`` (default), all polarizations present in the files are loaded. The returned
            ``self.hd`` will have its polarization metadata modified as well.
        """
        # Load the data
        files_here = configurator.bl_to_file_map[bl_str]

        # Load the cal files if they exist
        if hasattr(configurator, 'visfile_to_calfile_map'):
            cal_files = [
                configurator.visfile_to_calfile_map[visfile]
                for visfile in configurator.bl_to_file_map[bl_str]
            ]
        else:
            cal_files = None

        where_inpainted_files = ([reduce(lambda txt, pair: txt.replace(*pair), where_inpainted_file_rules, df) for df in files_here]
                                 if where_inpainted_file_rules is not None else None)
        hd = io.HERAData(files_here[-1])
        if pols is not None:
            hd.select(polarizations=pols)  # keep hd metadata consistent with loaded subset
            hd._attach_metadata()  # refresh HERAData-specific metadata, inlcuding hd.pols
        (bin_lst,
         data,
         flags,
         nsamples,
         where_inpainted,
         times_in_bins,
         lsts_in_bins) = lst_bin_files_for_baselines(
            data_files=files_here,
            lst_bin_edges=lst_bin_edges,
            antpairs=hd.antpairs,
            freqs=hd.freqs,
            pols=hd.pols,
            rephase=True,
            where_inpainted_files=where_inpainted_files,
            cal_files=cal_files,
            cal_file_loader=cal_file_loader,
            cal_file_loader_kwargs=cal_file_loader_kwargs,
            n_workers=n_workers,
         )

        # Cut out baseline dimension
        for list_obj in (data, flags, nsamples, where_inpainted):
            for j, arr in enumerate(list_obj):
                if arr is not None and arr.ndim == 4:
                    list_obj[j] = arr[:, 0, :, :].copy()

        # Roll the lists to the branch cut
        if lst_branch_cut is not None:
            cls._roll_lists_to_lst_branch_cut(lst_branch_cut, bin_lst, lsts_in_bins, data,
                                              flags, nsamples, where_inpainted, times_in_bins)

        # Figure out which times to keep and remove others
        if to_keep_slice is None:
            lst_has_data = np.array([~np.all(f) & (len(f) > 0) for f in flags])
            ts = true_stretches(~lst_has_data)
            start = ts[0].stop if (len(ts) and ts[0].start == 0) else 0
            stop = ts[-1].start if (len(ts) and ts[-1].stop == len(flags)) else len(flags)
            to_keep_slice = slice(start, stop)
        bin_lst, data, flags, nsamples, where_inpainted, times_in_bins, lsts_in_bins = (
            obj[to_keep_slice] for obj in (bin_lst, data, flags, nsamples, where_inpainted, times_in_bins, lsts_in_bins))

        # Create the SingleBaselineStacker object
        return cls(bin_lst=bin_lst,
                   data=data,
                   flags=flags,
                   nsamples=nsamples,
                   where_inpainted=where_inpainted,
                   times_in_bins=times_in_bins,
                   lsts_in_bins=lsts_in_bins,
                   bl_str=bl_str,
                   configurator=configurator,
                   lst_branch_cut=lst_branch_cut,
                   slice_kept=to_keep_slice,
                   hd=hd)

    @staticmethod
    def _roll_lists_to_lst_branch_cut(lst_branch_cut: float,
                                      bin_lst: np.ndarray,
                                      lsts_in_bins: list[np.ndarray],
                                      *list_objects: list[np.ndarray]
                                      ) -> None:
        '''Rolls the lists to the branch cut defined by lst_branch_cut. This is done in place,
        so the lists are modified directly. LSTs after the branch cut are adjusted by adding 2 pi.'''
        branch_cut_idx = np.searchsorted(bin_lst, lst_branch_cut)

        # Roll all internal arrays.
        bin_lst[:] = np.roll(bin_lst, -branch_cut_idx)  # modified in place
        lsts_in_bins[:] = lsts_in_bins[branch_cut_idx:] + lsts_in_bins[0:branch_cut_idx]  # modified in place
        for list_obj in list_objects:
            list_obj[:] = list_obj[branch_cut_idx:] + list_obj[0:branch_cut_idx]  # modified in place

        # adds 2 pi to lsts after the branch cut
        bin_lst[bin_lst < lst_branch_cut] += 2 * np.pi
        lsts_in_bins[:] = [np.where(lst < lst_branch_cut, lst + 2 * np.pi, lst) for lst in lsts_in_bins]

    def average_over_nights(self, inpainted_data_are_samples: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute nightly averaged data, flags, and nsamples. Data is averaged with nsamples as weights
        (or 0 if flagged), regardless of whether the data is inpainted. Flags are ANDed across nights.
        Nsamples are summed across nights, but inpainted data is not counted as samples (by default).

        Wherever self.where_inpainted is None, it is assumed that the data was not inpainted.

        Parameters
        ----------
        inpainted_data_are_samples : bool
            If False (default), inpainted data is not counted as samples in the output lst_avg_nsamples. They
            are still counted as samples when computing the weight with which to average data across nights.

        Returns
        -------
        lst_avg_data : np.ndarray
            The averaged data, shape (Nlst, Nfreqs, Npols)
        lst_avg_flags : np.ndarray
            The flags for the averaged data, shape (Nlst, Nfreqs, Npols). N.B. that inpainted data is not flagged.
        lst_avg_nsamples : np.ndarray
            The number of unflagged samples going into each data point, shape (Nlst, Nfreqs, Npols)
        """
        # Initialize empty arrays to hold the results
        lst_avg_data = np.zeros((len(self.bin_lst), len(self.hd.freqs), len(self.hd.pols)), dtype=complex)
        lst_avg_flags = np.ones((len(self.bin_lst), len(self.hd.freqs), len(self.hd.pols)), dtype=bool)
        lst_avg_nsamples = np.zeros((len(self.bin_lst), len(self.hd.freqs), len(self.hd.pols)), dtype=float)

        for lidx, (d, f, n, wip) in enumerate(zip(self.data, self.flags, self.nsamples, self.where_inpainted)):

            # If no data for this LST bin, continue, leaving the data 0, the flags True, and nsamples 0
            if d.shape[0] == 0:
                continue

            # If there's no information about inpainting, assume no inpainting
            if wip is None:
                wip = np.zeros_like(f, dtype=bool)

            # flag if all nights are flagged
            lst_avg_flags[lidx] = np.all(f, axis=0)

            # compute weights as flagged nsamples
            weights = np.where(f, 0, n)
            # set weights to 1 where it'd be flagged over all nights so that there's no issue with averaging
            for pidx in range(d.shape[-1]):
                weights[:, lst_avg_flags[lidx, :, pidx], pidx] = 1

            # compute average data, setting flagged data to 0 rather than np.nan since its weight is 0
            lst_avg_data[lidx] = np.average(np.where(f, 0, d), axis=0, weights=weights)

            # compute nsamples, where flagged (or inpainted) data are set to 0
            if inpainted_data_are_samples:
                lst_avg_nsamples[lidx] = np.sum(np.where(f, 0, n), axis=0)
            else:
                lst_avg_nsamples[lidx] = np.sum(np.where(f | wip, 0, n), axis=0)

        return lst_avg_data, lst_avg_flags, lst_avg_nsamples


class LSTStack:
    """A very simple validation layer on top of UVData for LST-stacked data."""
    def __init__(self, uvd: UVData | UVFlag):
        self._uvd = uvd
        self._validate_uvd()

    def _validate_uvd(self):
        if isinstance(self._uvd, UVData):
            if not self._uvd.blts_are_rectangular:
                raise ValueError("blts_are_rectangular must be True")

            if self._uvd.time_axis_faster_than_bls:
                raise ValueError("time_axis_faster_than_bls must be False")
        elif isinstance(self._uvd, UVFlag):
            # Here, for now we must _assume_ that the blts are rectangular and
            # that the time axis is the outer axis. This is because we don't have
            # a way to check this in UVFlag objects (yet)
            if self._uvd.type != "baseline":
                raise ValueError("UVFlag type must be 'baseline'")

    def __getattr__(self, item):
        return getattr(self._uvd, item)

    def __setattr__(self, key, value):
        if key == "_uvd":
            super().__setattr__(key, value)

        setattr(self._uvd, key, value)

    @cached_property
    def dt(self) -> units.Quantity[units.s]:
        """The median integration time of the data."""
        return np.median(self.integration_time) * units.s

    @cached_property
    def df(self) -> units.Quantity[units.Hz]:
        """The median frequency resolution of the data."""
        return np.median(np.diff(self.freq_array)) * units.Hz

    @property
    def data(self) -> np.ndarray:
        """A view into the data array, reshaped to (Nbls, Ntimes, Nfreqs, Npols)."""
        return self._uvd.data_array.reshape(
            (self.Ntimes, self.Nbls, len(self.freq_array), len(self.polarization_array))
        )

    @property
    def nsamples(self) -> np.ndarray:
        """A view into the nsamples array, reshaped to (Nbls, Ntimes, Nfreqs, Npols)."""
        return self._uvd.nsample_array.reshape(
            (self.Ntimes, self.Nbls, len(self.freq_array), len(self.polarization_array))
        )

    @property
    def flags(self) -> np.ndarray:
        """A view into the flags array, reshaped to (Nbls, Ntimes, Nfreqs, Npols)."""
        return self._uvd.flag_array.reshape(
            (self.Ntimes, self.Nbls, len(self.freq_array), len(self.polarization_array))
        )

    def inpainted(self) -> np.ndarray:
        """Flags representing data that is inpainted."""
        return self.nsamples <= 0

    def flagged_or_inpainted(self):
        """Flags representing data that is flagged or inpainted."""
        return self.flags | self.inpainted()

    @property
    def metrics(self) -> np.ndarray:
        """A view into the flags array, reshaped to (Nbls, Ntimes, Nfreqs, Npols)."""
        return self._uvd.metric_array.reshape(
            (self.Ntimes, self.Nbls, len(self.freq_array), len(self.polarization_array))
        )

    @property
    def times(self) -> np.ndarray:
        """The unique times of the data (same shape as first axis of ``data``)."""
        return self._uvd.time_array[::self.Nbls]

    @property
    def nights(self) -> np.ndarray:
        """The nights in the data as integer JDs"""
        return self.times.astype(int)

    @property
    def antpairs(self) -> list[Antpair]:
        """The antenna pairs in the data."""
        return list(zip(self.ant_1_array[:self.Nbls], self.ant_2_array[:self.Nbls]))

    @property
    def pols(self) -> list[str]:
        """The polarizations in the data."""
        return utils.polnum2str(self.polarization_array, x_orientation=self.telescope.get_x_orientation_from_feeds())

    def copy(self, *args, **kwargs):
        """Return a copy of the LSTStack object."""
        return LSTStack(self._uvd.copy(*args, **kwargs))


def lst_bin_files_from_config(
    config: LSTConfigSingle,
    bl_chunk_to_load: int | str = 0,
    nbl_chunks: int = 1,
    rephase: bool = True,
    freq_min: float | None = None,
    freq_max: float | None = None,
    n_workers: int = 1,
) -> list[LSTStack | None] | None:
    """Read and LST-bin data from a configuration object.

    This function is the main entry point for binning (not averaging) data into LST
    bins, given a :class:`LSTConfigSingle` object, which is the intended mode of
    operation of the `lststack` subpackage.

    Parameters
    ----------
    config : LSTConfigSingle
        The configuration object to read data from.
    bl_chunk_to_load : int or str, optional
        The chunk of baselines to load. If 'autos', will load only the autos. If an
        integer, will load the nth chunk of baselines, where the number of chunks
        is defined by ``nbl_chunks``. Default is 0.
    nbl_chunks : int, optional
        The number of chunks to split the baselines into. Default is 1. Use more chunks
        to reduce memory usage.
    rephase : bool, optional
        Whether to rephase the data to the LST bin centres. Default is True.
    freq_min : float, optional
        The minimum frequency to include in the data (Hz). Default is all frequencies.
    freq_max : float, optional
        The maximum frequency to include in the data (Hz). Default is all frequencies.
    n_workers : int, optional
        Number of parallel workers for concurrent file reads. Default is 1 (serial).
        Passed through to :func:`lst_bin_files_for_baselines`.

    Returns
    -------
    list[LSTStack] or None
        A list of LSTStack objects, one for each LST bin. If there is no data to read,
        returns None. The LSTStack object looks and feels just like a UVData object, but
        has some additional properties and methods that are useful for LST-stacked data,
        as well as validating that the data is in the correct format.

        In particular, the LSTStack object has "rectangular" baselines and times (i.e.
        at each time, the same set of baselines are present), and the time axis is
        slower than the baseline axis (i.e the data has virtual shape
        ``(Nnights, Nbls, Nfreqs, Npols)``). Attributes on the stack that are extra to
        base UVData are ``data``, ``nsamples`` and ``flags`` -- all of which are simply
        views into their UVData counterparts (e.g. ``data_array``), but where the
        baseline and time axis are explicitly split.
    """
    if not config.matched_files:
        # An empty list of files means there's no data to read for this outfile
        return None

    # get metadata
    meta = config.config.datameta

    # Split up the baselines into chunks that will be LST-binned together.
    # This is just to save on RAM.
    if bl_chunk_to_load == "autos":
        antpairs = config.autopairs
    else:
        nbls_to_load = int(np.ceil(len(config.antpairs) / nbl_chunks))
        antpairs = config.antpairs[nbls_to_load * bl_chunk_to_load: nbls_to_load * (bl_chunk_to_load + 1)]

    all_lsts = np.concatenate(config.get_lsts())

    _, data, flags, nsamples, where_inpainted, binned_times, binned_lsts = lst_bin_files_for_baselines(
        antpairs=antpairs,
        data_files=config.matched_files,
        lst_bin_edges=config.lst_grid_edges,
        freqs=meta.freq_array,
        pols=config.pols,
        cal_files=config.calfiles,
        time_idx=config.time_indices,
        ignore_flags=False,
        rephase=rephase,
        antpos=config.config.reds.antpos,
        lsts=all_lsts,
        redundantly_averaged=config.config.is_redundantly_averaged,
        reds=config.config.reds,
        freq_min=freq_min,
        freq_max=freq_max,
        where_inpainted_files=config.inpaint_files,
        n_workers=n_workers,
    )

    freqs, _ = _get_freqs_chans(meta.freq_array, freq_min, freq_max)

    out = []

    # mount_type didn't always exist so older uvh5 files don't have it and will
    # error.
    try:
        mount_type = meta.mount_type
    except KeyError:
        mount_type = None

    for (d, f, n, wf, bt) in zip(data, flags, nsamples, where_inpainted, binned_times):

        # To enable inpaint-mode, set nsamples where things are flagged and inpainted
        # to zero, and set the flags to false.
        if wf is not None:
            f[wf] = False
            n[wf] *= -1

        telescope = Telescope.new(
            location=EarthLocation.from_geocentric(*meta.telescope_location, unit="m"),
            name=meta.telescope_name,
            antenna_names=meta.antenna_names,
            antenna_numbers=meta.antenna_numbers,
            antenna_positions=meta.antenna_positions,
            instrument=meta.instrument,
            mount_type=mount_type,
            antenna_diameters=meta.antenna_diameters
        )
        telescope.set_feeds_from_x_orientation(meta.x_orientation, feeds=['x', 'y'])  # assumes linear polarization
        uv = UVData.new(
            freq_array=freqs,
            polarization_array=utils.polstr2num([_comply_vispol(p) for p in config.pols], x_orientation=meta.x_orientation),
            times=bt,
            antpairs=antpairs,
            do_blt_outer=True,
            integration_time=np.mean(meta.integration_time),
            telescope=telescope,
            blts_are_rectangular=True,
            data_array=d.reshape((-1, len(freqs), len(config.pols))),
            flag_array=f.reshape((-1, len(freqs), len(config.pols))),
            nsample_array=n.reshape((-1, len(freqs), len(config.pols))),
            vis_units="Jy",
            time_axis_faster_than_bls=False,
        )

        # These can be removed in future pyuvdata versions where they are set automatically.
        uv.blts_are_rectangular = True
        uv.time_axis_faster_than_bls = False

        out.append(LSTStack(uv))
    return out
