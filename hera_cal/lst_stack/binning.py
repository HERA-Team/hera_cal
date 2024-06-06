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
from functools import cached_property
from astropy import units

from pyuvdata import utils as uvutils
from .. import io
from ..datacontainer import DataContainer
from .. import apply_cal
from .config import LSTConfigSingle
logger = logging.getLogger(__name__)
from astropy.coordinates import EarthLocation
from ..utils import _comply_vispol


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
    lsts_in_bins
    """
    metas = [
        (
            fl
            if isinstance(fl, FastUVH5Meta)
            else FastUVH5Meta(fl, blts_are_rectangular=True)
        )
        for fl in data_files
    ]

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

    # This loop actually reads the associated data in this LST bin.
    ntimes_so_far = 0
    for meta, calfl, tind, inpfile in zip(
        metas, cal_files, time_idx, where_inpainted_files
    ):
        logger.info(f"Reading {meta.path}")
        slc = slice(ntimes_so_far, ntimes_so_far + len(tind))
        ntimes_so_far += len(tind)

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

        if not bls_to_load or len(tind) == 0:
            # If none of the requested baselines are in this file, then just
            # set stuff as nan and go to next file.
            logger.warning(f"None of the baseline-times are in {meta.path}. Skipping.")
            data[slc] = np.nan
            flags[slc] = True
            nsamples[slc] = 0
            continue

        # TODO: use Fast readers here instead, and select times directly on read.
        _data, _flags, _nsamples = io.HERAData(meta.path).read(
            bls=bls_to_load,
            freq_chans=freq_chans,
            polarizations=pols,
        )

        _data.select_or_expand_times(indices=tind, skip_bda_check=True)
        _flags.select_or_expand_times(indices=tind, skip_bda_check=True)
        _nsamples.select_or_expand_times(indices=tind, skip_bda_check=True)

        if inpfile is not None:
            # This returns a DataContainer (unless something went wrong) since it should
            # always be a 'baseline' type of UVFlag.
            inpainted = io.load_flags(inpfile)
            if not isinstance(inpainted, DataContainer):
                raise ValueError(f"Expected {inpfile} to be a DataContainer")

            # We need to down-selecton times/freqs (bls and pols will be sub-selected
            # based on those in the data through the next loop)
            inpainted.select_or_expand_times(indices=tind, skip_bda_check=True)
            inpainted.select_freqs(channels=freq_chans)
        else:
            inpainted = None

        # load calibration
        if calfl is not None:
            logger.info(f"Opening and applying {calfl}")
            uvc = io.to_HERACal(calfl)
            gains, cal_flags, _, _ = uvc.read(freq_chans=freq_chans)
            # down select times if necessary
            if len(tind) < uvc.Ntimes and uvc.Ntimes > 1:
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
                bls = reds.get_reds_in_bl_set(bl, _data.antpairs(), include_conj=True)
                if len(bls) > 1:
                    raise ValueError(
                        f"Expected only one baseline in group for {bl}, got {bls}"
                    )
                if bls:
                    # if there are no bls, just keep bl the same, and it won't be found,
                    # triggering the data to be filled with nans anyway.
                    bl = next(iter(bls))  # use next(iter) since bls is a set

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
        return utils.polnum2str(self.polarization_array, x_orientation=self.x_orientation)

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
        where_inpainted_files=config.inpaint_files
    )

    freqs, _ = _get_freqs_chans(meta.freq_array, freq_min, freq_max)

    out = []
    for (d, f, n, wf, bt) in zip(data, flags, nsamples, where_inpainted, binned_times):

        # To enable inpaint-mode, set nsamples where things are flagged and inpainted
        # to zero, and set the flags to false.
        if wf is not None:
            f[wf] = False
            n[wf] *= -1

        uv = UVData.new(
            freq_array=freqs,
            polarization_array=utils.polstr2num([_comply_vispol(p) for p in config.pols], x_orientation=meta.x_orientation),
            antenna_positions=meta.antenna_positions,
            telescope_location=EarthLocation.from_geocentric(*meta.telescope_location, unit="m"),
            telescope_name=meta.telescope_name,
            times=bt,
            antpairs=antpairs,
            do_blt_outer=True,
            integration_time=np.mean(meta.integration_time),
            antenna_names=meta.antenna_names,
            antenna_numbers=meta.antenna_numbers,
            blts_are_rectangular=True,
            data_array=d.reshape((-1, len(freqs), len(config.pols))),
            flag_array=f.reshape((-1, len(freqs), len(config.pols))),
            nsample_array=n.reshape((-1, len(freqs), len(config.pols))),
            vis_units="Jy",
            time_axis_faster_than_bls=False,
            x_orientation=meta.x_orientation,
        )

        # These can be removed in future pyuvdata versions where they are set automatically.
        uv.blts_are_rectangular = True
        uv.time_axis_faster_than_bls = False

        out.append(LSTStack(uv))
    return out
