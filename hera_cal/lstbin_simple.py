"""
An attempt at a simpler LST binner that makes more assumptions but runs faster.

In particular, we assume that all baselines have the same time array and frequency array,
and that each is present throughout the data array. This allows a vectorization.
"""
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
            f"flags should have shape {data.sahpe} but got {flags.shape}"
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
    while lst_bin_edges[0] < 0:
        lst_bin_edges += 2*np.pi
    while lst_bin_edges[0] >= 2*np.pi:
        lst_bin_edges -= 2*np.pi

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

def get_lst_bins(lsts: np.ndarray, edges: np.ndarray):
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
    data : list[np.ndarray]
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
        iterating over a set of inputs files, for example.
    mutable
        Whether the data (and flags and nsamples) can be modified in place within
        the algorithm. Setting to true saves memory, and is safe for a one-shot script.
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
    
    if median:
        logger.info("Calculating median")
        data = np.nanmedian(data, axis=0)
    else:
        logger.info("Calculating mean")
        data = np.nansum(data * nsamples, axis=0)
        data[norm>0] /= norm[norm>0]
        data[norm<=0] = 1  # any value, it's flagged anyway
        
    f_min = np.all(flags, axis=0)
    std[f_min] = 1.0
    norm[f_min] = 0  # This is probably redundant.

    return data, f_min, std, norm


def lst_bin_files_for_baselines(
    data_files: list[Path | FastUVH5Meta], 
    lst_bin_edges: np.ndarray, 
    baselines, 
    freqs: np.ndarray | None = None, 
    pols: np.ndarray | None = None,
    cal_files: list[Path | None] | None = None,
    time_arrays: list[np.ndarray] | None = None,
    time_idx: list[np.ndarray] | None = None,
    ignore_flags: bool = False,
    rephase: bool = True,
    antpos: dict[int, np.ndarray] | None = None,
    lsts: np.ndarray | None = None,
    assume_time_first = None,
):
    metas = [fl if isinstance(fl, FastUVH5Meta) else FastUVH5Meta(fl, blts_are_rectangular=True) for fl in data_files]
    
    if assume_time_first is None:
        assume_time_first = metas[0].time_axis_faster_than_bls

    lst_bin_edges = np.array(lst_bin_edges)

    if freqs is None:
        freqs = np.squeeze(metas[0].freq_array)
    if pols is None:
        pols = metas[0].pols

    if antpos is None and rephase:
        antpos = get_all_antpos_from_files(metas, baselines)

    if time_idx is None:
        while lst_bin_edges[0] < 0:
            lst_bin_edges += 2*np.pi
        while lst_bin_edges[0] >= 2*np.pi:
            lst_bin_edges -= 2*np.pi
        lst_bin_edges %= 2*np.pi

        op = np.logical_and if lst_bin_edges[0] < lst_bin_edges[-1] else np.logical_or
        time_idx = [op(obj.lsts >= lst_bin_edges[0], obj.lsts < lst_bin_edges[-1]) for obj in metas]

    if time_arrays is None:
        time_arrays = [obj.times[idx] for obj, idx in zip(metas, time_idx)]

    if lsts is None:
        lsts = np.concatenate(
            [obj.lsts[idx] for obj, idx in zip(metas, time_idx)]
        )            

    # Now we can set up our master arrays of data. 
    data, flags, nsamples = _allocate_dnf(
        (len(lsts), len(baselines), len(freqs), len(pols)),
        d=np.nan + np.nan*1j,
        f=True
    )

    # This loop actually reads the associated data in this LST bin.
    ntimes_so_far = 0
    for meta, calfl, tind, tarr in zip(metas, cal_files, time_idx, time_arrays):
        logger.info(f"Reading {meta.path}")
        slc = slice(ntimes_so_far,ntimes_so_far+len(tarr))
        ntimes_so_far += len(tarr)

        #hd = io.HERAData(str(fl.path), filetype='uvh5')
        antpairs = meta.antpairs
        bls_to_load = [bl for bl in baselines if bl in antpairs or bl[::-1] in antpairs]

        if not bls_to_load:
            # If none of the requested baselines are in this file, then just 
            # set stuff as nan and go to next file. 
            logger.info("None of the baselines are in this file. Skipping.")
            data[slc] = np.nan
            flags[slc] = True
            nsamples[slc] = 0
            continue

        # TODO: use Fast readers here instead.
        _data, _flags, _nsamples = io.HERAData(meta.path).read(
            bls=bls_to_load, times=tarr
        )
        # _data = meta.get_datacontainer('data', bls = bls_to_load, times=tarr)
        # _flags = meta.get_datacontainer('flags', bls = bls_to_load, times=tarr)
        # _nsamples = meta.get_datacontainer('nsamples', bls = bls_to_load, times=tarr)
        
        # load calibration
        if calfl is not None:
            logger.info(f"Opening and applying {calfl}")
            uvc = io.to_HERACal(calfl)
            gains, cal_flags, _, _ = uvc.read()
            # down select times if necessary
            if False in tind and uvc.Ntimes > 1:
                # If uvc has Ntimes == 1, then broadcast across time will work automatically
                uvc.select(times=uvc.time_array[tind])
                gains, cal_flags, _, _ = uvc.build_calcontainers()

            apply_cal.calibrate_in_place(
                _data, gains, data_flags=_flags, cal_flags=cal_flags,
                gain_convention=uvc.gain_convention
            )

        for i, bl in enumerate(baselines):
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
        baselines=baselines,
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

@profile
def lst_bin_files(
    data_files: list[list[str]], 
    calfile_rules: tuple[tuple[str, str]] = (), 
    input_cals: tuple[list[str]] | None = (),
    dlst: float | None=None, 
    n_lstbins_per_outfile: int=60,
    file_ext: str="{type}.{time:7.5f}.uvh5", 
    outdir: str | Path | None=None, 
    overwrite: bool=False, 
    history: str='', 
    lst_start: float | None = None,
    lst_width: float = 2*np.pi,
    atol: float=1e-6,  
    rephase: bool=False,
    output_file_select: int | Sequence[int] | None=None, 
    Nbls_to_load: int | None=None, 
    ignore_flags: bool=False, 
    include_autos: bool=True, 
    ex_ant_yaml_files=None, 
    ignore_ants: tuple[int]=(),
    write_kwargs: dict | None = None,
    ignore_missing_calfiles: bool = False,
    save_channels: tuple[int] = (),
    golden_lsts: tuple[int] = (),
    sigma_clip_thresh: float = 0.0,
    sigma_clip_min_N: int = 4,
):
    """
    LST bin a series of UVH5 files.
    
    This takes a series of UVH5 files where each file has the same frequency bins and 
    pols, grids them onto a common LST grid, and then averages all integrations
    that appear in that LST bin.

    Output file meta data (frequency bins, antennas positions, time_array)
    are taken from the zeroth file on the last day. Can only LST bin drift-phased data.

    Note: Only supports input data files that have nsample_array == 1, and a single
    integration_time equal to np.diff(time_array), i.e. doesn't support baseline-dependent
    averaging yet. Also, all input files must have the same integration_time, as this
    metadata is taken from zeroth file but applied to all files.

    Parameters:
    -----------
    data_files : type=list of lists: nested set of lists, with each nested list containing
        paths to files from a particular night. Frequency axis of each file must be identical.
        Metadata like x_orientation is inferred from the lowest JD file on the night with the
        highest JDs (i.e. the last night) and assumed to be the same for all files
    dlst : type=float, LST bin width. If None, will get this from the first file in data_files.
    lst_start : type=float, starting LST for binner as it sweeps from lst_start to lst_start + 2pi.
    ntimes_per_file : type=int, number of LST bins in a single output file
    file_ext : type=str, extension to "zen." for output files. This must have at least a ".{type}." field
        where either "LST" or "STD" is inserted for data average or data standard dev., and also a ".{time:7.5f}"
        field where the starting time of the data file is inserted. If this also has a ".{pol}." field, then
        the polarizations of data is also inserted. Example: "{type}.{time:7.5f}.uvh5"
    outdir : type=str, output directory
    overwrite : type=bool, if True overwrite output files
    history : history to insert into output files
    rephase : type=bool, if True, rephase data points in LST bin to center of bin
    bin_kwargs : type=dictionary, keyword arguments for lst_bin.
    atol : type=float, absolute tolerance for LST bin float comparison
    output_file_select : type=int or integer list, list of integer indices of the output files to run on.
        Default is all files.
    input_cals : type=list of lists: nested set of lists matching data_files containing
        filepath to calfits, UVCal or HERACal objects with gain solutions to
        apply to data on-the-fly before binning via hera_cal.apply_cal.calibrate_in_place.
        If no apply cal is desired for a particular file, feed as None in input_cals.
    Nbls_to_load : int, default=None, Number of baselines to load and bin simultaneously. If Nbls exceeds this
        than iterate over an outer loop until all baselines are binned. Default is to load all baselines at once.
    ignore_flags : bool, if True, ignore the flags in the input files, such that all input data in included in binning.
    average_redundant_baselines : bool, if True, baselines that are redundant between and within nights will be averaged together.
        When this is set to true, Nbls_to_load is interpreted as the number of redundant groups
        to load simultaneously. The number of data waterfalls that are loaded can be substantially larger in some
        cases.
    include_autos : bool, if True, include autocorrelations in redundant baseline averages.
                    default is True.
    bl_error_tol : float, tolerance within which baselines are considered redundant
                   between and within nights for purposes of average_redundant_baselines.
    ex_ant_yaml_files : list of strings, optional
        list of paths of yaml files specifying antennas to flag and remove from data on each night.
    kwargs : type=dictionary, keyword arguments to pass to io.write_vis()
    sigma_clip_thresh : float, threshold for sigma clipping. If 0, no sigma clipping is done.
    sigma_clip_min_N : int, minimum number of points to perform sigma clipping on.

    Result:
    -------
    zen.{pol}.LST.{file_lst}.uv : holds LST bin avg (data_array) and bin count (nsample_array)
    zen.{pol}.STD.{file_lst}.uv : holds LST bin stand dev along real and imag (data_array)
    """
    # Check that that there are the same number of input data files and 
    # calibration files each night.

    input_cals = input_cals or []
    if not input_cals and calfile_rules:
            
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
                elif ignore_missing_calfiles:
                    warnings.warn(f"Calibration file {cf} does not exist")
                    missing.append(df)
                else:
                    raise IOError(f"Calibration file {cf} does not exist")
            data_files[night] = [df for df in dflist if df not in missing]


    # Prune empty nights (some nights start with files, but have files removed because
    # they have no associated calibration)
    data_files = [df for df in data_files if df]
    input_cals = [cf for cf in input_cals if cf]

    logger.info("Got the following numbers of data files per night:")
    for dflist in data_files:
        logger.info(f"{dflist[0].split('/')[-1]}: {len(dflist)}")

    data_metas = [[FastUVH5Meta(df, blts_are_rectangular=True) for df in dflist] for dflist in data_files]

    # get file lst arrays
    _, dlst, file_lsts, _, lst_arrs, time_arrs = config_lst_bin_files(
        data_metas, 
        dlst=dlst, 
        atol=atol, 
        lst_start=lst_start,
        ntimes_per_file=n_lstbins_per_outfile, 
        lst_width=lst_width,
        verbose=False
    )

    nfiles = len(file_lsts)

    logger.info("Setting output files")

    # Set branch cut before trimming files -- want it to be the same for all files
    write_kwargs = write_kwargs or {}
    if 'lst_branch_cut' not in write_kwargs and lst_start is not None:
        write_kwargs['lst_branch_cut'] = file_lsts[0][0]

    # select file_lsts
    if output_file_select is not None:
        if isinstance(output_file_select, (int, np.integer)):
            output_file_select = [output_file_select]
        output_file_select = [int(o) for o in output_file_select]
        try:
            file_lsts = [file_lsts[i] for i in output_file_select]
        except IndexError:
            warnings.warn(
                f"One or more indices in output_file_select {output_file_select} "
                f"caused an index error with length {nfiles} file_lsts list, exiting..."
            )
            return

    # get metadata from the zeroth data file in the last day
    last_day_index = np.argmax([np.min([time for tarr in tarrs for time in tarr]) for tarrs in time_arrs])
    zeroth_file_on_last_day_index = np.argmin([np.min(tarr) for tarr in time_arrs[last_day_index]])

    logger.info("Getting metadata from last data...")
    
    # TODO: since hd is only used for metadata, we could use the FastUVH5Meta class.
    #       However, we need to figure out how to read the antpos properly first.
    #hd = io.HERAData(str(data_files[last_day_index][zeroth_file_on_last_day_index]))
    meta = data_metas[last_day_index][zeroth_file_on_last_day_index]
    x_orientation = meta.x_orientation

    # get metadata
    freq_array = np.squeeze(meta.freq_array)
    times = meta.times
    start_jd = np.floor(times.min())
    integration_time = np.median(meta.integration_time)
    if not  np.all(np.abs(np.diff(times) - np.median(np.diff(times))) < 1e-6):
        raise ValueError('All integrations must be of equal length (BDA not supported)')

    logger.info("Compiling all unflagged baselines...")
    all_baselines, all_pols, fls_with_ants = get_all_unflagged_baselines(
        data_metas, 
        ex_ant_yaml_files, 
        include_autos=include_autos, 
        ignore_ants=ignore_ants
    )
    all_baselines = sorted(all_baselines)

    antpos = get_all_antpos_from_files(fls_with_ants, all_baselines)
    # Split up the baselines into chunks that will be LST-binned together.
    # This is just to save on RAM.
    if Nbls_to_load is None:
        Nbls_to_load = len(all_baselines) + 1
    n_bl_chunks = len(all_baselines) // Nbls_to_load + 1
    bl_chunks = [all_baselines[i * Nbls_to_load:(i + 1) * Nbls_to_load] for i in range(n_bl_chunks)]
    bl_chunks = [blg for blg in bl_chunks if len(blg) > 0]

    time_first_ordering = data_metas[0][0].time_axis_faster_than_bls

    # iterate over output LST files
    for i, outfile_lsts in enumerate(file_lsts):
        logger.info(f"LST file {i+1} / {len(file_lsts)}")

        outfile_lst_min = outfile_lsts[0] - (dlst / 2 + atol)
        outfile_lst_max = outfile_lsts[-1] + (dlst / 2 + atol)

        tinds = []
        all_lsts = []
        file_list = []
        time_arrays = []
        cals = []
        # This loop just gets the number of times that we'll be reading.
        for night, night_files in enumerate(data_metas):
            # iterate over files in each night, and open files that fall into this output file LST range

            for k_file, fl in enumerate(night_files):

                # unwrap la relative to itself
                larr = lst_arrs[night][k_file]
                larr[larr < larr[0]] += 2 * np.pi

                # phase wrap larr to get it to fall within 2pi of file_lists
                while larr[0] + 2 * np.pi < outfile_lst_max:
                    larr += 2 * np.pi
                while larr[-1] - 2 * np.pi > outfile_lst_min:
                    larr -= 2 * np.pi

                tind = (larr > outfile_lst_min) & (larr < outfile_lst_max)

                if np.any(tind):
                    tinds.append(tind)
                    time_arrays.append(time_arrs[night][k_file][tind])
                    all_lsts.append(larr[tind])
                    file_list.append(fl)
                    if input_cals:
                        cals.append(input_cals[night][k_file])
                    else:
                        cals.append(None)

        all_lsts = np.concatenate(all_lsts)

        # If we have no times at all for this bin, just continue to the next bin.
        if len(all_lsts) == 0:
            continue

        # iterate over baseline groups (for memory efficiency)
        out_data, out_flags, out_nsamples = _allocate_dnf(
            (len(all_baselines), len(outfile_lsts), len(freq_array), len(all_pols)),
        )
        out_stds = np.zeros_like(out_data)

        nbls_so_far = 0
        lst_bin_edges = np.array(
            [x - dlst/2 for x in outfile_lsts] + [outfile_lsts[-1] + dlst/2]
        )
        
        # The "golden" data is the full data over all days for a small subset of LST
        # bins. This works best if the LST bins are small (similar to the size of the
        # raw integrations). Usually, the length of "bins" will be zero.
        # NOTE: we work under the assumption that the LST bins are small, so that 
        # each night only gets one integration in each LST bin. If there are *more*
        # than one integration in the bin, we take the first one only.
        bins, _, mask = get_lst_bins(golden_lsts, lst_bin_edges)
        bins = bins[mask]
        golden_data, golden_flags, golden_nsamples = [], [], []
        logger.info(f"golden_lsts bins in this output file: {bins}, lst_bin_edges={lst_bin_edges}, {len(lst_bin_edges)}")

        

        for bi, bl_chunk in enumerate(bl_chunks):
            logger.info(f"Baseline Chunk {bi+1} / {len(bl_chunks)}")
            # data/flags/nsamples are *lists*, with nlst_bins entries, each being an
            # array, with shape (times, bls, freqs, npols)
            bin_lst, data, flags, nsamples, binned_times = lst_bin_files_for_baselines(
                data_files = file_list, 
                lst_bin_edges=lst_bin_edges, 
                baselines=bl_chunk, 
                freqs=freq_array, 
                pols=all_pols,
                cal_files=cals,
                time_arrays=time_arrays,
                time_idx=tinds,
                ignore_flags=ignore_flags,
                rephase=rephase,
                antpos=antpos,
                lsts=all_lsts,
                assume_time_first=time_first_ordering,
            )

            slc = slice(nbls_so_far, nbls_so_far + len(bl_chunk))
            reduce_lst_bins(
                data, flags, nsamples,
                out_nsamples=out_nsamples[slc],
                out_data=out_data[slc],
                out_flags=out_flags[slc],
                out_std=out_stds[slc],
                sigma_clip_thresh = sigma_clip_thresh,
                sigma_clip_min_N = sigma_clip_min_N,
            )

            if len(bins):
                for nbin, b in enumerate(bins):
                    if bi == 0:
                        logger.info(f"length of data: {len(data)}")
                        nt, _, nf, npol = data[b].shape
                        d, f, n = _allocate_dnf(
                            (len(all_baselines), nt, nf, npol)
                        )
                        golden_data.append(d)
                        golden_flags.append(f)
                        golden_nsamples.append(n)
                    
                    golden_data[nbin][slc] = data[b].transpose((1,0,2,3))
                    golden_flags[nbin][slc] = flags[b].transpose((1,0,2,3))
                    golden_nsamples[nbin][slc] = nsamples[b].transpose((1,0,2,3))

            if len(save_channels):
                if bi == 0:
                    # The "chan" data is a subset of the full data, taking days, baselines
                    # and pols, but only a small subset of frequencies. We do this for the first 
                    # LST bin in each output file.
                    chan_data, chan_flags, chan_nsamples = _allocate_dnf(
                        (len(all_baselines), data[0].shape[0], len(save_channels), len(all_pols))
                    )

                for ichan, chan in enumerate(save_channels):
                    chan_data[slc, :, ichan] = data[0][:, :, chan].transpose((1, 0, 2))
                    chan_flags[slc, :, ichan] = flags[0][:, :, chan].transpose((1, 0, 2))
                    chan_nsamples[slc, :, ichan] = nsamples[0][:, :, chan].transpose((1, 0, 2))

            nbls_so_far += len(bl_chunk)

        logger.info("Writing output files")

        # get outdir
        if outdir is None:
            outdir = os.path.dirname(os.path.commonprefix(abscal.flatten(data_files)))

        # update kwrgs
        # update history
        file_list_str = "-".join(os.path.basename(ff.path)for ff in file_list)
        file_history = f"{history} Input files: {file_list_str}"
        _history = file_history + utils.history_string()

        # form integration time array
        integration_time = integration_time*np.ones(
            len(bin_lst) * len(all_baselines), 
            dtype=np.float64
        )

        # file in data ext
        fkwargs = {"type": "LST", "time": bin_lst[0] - dlst / 2.0}
        if "{pol}" in file_ext:
            fkwargs['pol'] = '.'.join(all_pols)

        # configure filenames
        bin_file = f"zen.{file_ext.format(**fkwargs)}"
        fkwargs['type'] = 'STD'
        std_file = f"zen.{file_ext.format(**fkwargs)}"

        logger.info(f"Writing {bin_file} and {std_file} to {outdir}")

        # check for overwrite
        if os.path.exists(os.path.join(outdir, bin_file)) and not overwrite:
            logger.warning(f"{bin_file} exists, not overwriting")
            continue

        write_kwargs.update(
            lst_array=bin_lst,
            freq_array=freq_array,
            antpos=antpos,
            pols=all_pols,
            antpairs=all_baselines,
            flags=out_flags,
            nsamples=out_nsamples,
            x_orientation=x_orientation,
            integration_time=integration_time,
            history=_history,
            start_jd=start_jd,
        )
        uvd_data = io.create_uvd_from_hera_data(data = out_data, **write_kwargs)
        uvd_data.write_uvh5(os.path.join(outdir, bin_file), clobber=overwrite)

        uvd_data = io.create_uvd_from_hera_data(data = out_stds, **write_kwargs)
        uvd_data.write_uvh5(os.path.join(outdir, std_file), clobber=overwrite)

        # Now write out the golden lsts
        for gd, gn, gf, bt, ib in zip(golden_data, golden_nsamples, golden_flags, binned_times, bins):
            fkwargs['type'] = 'GOLDEN'
            fkwargs['time'] = lst_bin_edges[ib]
            filename = f"zen.{file_ext.format(**fkwargs)}"

            guvd = io.create_uvd_from_hera_data(
                data = gd,
                time_array = bt,
                freq_array=freq_array,
                antpos=antpos,
                pols=all_pols,
                antpairs=all_baselines,
                flags=gf,
                nsamples=gn,
                x_orientation=x_orientation,
                integration_time=integration_time[0],
                history=_history,
            )
            # Don't check autos because we've set flagged stuff to NaN and that fails.
            guvd.write_uvh5(
                os.path.join(outdir, filename), clobber=overwrite, 
                check_autos=False, fix_autos=False
            )

        # Now write out reduced-channel data
        if save_channels:
            fkwargs['type'] = 'REDUCEDCHAN'
            fkwargs['time'] = bin_lst[0] - dlst / 2.0
            filename = f"zen.{file_ext.format(**fkwargs)}"
            guvd = io.create_uvd_from_hera_data(
                data = chan_data,
                time_array = binned_times[0],
                freq_array=freq_array[list(save_channels)],
                antpos=antpos,
                pols=all_pols,
                antpairs=all_baselines,
                flags=chan_flags,
                nsamples=chan_nsamples,
                x_orientation=x_orientation,
                integration_time=integration_time[0],
                history=_history,
            )
            logger.info(f"BINNED TIMES: {binned_times[0]}")
            logger.info(f"REDUCEDCHAN DATA SHAPE: {guvd.data_array.shape}")
            logger.info(f"Ntimes: {guvd.Ntimes}")

            # Don't check autos because we've set flagged stuff to NaN and that fails.
            guvd.write_uvh5(
                os.path.join(outdir, filename), clobber=overwrite, 
                check_autos=False, fix_autos=False
            )

@profile
def get_all_unflagged_baselines(
    data_files: list[list[str | Path | FastUVH5Meta]], 
    ex_ant_yaml_files: list[str] | None = None,
    include_autos: bool = True,
    ignore_ants: tuple[int] = (),
    only_last_file_per_night: bool = False,
) -> tuple[set[tuple[int, int]], list[str]]:
    """Generate a set of all antpairs that have at least one un-flagged entry.
    
    This is performed over a list of nights, each of which consists of a list of 
    individual uvh5 files. Each UVH5 file is *assumed* to have the same set of times
    for each baseline internally (different nights obviously have different times).
    
    Returns
    -------
    all_baselines
        The set of all antpairs in all files in the given list.
    all_pols
        A list of all polarizations in the files in the given list, as strings like 
        'ee' and 'nn' (i.e. with x_orientation information).
    """
    data_files = [[fl if isinstance(fl, FastUVH5Meta) else FastUVH5Meta(fl, blts_are_rectangular=True) for fl in fl_list] for fl_list in data_files]
    
    pols = None
    xorient = None

    all_baselines = set()
    files_with_ants = set()
    unique_ants = set()

    for night, fl_list in enumerate(data_files):
        if ex_ant_yaml_files:
            a_priori_antenna_flags = read_a_priori_ant_flags(
                ex_ant_yaml_files[night], ant_indices_only=True
            )
        else:
            a_priori_antenna_flags = set()

        if only_last_file_per_night:
            fl_list = fl_list[-1:]

        for fl in fl_list:
            antpairs = fl.antpairs
            
            if pols is not None and not np.all(pols == fl.polarization_array):
                raise ValueError(
                    f"The polarizations in {fl} are not the same as in {fl_list[0]}"
                )
            pols = fl.polarization_array

            if xorient is not None and fl.x_orientation != xorient:
                raise ValueError("Not all input files have the same x_orientation!")

            xorient = fl.x_orientation

            for a1, a2 in antpairs:
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

                    if a1 not in unique_ants:
                        unique_ants.add(a1)
                        files_with_ants.add(fl)
                    if a2 not in unique_ants:
                        unique_ants.add(a2)
                        files_with_ants.add(fl)
                    

    return all_baselines, fl.pols, files_with_ants


def get_all_antpos_from_files(
    data_files: list[FastUVH5Meta], 
    all_baselines: list[tuple[int, int]]
) -> dict[tuple[int, int], np.ndarray]:

    antpos_out = {}
    
    # ants will be a set of integers antenna numbers.
    ants = set(sum(all_baselines, start=()))
    nants = len(ants)

    for fl in data_files:
        for i, ant in enumerate(fl.antenna_numbers):
            if ant in ants and ant not in antpos_out:
                # We only access antpos_enu inside hte conditional, because it has
                # computation to do, and we don't want to do it if we don't need to.
                antpos_out[ant] = fl.antpos_enu[i]
        if len(antpos_out) == nants:
            break

    return antpos_out

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
    a.add_argument('data_files', nargs='*', type=str, help="quotation-bounded, space-delimited, glob-parsable search strings to nightly data files (UVH5)")
    a.add_argument(
        "--calfile-rules", nargs='*', type=str, 
        help="rules to convert datafile names to calfile names. A series of two strings where the first will be replaced by the latter"
    )
    a.add_argument("--dlst", type=float, default=None, help="LST grid bin width")
    a.add_argument("--ntimes_per_file", dest='n_lstbins_per_outfile', type=int, default=60, help="number of LST bins to write per output file")
    a.add_argument("--file_ext", type=str, default="{type}.{time:7.5f}.uvh5", help="file extension for output files. See lstbin.lst_bin_files doc-string for format specs.")
    a.add_argument("--outdir", default=None, type=str, help="directory for writing output")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output files")
    a.add_argument("--lst_start", type=float, default=None, help="starting LST for binner as it sweeps across 2pi LST. Default is first LST of first file.")
    a.add_argument("--lst-width", type=float, default=2*np.pi, help="how much LST to bin in total, default is full 2pi.")
    a.add_argument("--rephase", default=False, action='store_true', help="rephase data to center of LST bin before binning")
    a.add_argument("--history", default=' ', type=str, help="history to insert into output files")
    a.add_argument("--atol", default=1e-6, type=float, help="absolute tolerance when comparing LST bin floats")
    a.add_argument("--output_file_select", default=None, nargs='*', help="list of output file integers to run on. Default is all output files.")
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
    return a