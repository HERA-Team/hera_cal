"""
An attempt at a simpler LST binner that makes more assumptions but runs faster.

In particular, we assume that all baselines have the same time array and frequency array,
and that each is present throughout the data array. This allows a vectorization.
"""
import numpy as np
from . import utils
import warnings
from pathlib import Path
from .lstbin import config_lst_bin_files
from . import abscal
import os
from . import io
import logging
import h5py
from hera_qm.metrics_io import read_a_priori_ant_flags
from . import apply_cal
from .datacontainer import DataContainer
from .utils import mergedicts
import gc
from collections import OrderedDict as odict
from typing import Sequence

logger = logging.getLogger(__name__)

def simple_lst_bin(
    data: np.ndarray,
    data_lsts: np.ndarray,
    baselines: list[tuple[int, int]],
    pols: list[str],
    lst_bin_edges: np.ndarray,
    freq_array: np.ndarray,
    flags: np.ndarray | None = None,
    nsamples: np.ndarray | None = None,
    rephase: bool = True,
    antpos: np.ndarray | None = None,
    lat: float = -30.72152,
):
    required_shape = (len(data_lsts), len(baselines), len(pols), len(freq_array))
    if data.shape != required_shape:
        raise ValueError(
            f"data should have shape {required_shape} but got {data.shape}"
        )

    if flags is None:
        flags = np.zeros(data.shape, dtype=bool)

    if flags.shape != required_shape:
        raise ValueError(
            f"flags should have shape {required_shape} but got {flags.shape}"
        )

    if nsamples is None:
        nsamples = np.ones(data.shape, dtype=float)
    
    if nsamples.shape != required_shape:
        raise ValueError(
            f"nsampels should have shape {required_shape} but got {nsamples.shape}"
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
    data_lsts %= 2*np.pi
    data_lsts[data_lsts < lst_bin_edges[0]] += 2* np.pi

    lst_bin_centres = (lst_bin_edges[1:] + lst_bin_edges[:-1])/2

    grid_indices = np.digitize(data_lsts, lst_bin_edges, right=True) - 1

    # Now, any grid index that is less than zero, or len(edges) - 1 is not included in this grid.
    lst_mask = (grid_indices >= 0) & (grid_indices < len(lst_bin_centres))

    # TODO: check whether this creates a data copy. Don't want the extra RAM...
    data = data[lst_mask]  # actually good if this is copied, because we do LST rephase in-place
    flags = flags[lst_mask]
    nsamples = nsamples[lst_mask]
    data_lsts = data_lsts[lst_mask]
    grid_indices = grid_indices[lst_mask]

    # Now, rephase the data to the lst bin centres.
    if rephase:
        if freq_array is None or antpos is None:
            raise ValueError("freq_array and antpos is needed for rephase")

        bls = np.array([antpos[k[0]] - antpos[k[1]] for k in baselines])

        # get appropriate lst_shift for each integration, then rephase
        lst_shift = lst_bin_centres[grid_indices] - data_lsts

        # this makes a copy of the data in d
        utils.lst_rephase_vectorized(data, bls, freq_array, lst_shift, lat=lat, inplace=True)

    # TODO: check for baseline conjugation stuff.

    davg = np.zeros((len(lst_bin_centres), len(baselines), len(pols), len(freq_array)), dtype=complex)
    flag_min = np.zeros(davg.shape, dtype=bool)
    std = np.ones(davg.shape, dtype=complex)
    counts = np.zeros(davg.shape, dtype=float)
    for lstbin in range(len(lst_bin_centres)):
        # TODO: check that this doesn't make yet another copy...
        # This is just the data in this particular lst-bin.
        mask = grid_indices==lstbin
        d = data[mask]
        n = nsamples[ mask]
        f = flags[mask]

        (
            davg[lstbin], flag_min[lstbin], std[lstbin], counts[lstbin]
        ) = lst_average(d, n ,f)

    # TODO: should we put these back into datacontainers before returning?
    davg_dc = {}
    std_dc = {}
    flags_dc = {}
    counts_dc = {}
    for i, bl in enumerate(baselines):
        for j, pol in enumerate(pols):
            davg_dc[bl + (pol,)] = davg[:, i, j]
            std_dc[bl + (pol,)] = std[:, i, j]
            flags_dc[bl + (pol,)] = flag_min[:,i,j]
            counts_dc[bl + (pol,)] = counts[:,i,j]
            
    return lst_bin_centres, davg_dc, flags_dc, std_dc, counts_dc

def lst_average(
    data: np.ndarray, nsamples: np.ndarray, flags: np.ndarray, 
    flag_thresh: float = 0.7, median: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # data has shape (ntimes, nbl, npols, nfreqs)
    # all data is assumed to be in the same LST bin.

    assert data.shape == nsamples.shape == flags.shape
    
    flags[np.isnan(data) | np.isinf(data) | (nsamples == 0)] = True

    # Flag entire LST bins if there are too many flags over time
    flag_frac = np.sum(flags, axis=0) / flags.shape[0]
    flags |= flag_frac > flag_thresh

    data[flags] *= np.nan  # do this so that we can do nansum later. multiply to get both real/imag as nan

    # get other stats
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")
        std = np.nanstd(data.real, axis=0) + 1j*np.nanstd(data.imag, axis=0)
        
    nsamples[flags] = 0
    norm = np.sum(nsamples, axis=0)  # missing a "clip" between 1e-99 and inf here...
    
    if median:
        data = np.nanmedian(data, axis=0)
    else:
        data = np.nansum(data * nsamples, axis=0)
        data[norm>0] /= norm[norm>0]
        data[norm<=0] = 1  # any value, it's flagged anyway
        
    f_min = np.all(flags, axis=0)
    std[f_min] = 1.0
    norm[f_min] = 0  # This is probably redundant.

    return data, f_min, std, norm


def lst_bin_files(
    data_files: list[list[str]], 
    input_cals: list[list[str]] | None = None, 
    dlst: float | None=None, 
    n_lstbins_per_outfile: int=60,
    file_ext: str="{type}.{time:7.5f}.uvh5", 
    outdir: str | Path | None=None, 
    overwrite: bool=False, 
    history: str='', 
    lst_start: float | None=None,
    atol: float=1e-6, 
    sig_clip: bool=True, 
    sigma: float=5.0, 
    min_N: int=5, 
    flag_below_min_N: bool=False, 
    rephase: bool=False,
    output_file_select: int | Sequence[int] | None=None, 
    Nbls_to_load: int | None=None, 
    ignore_flags: bool=False, 
    include_autos: bool=True, 
    ex_ant_yaml_files=None, 
    **kwargs
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

    Result:
    -------
    zen.{pol}.LST.{file_lst}.uv : holds LST bin avg (data_array) and bin count (nsample_array)
    zen.{pol}.STD.{file_lst}.uv : holds LST bin stand dev along real and imag (data_array)
    """
    # get file lst arrays
    lst_grid, dlst, file_lsts, begin_lst, lst_arrs, time_arrs = config_lst_bin_files(
        data_files, 
        dlst=dlst, 
        atol=atol, 
        lst_start=lst_start,
        ntimes_per_file=n_lstbins_per_outfile, 
        verbose=False
    )

    nfiles = len(file_lsts)

    # make sure the JD corresponding to file_lsts[0][0] is the lowest JD in the LST-binned data set
    if (lst_start is not None) and ('lst_branch_cut' not in kwargs):
        kwargs['lst_branch_cut'] = file_lsts[0][0]

    logger.info("Setting output files")

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

    # get outdir
    if outdir is None:
        outdir = os.path.dirname(os.path.commonprefix(abscal.flatten(data_files)))

    # update kwrgs
    kwargs['outdir'] = outdir
    kwargs['overwrite'] = overwrite

    # get metadata from the zeroth data file in the last day
    last_day_index = np.argmax([np.min([time for tarr in tarrs for time in tarr]) for tarrs in time_arrs])
    zeroth_file_on_last_day_index = np.argmin([np.min(tarr) for tarr in time_arrs[last_day_index]])

    logger.info("Getting metadata from last data...")    
    hd = io.HERAData(data_files[last_day_index][zeroth_file_on_last_day_index])
    x_orientation = hd.x_orientation

    # get metadata
    freq_array = hd.freqs
    antpos = hd.antpos
    times = hd.times
    start_jd = np.floor(times.min())
    kwargs['start_jd'] = start_jd
    integration_time = np.median(hd.integration_time)
    assert np.all(np.abs(np.diff(times) - np.median(np.diff(times))) < 1e-6), 'All integrations must be of equal length (BDA not supported).'

    logger.info("Getting antenna positions from last file on each night...")
    # get antpos over all nights looking at last file on each night
    nightly_last_hds = []
    for dlist, tarrs in zip(data_files, time_arrs):
        last_file_index = np.argmin([np.min(tarr) for tarr in tarrs])
        hd = io.HERAData(dlist[last_file_index])
        for a in hd.antpos:
            if a not in antpos:
                antpos[a] = hd.antpos[a]
        nightly_last_hds.append(hd)

    logger.info("Compiling all unflagged baselines...")
    all_baselines = list(get_all_unflagged_baselines(data_files, ex_ant_yaml_files, include_autos=include_autos))

    # generate a list of dictionaries which contain the nights occupied by each unique baseline
    # (or unique baseline group if average_redundant_baselines is true)
    # bl_nightly_dicts = gen_bl_nightly_dicts(nightly_last_hds, bl_error_tol=bl_error_tol,
    #                                         include_autos=include_autos, redundant=average_redundant_baselines, ex_ant_yaml_files=ex_ant_yaml_files)
    
    # Split up the baselines into chunks that will be LST-binned together.
    # This is just to save on RAM.
    if Nbls_to_load is None:
        Nbls_to_load = len(all_baselines) + 1
    n_bl_chunks = len(all_baselines) // Nbls_to_load + 1
    bl_chunks = [all_baselines[i * Nbls_to_load:(i + 1) * Nbls_to_load] for i in range(n_bl_chunks)]
    bl_chunks = [blg for blg in bl_chunks if len(blg) > 0]

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
        for night, night_files in enumerate(data_files):
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
                    if input_cals is not None:
                        cals.append(input_cals[night][k_file])
                    else:
                        cals.append(None)

        all_lsts = np.concatenate(all_lsts)

        # If we have no times at all for this bin, just continue to the next bin.
        if len(all_lsts) == 0:
            continue
        
        # iterate over baseline groups (for memory efficiency)
        data_conts, flag_conts, std_conts, num_conts = [], [], [], []
        for bi, bl_chunk in enumerate(bl_chunks):
            logger.info(f"Baseline Chunk {bi+1} / {len(bl_chunks)}")

            # Now we can set up our master arrays of data. 
            data = np.full((
                len(all_lsts), len(bl_chunk), len(hd.pols), len(hd.freqs)), 
                np.nan+np.nan*1j, dtype=complex
            )
            flags = np.ones(data.shape, dtype=bool)
            nsamples = np.zeros(data.shape, dtype=float)

            # This loop actually reads the associated data in this LST bin.
            ntimes_so_far = 0
            for fl, calfl, tind, tarr in zip(file_list, cals, tinds, time_arrays):
                hd = io.HERAData(fl, filetype='uvh5')

                bls_to_load = [bl for bl in bl_chunk if bl in hd.antpairs]
                _data, _flags, _nsamples  = hd.read(
                    bls=bls_to_load, 
                    times=tarr
                )

                # load calibration
                if calfl is not None:
                    logger.info(f"Opening and applying {calfl}")
                    uvc = io.to_HERACal(calfl)
                    gains, cal_flags, _, _ = uvc.read()
                    # down select times in necessary
                    if False in tind and uvc.Ntimes > 1:
                        # If uvc has Ntimes == 1, then broadcast across time will work automatically
                        uvc.select(times=uvc.time_array[tind])
                        gains, cal_flags, _, _ = uvc.build_calcontainers()
                    
                    apply_cal.calibrate_in_place(
                        _data, gains, data_flags=_flags, cal_flags=cal_flags,
                        gain_convention=uvc.gain_convention
                    )

                slc = slice(ntimes_so_far,ntimes_so_far+_data.shape[0])
                for i, bl in enumerate(bl_chunk):
                    for j, pol in enumerate(_data._pols):
                        if bl + (pol,) in _data:
                            data[slc, i, j] = _data[bl+(pol,)]
                            flags[slc, i, j] = _flags[bl+(pol,)]
                            nsamples[slc, i, j] = _nsamples[bl+(pol,)]
                        else:
                            # This baseline+pol doesn't exist in this file. That's
                            # OK, we don't assume all baselines are in every file.
                            data[slc, i, j] = np.nan
                            flags[slc, i, j] = True
                            nsamples[slc, i, j] = 0

                ntimes_so_far += _data.shape[0]

            logger.info("About to run LST binning...")
            # LST bin edges are the actual edges of the bins, so should have length
            # +1 of the LST centres. We use +dlst instead of +dlst/2 on the top edge
            # so that np.arange definitely gets the last edge.
            lst_edges = np.arange(outfile_lsts[0] - dlst/2, outfile_lsts[-1] + dlst, dlst)
            bin_lst, bin_data, flag_data, std_data, num_data = simple_lst_bin(
                data=data, 
                flags=None if ignore_flags else flags,
                nsamples=nsamples,
                data_lsts=all_lsts,
                baselines=bl_chunk,
                pols=hd.pols,
                lst_bin_edges=lst_edges,
                freq_array = hd.freqs,
                rephase = rephase,
                antpos=antpos,
            )
            
            # append to lists
            data_conts.append(bin_data)
            flag_conts.append(flag_data)
            std_conts.append(std_data)
            num_conts.append(num_data)

        # join DataContainers across blgroups
        bin_data = DataContainer(mergedicts(*data_conts))
        flag_data = DataContainer(mergedicts(*flag_conts))
        std_data = DataContainer(mergedicts(*std_conts))
        num_data = DataContainer(mergedicts(*num_conts))
        
        # update history
        file_list_str = "-".join(os.path.basename(ff)for ff in file_list)
        file_history = f"{history} Input files: {file_list_str}"
        kwargs['history'] = file_history + utils.history_string()

        # form integration time array
        kwargs['integration_time'] = integration_time*np.ones(
            len(bin_lst) * len(all_baselines), 
            dtype=np.float64
        )

        # file in data ext
        fkwargs = {"type": "LST", "time": bin_lst[0] - dlst / 2.0}
        if "{pol}" in file_ext:
            fkwargs['pol'] = '.'.join(bin_data.pols())

        # configure filenames
        bin_file = "zen." + file_ext.format(**fkwargs)
        fkwargs['type'] = 'STD'
        std_file = "zen." + file_ext.format(**fkwargs)

        # check for overwrite
        if os.path.exists(bin_file) and overwrite is False:
            logger.warning(f"{bin_file} exists, not overwriting")
            continue

        # write to file
        io.write_vis(bin_file, bin_data, bin_lst, freq_array, antpos, flags=flag_data,
                     nsamples=num_data, filetype='uvh5', x_orientation=x_orientation, **kwargs)
        io.write_vis(std_file, std_data, bin_lst, freq_array, antpos, flags=flag_data,
                     nsamples=num_data, filetype='uvh5', x_orientation=x_orientation, **kwargs)

        del bin_file, std_file, bin_data, std_data, num_data, bin_lst, flag_data
        del data_conts, flag_conts, std_conts, num_conts
        gc.collect()

def get_all_unflagged_baselines(
    data_files: list[list[str | Path]], 
    ex_ant_yaml_files: list[str] | None = None,
    include_autos: bool = True,
) -> set[tuple[int, int]]:
    """Generate a set of all antpairs that have at least one un-flagged entry.
    
    This is performed over a list of nights, each of which consists of a list of 
    individual uvh5 files. Each UVH5 file is *assumed* to have the same set of times
    for each baseline.
    """
    # Get all baselines from all files
    all_baselines = set()
    for night, fl_list in enumerate(data_files):
        if ex_ant_yaml_files:
            a_priori_antenna_flags = read_a_priori_ant_flags(
                ex_ant_yaml_files[night], ant_indices_only=True
            )
        else:
            a_priori_antenna_flags = set()
                        
        for fl in fl_list:
            # To go faster, let's JUST read the antpairs and pols from the files.
            with h5py.File(fl, 'r') as hfl:
                times = hfl['Header']["time_array"][:]
                # This could be faster if we always knew that the order of the blt axis
                # was (t, bl). Not sure that is guaranteed though.
                time0_indx = np.where(times == times[0])[0]
                ant1 = hfl['Header']['ant_1_array'][time0_indx]
                ant2 = hfl['Header']['ant_2_array'][time0_indx]
                flags = hfl['Data']['flags'][time0_indx]

            for ipair, (a1, a2) in enumerate(zip(ant1, ant2)):
                if (a1, a2) in all_baselines:
                    # Check first if it's already in our baseline set, because this will
                    # most of the time be true after the first few files, and we don't
                    # want to always do the np.all() on the flags.
                    continue
                
                if not include_autos and a1 == a2:
                    continue

                if not np.all(flags[ipair]) and a1 not in a_priori_antenna_flags and a2 not in a_priori_antenna_flags:
                    all_baselines.add((a1, a2))

    return all_baselines