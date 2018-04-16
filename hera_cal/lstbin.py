"""
lstbin.py
---------

Routines for aligning and binning of visibility
data onto a universal Local Sidereal Time (LST) grid.
"""
import os
import sys
from collections import OrderedDict as odict
import copy
import argparse
import functools
import numpy as np
from pyuvdata import UVCal, UVData
from pyuvdata import utils as uvutils
from hera_cal import omni, utils, firstcal, cal_formats, abscal, redcal, io
from hera_cal.datacontainer import DataContainer
from scipy import signal
from scipy import interpolate
from scipy import spatial
import itertools
import operator
from astropy import stats as astats
import gc as garbage_collector
import datetime
import aipy


def lst_bin(data_list, lst_list, flags_list=None, dlst=None, lst_start=None, lst_low=None,
            lst_hi=None, flag_thresh=0.7, atol=1e-10, median=False, truncate_empty=True,
            sig_clip=False, sigma=4.0, min_N=4, return_no_avg=False, antpos=None, rephase=False,
            freq_array=None, lat=-30.72152, verbose=True):
    """
    Bin data in Local Sidereal Time (LST) onto an LST grid. An LST grid
    is defined as an array of points increasing in Local Sidereal Time, with each point marking
    the center of the LST bin.

    Parameters:
    -----------
    data_list : type=list, list of DataContainer dictionaries holding complex visibility data

    lst_list : type=list, list of ndarrays holding LST bin centers of each data dictionary in data_list.
        These LST arrays must be monotonically increasing, except for a possible wrap at 2pi.
    
    flags_list : type=list, list of DataContainer dictionaries holding flags for each data dict
        in data_list. Flagged data do not contribute to the average of an LST bin.

    dlst : type=float, delta-LST spacing for lst_grid. If None, will use the delta-LST of the first
        array in lst_list.

    lst_start : type=float, starting LST for making the lst_grid, extending from
        [lst_start, lst_start+2pi). Default is lst_start = 0 radians.

    lst_low : type=float, lower bound on LST bin centers used for contructing LST grid

    lst_hi : type=float, upper bound on LST bin centers used for contructing LST grid

    flag_thresh : type=float, minimum fraction of flagged points in an LST bin needed to
        flag the entire bin.

    atol : type=float, absolute tolerance for comparing LST bin center floats

    median : type=boolean, if True use median for LST binning. Warning: this is slower.

    truncate_empty : type=boolean, if True, truncate output time integrations that have no data
        in them.

    sig_clip : type=boolean, if True, perform a sigma clipping algorithm of the LST bins on the
        real and imag components separately. Warning: This is considerably slow.

    sigma : type=float, input sigma threshold to use for sigma clipping algorithm.

    min_N : type=int, minimum number of points in an LST bin to perform sigma clipping

    return_no_avg : type=boolean, if True, return binned but un-averaged data and flags.

    rephase : type=bool, if True, phase data to center of LST bin before binning.
        Note that this produces a copy of the data.

    antpos : type=dictionary, holds antenna position vectors in ENU frame in meters with 
        antenna integers as keys and 3D ndarrays as values. See io.load_vis(). Needed for rephase.

    freq_array : type=ndarray, 1D array of unique data frequencies channels in Hz. Needed for rephase.

    lat : type=float, latitude of array in degrees North. Needed for rephase.

    verbose : type=bool, if True report feedback to stdout

    Output: (lst_bins, data_avg, flags_min, data_std, data_count)
    -------
    lst_bins : ndarray containing final lst grid of data (marks bin centers)

    data_avg : dictionary of data having averaged in each LST bin

    flags_min : dictionary of minimum of data flags in each LST bin
    
    data_std : dictionary of data with real component holding LST bin std along real axis
               and imag component holding std along imag axis

    data_count : dictionary containing the number count of data points averaged in each LST bin.

    if return_no_avg:
        Output: (lst_bins, data_bin, flags_min)
        data_bin : dictionary with (ant1,ant2,pol) as keys and ndarrays holding
            un-averaged complex visibilities in each LST bin as values. 
        flags_min : dictionary with data flags
    """
    # get visibility shape
    Ntimes, Nfreqs = data_list[0][data_list[0].keys()[0]].shape

    # get dlst if not provided
    if dlst is None:
        dlst = np.median(np.diff(lst_list[0]))

    # construct lst_grid
    lst_grid = make_lst_grid(dlst, lst_start=lst_start, verbose=verbose)
    dlst = np.median(np.diff(lst_grid))

    # test for special case of lst grid restriction
    if lst_low is not None and lst_hi is not None and lst_hi < lst_low:
        lst_grid = lst_grid[(lst_grid > (lst_low - atol)) | (lst_grid < (lst_hi + atol))]
    else:
        # restrict lst_grid based on lst_low and lst_high
        if lst_low is not None:
            lst_grid = lst_grid[lst_grid > (lst_low - atol)]
        if lst_hi is not None:
            lst_grid = lst_grid[lst_grid < (lst_hi + atol)]

    # Raise Exception if lst_grid is empty
    if len(lst_grid) == 0:
        raise ValueError("len(lst_grid) == 0; consider changing lst_low and/or lst_hi.")

    # move lst_grid centers to the left
    lst_grid_left = lst_grid - dlst / 2

    # form new dictionaries
    # data is a dictionary that will hold other dictionaries as values, which will
    # themselves hold lists of ndarrays
    data = odict()
    flags = odict()
    all_lst_indices = set()

    # iterate over data_list
    for i, d in enumerate(data_list):
        # get lst array
        l = copy.copy(lst_list[i])

        # ensure l isn't wrapped relative to lst_grid
        l[l < lst_grid_left.min() - atol] += 2*np.pi

        # digitize data lst array "l"
        grid_indices = np.digitize(l, lst_grid_left[1:], right=True)

        # make data_in_bin boolean array, and set to False data that don't fall in any bin
        data_in_bin = np.ones_like(l, np.bool)
        data_in_bin[(l<lst_grid_left.min() - atol)] = False
        data_in_bin[(l>lst_grid_left.max() + dlst + atol)] = False

        # update all_lst_indices
        all_lst_indices.update(set(grid_indices[data_in_bin]))

        if rephase:
            # rephase each integration in d to nearest LST bin
            if freq_array is None or antpos is None:
                raise ValueError("freq_array and antpos is needed for rephase")

            # form baseline dictionary
            bls = odict(map(lambda k: (k, antpos[k[0]] - antpos[k[1]]), d.keys()))

            # get appropriate lst_shift for each integration, then rephase
            lst_shift = lst_grid[grid_indices] - l

            # this makes a copy of the data in d
            d = utils.lst_rephase(d, bls, freq_array, lst_shift, lat=lat, inplace=False)

        # iterate over keys in d
        for j, key in enumerate(d.keys()):

            # data[key] will be an odict. if data[key] doesn't exist
            # create data[key] as an empty odict. if data[key] already
            # exists, then pass
            if key in data:
                pass
            elif switch_bl(key) in data:
                # check to see if conj(key) exists in data
                key = switch_bl(key)
                d[key] = np.conj(d[switch_bl(key)])
                if flags_list is not None:
                    flags_list[i][key] = flags_list[i][switch_bl(key)]
            else:
                # if key or conj(key) not in data, insert key into data
                data[key] = odict()
                flags[key] = odict()

            # data[key] is an odict, with keys as grid index integers and 
            # values as lists holding the LST bin data: ndarrays of shape (Nfreqs)

            # iterate over grid_indices, and append to data if data_in_bin is True
            for k, ind in enumerate(grid_indices):
                # ensure data_in_bin is True for this grid index
                if data_in_bin[k]:
                    # if index not in data[key], insert it as empty list
                    if ind not in data[key]:
                        data[key][ind] = []
                        flags[key][ind] = []
                    # append data ndarray to LST bin
                    data[key][ind].append(d[key][k])
                    # also insert flags if fed
                    if flags_list is None:
                        flags[key][ind].append(np.zeros_like(d[key][k], np.bool))
                    else:
                        flags[key][ind].append(flags_list[i][key][k])

    # get final lst_bin array
    if truncate_empty:
        # use only lst_grid bins that have data in them
        lst_bins = lst_grid[sorted(all_lst_indices)]
    else:
        # keep all lst_grid bins and fill empty ones with unity data and mark as flagged
        for index in range(len(lst_grid)):
            if index in all_lst_indices:
                # skip if index already in data
                continue
            for key in data.keys():
                # fill data with blank data
                data[key][index] = [np.ones(Nfreqs, np.complex)]
                flags[key][index] = [np.ones(Nfreqs, np.bool)]

        # use all LST bins              
        lst_bins = lst_grid

    # wrap lst_bins if needed
    lst_bins = lst_bins % (2*np.pi)

    # make final dictionaries
    flags_min = odict()
    data_avg = odict()
    data_count = odict()
    data_std = odict()

    # return un-averaged data if desired
    if return_no_avg:
        # return all binned data instead of just bin average 
        data_bin = odict(map(lambda k: (k, np.array(odict(map(lambda k2: (k2, data[k][k2]), sorted(data[k].keys()))).values())), sorted(data.keys())))
        flags_bin = odict(map(lambda k: (k, np.array(odict(map(lambda k2: (k2, flags[k][k2]), sorted(flags[k].keys()))).values())), sorted(flags.keys())))

        return lst_bins, data_bin, flags_bin

    # iterate over data keys (baselines) and get statistics
    for i, key in enumerate(data.keys()):

        # create empty lists
        real_avg = []
        imag_avg = []
        f_min = []
        real_std = []
        imag_std = []
        bin_count = []

        # iterate over sorted LST grid indices in data[key]
        for j, ind in enumerate(sorted(data[key].keys())):

            # make data and flag arrays from lists
            d = np.array(data[key][ind]) # shape = (Ndays x Nfreqs)
            f = np.array(flags[key][ind])
            f[np.isnan(f)] = True

            # replace flagged data with nan
            d[f] *= np.nan

            # sigma clip if desired
            if sig_clip:
                # clip real
                real_f = sigma_clip(d.real, sigma=sigma, min_N=min_N, axis=0)

                # clip imag
                imag_f = sigma_clip(d.imag, sigma=sigma, min_N=min_N, axis=0)

                # get real +  imag flags
                clip_flags = real_f + imag_f

                # flag data
                d[clip_flags] *= np.nan

                # merge clip flags
                f += (real_f + imag_f)

            # check flag thresholds
            if len(f) == 1:
                flag_bin = np.zeros(f.shape[1], np.bool)
            else:
                flag_bin = np.sum(f, axis=0).astype(np.float) / len(f) > flag_thresh
            d[:, flag_bin] *= np.nan
            f[:, flag_bin] = True

            # take bin average
            if median:
                real_avg.append(np.nanmedian(d.real, axis=0))
                imag_avg.append(np.nanmedian(d.imag, axis=0))
            else:
                real_avg.append(np.nanmean(d.real, axis=0))
                imag_avg.append(np.nanmean(d.imag, axis=0))

            # get minimum bin flag
            f_min.append(np.nanmin(f, axis=0))

            # get other stats
            real_std.append(np.nanstd(d.real, axis=0))
            imag_std.append(np.nanstd(d.imag, axis=0))
            bin_count.append(np.nansum(~np.isnan(d), axis=0))

        # get final statistics
        d_avg = np.array(real_avg) + 1j*np.array(imag_avg)
        f_min = np.array(f_min)
        d_std = np.array(real_std) + 1j*np.array(imag_std)
        d_num = np.array(bin_count).astype(np.float)

        # fill nans
        d_nan = np.isnan(d_avg)
        d_avg[d_nan] = 1.0
        f_min[d_nan] = True
        d_std[d_nan] = 1.0
        d_num[d_nan] = 0.0

        # insert into dictionaries
        data_avg[key] = d_avg
        flags_min[key] = f_min
        data_std[key] = d_std
        data_count[key] = d_num

    # turn into DataContainer objects
    data_avg = DataContainer(data_avg)
    flags_min = DataContainer(flags_min)
    data_std = DataContainer(data_std)
    data_count = DataContainer(data_count)

    return lst_bins, data_avg, flags_min, data_std, data_count


def lst_align(data, data_lsts, flags=None, dlst=None,
              verbose=True, atol=1e-10, **interp_kwargs):
    """
    Interpolate complex visibilities to align time integrations with an LST grid. An LST grid
    is defined as an array of points increasing in Local Sidereal Time, with each point marking
    the center of the LST bin.

    Parameters:
    -----------
    data : type=dictionary, DataContainer object holding complex visibility data

    data_lsts : type=ndarray, 1D monotonically increasing LST array in radians, except for a possible
                              phase wrap at 2pi

    flags : type=dictionary, flag dictionary of data. Can also be a wgts dictionary and will
                            convert appropriately.

    dlst : type=float, delta-LST spacing for lst_grid
    
    atol : type=float, absolute tolerance in comparing LST bins

    verbose : type=boolean, if True, print feedback to stdout

    interp_kwargs : type=dictionary, keyword arguments to feed to abscal.interp2d_vis

    Output: (interp_data, interp_flags, interp_lsts)
    -------
    interp_data : dictionary containing lst-aligned data

    interp_flags : dictionary containing flags for lst-aligned data

    interp_lsts : ndarray holding centers of LST bins.
    """
    # get lst if not fed grid
    if dlst is None:
        dlst = np.median(np.diff(data_lsts))

    # unwrap lsts
    if data_lsts[-1] < data_lsts[0]:
        data_lsts[data_lsts < data_lsts[0]] += 2*np.pi

    # make lst_grid
    lst_start = np.max([data_lsts[0] - 1e-5, 0])
    lst_grid = make_lst_grid(dlst, lst_start=lst_start, verbose=verbose)

    # get frequency info
    Nfreqs = data[data.keys()[0]].shape[1]
    data_freqs = np.arange(Nfreqs)
    model_freqs = np.arange(Nfreqs)

    # restrict lst_grid based on interpolate-able points
    lst_start = data_lsts[0]
    lst_end = data_lsts[-1]
    lst_grid = lst_grid[(lst_grid > lst_start - dlst/2 - atol) & (lst_grid < lst_end + dlst/2 + atol)]

    # interpolate data
    interp_data, interp_flags = abscal.interp2d_vis(data, data_lsts, data_freqs, lst_grid, model_freqs, flags=flags, **interp_kwargs)

    # wrap lst_grid
    lst_grid = lst_grid % (2*np.pi)

    return interp_data, interp_flags, lst_grid


def lst_align_arg_parser():
    a = argparse.ArgumentParser(description='LST align files with a universal LST grid')
    a.add_argument("data_files", nargs='*', type=str, help="miriad file paths to run LST align on.")
    a.add_argument("--file_ext", default=".L.{:7.5f}", type=str, help="file extension for LST-aligned data. must have one placeholder for starting LST.")
    a.add_argument("--outdir", default=None, type=str, help='directory for output files')
    a.add_argument("--dlst", type=float, default=None, help="LST grid interval spacing")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output files")
    a.add_argument("--miriad_kwargs", type=dict, default={}, help="kwargs to pass to miriad_to_data function")
    a.add_argument("--align_kwargs", type=dict, default={}, help="kwargs to pass to lst_align function")
    a.add_argument("--silence", default=False, action='store_true', help='silence output to stdout')
    return a


def lst_align_files(data_files, file_ext=".L.{:7.5f}", dlst=None,
                    overwrite=None, outdir=None, miriad_kwargs={}, align_kwargs={}, verbose=True):
    """
    Align a series of data files with a universal LST grid.

    Parameters:
    -----------
    data_files : type=list, list of paths to miriad files, or a single miriad file path

    file_ext : type=str, file_extension for each file in data_files when writing to disk

    dlst : type=float, LST grid bin interval, if None get it from first file in data_files

    overwrite : type=boolean, if True overwrite output files

    miriad_kwargs : type=dictionary, keyword arguments to feed to miriad_to_data()

    align_kwargs : keyword arguments to feed to lst_align()

    Result:
    -------
    A series of "data_files + file_ext" miriad files written to disk.
    """
    # check type of data_files
    if type(data_files) == str:
        data_files = [data_files]

    # get dlst if None
    if dlst is None:
        start, stop, int_time = utils.get_miriad_times(data_files[0])
        dlst = int_time

    # iterate over data files
    for i, f in enumerate(data_files):
        # load data
        (data, flgs, apos, ants, freqs, times, lsts,
         pols) = io.load_vis(f, return_meta=True)

        # lst align
        interp_data, interp_flgs, interp_lsts = lst_align(data, lsts, flags=flgs, dlst=dlst, **align_kwargs)

        # check output
        output_fname = os.path.basename(f) + file_ext.format(interp_lsts[0])

        # write to miriad file
        if overwrite is not None:
            miriad_kwargs['overwrite'] = overwrite
        if outdir is not None:
            miriad_kwargs['outdir'] = outdir
        miriad_kwargs['start_jd'] = np.floor(times[0])
        io.write_vis(output_fname, interp_data, interp_lsts, freqs, apos, flags=interp_flgs, verbose=verbose, filetype='miriad', **miriad_kwargs)


def lst_bin_arg_parser():
    """
    arg parser for lst_bin_files() function. data_files argument must be quotation-bounded
    glob-parsable search strings to nightly data. For example:

    '2458042/zen.2458042.*.xx.HH.uv' '2458043/zen.2458043.*.xx.HH.uv'

    """
    a = argparse.ArgumentParser(description="drive script for lstbin.lst_bin_files(). "
        "data_files argument must be quotation-bounded "
        "glob-parsable search strings to nightly data. For example: \n"
        "'2458042/zen.2458042.*.xx.HH.uv' '2458043/zen.2458043.*.xx.HH.uv' \n"
        "Consult lstbin.lst_bin_files() for further details on functionality.")
    a.add_argument('data_files', nargs='*', type=str, help="quotation-bounded, space-delimited, glob-parsable search strings to time-contiguous nightly data files")
    a.add_argument("--dlst", type=float, default=None, help="LST grid bin width")
    a.add_argument("--lst_start", type=float, default=0, help="starting LST for binner as it sweeps across 2pi LST")
    a.add_argument("--ntimes_per_file", type=int, default=60, help="number of LST bins to write per output file")
    a.add_argument("--file_ext", type=str, default="{}.{}.{:7.5f}.uv", help="file extension for output files. See lstbin.lst_bin_files doc-string for format specs.")
    a.add_argument("--outdir", default=None, type=str, help="directory for writing output")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output files")
    a.add_argument("--sig_clip", default=False, action='store_true', help="perform robust sigma clipping before binning")
    a.add_argument("--sigma", type=float, default=4.0, help="sigma threshold for sigma clipping")
    a.add_argument("--min_N", type=int, default=4, help="minimum number of points in bin needed to proceed with sigma clipping")
    a.add_argument("--rephase", default=False, action='store_true', help="rephase data to center of LST bin before binning")
    a.add_argument("--history", default=' ', type=str, help="history to insert into output files")
    a.add_argument("--miriad_kwargs", default={}, type=dict, help="dict w/ kwargs to pass to miriad_to_data function")
    a.add_argument("--atol", default=1e-6, type=float, help="absolute tolerance when comparing LST bin floats")
    a.add_argument("--silence", default=False, action='store_true', help='stop feedback to stdout')
    a.add_argument("--output_file_select", default=None, nargs='*', type=int, help="list of output file integers ot run on. Default is all output files.")
    return a


def config_lst_bin_files(data_files, dlst=None, atol=1e-10, lst_start=0.0, verbose=True, ntimes_per_file=60):
    """
    Configure lst grid, starting LST and output files given input data files and LSTbin params.

    Parameters
    ----------
    data_files : type=list of lists: nested set of lists, with each nested list containing
                 paths to miriad files from a particular night. These files should be sorted
                 by ascending Julian Date. Frequency axis of each file must be identical.

    dlst : type=float, LST bin width. If None, will get this from the first file in data_files.

    lst_start : type=float, starting LST for binner as it sweeps from lst_start to lst_start + 2pi.

    ntimes_per_file : type=int, number of LST bins in a single output file

    Returns (lst_grid, dlst, file_lsts, start_lst)
    -------
    lst_grid : float ndarray holding LST bin centers

    dlst : float, LST bin width of output lst_grid

    file_lsts : list, contains the lst grid of each output file

    start_lst : float, starting lst for LST binner
    """
    # get dlst from first data file if None
    if dlst is None:
        start, stop, int_time = utils.get_miriad_times(data_files[0][0])
        dlst = int_time

    # get start and stop times for each list of files in data_files.
    # add_int_buffer adds an integration to the end time of df[:-1] files,
    # and the %(2pi) ensures everything is within a 2pi LST grid.
    data_times = []
    for df in data_files:
        data_times.append(np.array(utils.get_miriad_times(df, add_int_buffer=True))[:2, :].T % (2*np.pi))

    # unwrap data_times less than lst_start, get starting and ending lst
    start_lst = 100
    end_lst = -1
    for dt in data_times:
        # unwrap starts below lst_start
        dt[:, 0][dt[:, 0] < lst_start - atol] += 2*np.pi

        # unwrap ends below starts
        dt[:, 1][dt[:, 1] < dt[:, 0] - atol] += 2*np.pi
 
        # get start and end lst
        start_lst = np.min(np.append(start_lst, dt[:, 0]))
        end_lst = np.max(np.append(end_lst, dt.ravel()))

    # create lst_grid
    lst_grid = make_lst_grid(dlst, lst_start=start_lst, verbose=verbose)
    dlst = np.median(np.diff(lst_grid))

    # get starting and stopping lst_grid indices
    start_diff = lst_grid - start_lst
    start_diff[start_diff < -dlst/2 - atol] = 100
    start_index = np.argmin(start_diff)
    end_diff = lst_grid - end_lst
    end_diff[end_diff > dlst/2 + atol] = -100
    end_index = np.argmax(end_diff)

    # get number of output files
    nfiles = int(np.ceil(float(end_index - start_index) / ntimes_per_file))

    # get output file lsts
    file_lsts = [lst_grid[start_index:end_index][ntimes_per_file*i:ntimes_per_file*(i+1)] for i in range(nfiles)]

    return data_times, lst_grid, dlst, file_lsts, start_lst


def lst_bin_files(data_files, dlst=None, verbose=True, ntimes_per_file=60, file_ext="{}.{}.{:7.5f}.uv",
                  outdir=None, overwrite=False, history=' ', lst_start=0, atol=1e-6, sig_clip=True,
                  sigma=5.0, min_N=5, rephase=False, miriad_kwargs={}, output_file_select=None):
    """
    LST bin a series of miriad files with identical frequency bins, but varying
    time bins. Output miriad file meta data (frequency bins, antennas positions, time_array)
    are taken from the first file in data_files.

    Parameters:
    -----------
    data_files : type=list of lists: nested set of lists, with each nested list containing
                 paths to miriad files from a particular night. These files should be sorted
                 by ascending Julian Date. Frequency axis of each file must be identical.

    dlst : type=float, LST bin width. If None, will get this from the first file in data_files.

    lst_start : type=float, starting LST for binner as it sweeps from lst_start to lst_start + 2pi.

    ntimes_per_file : type=int, number of LST bins in a single output file

    file_ext : type=str, extension to "zen." for output miriad files. This must have three
               formatting placeholders, first for polarization(s), second for type of file
               Ex. ["LST", "STD", "NUM"] and third for starting LST bin of file.

    outdir : type=str, output directory

    overwrite : type=bool, if True overwrite output files

    rephase : type=bool, if True, rephase data points in LST bin to center of bin

    bin_kwargs : type=dictionary, keyword arguments for lst_bin.

    atol : type=float, absolute tolerance for LST bin float comparison

    miriad_kwargs : type=dictionary, keyword arguments to pass to io.write_vis()

    output_file_select : type=int or integer list, list of integer indices of the output files to run on.
        Default is all files.

    Result:
    -------
    zen.{pol}.LST.{file_lst}.uv : holds LST bin avg (data_array) and bin count (nsample_array)
    zen.{pol}.STD.{file_lst}.uv : holds LST bin stand dev along real and imag (data_array)
    """
    (data_times, lst_grid, dlst, file_lsts, 
     start_lst) = config_lst_bin_files(data_files, dlst=dlst, lst_start=lst_start, 
                                       ntimes_per_file=ntimes_per_file, verbose=verbose)
    nfiles = len(file_lsts)

    # select file_lsts
    if output_file_select is not None:
        if isinstance(output_file_select, (int, np.int)):
            output_file_select = [output_file_select]
        try:
            file_lsts = map(lambda i: file_lsts[i], output_file_select)
        except IndexError:
            print "Warning: one or more indices in output_file_select {} caused an index error with length {} file_lsts list, exiting...".format(output_file_select, nfiles)
            return

    # create data file status: None if not opened, data objects if opened
    data_status = map(lambda d: map(lambda f: None, d), data_files)

    # get outdir
    if outdir is None:
        outdir = os.path.dirname(os.path.commonprefix(abscal.flatten(data_files)))

    # update miriad_kwrgs
    miriad_kwargs['outdir'] = outdir
    miriad_kwargs['overwrite'] = overwrite
 
    # get metadata from the zeroth data file
    d, fl, ap, a, f, t, l, p = io.load_vis(data_files[0][0], return_meta=True, pick_data_ants=False)

    # push time and lst arrays forward by half an integration
    abscal.echo("pushing time and lst arrays forward by half an integration b/c pyuvdata "\
                "currently does not do this for us....", verbose=verbose)
    if len(t) > 1:
        t += np.median(np.diff(t)) / 2.0

    # get frequency, time and antenna position information
    freq_array = copy.copy(f)
    antpos = copy.deepcopy(ap)
    start_jd = np.floor(t)[0]
    miriad_kwargs['start_jd'] = start_jd
    miriad_kwargs['integration_time'] = np.median(np.diff(t)) * 24 * 3600.
    del d, fl, ap, a, f, t, l, p
    garbage_collector.collect()

    # iterate over output LST files
    for i, f_lst in enumerate(file_lsts):
        abscal.echo("LST file {} / {}: {}".format(i+1, len(file_lsts), datetime.datetime.now()), type=1, verbose=verbose)

        # create empty data_list and lst_list
        data_list = []
        file_list = []
        flgs_list = []
        lst_list = []

        # locate all data files that fall within the range of lst for this output file
        f_min = np.min(f_lst)
        f_max = np.max(f_lst)
        f_select = np.array(map(lambda d: map(lambda f: (f[1] > f_min)&(f[0] < f_max + atol), d), data_times))
        if i == 0:
            old_f_select = copy.copy(f_select)

        # open necessary files, close ones that are no longer needed
        # append necessary files to data_list
        for j in range(len(data_files)):
            nightly_data_list = []
            nightly_flgs_list = []
            nightly_lst_list = []
            for k in range(len(data_files[j])):
                if f_select[j][k] == True and data_status[j][k] is None:
                    # open file
                    d, fl, ap, a, f, t, l, p = io.load_vis(data_files[j][k], return_meta=True)
                    d.phase_type = 'drift'

                    # unwrap l relative to itself
                    l[l < l[0]] += 2*np.pi

                    # push time and lst arrays forward by half an integration
                    # b/c pyuvdata does not currently do this for us....
                    if len(l) > 1:
                        l += np.median(np.diff(l)) / 2.0
                        t += np.median(np.diff(t)) / 2.0

                    # unwrap l relative to start_lst
                    l[l < start_lst - atol] += 2*np.pi

                    # pass data references to data_status list
                    data_status[j][k] = [d, fl, ap, a, f, t, l, p]

                    # erase references from namespace
                    del d, fl, ap, a, f, t, l, p

                elif f_select[j][k] == False and old_f_select[j][k] == True:
                    # close file
                    del data_status[j][k]
                    data_status[j].insert(k, None)

                # if file is needed, append data references to data_list
                if f_select[j][k] == True:
                    file_list.append(data_files[j][k])
                    nightly_data_list.append(data_status[j][k][0])  # this is data
                    nightly_flgs_list.append(data_status[j][k][1])  # this is flgs
                    nightly_lst_list.append(data_status[j][k][6])  # this is lsts

            # skip if nothing accumulated in nightly files
            if len(nightly_data_list) == 0:
                continue

            # extend to data lists
            data_list.extend(nightly_data_list)
            flgs_list.extend(nightly_flgs_list)
            lst_list.extend(nightly_lst_list)

            del nightly_data_list, nightly_flgs_list, nightly_lst_list

        # skip if data_list is empty
        if len(data_list) == 0:
            abscal.echo("data_list is empty for beginning LST {}".format(f_lst[0]), verbose=verbose)

            # erase data references
            del file_list, data_list, flgs_list, lst_list

            # assign old f_select
            old_f_select = copy.copy(f_select)
            continue

        # pass through lst-bin function
        (bin_lst, bin_data, flag_data, std_data,
         num_data) = lst_bin(data_list, lst_list, flags_list=flgs_list, dlst=dlst, lst_start=start_lst,
                             lst_low=f_min, lst_hi=f_max, truncate_empty=False, sig_clip=sig_clip, 
                             sigma=sigma, min_N=min_N, rephase=rephase, freq_array=freq_array, antpos=antpos)

        # push lst_bins back to bin start rather than bin center
        # b/c pyuvdata does not do this for us yet...
        abscal.echo("pushing time and lst arrays backward by half an integration b/c pyuvdata "\
                    "currently does not do this for us....", verbose=verbose)
        # unwrap bin_lst
        bin_lst[bin_lst < bin_lst[0] - atol] += 2*np.pi
        # push back
        if len(bin_lst) > 1:
            bin_lst -= np.median(np.diff(bin_lst)) / 2.0
        # re-wrap bin_lst
        bin_lst = bin_lst % (2*np.pi)

        # update history
        file_history = history + " Input files: " + "-".join(map(lambda ff: os.path.basename(ff), file_list))
        miriad_kwargs['history'] = file_history

        # erase data references
        del file_list, data_list, flgs_list, lst_list
        garbage_collector.collect()

        # assign old f_select
        old_f_select = copy.copy(f_select)

        # get polarizations
        pols = bin_data.pols()

        # configure filenames
        bin_file = "zen.{}".format(file_ext.format('.'.join(pols), "LST", bin_lst[0]))
        std_file = "zen.{}".format(file_ext.format('.'.join(pols), "STD", bin_lst[0]))

        # check for overwrite
        if os.path.exists(bin_file) and overwrite is False:
            abscal.echo("{} exists, not overwriting".format(bin_file), verbose=verbose)
            continue

        # write to file
        io.write_vis(bin_file, bin_data, bin_lst, freq_array, antpos, flags=flag_data, verbose=verbose, nsamples=num_data, filetype='miriad', **miriad_kwargs)
        io.write_vis(std_file, std_data, bin_lst, freq_array, antpos, flags=flag_data, verbose=verbose, nsamples=num_data, filetype='miriad', **miriad_kwargs)

        del bin_file, std_file, bin_data, std_data, num_data, bin_lst, flag_data
        garbage_collector.collect()


def make_lst_grid(dlst, lst_start=None, verbose=True):
    """
    Make a uniform grid in local sidereal time spanning 2pi radians.

    Parameters:
    -----------
    dlst : type=float, delta-LST: width of a single LST bin in radians. 2pi must be equally divisible 
                by dlst. If not, will default to the closest dlst that satisfies this criterion that
                is also greater than the input dlst. There is a minimum allowed dlst of 6.283e-6 radians,
                or .0864 seconds.

    lst_start : type=float, starting point for lst_grid, which extends out 2pi from lst_start.
                lst_start must fall exactly on an LST bin given a dlst, within 0-2pi. If not, it is
                replaced with the closest bin. Default is lst_start at zero radians.

    Output:
    -------
    lst_grid : type=ndarray, dtype=float, uniform LST grid marking the center of each LST bin
    """
    # check 2pi is equally divisible by dlst
    if (np.isclose((2*np.pi / dlst) % 1, 0.0, atol=1e-5) is False) and (np.isclose((2*np.pi / dlst) % 1, 1.0, atol=1e-5) is False):
        # generate array of appropriate dlsts
        dlsts = 2*np.pi / np.arange(1, 1000000).astype(np.float)

        # get dlsts closest to dlst, but also greater than dlst
        dlst_diff = dlsts - dlst
        dlst_diff[dlst_diff < 0] = 10
        new_dlst = dlsts[np.argmin(dlst_diff)]
        abscal.echo("2pi is not equally divisible by input dlst ({:.16f}) at 1 part in 1e7.\n"
                    "Using {:.16f} instead.".format(dlst, new_dlst), verbose=verbose)
        dlst = new_dlst

    # make an lst grid from [0, 2pi), with the first bin having a left-edge at 0 radians.
    lst_grid = np.arange(0, 2*np.pi-1e-7, dlst) + dlst / 2

    # shift grid by lst_start
    if lst_start is not None:
        # enforce lst_start to be within 0-2pi, else replace with 0
        if lst_start < 0 or lst_start >= 2*np.pi:
            abscal.echo("lst_start was < 0 or >= 2pi, taking modulus with (2pi)", verbose=verbose)
            lst_start = lst_start % (2*np.pi)
        lst_start = lst_grid[np.argmin(np.abs(lst_grid - lst_start))] - dlst/2
        lst_grid += lst_start

    return lst_grid


def sigma_clip(array, flags=None, sigma=4.0, axis=0, min_N=4):
    """
    one-iteration robust sigma clipping algorithm. returns clip_flags array.
    Warning: this function will directly replace flagged and clipped data in array with
    a np.nan, so as to not make a copy of array.

    Parameters:
    -----------
    array : ndarray of complex visibility data. If 2D, [0] axis is samples and [1] axis is freq.

    flags : ndarray matching array shape containing boolean flags. True if flagged.

    sigma : float, sigma threshold to cut above

    axis : int, axis of array to sigma clip

    min_N : int, minimum length of array to sigma clip, below which no sigma
                clipping is performed.

    return_arrs : type=boolean, if True, return array and flags
    
    Output: flags
    -------
    clip_flags : type=boolean ndarray, has same shape as input array, but has clipped
                 values set to True. Also inherits any flagged data from flags array
                 if passed.
    """
    # ensure array is an array
    if type(array) is not np.ndarray:
        array = np.array(array)

    # ensure array passes min_N criteria:
    if array.shape[axis] < min_N:
        return np.zeros_like(array, np.bool)

    # create empty clip_flags array
    clip_flags = np.zeros_like(array, np.bool)

    # inherit flags if fed and apply flags to data
    if flags is not None:
        clip_flags += flags
        array[flags] *= np.nan

    # get robust location
    location = np.nanmedian(array, axis=axis)

    # get MAD! * 1.482579
    scale = np.nanmedian(np.abs(array - location), axis=axis) * 1.482579

    # get clipped data
    clip = np.abs(array-location)/scale > sigma

    # set clipped data to nan and set clipped flags to True
    array[clip] *= np.nan
    clip_flags[clip] = True

    return clip_flags


def switch_bl(key):
    """
    switch antenna ordering in (ant1, ant2, pol) key
    where ant1 and ant2 are ints and pol is a two-char str
    Ex. (1, 2, 'xx')
    """
    return (key[1], key[0], key[2][::-1])


class LSTBINPHASED(Exception):
    """ custom exception for lst_bin() """
    pass

