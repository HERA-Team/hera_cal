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
from hera_cal import omni, utils, firstcal, cal_formats, redcal, abscal
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


def lst_bin(data_list, lst_list, flags_list=None, dlst=None, lst_start=0, 
            lst_low=None, lst_hi=None, atol=1e-6, median=False, truncate_empty=True,
            sig_clip=False, sigma=2.0, min_N=4, return_no_avg=False, verbose=True):
    """
    Bin data in Local Sidereal Time (LST) onto an LST grid. An LST grid
    is defined as an array of points increasing in Local Sidereal Time, with each point marking
    the center of the LST bin.

    Parameters:
    -----------
    data_list : type=list, list of DataContainer dictionaries holding complex visibility data

    lst_list : type=list, list of ndarrays holding LST stamps of each data dictionary in data_list.
                These LST arrays must be monotonically increasing, except for a possible wrap at 2pi.
    
    flags_list : type=list, list of DataContainer dictionaries holding flags for each dict in data_list

    dlst : type=float, delta-LST spacing for lst_grid. If None, will use the delta-LST of the first
            array in lst_list.

    lst_low : type=float, lower bound in lst_grid centers used for LST binning

    lst_hi : type=float, upper bound in lst_grid centers used for LST binning

    atol : type=float, absolute tolerance for comparing floats in lst_list to floats in lst_grid

    median : type=boolean, if True use median for LST binning, else use mean

    sig_clip : type=boolean, if True, perform a sigma clipping algorithm of the LST bins
            on the real and imag components separately. Warning: This considerably slows down the code.

    sigma : type=float, input standard deviation to use for sigma clipping algorithm.

    min_N : type=int, minimum number of points in LST bin to perform LST binning

    return_no_avg : type=boolean, if True, return binned but un-averaged data and flags.

    Output: (lst_bins, data_avg, flags_min, data_std, data_count)
    -------
    lst_bins : ndarray containing final lst grid of data

    data_avg : dictionary of data having averaged the LST bins

    flags_min : dictionary of minimum data flag in each LST bin
    
    data_std : dictionary of data with real component holding std along real axis
        and imag component holding std along imag axis

    data_count : dictionary containing the number count of data points in each LST bin.

    if return_no_avg:
        Output: (data_bin, flags_min)
        data_bin : dictionary with (ant1,ant2,pol) as keys and ndarrays holding
            un-averaged complex visibilities in each LST bin as values. 
    """
    # get visibility shape
    Ntimes, Nfreqs = data_list[0][data_list[0].keys()[0]].shape

    # get dlst if not provided
    if dlst is None:
        dlst = np.median(np.diff(lst_list[0]))

    # construct lst_grid
    lst_grid = make_lst_grid(dlst, lst_start=lst_start, verbose=verbose)

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
        l = lst_list[i]

        # digitize data lst array "l"
        grid_indices = np.digitize(l, lst_grid_left[1:], right=True)

        # make data_in_bin boolean array, and set to False data that don't fall in any bin
        data_in_bin = np.ones_like(l, np.bool)
        data_in_bin[(l<lst_grid_left.min()-atol)] = False
        data_in_bin[(l>lst_grid_left.max()+dlst+atol)] = False

        # update all_lst_indices
        all_lst_indices.update(set(grid_indices[data_in_bin]))

        # iterate over keys in d
        for j, key in enumerate(d.keys()):
            # conjugate d[key] if necessary
            if key in data:
                pass
            elif switch_bl(key) in data:
                key = switch_bl(key)
                d[key] = np.conj(d[switch_bl(key)])
                if flags_list is not None:
                    flags_list[i][key] = flags_list[i][switch_bl(key)]
            else:
                data[key] = odict()
                flags[key] = odict()

            # iterate over grid_indices, and append to data if data_in_bin is True
            for k, ind in enumerate(grid_indices):
                if data_in_bin[k]:
                    if ind not in data[key]:
                        data[key][ind] = []
                        flags[key][ind] = []
                    data[key][ind].append(d[key][k])
                    if flags_list is None:
                        flags[key][ind].append(np.zeros_like(d[key][k], np.bool))
                    else:
                        flags[key][ind].append(flags_list[i][key][k])

    # get final lst_bin array
    if truncate_empty:
        # use only lst_grid bins that have data in them
        lst_bins = lst_grid[sorted(all_lst_indices)]
    else:
        # keep all lst_grid bins and fill data and flags appropriately
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

    # wrap lst_bins
    lst_bins = lst_bins % (2*np.pi)

    # make final dictionaries
    flags_min = odict()
    data_avg = odict()
    data_count = odict()
    data_std = odict()

    # return un-averaged data if desired
    if return_no_avg:
        # return all binned data instead of just bin average 
        data_bin = odict(map(lambda k: (k, np.array(odict(map(lambda k2: (k2, data[k][k2]), sorted(data[k].keys()))).values())), data.keys()))
        flags_bin = odict(map(lambda k: (k, np.array(odict(map(lambda k2: (k2, flags[k][k2]), sorted(flags[k].keys()))).values())), flags.keys()))

        return data_bin, flags_bin

    # iterate over data and get statistics
    for i, key in enumerate(data.keys()):
        if sig_clip:
            for j, ind in enumerate(data[key].keys()):
                d = np.array(data[key][ind])
                f = np.array(flags[key][ind])
                f[np.isnan(f)] = True
                data[key][ind] = sigma_clip(d.real, flags=f, sigma=sigma, min_N=min_N) + 1j*sigma_clip(d.imag, flags=f, sigma=sigma, min_N=min_N)
                flags[key][ind] = f
        if median:
            real_avg = np.array(map(lambda ind: np.nanmedian(map(lambda r: r.real, data[key][ind]), axis=0), sorted(data[key].keys())))
            imag_avg = np.array(map(lambda ind: np.nanmedian(map(lambda r: r.imag, data[key][ind]), axis=0), sorted(data[key].keys())))
            f_min = np.array(map(lambda ind: np.nanmin(flags[key][ind], axis=0), sorted(flags[key].keys())))
        else:
            real_avg = np.array(map(lambda ind: np.nanmean(map(lambda r: r.real, data[key][ind]), axis=0), sorted(data[key].keys())))
            imag_avg = np.array(map(lambda ind: np.nanmean(map(lambda r: r.imag, data[key][ind]), axis=0), sorted(data[key].keys())))
            f_min = np.array(map(lambda ind: np.nanmin(flags[key][ind], axis=0), sorted(flags[key].keys())))
        real_stan_dev = np.array(map(lambda ind: np.nanstd(map(lambda r: r.real, data[key][ind]), axis=0), sorted(data[key].keys())))
        imag_stan_dev = np.array(map(lambda ind: np.nanstd(map(lambda r: r.imag, data[key][ind]), axis=0), sorted(data[key].keys())))
        num_pix = np.array(map(lambda ind: np.nansum(map(lambda r: r.real*0+1, data[key][ind]), axis=0), sorted(data[key].keys())))

        data_avg[key] = real_avg + 1j*imag_avg
        flags_min[key] = f_min
        data_std[key] = real_stan_dev + 1j*imag_stan_dev
        data_count[key] = num_pix.astype(np.complex)

    # turn into DataContainer
    data_avg = DataContainer(data_avg)
    flags_min = DataContainer(flags_min)
    data_std = DataContainer(data_std)
    data_count = DataContainer(data_count)

    return lst_bins, data_avg, flags_min, data_std, data_count


def lst_align(data, data_lsts, flags=None, dlst=None,
              verbose=True, atol=1e-6, **interp_kwargs):
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
    a.add_argument("--longitude", type=float, default=21.42830, help="longitude of observer in degrees east")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output files")
    a.add_argument("--miriad_kwargs", type=dict, default={}, help="kwargs to pass to miriad_to_data function")
    a.add_argument("--align_kwargs", type=dict, default={}, help="kwargs to pass to lst_align function")
    a.add_argument("--silence", default=False, action='store_true', help='silence output to stdout')
    return a


def lst_align_files(data_files, file_ext=".L.{:7.5f}", dlst=None, longitude=21.42830,
                    overwrite=None, outdir=None, miriad_kwargs={}, align_kwargs={}, verbose=True):
    """
    Align a series of data files with a universal LST grid.

    Parameters:
    -----------
    data_files : type=list, list of paths to miriad files, or a single miriad file path

    file_ext : type=str, file_extension for each file in data_files when writing to disk

    dlst : type=float, LST grid bin interval, if None get it from first file in data_files

    longitude : type=float, longitude of observer in degrees east

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
         pols) = abscal.UVData2AbsCalDict(f, return_meta=True, return_wgts=False)

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
        data_to_miriad(output_fname, interp_data, interp_lsts, freqs, apos, flags=interp_flgs, verbose=verbose, **miriad_kwargs)


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
    a.add_argument("--lst_init", type=float, default=np.pi, help="starting point for universal LST grid")
    a.add_argument("--dlst", type=float, default=None, help="LST grid bin width")
    a.add_argument("--lst_start", type=float, default=0, help="starting LST for binner as it sweeps across 2pi LST")
    a.add_argument("--lst_low", default=None, type=float, help="enact a lower bound on LST grid")
    a.add_argument("--lst_hi", default=None, type=float, help="enact an upper bound on LST grid")
    a.add_argument("--ntimes_per_file", type=int, default=60, help="number of LST bins to write per output file")
    a.add_argument("--file_ext", type=str, default="{}.{}.{:7.5f}.uv", help="file extension for output files. See lstbin.lst_bin_files doc-string for format specs.")
    a.add_argument("--pol_select", nargs='*', type=str, default=None, help="polarization strings to use in data_files")
    a.add_argument("--outdir", default=None, type=str, help="directory for writing output")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output files")
    a.add_argument("--history", default=' ', type=str, help="history to insert into output files")
    a.add_argument("--atol", default=1e-6, type=float, help="absolute tolerance when comparing LST bin floats")
    a.add_argument('--align', default=False, action='store_true', help='perform LST align before binning')
    a.add_argument("--align_kwargs", default={}, type=dict, help="dict w/ kwargs for lst_align if --align")
    a.add_argument("--bin_kwargs", default={}, type=dict, help="dict w/ kwargs to pass to lst_bin function")
    a.add_argument("--miriad_kwargs", default={}, type=dict, help="dict w/ kwargs to pass to miriad_to_data function")
    a.add_argument("--silence", default=False, action='store_true', help='stop feedback to stdout')
    return a


def lst_bin_files(data_files, dlst=None, verbose=True, ntimes_per_file=60, file_ext="{}.{}.{:7.5f}.uv",
                  pol_select=None, outdir=None, overwrite=False, history=' ', lst_start=0,
                  align=False, align_kwargs={}, bin_kwargs={},
                  atol=1e-6, miriad_kwargs={}):
    """
    LST bin a series of miriad files with identical frequency bins, but varying
    time bins. Miriad file meta data (frequency bins, antennas positions, time_array)
    are taken from the first file in data_files.

    Parameters:
    -----------
    data_files : type=list of lists: nested set of lists, with each nested list containing
            paths to miriad files from a particular night. Frequency axis of each file must 
            be identical.

    dlst : type=float, LST grid bin spacing. If None will get this from the first file in data_files.

    lst_start : type=float, starting LST for binner as it sweeps from lst_start to lst_start + 2pi.

    ntimes_per_file : type=int, number of LST bins in a single file

    file_ext : type=str, extension to "zen." for output miriad files. must have three
            formatting placeholders, first for polarization(s), second for type of file
            Ex. ["LST", "STD", "NUM"] and third for starting LST bin of file.

    pol_select : type=list, list of polarization strings Ex. ['xx'] to select in data_files

    outdir : type=str, output directory

    overwrite : type=bool, if True overwite output files

    align : type=bool, if True, concatenate nightly data and LST align with the lst_grid.
        Warning : slows down code somewhat

    align_kwargs : type=dictionary, keyword arugments for lst_align not included in above kwars.

    bin_kwargs : type=dictionary, keyword arguments for lst_bin.

    atol : type=float, absolute tolerance for LST bin float comparison

    miriad_kwargs : type=dictionary, keyword arguments to pass to data_to_miriad()

    Result:
    -------
    if write_miriad:
        zen.pol.LST.*.*.uv : containing LST-binned data
        zen.pol.STD.*.*.uv : containing stand dev of LST bin
        zen.pol.NUM.*.*.uv : containing number of points in LST bin
    """
    # get dlst from first data file if None
    if dlst is None:
        start, stop, int_time = utils.get_miriad_times(data_files[0][0])
        dlst = int_time

    # get file start and stop times
    data_times = map(lambda f: np.array(utils.get_miriad_times(f, add_int_buffer=True)).T[:, :2] % (2*np.pi), data_files)

    # unwrap data_times less than lst_start, get starting and ending lst
    start_lst = 100
    end_lst = -1
    for dt in data_times:
        # unwrap starts below lst_start
        dt[:, 0][dt[:, 0] < lst_start] += 2*np.pi

        # get start and end lst
        start_lst = np.min(np.append(start_lst, dt[:, 0]))
        end_lst = np.max(np.append(end_lst, dt.ravel()))

    # create lst_grid
    lst_grid = make_lst_grid(dlst, lst_start=start_lst, verbose=verbose)
    dlst = np.median(np.diff(lst_grid))

    # get starting and stopping indices
    start_diff = lst_grid - start_lst
    start_diff[start_diff < -dlst/2 - atol] = 100
    start_index = np.argmin(start_diff)
    end_diff = lst_grid - end_lst
    end_diff[end_diff > dlst/2 + atol] = -100
    end_index = np.argmax(end_diff)

    # get number of files
    nfiles = int(np.ceil(float(end_index - start_index) / ntimes_per_file))

    # get file lsts
    file_lsts = [lst_grid[start_index:end_index][ntimes_per_file*i:ntimes_per_file*(i+1)] for i in range(nfiles)]

    # create data file status: None if not opened, data object if opened
    data_status = map(lambda d: map(lambda f: None, d), data_files)

    # get outdir
    if outdir is None:
        outdir = os.path.dirname(os.path.commonprefix(abscal.flatten(data_files)))

    # update miriad_kwrgs
    miriad_kwargs['outdir'] = outdir
    miriad_kwargs['overwrite'] = overwrite
 
    # get frequency and antennas position information from the first data_files
    d, fl, ap, a, f, t, l, p = abscal.UVData2AbsCalDict(data_files[0][0], return_meta=True, pick_data_ants=False)
    freq_array = copy.copy(f)
    antpos = copy.deepcopy(ap)
    start_jd = np.floor(t)[0]
    miriad_kwargs['start_jd'] = start_jd
    del d, fl, ap, a, f, t, l, p
    garbage_collector.collect()

    # iterate over end-result LST files
    for i, f_lst in enumerate(file_lsts):
        abscal.echo("LST file {} / {}: {}".format(i+1, nfiles, datetime.datetime.now()), type=1, verbose=verbose)
        # create empty data_list and lst_list
        data_list = []
        file_list = []
        flgs_list = []
        lst_list = []

        # locate all files that fall within this range of lsts
        f_min = np.min(f_lst)
        f_max = np.max(f_lst)
        f_select = np.array(map(lambda d: map(lambda f: (f[1] >= f_min)&(f[0] <= f_max), d), data_times))
        if i == 0:
            old_f_select = copy.copy(f_select)

        # open necessary files, close ones that are no longer needed
        for j in range(len(data_files)):
            nightly_data_list = []
            nightly_flgs_list = []
            nightly_lst_list = []
            for k in range(len(data_files[j])):
                if f_select[j][k] == True and data_status[j][k] is None:
                    # open file(s)
                    d, fl, ap, a, f, t, l, p = abscal.UVData2AbsCalDict(data_files[j][k], return_meta=True, pol_select=pol_select)

                    # unwrap l
                    l[np.where(l < start_lst)] += 2*np.pi

                    # pass reference to data_status
                    data_status[j][k] = [d, fl, ap, a, f, t, l, p]

                    # erase unnecessary references
                    del d, fl, ap, a, f, t, l, p

                elif f_select[j][k] == False and old_f_select[j][k] == True:
                    # erase reference
                    del data_status[j][k]
                    data_status[j].insert(k, None)

                # copy references to data_list
                if f_select[j][k] == True:
                    file_list.append(data_files[j][k])
                    nightly_data_list.append(data_status[j][k][0])
                    nightly_flgs_list.append(data_status[j][k][1])
                    nightly_lst_list.append(data_status[j][k][6])

            # skip if nothing accumulated in nightly files
            if len(nightly_data_list) == 0:
                continue

            # align nightly data if desired, this involves making a copy of the raw data,
            # and then interpolating it (another copy)
            if align:
                # concatenate data across night
                night_data = reduce(operator.add, nightly_data_list)
                night_flgs = reduce(operator.add, nightly_flgs_list)
                night_lsts = np.concatenate(nightly_lst_list)

                del nightly_data_list, nightly_flgs_list, nightly_lst_list

                # align data
                night_data, night_flgs, night_lsts = lst_align(night_data, night_lsts, flags=night_flgs,
                                                               dlst=dlst, atol=atol, **align_kwargs)

                nightly_data_list = [night_data]
                nightly_flgs_list = [night_flgs]
                nightly_lst_list = [night_lsts]

                del night_data, night_flgs, night_lsts

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
                             lst_low=f_min, lst_hi=f_max, truncate_empty=False, **bin_kwargs)

        # make sure bin_lst is wrapped
        bin_lst = bin_lst % (2*np.pi)

        # update history
        file_history = history + "input files: " + "-".join(map(lambda ff: os.path.basename(ff), file_list))
        miriad_kwargs['history'] = file_history

        # erase data references
        del file_list, data_list, flgs_list, lst_list
        garbage_collector.collect()

        # assign old f_select
        old_f_select = copy.copy(f_select)

        # get polarizations
        pols = bin_data.pols()

        # configure filename
        bin_file = "zen.{}".format(file_ext.format('.'.join(pols), "LST", bin_lst[0]))
        std_file = "zen.{}".format(file_ext.format('.'.join(pols), "STD", bin_lst[0]))
        num_file = "zen.{}".format(file_ext.format('.'.join(pols), "NUM", bin_lst[0]))

        # check for overwrite
        if os.path.exists(bin_file) and overwrite is False:
            abscal.echo("{} exists, not overwriting".format(bin_file), verbose=verbose)
            continue

        # write to file
        data_to_miriad(bin_file, bin_data, bin_lst, freq_array, antpos, flags=flag_data, verbose=verbose, **miriad_kwargs)
        data_to_miriad(std_file, std_data, bin_lst, freq_array, antpos, verbose=verbose, **miriad_kwargs)
        data_to_miriad(num_file, num_data, bin_lst, freq_array, antpos, verbose=verbose, **miriad_kwargs)

        del bin_file, std_file, num_file, bin_data, std_data, num_data, bin_lst, flag_data
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

    lst_start : type=float, starting point for lst_grid, extending out 2pi from lst_start.
                            lst_start must fall exactly on an LST bin given dlst. If not, it is
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
        abscal.echo("2pi is not equally divisible by input dlst ({:.16f}) at 1 part in 1e5.\n"
                    "Using {:.16f} instead.".format(dlst, new_dlst), verbose=verbose)
        dlst = new_dlst

    # make an lst grid from [0, 2pi), with the first bin having a left-edge at 0 radians.
    lst_grid = np.arange(0, 2*np.pi-1e-6, dlst) + dlst / 2

    # shift grid by lst_start
    if lst_start is not None:
        lst_start = lst_grid[np.argmin(np.abs(lst_grid - lst_start))] - dlst/2
        lst_grid += lst_start

    return lst_grid


def lst_rephase(data, bls, freqs, dlst, lat=-30.72152):
    """
    Shift phase center of each integration in data by amount dlst [radians] along right ascension axis.
    This function directly edits the arrays in 'data' in memory, so as not to make a copy of data.

    Parameters:
    -----------
    data : type=DataContainer, holding 2D visibility data, with [0] axis time and [1] axis frequency

    bls : type=dictionary, same keys as data, values are 3D float arrays holding baseline vector
                            in ENU frame in meters

    freqs : type=ndarray, frequency array of data [Hz]

    dlst : type=ndarray or float, delta-LST to rephase by [radians]. If a float, shift all integrations
                by dlst, elif an ndarray, shift each integration by different amount w/ shape=(Ntimes)

    lat : type=float, latitude of observer in degrees South
    """
    # get top2eq matrix
    top2eq = uvutils.top2eq_m(0, lat*np.pi/180)

    # check format of dlst
    if type(dlst) == list or type(dlst) == np.ndarray:
        lat = np.ones_like(dlst) * lat
        zero = np.zeros_like(dlst)

    else:
        zero = 0

    # get eq2top matrix
    eq2top = uvutils.eq2top_m(dlst, lat*np.pi/180)

    # get full rotation matrix
    rot = eq2top.dot(top2eq)

    # iterate over data keys
    for i, k in enumerate(data.keys()):

        # dot bls with new s-hat vector
        u = bls[k].dot(rot.dot(np.array([0, 0, 1])).T)

        # reshape u
        if type(u) == np.ndarray:
            pass
        else:
            u = np.array([u])

        # get phasor
        phs = np.exp(-2j*np.pi*freqs[None, :]*u[:, None]/aipy.const.c*100)

        # multiply into data
        data[k] *= phs


def data_to_miriad(fname, data, lst_array, freq_array, antpos, time_array=None, flags=None,
                   outdir="./", write_miriad=True, overwrite=False, verbose=True, history=" ", return_uvdata=False,
                   longitude=21.42830, start_jd=None, instrument="HERA", telescope_name="HERA",
                   object_name='EOR', phase_type='drift', vis_units='uncalib', dec=-30.72152,
                   telescope_location=np.array([5109325.85521063,2005235.09142983,-3239928.42475395])):
    """
    Take data dictionary, export to UVData object and write as a miriad file. See pyuvdata.UVdata
    documentation for more info on these attributes.

    Parameters:
    -----------
    data : type=dictinary, DataContainer dictionary of complex visibility data

    lst_array : type=ndarray, array containing unique LST time bins of data

    freq_array : type=ndarray, array containing frequency bins of data (Hz)

    antpos : type=dictionary, antenna position dictionary. keys are ant ints and values are position vectors

    time_array : type=ndarray, array containing unique Julian Date time bins of data

    flags : type=dictionary, DataContainer dictionary matching data in shape, holding flags of data.

    outdir : type=str, output directory

    write_miriad : type=boolean, if True, write data to miriad file

    overwrite : type=boolean, if True, overwrite output files

    verbose : type=boolean, if True, report feedback to stdout

    history : type=str, history string for UVData object

    return_uvdata : type=boolean, if True return UVData instance

    longitude : type=float, longitude of observer in degrees East

    dec : type=float, declination of observer in degrees South

    start_jd : type=float, starting julian date of time_array if time_array is None

    instrument : type=str, instrument name

    telescope_name : type=str, telescope name

    object_name : type=str, observing object name

    phase_type : type=str, phasing type

    vis_unit : type=str, visibility units

    telescope_location : type=ndarray, telescope location in xyz in ITRF (earth-centered frame)

    Output:
    -------
    if return_uvdata: return UVData instance
    """
    ## configure UVData parameters
    # get pols
    pols = np.unique(map(lambda k: k[-1], data.keys()))
    Npols = len(pols)
    pol2int = {'xx':-5, 'yy':-6, 'xy':-7, 'yx':-8}
    polarization_array = np.array(map(lambda p: pol2int[p], pols))

    # get times
    if time_array is None:
        if start_jd is None:
            raise AttributeError("if time_array is not fed, start_jd must be fed")
        time_array = np.array(map(lambda lst: utils.LST2JD(lst, start_jd, longitude=longitude), lst_array))
    Ntimes = len(time_array)
    integration_time = np.median(np.diff(time_array)) * 24 * 3600.

    # get freqs
    Nfreqs = len(freq_array)
    channel_width = np.median(np.diff(freq_array))
    freq_array = freq_array.reshape(1, -1)
    spw_array = np.array([0])
    Nspws = 1

    # get baselines keys
    bls = sorted(data.bls())
    Nbls = len(bls)
    Nblts = Nbls * Ntimes

    # reconfigure time_array and lst_array
    time_array = np.repeat(time_array[np.newaxis], Nbls, axis=0).ravel()
    lst_array = np.repeat(lst_array[np.newaxis], Nbls, axis=0).ravel()

    # get data array
    data_array = np.moveaxis(map(lambda p: map(lambda bl: data[str(p)][bl], bls), pols), 0, -1)

    # resort time and baseline axes
    data_array = data_array.reshape(Nblts, 1, Nfreqs, Npols)
    nsample_array = np.ones_like(data_array, np.float)

    # flags
    if flags is None:
        flag_array = np.zeros_like(data_array, np.float).astype(np.bool)
    else:
        flag_array = np.moveaxis(map(lambda p: map(lambda bl: flags[str(p)][bl].astype(np.bool), bls), pols), 0, -1)
        flag_array = flag_array.reshape(Nblts, 1, Nfreqs, Npols)

    # configure baselines
    bls = np.repeat(np.array(bls), Ntimes, axis=0)

    # get ant_1_array, ant_2_array
    ant_1_array = bls[:,0]
    ant_2_array = bls[:,1]

    # get baseline array
    baseline_array = 2048 * (ant_2_array+1) + (ant_1_array+1) + 2^16

    # get antennas in data
    data_ants = np.unique(np.concatenate([ant_1_array, ant_2_array]))
    Nants_data = len(data_ants)

    # get telescope ants
    antenna_numbers = np.unique(antpos.keys())
    Nants_telescope = len(antenna_numbers)
    antenna_names = map(lambda a: "HH{}".format(a), antenna_numbers)

    # get antpos and uvw
    antenna_positions = np.array(map(lambda k: antpos[k], antenna_numbers))
    uvw_array = np.array([antpos[k[0]] - antpos[k[1]] for k in zip(ant_1_array, ant_2_array)])

    # get zenith location
    zenith_dec_degrees = np.ones_like(baseline_array) * dec
    zenith_ra_degrees = utils.JD2RA(time_array, longitude)
    zenith_dec = zenith_dec_degrees * np.pi / 180
    zenith_ra = zenith_ra_degrees * np.pi / 180

    # instantiate object
    uvd = UVData()

    # assign parameters
    params = ['Nants_data', 'Nants_telescope', 'Nbls', 'Nblts', 'Nfreqs', 'Npols', 'Nspws', 'Ntimes',
              'ant_1_array', 'ant_2_array', 'antenna_names', 'antenna_numbers', 'baseline_array',
              'channel_width', 'data_array', 'flag_array', 'freq_array', 'history', 'instrument',
              'integration_time', 'lst_array', 'nsample_array', 'object_name', 'phase_type',
              'polarization_array', 'spw_array', 'telescope_location', 'telescope_name', 'time_array',
              'uvw_array', 'vis_units', 'antenna_positions', 'zenith_dec', 'zenith_ra']              
    for p in params:
        uvd.__setattr__(p, locals()[p])

    # write uvdata
    if write_miriad:
        # check output
        fname = os.path.join(outdir, fname)
        if os.path.exists(fname) and overwrite is False:
            abscal.echo("{} exists, not overwriting".format(fname), verbose=verbose)
        else:
            abscal.echo("saving {}".format(fname), type=0, verbose=verbose)
            uvd.write_miriad(fname, clobber=True)

    if return_uvdata:
        return uvd


def sigma_clip(array, flags=None, sigma=4.0, axis=0, min_N=4):
    """
    one-iteration sigma clipping algorithm. set clipped values to nan.

    Parameters:
    -----------
    array : ndarray of complex visibility data. If 2D, [0] axis is samples and [1] axis is freq.

    flags : ndarray matching array shape containing boolean flags. True if flagged.

    sigma : float, sigma threshold to cut above

    axis : int, axis of array to sigma clip

    min_N : int, minimum length of array to sigma clip, below which no sigma
                clipping is performed.

    Output:
    -------
    array : same as input array, but clipped values have been set to np.nan
    """
    # ensure array is an array
    if type(array) is not np.ndarray:
        array = np.array(array)

    # ensure array passes min_N criteria:
    if array.shape[axis] < min_N:
        return array

    # apply flags
    if flags is not None:
        array[flags] *= np.nan

    # get robust location
    mean = np.nanmedian(array, axis=axis)

    # get MAD
    std = np.nanmedian(np.abs(array - mean), axis=axis) * 1.4

    # set cut data to nans
    cut = np.where(np.abs(array-mean)/std > sigma)
    array[cut] *= np.nan

    return array


def switch_bl(key):
    """
    switch antenna ordering in (ant1, ant2, pol) key
    where ant1 and ant2 are ints and pol is a two-char str
    Ex. (1, 2, 'xx')
    """
    return key[:2][::-1] + (key[-1],)


