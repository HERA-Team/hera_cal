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
from memory_profiler import memory_usage as memuse
import gc as garbage_collector
import datetime


def lst_bin(data_list, lst_list, lst_grid=None, wgts_list=None, lst_init=np.pi, dlst=None,
            lst_low=None, lst_hi=None, wrap_point=2*np.pi, atol=1e-8, median=False,
            sigma_clip=False, sigma=2.0, return_no_avg=False, verbose=True):
    """
    Bin data in Local Sidereal Time (LST) onto an LST grid. An LST grid
    is defined as an array of points increasing in Local Sidereal Time, with each point marking
    the center of the LST bin.

    Parameters:
    -----------
    data_list : type=list, list of DataContainer dictionaries holding complex visibility data

    lst_list : type=list, list of ndarrays holding LST stamps of each data dictionary in data_list
    
    lst_grid : type=ndarray, uniform 1D ndarray of LST grid to bin data into (marking center of bin).
                        Note: ideally this should not start at 0, but at 0 + dlst / 2.

    wgts_list : type=list, list of DataContainer dictionaries holding weights for each dict in data_list

    lst_init : type=float, starting LST to initialize lst_grid at (if not fed). This marks the left-edge
                        of the first LST bin.

    dlst : type=float, delta-LST spacing for lst_grid.

    lst_low : type=float, lower bound in lst_grid centers used for LST binning

    lst_hi : type=float, upper bound in lst_grid centers used for LST binning

    wrap_point : type=float, end point of a single day in LST

    atol : type=float, absolute tolerance for comparing floats in lst_list to floats in lst_grid

    median : type=boolean, if True use median for LST binning, else use mean

    sigma_clip : type=boolean, if True, perform a sigma clipping algorithm of the LST bins
        on the real and imag components separately. Warning: This considerably slows down the code.

    sigma : type=float, input standard deviation to use for sigma clipping algorithm.

    return_no_avg : type=boolean, if True, return binned but un-averaged data and wgts.

    Output: (data_avg, wgts_avg, data_std, lst_bins, data_num)
    -------
    data_avg : dictionary of data having averaged the LST bins

    wgts_avg : dictionary of data weights averaged in LST bins
    
    data_std : dictionary of data with real component holding std along real axis
        and imag component holding std along imag axis

    lst_bins : ndarray containing final lst grid of data

    data_num : dictionary containing the number of data points averaged in each LST bin.

    if return_no_avg:
        Output: (data_bin, wgts_bin)
        data_bin : dictionary with (ant1,ant2,pol) as keys and ndarrays holding
            un-averaged complex visibilities in each LST bin as values. 
    """
    # construct lst_grid if not provided
    if lst_grid is None:
        # get dlst if not provided
        if dlst is None:
            dlst = np.median(np.diff(lst_list[0]))
        # construct lst_grid
        lst_grid = np.arange(lst_init, lst_init + wrap_point - dlst/2, dlst) + dlst/2
    else:
        dlst = np.median(np.diff(lst_grid))
        lst_init = lst_grid[0] - dlst/2
        if lst_init < 0:
            lst_init += wrap_point

    # restrict lst_grid based on lst_low and lst_high
    if lst_low is not None:
        lst_grid = lst_grid[lst_grid > lst_low - atol]
    if lst_hi is not None:
        lst_grid = lst_grid[lst_grid < lst_hi + atol]

    # move lst_grid centers to the left
    lst_grid_left = lst_grid - dlst / 2
    lst_grid_left = unwrap(lst_grid_left)

    # form new dictionaries
    # data is a dictionary that will hold other dictionaries as values, which will
    # themselves hold lists of ndarrays
    data = odict()
    wgts = odict()
    all_lst_indices = set()

    # iterate over data_list
    for i, d in enumerate(data_list):
        # get lst array
        l = lst_list[i]

        # unwrap LST array
        l = unwrap(l)

        # add wrap_point to l where less than lst_init
        l[np.where(l < lst_init)] += wrap_point

        # digitize data lst array "l", moving bin centers to the left
        grid_indices = np.digitize(l, lst_grid_left[1:], right=True)

        # make data_in_bin boolean array, and set to False data that don't fall in any bin
        data_in_bin = np.ones_like(l, np.bool)
        data_in_bin[(l<lst_grid_left.min()-atol)|(l>lst_grid_left.max()+dlst+atol)] = False

        # update all_lst_indices
        all_lst_indices.update(set(grid_indices[data_in_bin]))

        # iterate over keys in d
        for j, key in enumerate(d.keys()):
            # conjugate d[key] if necessary
            if key in data:
                pass
            elif switch_bl(key) in data:
                key = switch_bl[key]
                d[key] = np.conj(d[switch_bl(key)])
                if wgts_list is not None:
                    wgts_list[i][key] = wgts_list[i][switch_bl(key)]
            else:
                data[key] = odict()
                wgts[key] = odict()

            # iterate over grid_indices, and append to data if data_in_bin is True
            for k, ind in enumerate(grid_indices):
                if data_in_bin[k]:
                    if ind not in data[key]:
                        data[key][ind] = []
                        wgts[key][ind] = []
                    data[key][ind].append(d[key][k])
                    if wgts_list is None:
                        wgts[key][ind].append(np.ones_like(d[key][k], np.float))
                    else:
                        wgts[key][ind].append(wgts_list[i][key][k])

    # get final lst_bin array from all_lst_indices
    lst_bins = lst_grid[sorted(all_lst_indices)]

    # wrap lst_bins
    lst_bins = wrap(lst_bins)

    # make final dictionaries
    wgts_avg = odict()
    data_avg = odict()
    data_num = odict()
    data_std = odict()

    # return un-averaged data if desired
    if return_no_avg:
        # return all binned data instead of just bin average 
        data_bin = odict(map(lambda k: (k, np.array(odict(map(lambda k2: (k2, data[k][k2]), sorted(data[k].keys()))).values())), data.keys()))
        wgts_bin = odict(map(lambda k: (k, np.array(odict(map(lambda k2: (k2, wgts[k][k2]), sorted(wgts[k].keys()))).values())), wgts.keys()))

        return data_bin, wgts_bin

    # iterate over data and get statistics
    for i, key in enumerate(data.keys()):
        if sigma_clip:
            for j, ind in enumerate(data[key].keys()):
                data[key][ind] = astats.sigma_clip(np.array(data[key][ind]).real, sigma=sigma) + 1j*astats.sigma_clip(np.array(data[key][ind]).imag, sigma=sigma)
                wgts[key][ind] = np.array(wgts[key][ind])
                wgts[key][ind][data[key][ind].mask] *= 0
        if median:
            real_avg = np.array(map(lambda ind: np.nanmedian(map(lambda r: r.real, data[key][ind]), axis=0), data[key].keys()))
            imag_avg = np.array(map(lambda ind: np.nanmedian(map(lambda r: r.imag, data[key][ind]), axis=0), data[key].keys()))
            w_avg = np.array(map(lambda ind: np.nanmedian(wgts[key][ind], axis=0), wgts[key].keys()))
        else:
            real_avg = np.array(map(lambda ind: np.nanmean(map(lambda r: r.real, data[key][ind]), axis=0), data[key].keys()))
            imag_avg = np.array(map(lambda ind: np.nanmean(map(lambda r: r.imag, data[key][ind]), axis=0), data[key].keys()))
            w_avg = np.array(map(lambda ind: np.nanmean(wgts[key][ind], axis=0), wgts[key].keys()))
        real_stan_dev = np.array(map(lambda ind: np.nanstd(map(lambda r: r.real, data[key][ind]), axis=0), data[key].keys()))
        imag_stan_dev = np.array(map(lambda ind: np.nanstd(map(lambda r: r.imag, data[key][ind]), axis=0), data[key].keys()))
        num_pix = np.array(map(lambda ind: np.nansum(map(lambda r: r.real*0+1, data[key][ind]), axis=0), data[key].keys()))

        data_avg[key] = real_avg + 1j*imag_avg
        wgts_avg[key] = w_avg
        data_std[key] = real_stan_dev + 1j*imag_stan_dev
        data_num[key] = num_pix.astype(np.complex)

    # turn into DataContainer
    data_avg = DataContainer(data_avg)
    wgts_avg = DataContainer(wgts_avg)
    data_std = DataContainer(data_std)
    data_num = DataContainer(data_num)

    return data_avg, wgts_avg, data_std, lst_bins, data_num


def lst_align(data, data_lsts, wgts=None, lst_grid=None, lst_init=np.pi, dlst=None, wrap_point=2*np.pi,
              verbose=True, atol=1e-5, **interp_kwargs):
    """
    Interpolate complex visibilities to align time integrations with an LST grid. An LST grid
    is defined as an array of points increasing in Local Sidereal Time, with each point marking
    the center of the LST bin.

    Parameters:
    -----------
    data : type=dictionary, 

    data_lsts : type=ndarray

    wgts : type=dictionary, weight dictionary

    lst_grid : type=ndarray, 1D ndarray of LST grid to bin data into

    lst_init : type=float, starting LST to create lst_grid from

    dlst : type=float, delta-LST spacing for lst_grid

    wrap_point : type=float, end point of a single day in LST

    verbose : type=boolean, if True, print feedback to stdout

    atol : type=float, absolute tolerance for grid point float comparison

    interp_kwargs : type=dictionary, keyword arguments to feed to abscal.interp2d_vis

    Output: (interp_data, interp_flags, interp_lsts)
    -------
    interp_data : dictionary containing lst-binned data

    interp_flags : dictionary containing weights for lst-binned data

    interp_lsts : ndarray holding centers of LST bins.
    """
    # get lst if not fed grid
    if dlst is None:
        dlst = np.median(np.diff(data_lsts))
    if lst_grid is None:
        lst_grid = np.arange(lst_init, lst_init + wrap_point, dlst) + dlst/2

    # wrap data_lsts if below lst_init
    data_lsts[np.where(data_lsts < lst_init)] += wrap_point

    # pick out interpolate-able grid points
    lst_grid = lst_grid[np.where((lst_grid >= data_lsts.min() - atol)&(lst_grid <= data_lsts.max() + atol))]

    # get frequency info
    Nfreqs = data[data.keys()[0]].shape[1]
    data_freqs = np.arange(Nfreqs)
    model_freqs = np.arange(Nfreqs)

    # interpolate data
    interp_data, interp_wgts = abscal.interp2d_vis(data, data_lsts, data_freqs, lst_grid, model_freqs, wgts=wgts, **interp_kwargs)

    return interp_data, interp_wgts, lst_grid


def lst_align_arg_parser():
    a = argparse.ArgumentParser(description='LST align files with a universal LST grid')
    a.add_argument("data_files", nargs='*', type=str, help="miriad file paths to run LST align on.")
    a.add_argument("--file_ext", default=".L.{:7.5f}", type=str, help="file extension for LST-aligned data. must have one placeholder for starting LST.")
    a.add_argument("--lst_init", type=float, default=np.pi, help="start of LST grid (left-side of starting LST bin)")
    a.add_argument("--wrap_point", type=float, default=2*np.pi, help="full day of LST")
    a.add_argument("--dlst", type=float, default=None, help="LST grid interval spacing")
    a.add_argument("--longitude", type=float, default=21.42830, help="longitude of observer in degrees east")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output files")
    a.add_argument("--miriad_kwargs", type=dict, default={}, help="kwargs to pass to miriad_to_data function")
    a.add_argument("--align_kwargs", type=dict, default={}, help="kwargs to pass to lst_align function")
    a.add_argument("--silence", default=False, action='store_true', help='silence output to stdout')
    return a


def lst_align_files(data_files, file_ext=".L.{:7.5f}", lst_init=np.pi, wrap_point=2*np.pi, dlst=None,
                    longitude=21.42830, overwrite=None, outdir=None, miriad_kwargs={}, align_kwargs={},
                    verbose=True):
    """
    Align a series of data files with a universal LST grid.

    Parameters:
    -----------
    data_files : type=list, list of paths to miriad files, or a single miriad file path

    file_ext : type=str, file_extension for each file in data_files when writing to disk

    lst_init : type=float, starting point for LST grid

    wrap_point : type=float, a full day of LST

    dlst : type=float, LST grid bin interval, if None get it from first file in data_files

    longitude : type=float, longitude of observer in degrees east

    overwrite : type=boolean, if True overwrite output files

    miriad_kwargs : type=dictionary, keyword arguments to feed to miriad_to_data()

    align_kwargs : keyword arguments to feed to lst_align()

    Result:
    -------
    data_files + file_ext miriad files
    """
    # check type of data_files
    if type(data_files) == str:
        data_files = [data_files]

    # get dlst if None
    if dlst is None:
        start, stop, int_time = utils.get_miriad_times(data_files[0])
        dlst = int_time

    # make lst grid
    lst_grid = np.arange(lst_init, lst_init + wrap_point, dlst) + dlst/2

    # iterate over data files
    for i, f in enumerate(data_files):
        # load data
        (data, wgts, apos, ants, freqs, times, lsts,
         pols) = abscal.UVData2AbsCalDict(f, return_meta=True, return_wgts=True)

        # lst align
        interp_data, interp_wgts, interp_lsts = lst_align(data, lsts, wgts=wgts, lst_grid=lst_grid,
                                                          wrap_point=wrap_point, lst_init=lst_init,
                                                          dlst=dlst, **align_kwargs)

        # wrap lsts
        interp_lsts = wrap(interp_lsts)

        # check output
        output_fname = os.path.basename(f) + file_ext.format(interp_lsts[0])

        # write to miriad file
        if overwrite is not None:
            miriad_kwargs['overwrite'] = overwrite
        if outdir is not None:
            miriad_kwargs['outdir'] = outdir
        miriad_kwargs['start_jd'] = np.floor(times[0])
        interp_flags = DataContainer(odict(map(lambda k: (k, ~(interp_wgts[k].astype(np.bool))), interp_wgts.keys())))
        data_to_miriad(output_fname, interp_data, interp_lsts, freqs, apos, flags=interp_flags, verbose=verbose, **miriad_kwargs)


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
    a.add_argument("--wrap_point", type=float, default=2*np.pi, help="full day of LST")
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


def lst_bin_files(data_files, lst_init=np.pi, dlst=None, wrap_point=2*np.pi, verbose=True,
                  ntimes_per_file=60, file_ext="{}.{}.{:7.5f}.uv", pol_select=None,
                  outdir=None, overwrite=False, history=' ', lst_low=None, lst_hi=None,
                  align=False, align_kwargs={}, bin_kwargs={}, atol=1e-6, miriad_kwargs={}):
    """
    LST bin a series of miriad files with identical frequency bins, but varying
    time bins. Miriad file meta data (frequency bins, antennas positions, time_array)
    are taken from the first file in data_files.

    Parameters:
    -----------
    data_files : type=list of lists: nested set of lists, with each nested list containing
            paths to miriad files from a particular night

    lst_init : type=float, starting point of LST grid

    dlst : type=float, LST grid bin spacing. If None will get this from the first file in data_files

    wrap_point : type=float, duration of a full LST day

    lst_low : type=float, lower bound in lst grid

    lst_hi : type=float, upper bound in lst grid
    
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

    # create LST grid 
    lst_grid = np.arange(lst_init, lst_init + wrap_point, dlst) + dlst/2

    # cut lst grid bounds
    if lst_low is not None:
        lst_grid = lst_grid[np.where(lst_grid >= lst_low - atol)]

    if lst_hi is not None:
        lst_grid = lst_grid[np.where(lst_grid <= lst_hi + atol)]

    # check lst_grid isn't empty
    if len(lst_grid) == 0:
        raise ValueError("length of lst_grid is zero. try different lst_low or lst_hi values.")

    # get file start and stop times
    data_times = np.array(map(lambda f: np.array(utils.get_miriad_times(f, add_int_buffer=True)).T, data_files))[:, :, :2]

    # unwrap times
    data_times[np.where(data_times < lst_init)] += wrap_point

    # create data file status: None if not opened, data object if opened
    data_status = map(lambda d: map(lambda f: None, d), data_files)

    # get start and end lst
    start_lst = np.min(data_times)
    start_diff = lst_grid - start_lst
    start_diff[np.where(start_diff < 0)] = 100
    start_index = np.argmin(start_diff)

    end_lst = np.max(data_times)
    end_diff = lst_grid - end_lst
    end_diff[np.where(end_diff > 0)] = -100
    end_index = np.argmax(end_diff)
    nfiles = int(np.ceil(float((end_index - start_index)) / ntimes_per_file))

    # get outdir
    if outdir is None:
        outdir = os.path.dirname(os.path.commonprefix(abscal.flatten(data_files)))

    # update miriad_kwrgs
    miriad_kwargs['outdir'] = outdir
    miriad_kwargs['overwrite'] = overwrite

    # create lst-grid of files
    file_lsts = [lst_grid[start_index:end_index][ntimes_per_file*i:ntimes_per_file*(i+1)] for i in range(nfiles)]
 
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
        abscal.echo("starting LST file {} / {}: {}".format(i+1, nfiles, datetime.datetime.now()), type=1, verbose=verbose)
        # create empty data_list and lst_list
        data_list = []
        file_list = []
        wgts_list = []
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
            nightly_wgts_list = []
            nightly_lst_list = []
            for k in range(len(data_files[j])):
                if f_select[j][k] == True and data_status[j][k] is None:
                    # open file(s)
                    d, w, ap, a, f, t, l, p = abscal.UVData2AbsCalDict(data_files[j][k], return_meta=True, return_wgts=True, pol_select=pol_select)

                    # unwrap l
                    l[np.where(l < lst_init)] += wrap_point

                    # pass reference to data_status
                    data_status[j][k] = [d, w, ap, a, f, t, l, p]

                    # erase unnecessary references
                    del d, w, ap, a, f, t, l, p

                elif f_select[j][k] == False and old_f_select[j][k] == True:
                    # erase reference
                    del data_status[j][k]
                    data_status[j].insert(k, None)

                # copy references to data_list
                if f_select[j][k] == True:
                    file_list.append(data_files[j][k])
                    nightly_data_list.append(data_status[j][k][0])
                    nightly_wgts_list.append(data_status[j][k][1])
                    nightly_lst_list.append(data_status[j][k][6])

            # skip if nothing accumulated in nightly files
            if len(nightly_data_list) == 0:
                continue

            # align nightly data if desired, this involves making a copy of the raw data,
            # and then interpolating it (another copy)
            if align:
                # concatenate data across night
                night_data = reduce(operator.add, nightly_data_list)
                night_wgts = reduce(operator.add, nightly_wgts_list)
                night_lsts = np.concatenate(nightly_lst_list)

                del nightly_data_list, nightly_wgts_list, nightly_lst_list

                # align data
                night_data, night_wgts, night_lsts = lst_align(night_data, night_lsts, wgts=night_wgts,
                    lst_grid=f_lst, lst_init=lst_init, wrap_point=wrap_point, atol=atol, **align_kwargs)

                nightly_data_list = [night_data]
                nightly_wgts_list = [night_wgts]
                nightly_lst_list = [night_lsts]

                del night_data, night_wgts, night_lsts

            # extend to data lists
            data_list.extend(nightly_data_list)
            wgts_list.extend(nightly_wgts_list)
            lst_list.extend(nightly_lst_list)

            del nightly_data_list, nightly_wgts_list, nightly_lst_list

        # skip if data_list is empty
        if len(data_list) == 0:
            abscal.echo("data_list is empty for beginning LST {}".format(f_lst[0]), verbose=verbose)
            # erase data references
            del file_list, data_list, wgts_list, lst_list

            # assign old f_select
            old_f_select = copy.copy(f_select)
            continue

        # pass through lst-bin function
        (bin_data, wgt_data, std_data, bin_lst,
         num_data) = lst_bin(data_list, lst_list, wgts_list=wgts_list, lst_grid=f_lst, lst_init=lst_init,
                             wrap_point=wrap_point, lst_low=f_min, lst_hi=f_max, **bin_kwargs)

        # update history
        file_history = history + "input files: " + "-".join(map(lambda ff: os.path.basename(ff), file_list))
        miriad_kwargs['history'] = file_history

        # erase data references
        del file_list, data_list, wgts_list, lst_list
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
        flag_data = DataContainer(odict([(k, ~(wgt_data[k].astype(np.bool))) for k in wgt_data.keys()]))
        data_to_miriad(bin_file, bin_data, bin_lst, freq_array, antpos, flags=flag_data, verbose=verbose, **miriad_kwargs)
        data_to_miriad(std_file, std_data, bin_lst, freq_array, antpos, verbose=verbose, **miriad_kwargs)
        data_to_miriad(num_file, num_data, bin_lst, freq_array, antpos, verbose=verbose, **miriad_kwargs)

        del bin_file, std_file, num_file, bin_data, std_data, num_data, bin_lst, flag_data, wgt_data
        garbage_collector.collect()


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


def sigma_clip(array, sigma=4.0, axis=0):
    """
    one-iteration sigma clipping algorithm

    Parameters:
    -----------
    array : ndarray of complex visibility data

    sigma : float, sigma threshold to cut above

    axis : int, axis of array to sigma clip

    Output:
    -------
    array : same as input array, but clipped values have been set to np.nan
    """
    # ensure array is an array
    array = np.array(array)

    # get robust location
    mean = np.nanmedian(array, axis=axis)

    # get robust std
    std = np.sqrt(astats.biweight_midvariance(array, axis=axis))

    # set cut data to nans
    cut = np.where(np.abs(array-mean)/std > sigma)
    array[cut] *= np.nan

    return array


def unwrap(array, wrap_point=2*np.pi):
    """
    Unwrap 1D LST ndarray that is periodic about wrap_point.

    Parameters:
    -----------
    array : type=ndarray, 1D array containing LST values in units of wrap_point.

    wrap_point : type=float, full day of LST.

    Output:
    -------
    new_array : type=ndarray, 1D array with unwrapped LST values
    """
    array = copy.copy(array)
    new_array = np.empty_like(array)
    for i, v in enumerate(array):
        if i == 0:
            while v < 0:
                v += wrap_point
            start = v
        else:
            if v < start:
                array[i:] += wrap_point
                v += wrap_point
        new_array[i] = v

    return new_array


def wrap(array, wrap_point=2*np.pi):
    """
    Wrap 1D LST ndarray about a periodic bound ending at wrap_point.

    Parameters:
    -----------
    array : type=ndarray, 1D array containing LST values in units of wrap_point.

    wrap_point : type=float, full day of LST.

    Output:
    -------
    new_array : type=ndarray, 1D array containing wrapped LST values
    """
    array = copy.copy(array)
    new_array = np.empty_like(array)
    for i, v in enumerate(array):
        while v < 0:
            v += wrap_point
        if v >= wrap_point:
            wrap_diff = (v // wrap_point) * wrap_point
            v -= wrap_diff
            array[i:] -= wrap_diff
        new_array[i] = v

    return new_array


def switch_bl(key):
    """
    switch antenna ordering in (ant1, ant2, pol) key
    where ant1 and ant2 are ints and pol is a two-char str
    Ex. (1, 2, 'xx')
    """
    return key[:2][::-1] + (key[-1],)


