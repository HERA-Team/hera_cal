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


def lst_bin(data_list, lst_list, lst_grid=None, wgts_list=None, lst_init=np.pi, dlst=None,
            lst_low=None, lst_hi=None, wrap_point=2*np.pi, atol=1e-8, median=False,
            sigma_clip=False, sigma=2.0):
    """
    Bin data in Local Sidereal Time (LST) onto an LST grid. An LST grid
    is defined as an array of points increasing in Local Sidereal Time, with each point marking
    the center of the LST bin.

    Parameters:
    -----------
    data_list : type=list, list of DataContainer dictionaries holding complex visibility data

    lst_list : type=list, list of ndarrays holding LST stamps of each data dictionary in data_list
    
    lst_grid : type=ndarray, 1D ndarray of LST grid to bin data into

    wgts_list : type=list, list of DataContainer dictionaries holding weights for each dict in data_list

    lst_init : type=float, starting LST to create lst_grid from

    dlst : type=float, delta-LST spacing for lst_grid

    lst_low : type=float, lower bound in LST for binning

    lst_hi : type=float, upper bound in LST for binning

    wrap_point : type=float, end point of a single day in LST

    atol : type=float, absolute tolerance for comparing floats in lst_list to floats in lst_grid

    median : type=boolean, if True use median for LST binning, else use mean

    sigma_clip : type=boolean, if True, perform a sigma clipping algorithm of the LST bins
        on the real and imag components separately. Warning: This considerably slows down the code.

    sigma : type=float, input standard deviation to use for sigma clipping algorithm.

    Output: (data_avg, data_std, lst_bins, data_num)
    -------
    data_avg : dictionary of data having averaged the LST bins

    data_std : dictionary of data with real component holding std along real axis
        and imag component holding std along imag axis

    lst_bins : ndarray containing final lst grid of data

    data_num : dictionary containing the number of data points averaged in each LST bin.
    """
    # construct lst_grid if not provided
    if lst_grid is None:
        # get dlst if not provided
        if dlst is None:
            dlst = np.median(np.diff(lst_list[0]))
        # construct lst_grid
        lst_grid = np.arange(lst_init, lst_init + wrap_point, dlst) + dlst/2
    else:
        lst_init = lst_grid[0]

    # form new dictionaries
    data = odict()
    data_avg = odict()
    data_num = odict()
    data_std = odict()
    all_indices = odict()

    # iterate over data_list
    for i, d in enumerate(data_list):
        # get lst array
        l = lst_list[i]

        # unwrap LST array
        l = unwrap(l)

        # periodicize l where less than lst_init
        l[np.where(l < lst_init)] += wrap_point

        # get indices in lst_grid
        indices = np.array(map(lambda x: np.argmin(np.abs(lst_grid-x)), l))

        # restrict lst array if desired
        if lst_low is not None:
            indices = indices[np.where(l >= lst_low - atol)]
        if lst_hi is not None:
            if lst_hi < l.min():
                lst_hi += wrap_point
            indices = indices[np.where(l <= lst_hi + atol)]

        # update all_indices
        all_indices.update(odict(map(lambda x: (x, None), indices)))

        # iterate over keys in d
        for j, key in enumerate(d.keys()):
            if key not in data:
                data[key] = odict()

            # iterate over indices, append to data
            for k, ind in enumerate(indices):
                if ind not in data[key]:
                    data[key][ind] = []
                data[key][ind].append(d[key][k])

    # get all lst array from keys of all_indices
    all_lst = lst_grid[all_indices.keys()]

    # wrap lst array
    all_lst = wrap(all_lst)

    # iterate over data and get statistics
    for i, key in enumerate(data.keys()):
        if sigma_clip:
            for j, ind in enumerate(data[key].keys()):
                data[key][ind] = astats.sigma_clip(np.array(data[key][ind]).real, sigma=sigma) + 1j*astats.sigma_clip(np.array(data[key][ind]).imag, sigma=sigma)
        if median:
            real_avg = np.array(map(lambda ind: np.nanmedian(map(lambda r: r.real, data[key][ind]), axis=0), data[key].keys()))
            imag_avg = np.array(map(lambda ind: np.nanmedian(map(lambda r: r.imag, data[key][ind]), axis=0), data[key].keys()))
        else:
            real_avg = np.array(map(lambda ind: np.nanmean(map(lambda r: r.real, data[key][ind]), axis=0), data[key].keys()))
            imag_avg = np.array(map(lambda ind: np.nanmean(map(lambda r: r.imag, data[key][ind]), axis=0), data[key].keys()))
        real_stan_dev = np.array(map(lambda ind: np.nanstd(map(lambda r: r.real, data[key][ind]), axis=0), data[key].keys()))
        imag_stan_dev = np.array(map(lambda ind: np.nanstd(map(lambda r: r.imag, data[key][ind]), axis=0), data[key].keys()))
        num_pix = np.array(map(lambda ind: np.nansum(map(lambda r: r.real*0+1, data[key][ind]), axis=0), data[key].keys()))

        data_avg[key] = real_avg + 1j*imag_avg
        data_std[key] = real_stan_dev + 1j*imag_stan_dev
        data_num[key] = num_pix

    return DataContainer(data_avg), DataContainer(data_std), all_lst, DataContainer(data_num)


def lst_align(data, data_lsts, wgts=None, lst_grid=None, lst_init=np.pi, dlst=None, wrap_point=2*np.pi,
              match='nearest', verbose=True, **interp_kwargs):
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

    match : type=str, LST-bin matching method, options=['nearest','forward','backward']

    verbose : type=boolean, if True, print feedback to stdout

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

    # get closest lsts
    nn = np.array(map(lambda l: np.argmin(np.abs(lst_grid-l)), data_lsts))
    if match == 'nearest':
        indices = nn
    elif match == 'forward':
        indices = nn + 1
    elif match == 'backward':
        indices = nn - 1

    # get frequency info
    Nfreqs = data[data.keys()[0]].shape[1]
    data_freqs = np.arange(Nfreqs)
    model_freqs = np.arange(Nfreqs)
    model_lsts = lst_grid[indices]

    # interpolate data
    interp_data, interp_wgts = abscal.interp2d_vis(data, data_lsts, data_freqs, model_lsts, model_freqs, wgts=wgts, **interp_kwargs)

    return interp_data, interp_wgts, model_lsts


def lst_align_arg_parser():
    a = argparse.ArgumentParser(description='')
    a.add_argument("--data_files", type=str, nargs='*', help="list of miriad files of data to-be-calibrated.", required=True)
    a.add_argument("--model_files", type=str, nargs='*', default=[], help="list of data-overlapping miriad files for visibility model.", required=True)
    a.add_argument("--calfits_fname", type=str, default=None, help="name of output calfits file.")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output calfits file if it exists.")
    a.add_argument("--silence", default=False, action='store_true', help="silence output from abscal while running.")
    a.add_argument("--zero_psi", default=False, action='store_true', help="set overall gain phase 'psi' to zero in linsolve equations.")
    return a


def lst_align_files(data_files, ext="L.{:8.6f}", lst_init=np.pi, dlst=0.00078298496, **kwargs):
    """
    align a series of data files with an LST grid

    Parameters:
    -----------


    Result:
    -------

    """

    # iterate over data files
    for i, f in enumerate(data_files):
        # load data
        uvd = UVData()
        uvd.read_miriad(f)
        data, flags = abscal.UVData2AbsCalDict(uvd)
        lsts = np.unique(uvd.lst_array)

        # lst align
        interp_data, interp_flags, interp_lsts = lst_align(data, lsts, lst_init=lst_init, dlst=dlst, **kwargs)

        # get JD
        jds = utils.LST2JD(interp_lsts, np.floor(np.min(uvd.time_array)), longitude=21.4283)

        # reorder into arrays
        uvd_data = np.array(interp_data.values())
        uvd_data = uvd_data.reshape(-1, 1, Nfreqs, 1)
        uvd_flags = np.array(map(lambda k: interp_flags[k], flags.keys())).astype(np.bool) + \
                    np.array(map(lambda k: flags[k], flags.keys())).astype(np.bool) 
        uvd_flags = uvd_flags.reshape(-1, 1, Nfreqs, 1)
        uvd_keys = np.repeat(np.array(interp_data.keys()).reshape(-1, 1, 2), Ntimes, axis=1).reshape(-1, 2)
        uvd_bls = np.array(map(lambda k: uvd.antnums_to_baseline(k[0], k[1]), uvd_keys))
        uvd_times = np.array(map(lambda x: utils.JD2LST.LST2JD(x, np.median(np.floor(uvd.time_array)), uvd.telescope_location_lat_lon_alt_degrees[1]), model_lsts))
        uvd_times = np.repeat(uvd_times[np.newaxis], Nbls, axis=0).reshape(-1)
        uvd_lsts = np.repeat(model_lsts[np.newaxis], Nbls, axis=0).reshape(-1)
        uvd_freqs = model_freqs.reshape(1, -1)

        # assign to uvdata object
        uvd.data_array = uvd_data
        uvd.flag_array = uvd_flags
        uvd.baseline_array = uvd_bls
        uvd.ant_1_array = uvd_keys[:, 0]
        uvd.ant_2_array = uvd_keys[:, 1]
        uvd.time_array = uvd_times
        uvd.lst_array = uvd_lsts
        uvd.freq_array = uvd_freqs
        uvd.Nfreqs = Nfreqs

        # check output
        if outdir is None:
            outdir = os.path.dirname(data_fname)
        if output_fname is None:
            output_fname = os.path.basename(data_fname) + ext.format(model_lsts[0])
        output_fname = os.path.join(outdir, output_fname)
        if os.path.exists(output_fname) and overwrite is False:
            raise IOError("{} exists, not overwriting".format(output_fname))

        # write to file
        abscal.echo("saving {}".format(output_fname), type=1, verbose=verbose)
        uvd.write_miriad(output_fname, clobber=True)


def lst_bin_arg_parser():
    a = argparse.ArgumentParser(description='')


    return a


def lst_bin_files(data_files, lst_init=np.pi, dlst=0.00078298496, wrap_point=2*np.pi,
                  ntimes_per_file=60, file_ext="{}LST.{}.uv" , outdir=None, overwrite=True,
                  align=False, align_kwargs={}):
    """
    LST bin a series of miriad files with identical frequency bins, but varying
    time bins.

    Parameters:
    -----------
    data_files : type=list of lists: nested set of lists, with each nested list containing
            paths to miriad files from a particular night
    """
    # create LST grid 
    lst_grid = np.arange(lst_init, lst_init + wrap_point, dlst) + dlst/2

    # get file start and stop times
    data_times = np.array(map(lambda f: np.array(utils.get_miriad_times(f)).T, data_files))

    # unwrap times
    data_times[np.where(data_times < lst_init)] += wrap_point

    # create data file status: None if not opened, data object if opened
    data_status = map(lambda d: map(lambda f: None, d), data_files)

    # get start and end lst
    start_lst = np.min(data_times)
    start_index = np.argmin(np.abs(lst_grid - start_lst)) + 1
    end_lst = np.max(data_times)
    end_index = np.argmin(np.abs(lst_grid - end_lst)) - 1
    nfiles = int(np.ceil(float((end_index - start_index)) / ntimes_per_file))

    # get outdir
    if outdir is None:
        outdir = os.path.dirname(os.path.commonprefix(abscal.flatten(data_files)))

    # create lst-grid of files
    file_lsts = [lst_grid[start_index:end_index][ntimes_per_file*i:ntimes_per_file*(i+1)] for i in range(nfiles)]
 
    # iterate over end-result LST files
    for i, f_lst in enumerate(file_lsts):
        # create empty data_list and lst_list
        data_list = []
        wgts_list = []
        lst_list = []

        # locate all files that fall within this range of lsts
        f_min = np.min(f_lst)
        f_max = np.max(f_lst)
        f_select = np.array(map(lambda d: map(lambda f: (f[1] > f_min)&(f[0] < f_max), d), data_times))
        if i == 0:
            old_f_select = copy.copy(f_select)

        # open necessary files, close ones that are no longer needed
        for j in range(len(data_files)):
            for k in range(len(data_files[j])):
                if f_select[j][k] == True and data_status[j][k] is None:
                    # open file(s)
                    d, w, ap, a, f, t, l, p = abscal.UVData2AbsCalDict(data_files[j][k], return_meta=True, return_wgts=True)

                    # unwrap l
                    l[np.where(l < lst_init)] += wrap_point

                    # lst-align if desired
                    if align:
                        d, w, l = lst_align(d, l, wgts=w, lst_grid=f_lst, lst_init=lst_init, wrap_point=wrap_point, match='nearest', verbose=True, bounds_error=False, **align_kwargs)

                    # pass reference to data_status
                    data_status[j][k] = (d, w, ap, a, f, t, l, p)

                    # erase unnecessary references
                    del(d,w,ap,a,f,t,l,p)

                elif f_select[j][k] == False and old_f_select[j][k] == True:
                    # erase reference
                    data_status[j][k] = None

                # copy references to data_list
                if f_select[j][k] == True:
                    data_list.append(data_status[j][k][0])
                    wgts_list.append(data_status[j][k][1])
                    lst_list.append(data_status[j][k][6])

        # pass through lst-bin function
        (bin_data, std_data, all_lst,
         num_data) = lst_bin(data_list, lst_list, wgts_list=wgts_list, lst_grid=f_lst,
                             lst_init=lst_init, wrap_point=wrap_point, lst_low=f_min, lst_hi=f_max)

        # erase data references
        del data_list
        del wgts_list
        del lst_list

        # assign old f_select
        old_f_select = copy.copy(f_select)

        # configure filename
        bin_file = ""
        std_file = ""
        num_file = ""

        # check for overwrite
        if os.path.exists(bin_file) and overwrite is False:
            abscal.echo("{} exists, not overwriting".format(bin_file), verbose=verbose)
            continue

        # configure history string
        history = ""

        # write to file
        data_to_miriad(bin_file, bin_data, all_lst, freq_array, antpos, history=history)
        data_to_miriad(std_file, bin_data, all_lst, freq_array, antpos, history=history)
        data_to_miriad(num_file, bin_data, all_lst, freq_array, antpos, history=history)


def data_to_miriad(fname, data, lst_array, freq_array, antpos,


                   outdir="./", overwrite=True, verbose=True,
                   history=""):
    """
    take data dictionary, export to UVData object and write as a miriad file.

    Parameters:
    -----------


    """
    # check output
    fname = os.path.join(outdir, fname)
    if os.path.exists(fname) and overwrite is False:
        abscal.echo("{} exists, not overwriting".format(fname), verbose=verbose)

    ## configure UVData parameters







def unwrap(array, wrap_point=2*np.pi):
    """
    unwrap 1D LST ndarray

    Parameters:
    -----------
    array : type=ndarray, 1D array containing LST values (default=radians)

    wrap_point : type=float, point of LST wrap, default=radians

    Output:
    -------
    new_array : type=ndarray, 1D array with unwrapped LST values
    """
    array = copy.copy(array)
    new_array = np.empty_like(array)
    for i, v in enumerate(array):
        if i == 0:
            start = v
        else:
            if v < start:
                array[i:] += wrap_point
                v += wrap_point
        new_array[i] = v

    return new_array


def wrap(array, wrap_point=2*np.pi):
    """
    wrap 1D LST ndarray

    Parameters:
    -----------
    array : type=ndarray, 1D array containing LST values (default=radians)

    wrap_point : type=float, point of LST wrap, default=radians

    Output:
    -------
    new_array : type=ndarray, 1D array containing wrapped LST values
    """
    array = copy.copy(array)
    new_array = np.empty_like(array)
    for i, v in enumerate(array):
        if v >= wrap_point:
            wrap_diff = (v // wrap_point) * wrap_point
            v -= wrap_diff
            array[i:] -= wrap_diff
        new_array[i] = v

    return new_array




