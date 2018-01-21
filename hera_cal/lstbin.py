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


def lst_bin(data_list, lst_list, wgts_list=None, lst_init=np.pi, dlst=None,
            lst_low=None, lst_hi=None, wrap_point=2*np.pi):
    """
    Bin data in Local Sidereal Time (LST)

    Parameters:
    -----------


    Output: (data, real_std, imag_std, all_lst)
    -------

    """

    # get dlst
    if dlst is None:
        dlst = np.median(np.diff(lst_list[0]))

    # construct lst_grid
    lst_grid = np.arange(lst_init, lst_init + wrap_point, dlst) + dlst/2

    # form new dictionaries
    data = odict()
    num_pix = odict()
    real_std = odict()
    imag_std = odict()
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
            indices = indices[np.where(l > lst_low)]
        if lst_hi is not None:
            if lst_hi < l.min():
                lst_hi += wrap_point
            indices = indices[np.where(l < lst_hi)]

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
        real_mean = np.array(map(lambda ind: np.nanmean(map(lambda r: r.real, data[key][ind]), axis=0), data[key].keys()))
        imag_mean = np.array(map(lambda ind: np.nanmean(map(lambda r: r.imag, data[key][ind]), axis=0), data[key].keys()))
        real_stan_dev = np.array(map(lambda ind: np.nanstd(map(lambda r: r.real, data[key][ind]), axis=0), data[key].keys()))
        imag_stan_dev = np.array(map(lambda ind: np.nanstd(map(lambda r: r.imag, data[key][ind]), axis=0), data[key].keys()))
        pixels = np.array(map(lambda ind: np.nansum(map(lambda r: r.real/r.real, data[key][ind]), axis=0), data[key].keys()))

        data[key] = real_mean + 1j*imag_mean
        real_std[key] = real_stan_dev
        imag_std[key] = imag_stan_dev
        num_pix[key] = pixels

    return data, real_std, imag_std, all_lst, num_pix


def lst_align(data, data_lsts, lst_grid=None, lst_init=np.pi, dlst=None, wrap_point=2*np.pi, match='nearest', verbose=True, **kwargs):
    """
    Interpolate complex visibilities to align time integrations with an LST grid.

    Parameters:
    -----------
    data : type=dictionary, 

    data_lsts : type=ndarray

    lst_grid : 

    lst_init : 

    dlst : 

    wrap_point : type=float, total duration of LST day
                 2*np.pi for radians and 23.9344699 for hours

    match : type=str, LST-bin matching method, options=['nearest','forward','backward']

    verbose : 

    kwargs : 

    Output: (interp_data, interp_flags, interp_lsts)
    -------
    interp_data : 

    interp_flags : 

    interp_lsts : 
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
    interp_data, interp_flags = abscal.interp2d_vis(data, data_lsts, data_freqs, model_lsts, model_freqs, **kwargs)

    return interp_data, interp_flags, model_lsts


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
                  lst_align=False):
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
    data_times = np.array(map(lambda f: utils.get_miriad_times(f), data_files))
   
    # unwrap times
    data_times[np.where(data_times < lst_init)] += wrap_point

    # get start and end lst
    start_lst = np.min(data_times)
    start_index = np.argmin(np.abs(lst_grid - start_lst))
    end_lst = np.max(data_times)
    end_index = np.argmin(np.abs(lst_grid - start_lst))
    nfiles = int(np.ceil(float((end_index - start_index)) / ntimes_per_file))

    # get outdir
    if outdir is None:
        outdir = os.path.dirname(os.path.commonprefix(abscal.flatten(data_files)))

    # create lst-grid of files
    file_lsts = [lst_grid[start_index:end_index][ntimes_per_file*i:ntimes_per_file*(i+1)] for i in range(nfiles)]
 
    # iterate over end-result LST files
    for i, f_lst in enumerate(file_lsts):

        # locate all files that fall within this range of lsts




        # write log-file











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




