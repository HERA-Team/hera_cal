# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import os
from collections import OrderedDict as odict
import copy
import argparse
import functools
import numpy as np
import operator
import gc as garbage_collector
import datetime
import warnings

from . import utils
from . import abscal
from . import io
from . import redcal
from . import apply_cal
from .datacontainer import DataContainer


def baselines_same_across_nights(data_list):
    """
    Check whether the sets of baselines in the datacontainers are consistent.

    Parameters
    ----------
    data_list: list of data-container dictionaries
               list of data-containers holding data for different nights.

    Returns
    -------
    same_across_nights: bool
        True, if all datacontainers in data_list have the same baselines.
        False if they do not.
    """
    # check whether baselines are the same across all nights
    # by checking that every baseline occurs in data_list the same number times.
    same_across_nights = False
    baseline_counts = DataContainer({})
    for dlist in data_list:
        for k in dlist:
            if k in baseline_counts:
                baseline_counts[k] += 1
            else:
                baseline_counts[k] = 1
    same_across_nights = np.all([baseline_counts[k] == baseline_counts[bl] for bl in baseline_counts])
    return same_across_nights


def lst_bin(data_list, lst_list, flags_list=None, nsamples_list=None, dlst=None, begin_lst=None, lst_low=None,
            lst_hi=None, flag_thresh=0.7, atol=1e-10, median=False, truncate_empty=True,
            sig_clip=False, sigma=4.0, min_N=4, flag_below_min_N=False, return_no_avg=False, antpos=None,
            rephase=False, freq_array=None, lat=-30.72152, verbose=True, bl_list=None):
    """
    Bin data in Local Sidereal Time (LST) onto an LST grid. An LST grid
    is defined as an array of points increasing in Local Sidereal Time, with each point marking
    the center of the LST bin.

    Parameters:
    -----------
    data_list : type=list, list of DataContainer dictionaries holding
        complex visibility data for each night to average.
    lst_list : type=list, list of ndarrays holding LST bin centers of each data dictionary in data_list.
        These LST arrays must be monotonically increasing, except for a possible wrap at 2pi.
    flags_list : type=list, list of DataContainer dictionaries holding flags for each data dict
        in data_list. Flagged data do not contribute to the average of an LST bin.
    nsamples_list : type=list. List of DataContainer dictionaries holding nsamples for each data dict
        in data_list. nsamples_list values are used to weight the data being averaged
        when median=False. Default is None -> all non-flagged nsamples are set to unity.
        median=False not supported if nsamples_list is not None!
    dlst : type=float, delta-LST spacing for lst_grid. If None, will use the delta-LST of the first
        array in lst_list.
    begin_lst : type=float, beginning LST for making the lst_grid, extending from
        [begin_lst, begin_lst+2pi). Default is begin_lst = 0 radians.
    lst_low : type=float, truncate lst_grid below this lower bound on the LST bin center
    lst_hi : type=float, truncate lst_grid above this upper bound on the LST bin center
    flag_thresh : type=float, minimum fraction of flagged points in an LST bin needed to
        flag the entire bin.
    atol : type=float, absolute tolerance for comparing LST bin center floats
    median : type=boolean, if True use median for LST binning. Warning: this is slower.
             median=True is not supported when nsamples_list is not None.
    truncate_empty : type=boolean, if True, truncate output time bins that have
        no averaged data in them.
    sig_clip : type=boolean, if True, perform a sigma clipping algorithm of the LST bins on the
        real and imag components separately. Resultant clip flags are OR'd between real and imag.
        Warning: This is considerably slow.
    sigma : type=float, input sigma threshold to use for sigma clipping algorithm.
    min_N : type=int, minimum number of points in averaged LST bin needed to perform sigma clipping
    flag_below_min_N : type=bool, if True, flag frequency slices with fewer than min_N data points if sigma clipping
    return_no_avg : type=boolean, if True, return binned but un-averaged data and flags.
    rephase : type=bool, if True, phase data to center of the LST bin before binning.
        Note this produces a copy of the data.
    antpos : type=dictionary, holds antenna position vectors in ENU frame in meters with
        antenna integers as keys and 3D ndarrays as values. Needed for rephase.
    freq_array : type=ndarray, 1D array of unique data frequencies channels in Hz. Needed for rephase.
    lat : type=float, latitude of array in degrees North. Needed for rephase.
    verbose : type=bool, if True report feedback to stdout
    bl_list : optional list of antenna pairs that includes baselines that may not be in the data for the chunk of lsts
              being processed but may be present in other lst chunks.
              baselines not in data will be replaced with completely flagged
              placeholder data in the average to keep the sets of baselines the same across
              different LST-chunks. Placeholder data has
              flags set to True, nsamples set to zero, data set to zero
              consistent across all lst bins.
              DOES NOT ACCEPT ANTPAIRPOLS!

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
    if nsamples_list is not None and median:
        raise NotImplementedError("LST binning with median not supported with nsamples_list is not None.")
    # get visibility shape
    Ntimes, Nfreqs = data_list[0][list(data_list[0].keys())[0]].shape
    # check whether baselines are the same across all nights
    # by checking that every baseline occurs in data_list the same number times.
    bls_same_across_nights = baselines_same_across_nights(data_list)

    # get dlst if not provided
    if dlst is None:
        dlst = np.median(np.diff(lst_list[0]))

    # construct lst_grid
    lst_grid = make_lst_grid(dlst, begin_lst=begin_lst, verbose=verbose)
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
    nsamples = odict()
    all_lst_indices = set()
    pols = list(set([pol for dc in data_list for pol in dc.pols()]))
    # iterate over data_list
    for i, d in enumerate(data_list):
        # get lst array
        li = copy.copy(lst_list[i])

        # ensure l isn't wrapped relative to lst_grid
        li[li < lst_grid_left.min() - atol] += 2 * np.pi

        # digitize data lst array "l"
        grid_indices = np.digitize(li, lst_grid_left[1:], right=True)

        # make data_in_bin boolean array, and set to False data that don't fall in any bin
        data_in_bin = np.ones_like(li, bool)
        data_in_bin[(li < lst_grid_left.min() - atol)] = False
        data_in_bin[(li > lst_grid_left.max() + dlst + atol)] = False

        # update all_lst_indices
        all_lst_indices.update(set(grid_indices[data_in_bin]))

        if rephase:
            # rephase each integration in d to nearest LST bin
            if freq_array is None or antpos is None:
                raise ValueError("freq_array and antpos is needed for rephase")

            # form baseline dictionary
            bls = odict([(k, antpos[k[0]] - antpos[k[1]]) for k in d.keys()])

            # get appropriate lst_shift for each integration, then rephase
            lst_shift = lst_grid[grid_indices] - li

            # this makes a copy of the data in d
            d = utils.lst_rephase(d, bls, freq_array, lst_shift, lat=lat, inplace=False)

        # iterate over keys in d
        klist = list(d.keys())
        for j, key in enumerate(klist):
            # if bl_list is not None, use it to determine conjugation:
            # this is to prevent situations where conjugation of bl in
            # data_list is different from bl in data which can cause
            # inconsistent conjugation conventions in different LST chunks.
            if bl_list is not None:
                if utils.reverse_bl(key)[:2] in bl_list:
                    key = utils.reverse_bl(key)
            # data[key] will be an odict. if data[key] doesn't exist
            # create data[key] as an empty odict. if data[key] already
            # exists, then pass
            if key in data:
                pass
            elif utils.reverse_bl(key) in data:
                # check to see if conj(key) exists in data
                key = utils.reverse_bl(key)
                d[key] = np.conj(d[utils.reverse_bl(key)])
                if flags_list is not None:
                    flags_list[i][key] = flags_list[i][utils.reverse_bl(key)]
                if nsamples_list is not None:
                    nsamples_list[i][key] = nsamples_list[i][utils.reverse_bl(key)]
            else:
                # if key or conj(key) not in data, insert key into data as an odict
                data[key] = odict()
                flags[key] = odict()
                nsamples[key] = odict()
            # data[key] is an odict, with keys as grid index integers and
            # values as lists holding the LST bin data: ndarrays of shape (Nfreqs)

            # iterate over grid_indices, and append to data if data_in_bin is True
            for k, ind in enumerate(grid_indices):
                # ensure data_in_bin is True for this grid index
                if data_in_bin[k]:
                    # if index not in data[key], insert it as empty list
                    if ind not in data[key]:
                        data[key][ind] = np.empty((0, Nfreqs), dtype=d[key].dtype)
                        flags[key][ind] = np.empty((0, Nfreqs), dtype=bool)
                        nsamples[key][ind] = np.empty((0, Nfreqs), dtype=np.int8)
                    # append data ndarray to LST bin
                    data[key][ind] = np.vstack((data[key][ind], d[key][k]))
                    # also insert flags if fed
                    if flags_list is None:
                        flags[key][ind] = np.vstack((flags[key][ind], np.zeros_like(d[key][k], dtype=bool)))
                    else:
                        flags[key][ind] = np.vstack((flags[key][ind], flags_list[i][key][k]))
                    if nsamples_list is None:
                        nsamples[key][ind] = np.vstack((nsamples[key][ind], np.ones_like(d[key][k], dtype=np.int8)))
                    else:
                        nsamples[key][ind] = np.vstack((nsamples[key][ind], nsamples_list[i][key][k]))

        # add in spoofed baselines to keep baselines in different LST files consistent.
        if bl_list is not None:
            for antpair in bl_list:
                for pol in pols:
                    key = antpair + (pol,)
                    if key not in data and ((key[0] != key[1] and utils.reverse_bl(key) not in data) or key[0] == key[1]):
                        # last part lets us spoof ne and en for autocorrs. If we dont include it, only en xor ne will be spoofed.
                        # using np.int8 and complex64 to allow numpy to promote the precision of these arrays only if needed
                        nsamples[key] = odict({ind: np.empty((0, Nfreqs), dtype=np.int8) for ind in range(len(lst_grid))})
                        data[key] = odict({ind: np.empty((0, Nfreqs), dtype=np.complex64) for ind in range(len(lst_grid))})
                        flags[key] = odict({ind: np.empty((0, Nfreqs), dtype=bool) for ind in range(len(lst_grid))})

                    # Since different nights have different sets of baselines and different LST bins have different sets of nights,
                    # it is possible to get a baseline that appears in a subset of the LSTs within an LST chunk
                    # (for example, a baseline that exists in one of the nights that only contained a subset of
                    # the LSTs in the LST chunk being processed).
                    # The following lines address this case.
                    for ind in range(len(lst_grid)):
                        if ind not in nsamples[key]:
                            nsamples[key][ind] = np.empty((0, Nfreqs), dtype=np.int8)
                            flags[key][ind] = np.empty((0, Nfreqs), dtype=bool)
                            data[key][ind] = np.empty((0, Nfreqs), dtype=np.complex64)
                        if len(nsamples[key][ind]) == 0:
                            nsamples[key][ind] = np.vstack((nsamples[key][ind], np.zeros(Nfreqs, dtype=np.int8)))
                            flags[key][ind] = np.vstack((flags[key][ind], np.ones(Nfreqs, dtype=bool)))
                            data[key][ind] = np.vstack((data[key][ind], np.zeros(Nfreqs, dtype=np.complex64)))

    # get final lst_bin array
    if truncate_empty:
        # use only lst_grid bins that have data in them
        lst_bins = lst_grid[sorted(all_lst_indices)]
    else:
        # keep all lst_grid bins and fill empty ones with zero data and mark as flagged
        for index in range(len(lst_grid)):
            if index in all_lst_indices and bls_same_across_nights:
                continue
            else:
                for key in list(data.keys()):
                    # fill data with blank data
                    # if the index is not present.
                    if index not in data[key]:
                        data[key][index] = np.array([np.zeros(Nfreqs, dtype=np.complex64)])
                        flags[key][index] = np.array([np.ones(Nfreqs, dtype=bool)])
                        nsamples[key][index] = np.array([np.zeros(Nfreqs, dtype=np.int8)])

        # use all LST bins
        lst_bins = lst_grid

    # wrap lst_bins if needed
    lst_bins = lst_bins % (2 * np.pi)

    # make final dictionaries
    flags_min = odict()
    data_avg = odict()
    data_count = odict()
    data_std = odict()

    # return un-averaged data if desired
    if return_no_avg:
        # return all binned data instead of just the bin average
        data_bins = odict([(k1, [data[k1][k2] for k2 in data[k1].keys()]) for k1 in data.keys()])
        flag_bins = odict([(k1, [flags[k1][k2] for k2 in flags[k1].keys()]) for k1 in flags.keys()])

        return lst_bins, data_bins, flag_bins

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
            d = np.array(data[key][ind])  # shape = (Ndays x Nfreqs)
            f = np.array(flags[key][ind])
            n = np.array(nsamples[key][ind])
            f[np.isnan(f)] = True

            # replace flagged data with nan
            d[f] *= np.nan  # multiplication (instead of assignment) gets real and imag

            # sigma clip if desired
            if sig_clip:
                # flag frequency slices with fewer than min_N data points
                if flag_below_min_N:
                    below_min_N_freqs = np.where(f.sum(axis=0) < min_N)[0]
                    if below_min_N_freqs.size > 0:
                        d[:, below_min_N_freqs] *= np.nan
                        f[:, below_min_N_freqs] = True

                # clip real
                clip_flags = sigma_clip(d.real, sigma=sigma, min_N=min_N)

                # clip imag and combine, skipping autocorrelations
                if utils.split_bl(key)[0] != utils.split_bl(key)[1]:
                    imag_f = sigma_clip(d.imag, sigma=sigma, min_N=min_N)
                    clip_flags |= imag_f

                # apply min_N condition as sigma_clip only checks axis size, not number of flags
                sc_min_N = np.logical_not(f).sum(axis=0) < min_N
                clip_flags[:, sc_min_N] = False

                # set clipped data to nan
                d[clip_flags] *= np.nan

                # merge clip flags
                f += clip_flags

            # check thresholds for flagging entire output LST bins
            if len(f) == 1:
                flag_bin = np.zeros(f.shape[1], bool)
            else:
                flag_bin = np.sum(f, axis=0).astype(float) / len(f) > flag_thresh
            d[:, flag_bin] *= np.nan
            f[:, flag_bin] = True

            # take bin average: real and imag separately
            if median:
                real_avg.append(np.nanmedian(d.real, axis=0))
                imag_avg.append(np.nanmedian(d.imag, axis=0))
            else:
                # for mean to account for varying nsamples, take nsamples weighted sum.
                # (inverse variance weighted sum).
                isfinite = np.isfinite(d)
                n[~isfinite] = 0.0

                norm = np.sum(n, axis=0).clip(1e-99, np.inf)
                real_avg_t = np.nansum(d.real * n, axis=0) / norm
                imag_avg_t = np.nansum(d.imag * n, axis=0) / norm

                # add back nans as np.nansum sums nan slices to 0
                flagged_f = np.logical_not(isfinite).all(axis=0)
                real_avg_t[flagged_f] = np.nan
                imag_avg_t[flagged_f] = np.nan

                real_avg.append(real_avg_t)
                imag_avg.append(imag_avg_t)

            # get minimum bin flag
            f_min.append(np.nanmin(f, axis=0))

            # get other stats
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")
                real_std.append(np.nanstd(d.real, axis=0))
                imag_std.append(np.nanstd(d.imag, axis=0))
            bin_count.append(np.nansum(~np.isnan(d) * n, axis=0))

        # get final statistics
        d_avg = np.array(real_avg) + 1j * np.array(imag_avg)
        f_min = np.array(f_min)
        d_std = np.array(real_std) + 1j * np.array(imag_std)
        d_num = np.array(bin_count).astype(float)

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
        data_lsts[data_lsts < data_lsts[0]] += 2 * np.pi

    # make lst_grid
    begin_lst = np.max([data_lsts[0] - 1e-5, 0])
    lst_grid = make_lst_grid(dlst, begin_lst=begin_lst, verbose=verbose)

    # get frequency info
    Nfreqs = data[list(data.keys())[0]].shape[1]
    data_freqs = np.arange(Nfreqs)
    model_freqs = np.arange(Nfreqs)

    # restrict lst_grid based on interpolate-able points
    begin_lst = data_lsts[0]
    lst_end = data_lsts[-1]
    lst_grid = lst_grid[(lst_grid > begin_lst - dlst / 2 - atol) & (lst_grid < lst_end + dlst / 2 + atol)]

    # interpolate data
    interp_data, interp_flags = abscal.interp2d_vis(data, data_lsts, data_freqs, lst_grid, model_freqs, flags=flags, **interp_kwargs)

    # wrap lst_grid
    lst_grid = lst_grid % (2 * np.pi)

    return interp_data, interp_flags, lst_grid


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
    a.add_argument('data_files', nargs='*', type=str, help="quotation-bounded, space-delimited, glob-parsable search strings to nightly data files (UVH5)")
    a.add_argument("--input_cals", nargs='*', type=str, help="quotation-bounded, space-delimited, glob-parsable search strings to corresponding nightly calibration files")
    a.add_argument("--dlst", type=float, default=None, help="LST grid bin width")
    a.add_argument("--ntimes_per_file", type=int, default=60, help="number of LST bins to write per output file")
    a.add_argument("--file_ext", type=str, default="{type}.{time:7.5f}.uvh5", help="file extension for output files. See lstbin.lst_bin_files doc-string for format specs.")
    a.add_argument("--outdir", default=None, type=str, help="directory for writing output")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output files")
    a.add_argument("--lst_start", type=float, default=None, help="starting LST for binner as it sweeps across 2pi LST. Default is first LST of first file.")
    a.add_argument("--sig_clip", default=False, action='store_true', help="perform robust sigma clipping before binning")
    a.add_argument("--sigma", type=float, default=4.0, help="sigma threshold for sigma clipping")
    a.add_argument("--min_N", type=int, default=4, help="minimum number of points in bin needed to proceed with sigma clipping")
    a.add_argument("--flag_below_min_N", default=False, action='store_true', help="flag frequency slices if there are fewer than min_N data points when sigma clipping")
    a.add_argument("--rephase", default=False, action='store_true', help="rephase data to center of LST bin before binning")
    a.add_argument("--history", default=' ', type=str, help="history to insert into output files")
    a.add_argument("--atol", default=1e-6, type=float, help="absolute tolerance when comparing LST bin floats")
    a.add_argument("--silence", default=False, action='store_true', help='stop feedback to stdout')
    a.add_argument("--output_file_select", default=None, nargs='*', help="list of output file integers ot run on. Default is all output files.")
    a.add_argument("--vis_units", default='Jy', type=str, help="visibility units of output files.")
    a.add_argument("--ignore_flags", default=False, action='store_true', help="Ignore flags in data files, such that all input data is included in binning.")
    a.add_argument("--Nbls_to_load", default=None, type=int, help="Number of baselines to load and bin simultaneously. Default is all.")
    a.add_argument("--average_redundant_baselines", action="store_true", default=False, help="Redundantly average baselines within and between nights.")
    a.add_argument("--flag_thresh", default=0.7, type=float, help="fraction of flags over all nights in an LST bin on a baseline to flag that baseline.")
    a.add_argument("--ex_ant_yaml_files", default=None, type=str, nargs='+', help="list of paths to yamls with lists of antennas from each night to exclude lstbinned data files.")
    return a


def config_lst_bin_files(data_files, dlst=None, atol=1e-10, lst_start=None, verbose=True, ntimes_per_file=60):
    """
    Configure data for LST binning.

    Make a 24 hour lst grid, starting LST and output files given
    input data files and LSTbin params.

    Parameters
    ----------
    data_files : type=list of lists: nested set of lists, with each nested list containing paths to
                 data files from a particular night. Frequency axis of each file must be identical.
    dlst : type=float, LST bin width. If None, will get this from the first file in data_files.
    atol : type=float, absolute tolerance for LST bin float comparison
    lst_start : type=float, starting LST for binner as it sweeps from lst_start to lst_start + 2pi.
        Default is first LST of the first file of the first night.
    ntimes_per_file : type=int, number of LST bins in a single output file

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

    # get dlst from first data file if None
    if dlst is None:
        dlst, _, _, _ = io.get_file_times(data_files[0][0], filetype='uvh5')

    # get time arrays for each file
    lst_arrays = []
    time_arrays = []
    for di, dfs in enumerate(data_files):
        # get times
        _, _, larrs, tarrs = io.get_file_times(dfs, filetype='uvh5')
        # append
        lst_arrays.append(larrs)
        time_arrays.append(tarrs)

    # get begin_lst from lst_start or from the first JD in the data_files
    if lst_start is None:
        all_lsts = [lst for larrs in lst_arrays for larr in larrs for lst in larr]
        all_times = [time for tarrs in time_arrays for tarr in tarrs for time in tarr]
        lst_start = all_lsts[np.argmin(all_times)]
    begin_lst = lst_start

    # make 24 hour LST grid
    lst_grid = make_lst_grid(dlst, begin_lst=begin_lst, verbose=verbose)
    dlst = np.median(np.diff(lst_grid))

    # enforce that lst_arrays are in the same range as the lst_grid
    for larrs in lst_arrays:
        for larr in larrs:
            while np.any(larr < np.min(lst_grid)):
                larr[larr < np.min(lst_grid)] += 2 * np.pi
            while np.any(larr > np.max(lst_grid)):
                larr[larr > np.max(lst_grid)] -= 2 * np.pi

    # get number of output files
    nfiles = int(np.ceil(len(lst_grid) / ntimes_per_file))

    # flattened lsts across days, nights, files
    flat_lsts = [lst for larrs in lst_arrays for larr in larrs for lst in larr]

    # get output file lsts that are not empty
    all_file_lsts = [lst_grid[ntimes_per_file * i:ntimes_per_file * (i + 1)] for i in range(nfiles)]
    file_lsts = []
    for i, f_lst in enumerate(all_file_lsts):
        fmin = f_lst[0] - (dlst / 2 + atol)
        fmax = f_lst[-1] + (dlst / 2 + atol)
        for lst in flat_lsts:
            if (lst >= fmin) and (lst <= fmax):
                file_lsts.append(f_lst)
                break
    lst_grid = np.array([lst for file_lsts in all_file_lsts for lst in file_lsts])

    return lst_grid, dlst, file_lsts, begin_lst, lst_arrays, time_arrays


def lst_bin_files(data_files, input_cals=None, dlst=None, verbose=True, ntimes_per_file=60,
                  file_ext="{type}.{time:7.5f}.uvh5", outdir=None, overwrite=False, history='', lst_start=None,
                  atol=1e-6, sig_clip=True, sigma=5.0, min_N=5, flag_below_min_N=False, rephase=False,
                  output_file_select=None, Nbls_to_load=None, ignore_flags=False, average_redundant_baselines=False,
                  bl_error_tol=1.0, include_autos=True, ex_ant_yaml_files=None, **kwargs):
    """
    LST bin a series of UVH5 files with identical frequency bins, but varying
    time bins. Output file meta data (frequency bins, antennas positions, time_array)
    are taken from the first file in data_files. Can only LST bin drift-phased data.

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
    (lst_grid, dlst, file_lsts, begin_lst, lst_arrs,
     time_arrs) = config_lst_bin_files(data_files, dlst=dlst, atol=atol, lst_start=lst_start,
                                       ntimes_per_file=ntimes_per_file, verbose=verbose)
    nfiles = len(file_lsts)

    # make sure the JD corresponding to file_lsts[0][0] is the lowest JD in the LST-binned data set
    if (lst_start is not None) and ('lst_branch_cut' not in kwargs):
        kwargs['lst_branch_cut'] = file_lsts[0][0]

    # select file_lsts
    if output_file_select is not None:
        if isinstance(output_file_select, (int, np.integer)):
            output_file_select = [output_file_select]
        output_file_select = [int(o) for o in output_file_select]
        try:
            file_lsts = [file_lsts[i] for i in output_file_select]
        except IndexError:
            print("Warning: one or more indices in output_file_select {} caused an index error with length {} "
                  "file_lsts list, exiting...".format(output_file_select, nfiles))
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

    # get antpos over all nights looking at last file on each night
    nightly_last_hds = []
    for dlist, tarrs in zip(data_files, time_arrs):
        last_file_index = np.argmin([np.min(tarr) for tarr in tarrs])
        hd = io.HERAData(dlist[last_file_index])
        for a in hd.antpos:
            if a not in antpos:
                antpos[a] = hd.antpos[a]
        nightly_last_hds.append(hd)

    # generate a list of dictionaries which contain the nights occupied by each unique baseline
    # (or unique baseline group if average_redundant_baselines is true)
    bl_nightly_dicts = gen_bl_nightly_dicts(nightly_last_hds, bl_error_tol=bl_error_tol,
                                            include_autos=include_autos, redundant=average_redundant_baselines, ex_ant_yaml_files=ex_ant_yaml_files)
    if Nbls_to_load in [None, 'None', 'none']:
        Nbls_to_load = len(bl_nightly_dicts) + 1
    Nblgroups = len(bl_nightly_dicts) // Nbls_to_load + 1
    blgroups = [bl_nightly_dicts[i * Nbls_to_load:(i + 1) * Nbls_to_load] for i in range(Nblgroups)]
    blgroups = [blg for blg in blgroups if len(blg) > 0]

    # iterate over output LST files
    for i, f_lst in enumerate(file_lsts):
        utils.echo("LST file {} / {}: {}".format(i + 1, len(file_lsts), datetime.datetime.now()), type=1, verbose=verbose)
        fmin = f_lst[0] - (dlst / 2 + atol)
        fmax = f_lst[-1] + (dlst / 2 + atol)
        # iterate over baseline groups (for memory efficiency)
        data_conts, flag_conts, std_conts, num_conts = [], [], [], []
        for bi, blgroup in enumerate(blgroups):
            utils.echo("starting baseline-group {} / {}: {}".format(bi + 1, len(blgroups), datetime.datetime.now()), type=0, verbose=verbose)
            # create empty data lists
            data_list = []
            file_list = []
            flgs_list = []
            lst_list = []
            nsamples_list = []
            # iterate over individual nights to bin
            any_lst_overlap = False
            for j in range(len(data_files)):
                nightly_data_list = []
                nightly_flgs_list = []
                nightly_lst_list = []
                nightly_nsamples_list = []
                # iterate over files in each night, and open files that fall into this output file LST range
                for k in range(len(data_files[j])):
                    # unwrap la relative to itself
                    larr = lst_arrs[j][k]
                    tarr = time_arrs[j][k]
                    larr[larr < larr[0]] += 2 * np.pi

                    # phase wrap larr to get it to fall within 2pi of file_lists
                    while larr[0] + 2 * np.pi < fmax:
                        larr += 2 * np.pi
                    while larr[-1] - 2 * np.pi > fmin:
                        larr -= 2 * np.pi

                    # check if this file has overlap with output file
                    if larr[-1] < fmin or larr[0] > fmax:
                        continue

                    any_lst_overlap = True

                    # if overlap, get relevant time indicies
                    tinds = (larr > fmin) & (larr < fmax)

                    # load data: only times needed for this output LST-bin file
                    hd = io.HERAData(data_files[j][k], filetype='uvh5')
                    try:
                        bls_to_load = []
                        key_baselines = []  # map first baseline in each group to
                        # first baseline in group on earliest night that has the baseline.
                        reds = []
                        for bl_nightly_dict in blgroup:
                            # only load group if present in the current night.
                            if j in bl_nightly_dict:
                                # key to earliest night with this redundant group.
                                key_bl = bl_nightly_dict[np.min(list(bl_nightly_dict.keys()))][0]
                                key_baselines.append(key_bl)
                                reds.append(bl_nightly_dict[j])
                                bls_to_load.extend(bl_nightly_dict[j])

                        data, flags, nsamps = hd.read(bls=bls_to_load, times=tarr[tinds])
                        # if we want to throw away data associated with flagged antennas, throw it away.
                        if ex_ant_yaml_files is not None:
                            from hera_qm.utils import apply_yaml_flags
                            hd = apply_yaml_flags(hd, a_priori_flag_yaml=ex_ant_yaml_files[j], ant_indices_only=True, flag_ants=True,
                                                  flag_freqs=False, flag_times=False, throw_away_flagged_ants=True)
                            data, flags, nsamps = hd.build_datacontainers()
                        data.phase_type = 'drift'
                    except ValueError:
                        # if no baselines in the file, skip this file
                        utils.echo("No baselines from blgroup {} found in {}, skipping file for these bls".format(bi + 1, data_files[j][k]), verbose=verbose)
                        # check that the current night is not present in any of the baselines in the current blgroup.
                        if np.all([j not in list(bl_nightly_dict.keys()) for bl_nightly_dict in blgroup]):
                            utils.echo(f"The current night {j} is not present in any of the baseline dicts in the current blgroup.", verbose=verbose)
                        continue

                    # load calibration
                    if input_cals is not None:
                        if input_cals[j][k] is not None:
                            utils.echo("Opening and applying {}".format(input_cals[j][k]), verbose=verbose)
                            uvc = io.to_HERACal(input_cals[j][k])
                            gains, cal_flags, quals, totquals = uvc.read()
                            # down select times in necessary
                            if False in tinds and uvc.Ntimes > 1:
                                # If uvc has Ntimes == 1, then broadcast across time will work automatically
                                uvc.select(times=uvc.time_array[tinds])
                                gains, cal_flags, quals, totquals = uvc.build_calcontainers()
                            apply_cal.calibrate_in_place(data, gains, data_flags=flags, cal_flags=cal_flags,
                                                         gain_convention=uvc.gain_convention)

                    # redundantly average baselines, keying to baseline group key
                    # on earliest night.
                    if average_redundant_baselines:
                        if ignore_flags:
                            raise NotImplementedError("average_redundant_baselines with ignore_flags True is not implemented.")
                        utils.red_average(data=data, flags=flags, nsamples=nsamps,
                                          bl_tol=bl_error_tol, inplace=True,
                                          reds=reds, red_bl_keys=key_baselines)
                    file_list.append(data_files[j][k])
                    nightly_data_list.append(data)  # this is data
                    nightly_flgs_list.append(flags)  # this is flgs
                    nightly_lst_list.append(larr[tinds])  # this is lsts
                    nightly_nsamples_list.append(nsamps)

                # skip if nothing accumulated in nightly files
                if len(nightly_data_list) == 0:
                    continue

                # extend to data lists
                data_list.extend(nightly_data_list)
                flgs_list.extend(nightly_flgs_list)
                lst_list.extend(nightly_lst_list)
                nsamples_list.extend(nightly_nsamples_list)
                del nightly_data_list, nightly_flgs_list, nightly_lst_list, nightly_nsamples_list

            all_blgroup_baselines = [list(bl_nightly_dict.values())[0][0] for bl_nightly_dict in blgroup]
            all_blgroup_antpairpols = []
            if len(data_list) == 0:
                if any_lst_overlap:
                    # spoof data  if data_list is empty but there are some data files with overlap with this lst.
                    # this is to avoid creating lstbinned files with varying numbers of baselines
                    # if we happen to be at an lst bin where one of the blgroups is empty.
                    for pol in hd.pols:
                        for bl in all_blgroup_baselines:
                            all_blgroup_antpairpols.append(bl + (pol,))
                    data_list = [DataContainer({bl: np.ones((len(f_lst), hd.Nfreqs), dtype=complex) for bl in all_blgroup_antpairpols})]
                    flgs_list = [DataContainer({bl: np.ones((len(f_lst), hd.Nfreqs), dtype=bool) for bl in all_blgroup_antpairpols})]
                    lst_list = [f_lst]
                    nsamples_list = [DataContainer({bl: np.zeros((len(f_lst), hd.Nfreqs)) for bl in all_blgroup_antpairpols})]
                else:
                    continue
            # pass through lst-bin function
            if ignore_flags:
                flgs_list = None
            (bin_lst, bin_data, flag_data, std_data,
             num_data) = lst_bin(data_list, lst_list, flags_list=flgs_list, dlst=dlst, begin_lst=begin_lst,
                                 lst_low=fmin, lst_hi=fmax, truncate_empty=False, sig_clip=sig_clip, nsamples_list=nsamples_list,
                                 sigma=sigma, min_N=min_N, flag_below_min_N=flag_below_min_N, rephase=rephase, freq_array=freq_array,
                                 antpos=antpos, bl_list=all_blgroup_baselines)
            # append to lists
            data_conts.append(bin_data)
            flag_conts.append(flag_data)
            std_conts.append(std_data)
            num_conts.append(num_data)
        # if all blgroups were empty skip
        if len(data_conts) == 0:
            utils.echo("data_list is empty for beginning LST {}".format(f_lst[0]), verbose=verbose)
            continue

        # join DataContainers across blgroups
        bin_data = DataContainer(dict(functools.reduce(operator.add, [list(dc.items()) for dc in data_conts])))
        flag_data = DataContainer(dict(functools.reduce(operator.add, [list(dc.items()) for dc in flag_conts])))
        std_data = DataContainer(dict(functools.reduce(operator.add, [list(dc.items()) for dc in std_conts])))
        num_data = DataContainer(dict(functools.reduce(operator.add, [list(dc.items()) for dc in num_conts])))

        # update history
        file_history = history + " Input files: " + "-".join([os.path.basename(ff) for ff in file_list])
        kwargs['history'] = file_history + utils.history_string()

        # form integration time array
        _Nbls = len(set([bl[:2] for bl in list(bin_data.keys())]))
        kwargs['integration_time'] = np.ones(len(bin_lst) * _Nbls, dtype=np.float64) * integration_time

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
            utils.echo("{} exists, not overwriting".format(bin_file), verbose=verbose)
            continue
        # write to file
        io.write_vis(bin_file, bin_data, bin_lst, freq_array, antpos, flags=flag_data, verbose=verbose,
                     nsamples=num_data, filetype='uvh5', x_orientation=x_orientation, **kwargs)
        io.write_vis(std_file, std_data, bin_lst, freq_array, antpos, flags=flag_data, verbose=verbose,
                     nsamples=num_data, filetype='uvh5', x_orientation=x_orientation, **kwargs)

        del bin_file, std_file, bin_data, std_data, num_data, bin_lst, flag_data
        del data_conts, flag_conts, std_conts, num_conts
        garbage_collector.collect()


def make_lst_grid(dlst, begin_lst=None, verbose=True):
    """
    Make a uniform grid in local sidereal time spanning 2pi radians.

    Parameters:
    -----------
    dlst : type=float, delta-LST: width of a single LST bin in radians. 2pi must be equally divisible
                by dlst. If not, will default to the closest dlst that satisfies this criterion that
                is also greater than the input dlst. There is a minimum allowed dlst of 6.283e-6 radians,
                or .0864 seconds.

    begin_lst : type=float, beginning point for lst_grid, which extends out 2pi from begin_lst.
                begin_lst must fall exactly on an LST bin given a dlst, within 0-2pi. If not, it is
                replaced with the closest bin. Default is zero radians.

    Output:
    -------
    lst_grid : type=ndarray, dtype=float, uniform LST grid marking the center of each LST bin
    """
    # check 2pi is equally divisible by dlst
    if not np.isclose((2 * np.pi / dlst) % 1, 0.0, atol=1e-5) and not np.isclose((2 * np.pi / dlst) % 1, 1.0, atol=1e-5):
        # generate array of appropriate dlsts
        dlsts = 2 * np.pi / np.arange(1, 1000000).astype(float)

        # get dlsts closest to dlst, but also greater than dlst
        dlst_diff = dlsts - dlst
        dlst_diff[dlst_diff < 0] = 10
        new_dlst = dlsts[np.argmin(dlst_diff)]
        utils.echo("2pi is not equally divisible by input dlst ({:.16f}) at 1 part in 1e7.\n"
                   "Using {:.16f} instead.".format(dlst, new_dlst), verbose=verbose)
        dlst = new_dlst

    # make an lst grid from [0, 2pi), with the first bin having a left-edge at 0 radians.
    lst_grid = np.arange(0, 2 * np.pi - 1e-7, dlst) + dlst / 2

    # shift grid by begin_lst
    if begin_lst is not None:
        # enforce begin_lst to be within 0-2pi
        if begin_lst < 0 or begin_lst >= 2 * np.pi:
            utils.echo("begin_lst was < 0 or >= 2pi, taking modulus with (2pi)", verbose=verbose)
            begin_lst = begin_lst % (2 * np.pi)
        begin_lst = lst_grid[np.argmin(np.abs(lst_grid - begin_lst))] - dlst / 2
        lst_grid += begin_lst

    return lst_grid


def sigma_clip(array, sigma=4.0, min_N=4):
    """
    One-iteration robust sigma clipping algorithm. Returns clip_flags array.

    Parameters:
    -----------
    array : ndarray of complex visibility data. Clipping performed on 0th axis.

    sigma : float, sigma threshold to cut above.

    min_N : int, minimum length of array to sigma clip, below which no sigma
            clipping is performed.

    Output: flags
    -------
    clip_flags : type=boolean ndarray, has same shape as input array, but has clipped
                 values set to True.
    """
    # ensure array is an array
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    # ensure array passes min_N criterion:
    if array.shape[0] < min_N:
        return np.zeros_like(array, dtype=bool)

    # get robust location
    location = np.nanmedian(array, axis=0)

    # get MAD * 1.482579 for consistency with Gaussian white noise
    scale = np.nanmedian(np.abs(array - location), axis=0) * 1.482579

    # get clipped data
    clip_flags = np.abs(array - location) / scale > sigma

    return clip_flags


def gen_bl_nightly_dicts(hds, bl_error_tol=1.0, include_autos=True, redundant=False, ex_ant_yaml_files=None):
    """
    Helper function to generate baseline dicts to keep track of reds between nights.

    Parameters:
    -----------
    hds : list of HERAData objects. Can have no data loaded (preferable) and should refer to single files.
    bl_error_tol : float (meters), optional. baselines whose vector difference are within this tolerance are considered
                   redundant. Default is 1.0 meter.
    include_autos : bool, if True, include autos in bl_nightly_dicts.
                    default is True.
    redundant : optional, if True, each bl_nightly_dict stores redundant group. If False, each bl_nightly_dict will contain a length-1 group for each baseline
                over all the nights.
                default is False.
    ex_ant_yaml_files : list of strings, optional
                list of paths to yaml files with antennas to throw out on each night.
                default is None (dont throw out any flagged antennas)
    Outputs:
    ---------
    If redundant:
        list of dictionaries of the form {0: [(a0, b0), (a0, c0)...], 1: [(a1, b1),.., ], ... Nnight: [(ANnight, BNnight), ...,]}.
        Each dictionary represents a unique baseline length and orientation.
        where the key of each dictionary is an index for each night to be LST binned and each value
        is the antenna pair representing the unique baseline on that night.
        some baseline dicts will only have a subset of nights.
    If not redundant:
        list of dictionaries of form {0: [(a,b)], 1:[(a, b)], 4:[(a,b)]}
        In other words, each baseline dict corresponds to a unique baseline rather then a group
        and each value is that baseline in a length 1 list and each key is each night in which the baseline is present.
    """
    # check that all hds are minimally redundant
    blvecs = {}
    bl_nightly_dicts = []
    for night, hd in enumerate(hds):
        assert len(hd.filepaths) == 1, 'HERAData objects must be for single data files.'
        reds = redcal.get_reds(hd.antpos, bl_error_tol=bl_error_tol, pols=[hd.pols[0]], include_autos=include_autos)
        # get data bls from ant_1_array and ant_2_array
        data_bls = sorted(set(zip(hd.ant_1_array, hd.ant_2_array, [hd.pols[0]] * len(hd.ant_1_array))))
        reds = redcal.filter_reds(reds, bls=data_bls)
        # if we are throwing away data with flagged ants, do it here.
        if ex_ant_yaml_files is not None:
            from hera_qm.metrics_io import read_a_priori_ant_flags
            a_priori_antenna_flags = read_a_priori_ant_flags(ex_ant_yaml_files[night], ant_indices_only=True)
            reds = redcal.filter_reds(reds, ex_ants=a_priori_antenna_flags)
        reds = [[bl[:2] for bl in grp] for grp in reds]
        for grp in reds:
            for bl in grp:
                # store baseline vectors for all data.
                blvecs[bl] = hd.antpos[bl[1]] - hd.antpos[bl[0]]
                blvecs[bl[::-1]] = hd.antpos[bl[0]] - hd.antpos[bl[1]]
        # otherwise, loop through baselines, for each bl_nightly_dict, see if the first
        # entry matches (or conjugate matches). If yes, append to that bl_nightly_dict
        for grp in reds:
            if redundant:
                present = False
                for bl_nightly_dict in bl_nightly_dicts:
                    # check if baseline group occured in previous nights
                    # if it did, add it to the appropriate bl_nightly_dict.
                    for i in bl_nightly_dict:
                        if np.linalg.norm(blvecs[grp[0]] - blvecs[bl_nightly_dict[i][0]]) <= bl_error_tol or np.linalg.norm(blvecs[grp[0]] + blvecs[bl_nightly_dict[i][0]]) <= bl_error_tol:
                            # I was having a lot of trouble getting a unittest for two separate cases here because
                            # I'm not sure how to conveniently create a UVData object with baselines conjugated.
                            # two cases is more readable then one so I'd prefer to have two.
                            sign = -1
                            if np.linalg.norm(blvecs[grp[0]] - blvecs[bl_nightly_dict[i][0]]) <= bl_error_tol:
                                sign = 1
                            bl_nightly_dict[night] = [bl[::sign] for bl in grp]
                            present = True
                            break
                if not present:
                    # this baseline group has not occured in previous nights
                    # add it.
                    bl_nightly_dicts.append({night: grp})
            else:
                for bl in grp:
                    present = False
                    # check if baseline occured in previous nights
                    # if it did, add it to its corresponding baseline dictionary.
                    for bl_nightly_dict in bl_nightly_dicts:
                        for i in bl_nightly_dict:
                            if bl in bl_nightly_dict[i] or bl[::-1] in bl_nightly_dict[i]:
                                # I was having a lot of trouble getting a unittest for two separate cases here because
                                # I'm not sure how to conveniently create a UVData object with baselines conjugated.
                                # two cases is more readable then one so I'd prefer to have two.
                                sign = -1
                                if np.linalg.norm(blvecs[grp[0]] - blvecs[bl_nightly_dict[i][0]]) <= bl_error_tol:
                                    sign = 1
                                bl_nightly_dict[night] = [bl[::sign]]
                                present = True
                                break
                    if not present:
                        # if this baseline does not appear in previous nights
                        # add it with this night.
                        bl_nightly_dicts.append({night: [bl]})
    return bl_nightly_dicts
