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

from . import utils
from . import version
from . import abscal
from . import io
from . import redcal
from . import apply_cal
from .datacontainer import DataContainer


def lst_bin(data_list, lst_list, flags_list=None, dlst=None, begin_lst=None, lst_low=None,
            lst_hi=None, flag_thresh=0.7, atol=1e-10, median=False, truncate_empty=True,
            sig_clip=False, sigma=4.0, min_N=4, return_no_avg=False, antpos=None, rephase=False,
            freq_array=None, lat=-30.72152, verbose=True, nsamp_list=None, bl_list=None):
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
    truncate_empty : type=boolean, if True, truncate output time bins that have
        no averaged data in them.
    sig_clip : type=boolean, if True, perform a sigma clipping algorithm of the LST bins on the
        real and imag components separately. Resultant clip flags are OR'd between real and imag.
        Warning: This is considerably slow.
    sigma : type=float, input sigma threshold to use for sigma clipping algorithm.
    min_N : type=int, minimum number of points in averaged LST bin needed to perform sigma clipping
    return_no_avg : type=boolean, if True, return binned but un-averaged data and flags.
    rephase : type=bool, if True, phase data to center of the LST bin before binning.
        Note this produces a copy of the data.
    antpos : type=dictionary, holds antenna position vectors in ENU frame in meters with
        antenna integers as keys and 3D ndarrays as values. Needed for rephase.
    freq_array : type=ndarray, 1D array of unique data frequencies channels in Hz. Needed for rephase.
    lat : type=float, latitude of array in degrees North. Needed for rephase.
    verbose : type=bool, if True report feedback to stdout
    nsamp_list : list of nsamples arrays
    bl_list : optional list of antenna pairs that includes baselines that may not be in the data for the chunk of lsts
              being processed but may be present in other lst chunks.
              baselines not in data will be spoofed in the average to keep baselines in lst-averaged files.
              flags set to True, nsamples set to zero, data set to one.
              consistent across all lst bins.

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
    Ntimes, Nfreqs = data_list[0][list(data_list[0].keys())[0]].shape

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
    pols = []
    for key in data_list[0].keys():
        if key[-1] not in pols:
            pols.append(key[-1])
    # iterate over data_list
    for i, d in enumerate(data_list):
        # get lst array
        li = copy.copy(lst_list[i])

        # ensure l isn't wrapped relative to lst_grid
        li[li < lst_grid_left.min() - atol] += 2 * np.pi

        # digitize data lst array "l"
        grid_indices = np.digitize(li, lst_grid_left[1:], right=True)

        # make data_in_bin boolean array, and set to False data that don't fall in any bin
        data_in_bin = np.ones_like(li, np.bool)
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
        for j, key in enumerate(d.keys()):
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
                if nsamp_list is not None:
                    nsamp_list[i][key] = nsamp_list[i][utils.reverse_bl(key)]
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
                        data[key][ind] = []
                        flags[key][ind] = []
                        nsamples[key][ind] = []
                    # append data ndarray to LST bin
                    data[key][ind].append(d[key][k])
                    # also insert flags if fed
                    if flags_list is None:
                        flags[key][ind].append(np.zeros_like(d[key][k], np.bool))
                    else:
                        flags[key][ind].append(flags_list[i][key][k])
                    if nsamp_list is None:
                        nsamples[key][ind].append(np.ones_like(d[key][k], np.int))
                    else:
                        nsamples[key][ind].append(nsamp_list[i][key][k])
        # add in spoofed baselines to keep baselines in different LST files consistent.
        if bl_list is not None:
            for antpair in bl_list:
                for pol in pols:
                    key = antpair + (pol,)
                    if key not in data and utils.reverse_bl(key) not in data:
                        nsamples[key] = odict({ind: [] for ind in grid_indices})
                        data[key] = odict({ind: [] for ind in grid_indices})
                        flags[key] = odict({ind: [] for ind in grid_indices})
                        for k, ind in enumerate(grid_indices):
                            nsamples[key][ind].append(np.zeros(Nfreqs))
                            flags[key][ind].append(np.ones(Nfreqs, dtype=bool))
                            data[key][ind].append(np.ones(Nfreqs, dtype=complex))

    # get final lst_bin array
    if truncate_empty:
        # use only lst_grid bins that have data in them
        lst_bins = lst_grid[sorted(all_lst_indices)]
    else:
        # keep all lst_grid bins and fill empty ones with unity data and mark as flagged
        for index in range(len(lst_grid)):
            # I got rid of this statement because it causes failures if a
            # subset of baselines dont have data in every LST bin which can happen
            # if a baseline is flagged on some days an not others.
            for key in list(data.keys()):
                # fill data with blank data
                # if the index is not present.
                if index not in data[key]:
                    data[key][index] = [np.ones(Nfreqs, np.complex)]
                    flags[key][index] = [np.ones(Nfreqs, np.bool)]
                    nsamples[key][index] = [np.ones(Nfreqs, np.int)]

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
                # clip real
                real_f = sigma_clip(d.real, sigma=sigma, min_N=min_N, axis=0)

                # clip imag
                imag_f = sigma_clip(d.imag, sigma=sigma, min_N=min_N, axis=0)

                # get real + imag flags
                clip_flags = real_f + imag_f

                # set clipped data to nan
                d[clip_flags] *= np.nan

                # merge clip flags
                f += clip_flags

            # check thresholds for flagging entire output LST bins
            if len(f) == 1:
                flag_bin = np.zeros(f.shape[1], np.bool)
            else:
                flag_bin = np.sum(f, axis=0).astype(np.float) / len(f) > flag_thresh
            d[:, flag_bin] *= np.nan
            f[:, flag_bin] = True

            # take bin average: real and imag separately
            if median:
                # for median to account for varying nsamples, copy each day by nsamps
                # this code is loopy landscapes. So avoid execution unless we have to.
                if not np.all(np.isclose(n, np.median(n))):
                    dnsamps = np.zeros(d.shape[1], dtype=complex)
                    for chan in range(d.shape[1]):
                        samples_r = []
                        samples_i = []
                        for dd, nn in zip(d[:, chan], n[:, chan]):
                            for m in range(int(nn)):
                                if np.isfinite(dd):
                                    samples_r.append(dd.real)
                                    samples_i.append(dd.imag)
                        dnsamps[chan] = np.median(samples_r) + 1j * np.median(samples_i)
                    real_avg.append(dnsamps.real)
                    imag_avg.append(dnsamps.imag)
                else:
                    real_avg.append(np.nanmedian(d.real, axis=0))
                    imag_avg.append(np.nanmedian(d.imag, axis=0))
            else:
                # for mean ot account for varying nsamples, take nsamples weighted sum.
                # (inverse variance weighted sum).
                resum = np.asarray([np.sum((d.real[:, chan] * n[:, chan])[np.isfinite(d.real[:, chan])]) for chan in range(d.shape[1])])
                re_nsum = np.asarray([np.sum((n[np.isfinite(d.real[:, chan]), chan])) for chan in range(d.shape[1])])
                imsum = np.asarray([np.sum((d.imag[:, chan] * n[:, chan])[np.isfinite(d.imag[:, chan])]) for chan in range(d.shape[1])])
                im_nsum = np.asarray([np.sum((n[np.isfinite(d.imag[:, chan]), chan])) for chan in range(d.shape[1])])

                resum[re_nsum > 0.] /= re_nsum[re_nsum > 0.]
                imsum[im_nsum > 0.] /= im_nsum[im_nsum > 0.]

                real_avg.append(resum)
                imag_avg.append(imsum)

            # get minimum bin flag
            f_min.append(np.nanmin(f, axis=0))

            # get other stats
            real_std.append(np.nanstd(d.real, axis=0))
            imag_std.append(np.nanstd(d.imag, axis=0))
            bin_count.append(np.nansum(~np.isnan(d) * n, axis=0))

        # get final statistics
        d_avg = np.array(real_avg) + 1j * np.array(imag_avg)
        f_min = np.array(f_min)
        d_std = np.array(real_std) + 1j * np.array(imag_std)
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
    a.add_argument('data_files', nargs='*', type=str, help="quotation-bounded, space-delimited, glob-parsable search strings to time-contiguous nightly data files (UVH5)")
    a.add_argument("--input_cals", nargs='*', type=str, help="quotation-bounded, space-delimited, glob-parsable search strings to time-contiguous nightly calibration files")
    a.add_argument("--dlst", type=float, default=None, help="LST grid bin width")
    a.add_argument("--lst_start", type=float, default=None, help="starting LST for binner as it sweeps across 2pi LST. Default is first LST of first file.")
    a.add_argument("--lst_stop", type=float, default=None, help="starting LST for binner as it sweeps across 2pi LST. Default is lst_start + 2pi")
    a.add_argument("--fixed_lst_start", action='store_true', default=False, help="If True, make the start of the LST grid equal to lst_start, rather than the LST of the first data record.")
    a.add_argument("--ntimes_per_file", type=int, default=60, help="number of LST bins to write per output file")
    a.add_argument("--file_ext", type=str, default="{type}.{time:7.5f}.uvh5", help="file extension for output files. See lstbin.lst_bin_files doc-string for format specs.")
    a.add_argument("--outdir", default=None, type=str, help="directory for writing output")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output files")
    a.add_argument("--sig_clip", default=False, action='store_true', help="perform robust sigma clipping before binning")
    a.add_argument("--sigma", type=float, default=4.0, help="sigma threshold for sigma clipping")
    a.add_argument("--min_N", type=int, default=4, help="minimum number of points in bin needed to proceed with sigma clipping")
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
    return a


def config_lst_bin_files(data_files, dlst=None, atol=1e-10, lst_start=None, lst_stop=None, fixed_lst_start=False, verbose=True,
                         ntimes_per_file=60):
    """
    Configure data for LST binning.

    Make an lst grid, starting LST and output files given
    input data files and LSTbin params.

    Parameters
    ----------
    data_files : type=list of lists: nested set of lists, with each nested list containing
                 paths to data files from a particular night. These files should be sorted
                 by ascending Julian Date. Frequency axis of each file must be identical.
    dlst : type=float, LST bin width. If None, will get this from the first file in data_files.
    lst_start : type=float, starting LST for binner as it sweeps from lst_start to lst_start + 2pi.
        Default is first LST of first file.
    lst_stop : type=float, stopping LST for binner as it sweeps from lst_start to lst_start + 2pi.
        Default is lst_start + 2pi.
    fixed_lst_start : type=bool, if True, LST grid starts at lst_start, regardless of LST of first data
        record. Otherwise, LST grid starts at LST of first data record.
    ntimes_per_file : type=int, number of LST bins in a single output file

    Returns
    -------
    lst_grid : float ndarray holding LST bin centers
    dlst : float, LST bin width of output lst_grid
    file_lsts : list, contains the lst grid of each output file
    begin_lst : float, starting lst for LST binner. If fixed_lst_start, this equals lst_start.
    lst_arrays : list, list of lst arrays for each file
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
        dlsts, dtimes, larrs, tarrs = io.get_file_times(dfs, filetype='uvh5')

        # get lmin: LST of first integration from first file
        if di == 0:
            lmin = larrs[0][0]

        # unwrap relative to lmin
        for la in larrs:
            if la[0] < lmin:
                la += 2 * np.pi

        # get lmax
        if di == (len(data_files) - 1):
            lmax = larrs[-1][-1]

        # append
        lst_arrays.append(larrs)
        time_arrays.append(tarrs)

    # get starting LST for output binning
    if lst_start is None:
        lst_start = lmin

    # if lst_start is sufficiently below lmin, shift everything down an octave
    if lst_start < (lmin - np.pi):
        lst_arrays -= 2 * np.pi
        lmin -= 2 * np.pi
        lmax -= 2 * np.pi

    # get beginning LST for lst_grid
    if fixed_lst_start:
        begin_lst = lst_start
    else:
        begin_lst = lmin

    # get stopping LST for output binning
    if lst_stop is None:
        lst_stop = lmax
    if lst_stop < begin_lst:
        lst_stop += 2 * np.pi

    # make LST grid
    lst_grid = make_lst_grid(dlst, begin_lst=begin_lst, verbose=verbose)
    dlst = np.median(np.diff(lst_grid))

    # get starting and stopping indicies
    start_index = np.argmin(np.abs(lst_grid - lst_start))
    stop_index = np.argmin(np.abs(lst_grid - lst_stop))

    # get number of output files
    nfiles = int(np.ceil(float(stop_index - start_index) / ntimes_per_file))

    # get output file lsts
    file_lsts = [lst_grid[start_index:stop_index][ntimes_per_file * i:ntimes_per_file * (i + 1)] for i in range(nfiles)]

    return lst_grid, dlst, file_lsts, begin_lst, lst_arrays, time_arrays


def lst_bin_files(data_files, input_cals=None, dlst=None, verbose=True, ntimes_per_file=60,
                  file_ext="{type}.{time:7.5f}.uvh5", outdir=None, overwrite=False, history='', lst_start=None,
                  lst_stop=None, fixed_lst_start=False, atol=1e-6, sig_clip=True, sigma=5.0, min_N=5, rephase=False,
                  output_file_select=None, Nbls_to_load=None, ignore_flags=False, average_redundant_baselines=False,
                  bl_error_tol=1.0, include_autos=True,
                  **kwargs):
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
        paths to files from a particular night. These files should be sorted
        by ascending Julian Date. Frequency axis of each file must be identical.
        x_orientation is inferred from the first item in this list and assumed to be the same for all files
    dlst : type=float, LST bin width. If None, will get this from the first file in data_files.
    lst_start : type=float, starting LST for binner as it sweeps from lst_start to lst_start + 2pi.
    lst_stop : type=float, stopping LST for binner as it sweeps from lst_start to lst_start + 2pi.
    fixed_lst_start : type=bool, if True, LST grid starts at lst_start, regardless of LST of first data
        record. Otherwise, LST grid starts at LST of first data record.
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
    average_redundant_baselines : bool, if True, baselines that are redundant between nights will be averaged together.
                                  When this is set to true, Nbls_to_load is interpreted as the number of redundant groups
                                  to load simultaneously. The number of data waterfalls can be substantially larger in some
                                  cases.
    include_autos : bool, if True, include autocorrelations in redundant baseline averages.
                    default is True.
    bl_error_tol : float, tolerance within which baselines are considered redundant
                   between nights for purposes of average_redundant_baselines.
    kwargs : type=dictionary, keyword arguments to pass to io.write_vis()

    Result:
    -------
    zen.{pol}.LST.{file_lst}.uv : holds LST bin avg (data_array) and bin count (nsample_array)
    zen.{pol}.STD.{file_lst}.uv : holds LST bin stand dev along real and imag (data_array)
    """
    # get file lst arrays
    (lst_grid, dlst, file_lsts, begin_lst, lst_arrs,
     time_arrs) = config_lst_bin_files(data_files, dlst=dlst, lst_start=lst_start, lst_stop=lst_stop, fixed_lst_start=fixed_lst_start,
                                       ntimes_per_file=ntimes_per_file, verbose=verbose)
    nfiles = len(file_lsts)

    # select file_lsts
    if output_file_select is not None:
        if isinstance(output_file_select, (int, np.integer)):
            output_file_select = [output_file_select]
        output_file_select = [int(o) for o in output_file_select]
        try:
            file_lsts = list(map(lambda i: file_lsts[i], output_file_select))
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
    hd = io.HERAData(data_files[-1][0])
    x_orientation = hd.x_orientation

    # get metadata
    freq_array = hd.freqs
    antpos = hd.antpos
    times = hd.times
    start_jd = np.floor(times.min())
    kwargs['start_jd'] = start_jd
    integration_time = np.median(hd.integration_time)
    assert np.all(np.abs(np.diff(times) - np.median(np.diff(times))) < 1e-6), 'All integrations must be of equal length (BDA not supported).'
    # get antpos over all nights
    for dlist in data_files:
        hd = io.HERAData(dlist[-1])
        for a in hd.antpos:
            if a not in antpos:
                antpos[a] = hd.antpos[a]
    # generate a list of dictionaries which contain the nights occupied by each unique baseline
    # (or unique baseline group if average_redundant_baselines is true)
    bldicts = gen_bldicts([io.HERAData(dlists[-1]) for dlists in data_files], bl_error_tol=bl_error_tol, include_autos=include_autos,
                          redundant=average_redundant_baselines)
    # store all baselines in this list. They need to be spoofed
    # if there are some times that dont overlap for nights with different baselines.
    # all_key_baselines = [list(bldict.values())[0] for bldict in blgroups]
    # iterate over output LST files
    if Nbls_to_load in [None, 'None', 'none']:
        Nbls_to_load = len(bldicts)
    Nblgroups = len(bldicts) // Nbls_to_load + 1
    blgroups = [bldicts[i * Nbls_to_load:(i + 1) * Nbls_to_load] for i in range(Nblgroups)]
    blgroups = [blg for blg in blgroups if len(blg) > 0]
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
            nsamp_list = []
            # iterate over individual nights to bin
            for j in range(len(data_files)):
                nightly_data_list = []
                nightly_flgs_list = []
                nightly_lst_list = []
                nightly_nsamp_list = []
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

                    # if overlap, get relevant time indicies
                    tinds = (larr > fmin) & (larr < fmax)

                    # load data: only times needed for this output LST-bin file
                    hd = io.HERAData(data_files[j][k], filetype='uvh5')
                    try:
                        bls_to_load = []
                        key_baselines = []  # map first baseline in each group to
                        # first baseline in group on earliest night.
                        reds = []
                        for bldict in blgroup:
                            # only load group if present in the current night.
                            if j in bldict:
                                # key to earliest night with this redundant group.
                                key_bl = bldict[np.min(list(bldict.keys()))][0]
                                key_baselines.append(key_bl)
                                reds.append(bldict[j])
                                for bl in bldict[j]:
                                    bls_to_load.append(bl)
                        data, flags, nsamps = hd.read(bls=bls_to_load, times=tarr[tinds])
                        data.phase_type = 'drift'
                    except ValueError:
                        # if no baselines in the file, skip this file
                        utils.echo("No baselines from blgroup {} found in {}, skipping file for these bls".format(bi + 1, data_files[j][k]), verbose=verbose)
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
                    nightly_nsamp_list.append(nsamps)

                # skip if nothing accumulated in nightly files
                if len(nightly_data_list) == 0:
                    continue

                # extend to data lists
                data_list.extend(nightly_data_list)
                flgs_list.extend(nightly_flgs_list)
                lst_list.extend(nightly_lst_list)
                nsamp_list.extend(nightly_nsamp_list)
                del nightly_data_list, nightly_flgs_list, nightly_lst_list, nightly_nsamp_list

            all_blgroup_baselines = [list(bldict.values())[0][0] for bldict in blgroup]
            all_blgroup_antpairpols = []
            if len(data_list) == 0:
                # spoof data  if data_list is empty.
                # this is to avoid creating lstbinned files with varying numbers of baselines
                # if we happen to be at an lst bin where one of the blgroups doesn't appear.
                for pol in hd.pols:
                    for bl in all_blgroup_baselines:
                        all_blgroup_antpairpols.append(bl + (pol,))
                data_list = [DataContainer({bl: np.ones((1, hd.Nfreqs), dtype=complex) for bl in all_blgroup_antpairpols})]
                flgs_list = [DataContainer({bl: np.ones((1, hd.Nfreqs), dtype=bool) for bl in all_blgroup_antpairpols})]
                lst_list = [make_lst_grid(dlst, begin_lst=begin_lst, verbose=verbose)[:1]]
                nsamp_list = [DataContainer({bl: np.zeros((1, hd.Nfreqs)) for bl in all_blgroup_antpairpols})]
            # pass through lst-bin function
            if ignore_flags:
                flgs_list = None
            (bin_lst, bin_data, flag_data, std_data,
             num_data) = lst_bin(data_list, lst_list, flags_list=flgs_list, dlst=dlst, begin_lst=begin_lst,
                                 lst_low=fmin, lst_hi=fmax, truncate_empty=False, sig_clip=sig_clip, nsamp_list=nsamp_list,
                                 sigma=sigma, min_N=min_N, rephase=rephase, freq_array=freq_array, antpos=antpos, bl_list=all_blgroup_baselines)
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
        file_history = history + " Input files: " + "-".join(list(map(lambda ff: os.path.basename(ff), file_list)))
        kwargs['history'] = file_history + version.history_string()

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
        dlsts = 2 * np.pi / np.arange(1, 1000000).astype(np.float)

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
    if not isinstance(array, np.ndarray):
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
    clip = np.abs(array - location) / scale > sigma

    # set clipped data to nan and set clipped flags to True
    array[clip] *= np.nan
    clip_flags[clip] = True

    return clip_flags


def gen_bldicts(hds, bl_error_tol=1.0, include_autos=True, redundant=False):
    """
    Helper function to generate baseline dicts to keep track of reds between nights.

    Parameters:
    -----------

    hds : list of HERAData objects. Can have no data loaded (preferable) and should refer to single files.
    bl_error_tol : float (meters), optional. baselines whose vector difference are within this tolerance are considered
                   redundant. Default is 1.0 meter.
    include_autos : bool, if True, include autos in bldicts.
                    default is True.
    redundant : optional, if True, each bldict stores redundant group. If False, each bldict will contain a length-1 group for each baseline
                over all the nights.
                default is False.
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
    bldicts = []
    for night, hd in enumerate(hds):
        assert len(hd.filepaths) == 1, 'HERAData objects must be for single data files.'
        reds = redcal.get_reds(hd.antpos, bl_error_tol=bl_error_tol, pols=hd.pols[0], include_autos=include_autos)
        # read to get baselines in data
        d, _, _ = hd.read()
        data_bls = d.antpairs()
        reds = [[bl for bl in grp if bl[:2] in data_bls or bl[:2][::-1] in data_bls] for grp in reds]
        reds = [grp for grp in reds if len(grp) > 0]
        reds = [[bl[:2] for bl in grp] for grp in reds]
        for grp in reds:
            for bl in grp:
                # store baseline vectors for all data.
                blvecs[bl] = hd.antpos[bl[1]] - hd.antpos[bl[0]]
        # otherwise, loop through baselines, for each bldict, see if the first
        # entry matches (or conjugate matches). If yes, append to that bldict
        for grp in reds:
            if redundant:
                present = False
                for bldict in bldicts:
                    # check if baseline group occured in previous nights
                    # if it did, add it to the appropriate bldict.
                    for i in bldict:
                        if np.linalg.norm(blvecs[grp[0]] - blvecs[bldict[i][0]]) <= bl_error_tol or np.linalg.norm(blvecs[grp[0]] + blvecs[bldict[i][0]]) <= bl_error_tol:
                            # The things I do for coverage.
                            # unittests for more readable code was just too painful.
                            sign = int((-1) ** float(np.linalg.norm(blvecs[grp[0]] + blvecs[bldict[i][0]]) <= bl_error_tol))
                            bldict[night] = [bl[::sign] for bl in grp]
                            present = True
                            break
                if not present:
                    # this baseline group has not occured in previous nights
                    # add it.
                    bldicts.append({night: grp})
            else:
                for bl in grp:
                    present = False
                    # check if baseline occured in previous nights
                    # if it did, add it to its corresponding baseline dictionary.
                    for bldict in bldicts:
                        for i in bldict:
                            if bl in bldict[i] or bl[::-1] in bldict[i]:
                                sign = bl[::-1] in bldict[i]
                                bldict[night] = [bl[::sign]]
                                present = True
                                break
                    if not present:
                        # if this baseline does not appear in previous nights
                        # add it with this night.
                        bldicts.append({night: [bl]})
    return bldicts
