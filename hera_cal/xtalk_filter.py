# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Module for xtalk filtering data and related operations."""

import numpy as np

from . import io
from . import version
from .vis_clean import VisClean
from . import vis_clean

import pickle
import random
import glob
import os
import warnings
from copy import deepcopy
from pyuvdata import UVCal
import argparse


class XTalkFilter(VisClean):
    """
    XTalkFilter object.

    Used for fringe-rate Xtalk CLEANing and filtering.
    See vis_clean.VisClean for CLEAN functions.
    """

    def run_xtalk_filter(self, to_filter=None, weight_dict=None, max_frate_coeffs=[0.024, -0.229], mode='clean',
                         skip_wgt=0.1, tol=1e-9, verbose=False, cache_dir=None, read_cache=False,
                         write_cache=False, **filter_kwargs):
        '''
        Run a cross-talk filter on data where the maximum fringe rate is set by the baseline length.

        Run a delay-filter on (a subset of) the data stored in the object.
        Uses stored flags unless explicitly overridden with weight_dict.

        Arguments:
            to_filter: list of visibilities to filter in the (i,j,pol) format.
                If None (the default), all visibilities are filtered.
            weight_dict: dictionary or DataContainer with all the same keys as self.data.
                Linear multiplicative weights to use for the delay filter. Default, use np.logical_not
                of self.flags. uvtools.dspec.xtalk_filter will renormalize to compensate.
            max_frate_coeffs: All fringe-rates below this value are filtered (or interpolated) (in milliseconds).
                              max_frate [mHz] = x1 * EW_bl_len [ m ] + x2
            mode: string specifying filtering mode. See fourier_filter or uvtools.dspec.xtalk_filter for supported modes.
            skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
                Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
                time. Skipped channels are then flagged in self.flags.
                Only works properly when all weights are all between 0 and 1.
            tol : float, optional. To what level are foregrounds subtracted.
            verbose: If True print feedback to stdout
            cache_dir: string, optional, path to cache file that contains pre-computed dayenu matrices.
                        see uvtools.dspec.dayenu_filter for key formats.
            read_cache: bool, If true, read existing cache files in cache_dir before running.
            write_cache: bool. If true, create new cache file with precomputed matrices
                               that were not in previously loaded cache files.
            cache: dictionary containing pre-computed filter products.
            filter_kwargs: see fourier_filter for a full list of filter_specific arguments.

        Results are stored in:
            self.clean_resid: DataContainer formatted like self.data with only high-delay components
            self.clean_model: DataContainer formatted like self.data with only low-delay components
            self.clean_info: Dictionary of info from uvtools.dspec.xtalk_filter with the same keys as self.data
        '''
        # read in cache
        if not mode == 'clean':
            if read_cache:
                filter_cache = io.read_filter_cache_scratch(cache_dir)
            else:
                filter_cache = {}
            keys_before = list(filter_cache.keys())
        else:
            filter_cache = None
        # compute maximum fringe rate dict based on EW baseline lengths.
        if self.round_up_bllens:
            max_frate = io.DataContainer({k: np.max([max_frate_coeffs[0] * np.ceil(self.blvecs[k[:2]][0]) + max_frate_coeffs[1], 0.0]) for k in self.data})
        else:
            max_frate = io.DataContainer({k: np.max([max_frate_coeffs[0] * self.blvecs[k[:2]][0] + max_frate_coeffs[1], 0.0]) for k in self.data})
        # loop over all baselines in increments of Nbls
        self.vis_clean(keys=to_filter, data=self.data, flags=self.flags, wgts=weight_dict,
                       ax='time', x=(self.times - np.mean(self.times)) * 24. * 3600.,
                       cache=filter_cache, mode=mode, tol=tol, skip_wgt=skip_wgt, max_frate=max_frate,
                       overwrite=True, verbose=verbose, **filter_kwargs)
        if not mode == 'clean':
            if write_cache:
                filter_cache = io.write_filter_cache_scratch(filter_cache, cache_dir, skip_keys=keys_before)


def load_xtalk_filter_and_write(infilename, calfile=None, Nbls_per_load=None, spw_range=None, cache_dir=None,
                                read_cache=False, write_cache=False,
                                factorize_flags=False, time_thresh=0.05, trim_edges=False,
                                res_outfilename=None, CLEAN_outfilename=None, filled_outfilename=None,
                                clobber=False, add_to_history='', round_up_bllens=False, **filter_kwargs):
    '''
    Uses partial data loading and writing to perform xtalk filtering.

    Arguments:
        infilename: string path to data to uvh5 file to load
        cal: optional string path to calibration file to apply to data before xtalk filtering
        Nbls_per_load: int, the number of baselines to load at once.
                       If None, load all baselines at once. default : None.
        spw_range: spw_range of data to delay-filter.
        cache_dir: string, optional, path to cache file that contains pre-computed dayenu matrices.
                    see uvtools.dspec.dayenu_filter for key formats.
        read_cache: bool, If true, read existing cache files in cache_dir before running.
        write_cache: bool. If true, create new cache file with precomputed matrices
                           that were not in previously loaded cache files.
        factorize_flags: bool, optional
            If True, factorize flags before running delay filter. See vis_clean.factorize_flags.
        time_thresh : float
            Fractional threshold of flagged pixels across time needed to flag all times
            per freq channel. It is not recommend to set this greater than 0.5.
            Fully flagged integrations do not count towards triggering time_thresh.
        trim_edges : bool, optional
            if true, trim fully flagged edge channels and times. helps to avoid edge popups.
            default is false.
        res_outfilename: path for writing the filtered visibilities with flags
        CLEAN_outfilename: path for writing the CLEAN model visibilities (with the same flags)
        filled_outfilename: path for writing the original data but with flags unflagged and replaced
            with CLEAN models wherever possible
        clobber: if True, overwrites existing file at the outfilename
        add_to_history: string appended to the history of the output file
        round_up_bllens: bool, if True, round up baseline lengths. Default is False.
        filter_kwargs: additional keyword arguments to be passed to XTalkFilter.run_xtalk_filter()
    '''
    hd = io.HERAData(infilename, filetype='uvh5')
    if calfile is not None:
        calfile = io.HERACal(calfile)
        calfile.read()
    if spw_range is None:
        spw_range = [0, hd.Nfreqs]
    freqs = hd.freqs[spw_range[0]:spw_range[1]]
    if Nbls_per_load is None:
        xf = XTalkFilter(hd, input_cal=calfile, round_up_bllens=round_up_bllens)
        xf.read(frequencies=freqs)
        if factorize_flags:
            xf.factorize_flags(time_thresh=time_thresh, inplace=True)
        if trim_edges:
            xf.trim_edges()
        xf.run_xtalk_filter(cache_dir=cache_dir, read_cache=read_cache, write_cache=write_cache, **filter_kwargs)
        xf.write_filtered_data(res_outfilename=res_outfilename, CLEAN_outfilename=CLEAN_outfilename,
                               filled_outfilename=filled_outfilename, partial_write=False,
                               clobber=clobber, add_to_history=add_to_history,
                               extra_attrs={'Nfreqs': xf.Nfreqs, 'freq_array': np.asarray([xf.freqs])})
    else:
        for i in range(0, hd.Nbls, Nbls_per_load):
            xf = XTalkFilter(hd, input_cal=calfile, round_up_bllens=round_up_bllens)
            xf.read(bls=hd.bls[i:i + Nbls_per_load], frequencies=freqs)
            if factorize_flags:
                xf.factorize_flags(time_thresh=time_thresh, inplace=True)
            if trim_edges:
                raise NotImplementedError("trim_edges not implemented for partial baseline loading.")
            xf.run_xtalk_filter(cache_dir=cache_dir, read_cache=read_cache, write_cache=write_cache, **filter_kwargs)
            xf.write_filtered_data(res_outfilename=res_outfilename, CLEAN_outfilename=CLEAN_outfilename,
                                   filled_outfilename=filled_outfilename, partial_write=True,
                                   clobber=clobber, add_to_history=add_to_history,
                                   freq_array=np.asarray([xf.freqs]), Nfreqs=xf.Nfreqs)
            xf.hd.data_array = None  # this forces a reload in the next loop


def load_xtalk_filter_and_write_baseline_list(datafile_list, baseline_list, calfile_list=None, spw_range=None, cache_dir=None,
                                              read_cache=False, write_cache=False,
                                              factorize_flags=False, time_thresh=0.05, trim_edges=False,
                                              res_outfilename=None, CLEAN_outfilename=None, filled_outfilename=None,
                                              clobber=False, add_to_history='', round_up_bllens=False, **filter_kwargs):
    '''
    A xtalk filtering method that only simultaneously loads and writes user-provided
    list of baselines. This is to support parallelization over baseline (rather then time).

    Arguments:
        datafile_list: list of data files to perform cross-talk filtering on
        baseline_list: list of antenna-pair-pol triplets to filter and write out from the datafile_list.
        calfile_list: optional list of calibration files to apply to data before xtalk filtering
        spw_range: 2-tuple or 2-list, spw_range of data to filter.
        cache_dir: string, optional, path to cache file that contains pre-computed dayenu matrices.
                    see uvtools.dspec.dayenu_filter for key formats.
        read_cache: bool, If true, read existing cache files in cache_dir before running.
        write_cache: bool. If true, create new cache file with precomputed matrices
                           that were not in previously loaded cache files.
        factorize_flags: bool, optional
            If True, factorize flags before running delay filter. See vis_clean.factorize_flags.
        time_thresh : float, optional
            Fractional threshold of flagged pixels across time needed to flag all times
            per freq channel. It is not recommend to set this greater than 0.5.
            Fully flagged integrations do not count towards triggering time_thresh.
        trim_edges : bool, optional
            if true, trim fully flagged edge channels and times. helps to avoid edge popups.
            default is false.
        res_outfilename: path for writing the filtered visibilities with flags
        CLEAN_outfilename: path for writing the CLEAN model visibilities (with the same flags)
        filled_outfilename: path for writing the original data but with flags unflagged and replaced
            with CLEAN models wherever possible
        clobber: if True, overwrites existing file at the outfilename
        add_to_history: string appended to the history of the output file
        round_up_bllens: bool, if True, round up baseline lengths. Default is False.
        filter_kwargs: additional keyword arguments to be passed to XTalkFilter.run_xtalk_filter()
    '''
    hd = io.HERAData(datafile_list, filetype='uvh5')
    if spw_range is None:
        spw_range = [0, hd.Nfreqs]
    freqs = hd.freq_array.flatten()[spw_range[0]:spw_range[1]]
    baseline_antennas = []
    for blpolpair in baseline_list:
        baseline_antennas += list(blpolpair[:2])
    baseline_antennas = np.unique(baseline_antennas).astype(int)
    if calfile_list is not None:
        # initialize calfile by iterating through calfile_list, selecting the antennas we need,
        # and concatenating.
        for filenum, calfile in enumerate(calfile_list):
            cal = UVCal()
            cal.read_calfits(calfile)
            # only select calibration antennas that are in the intersection of antennas in
            # baselines to be filtered and the calibration solution.
            ants_overlap = np.intersect1d(cal.ant_array, baseline_antennas).astype(int)
            cal.select(antenna_nums=ants_overlap, frequencies=freqs)
            if filenum == 0:
                cals = deepcopy(cal)
            else:
                cals = cals + cal
        cals = io.to_HERACal(cals)
    else:
        cals = None
    xf = XTalkFilter(hd, input_cal=cals, round_up_bllens=round_up_bllens)
    xf.read(bls=baseline_list, frequencies=freqs)
    if factorize_flags:
        xf.factorize_flags(time_thresh=time_thresh, inplace=True)
    if trim_edges:
        xf.trim_edges()
    xf.run_xtalk_filter(cache_dir=cache_dir, read_cache=read_cache, write_cache=write_cache, **filter_kwargs)
    xf.write_filtered_data(res_outfilename=res_outfilename, CLEAN_outfilename=CLEAN_outfilename,
                           filled_outfilename=filled_outfilename, partial_write=False,
                           clobber=clobber, add_to_history=add_to_history,
                           extra_attrs={'Nfreqs': xf.Nfreqs, 'freq_array': np.asarray([xf.freqs])})

# ------------------------------------------
# Here are arg-parsers for xtalk-filtering.
# ------------------------------------------


def xtalk_filter_argparser(mode='clean', multifile=False):
    '''Arg parser for commandline operation of xtalk filters.

    Parameters
    ----------
    mode : string, optional.
        Determines sets of arguments to load.
        Can be 'clean', 'dayenu', or 'dpss_leastsq'.
    multifile: bool, optional.
        If True, add calfilelist and filelist
        arguments.

    Returns
    -------
    argparser
        argparser for xtalk (time-domain) filtering for specified filtering mode

    '''
    if mode == 'clean':
        a = vis_clean._clean_argparser(multifile=multifile)
    elif mode in ['linear', 'dayenu', 'dpss_leastsq']:
        a = vis_clean._linear_argparser(multifile=multifile)
    filt_options = a.add_argument_group(title='Options for the cross-talk filter')
    a.add_argument("--max_frate_coeffs", type=float, default=None, nargs=2, help="Maximum fringe-rate coefficients for the model max_frate [mHz] = x1 * EW_bl_len [ m ] + x2.")
    return a
