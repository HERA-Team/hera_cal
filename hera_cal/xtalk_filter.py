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
                         skip_wgt=0.1, tol=1e-9, cache_dir=None, read_cache=False,
                         write_cache=False, skip_flagged_edges=False, keep_flags=True,
                         data=None, flags=None, **filter_kwargs):
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
            cache_dir: string, optional, path to cache file that contains pre-computed dayenu matrices.
                see uvtools.dspec.dayenu_filter for key formats.
            read_cache: bool, If true, read existing cache files in cache_dir before running.
            write_cache: bool. If true, create new cache file with precomputed matrices
                that were not in previously loaded cache files.
            cache: dictionary containing pre-computed filter products.
            skip_flagged_edges : bool, if true do not include edge times in filtering region (filter over sub-region).
            keep_flags : bool, if true, retain data flags in filled data.
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
        max_frate = io.DataContainer({k: np.max([max_frate_coeffs[0] * self.blvecs[k[:2]][0] + max_frate_coeffs[1], 0.0]) for k in self.data})
        # loop over all baselines in increments of Nbls
        self.vis_clean(keys=to_filter, data=self.data, flags=self.flags, wgts=weight_dict,
                       ax='time', x=(self.times - np.mean(self.times)) * 24. * 3600.,
                       cache=filter_cache, mode=mode, tol=tol, skip_wgt=skip_wgt, max_frate=max_frate,
                       overwrite=True, skip_flagged_edges=skip_flagged_edges,
                       keep_flags=keep_flags, **filter_kwargs)
        if not mode == 'clean':
            if write_cache:
                filter_cache = io.write_filter_cache_scratch(filter_cache, cache_dir, skip_keys=keys_before)


def load_xtalk_filter_and_write(datafile_list, baseline_list=None, calfile_list=None,
                                Nbls_per_load=None, spw_range=None, cache_dir=None,
                                read_cache=False, write_cache=False, external_flags=None,
                                factorize_flags=False, time_thresh=0.05,
                                res_outfilename=None, CLEAN_outfilename=None, filled_outfilename=None,
                                clobber=False, add_to_history='', avg_red_bllens=False, polarizations=None,
                                skip_flagged_edges=False, overwrite_flags=False,
                                flag_yaml=None,
                                clean_flags_in_resid_flags=True, **filter_kwargs):
    '''
    A xtalk filtering method that only simultaneously loads and writes user-provided
    list of baselines. This is to support parallelization over baseline (rather then time).
    While this function reads from multiple files (in datafile_list)
    it always writes to a single file for the resid, filled, and model files.

    Arguments:
        datafile_list: list of data files to perform cross-talk filtering on
        baseline_list: list of antenna-pair 2-tuples.
                       to filter and write out from the datafile_list.
                       If None, load all baselines in files in datafile_list. Default is None.
        calfile_list: optional list of calibration files to apply to data before xtalk filtering
        Nbls_per_load: int, the number of baselines to load at once.
            If None, load all baselines at once. default : None.
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
        res_outfilename: path for writing the filtered visibilities with flags
        CLEAN_outfilename: path for writing the CLEAN model visibilities (with the same flags)
        filled_outfilename: path for writing the original data but with flags unflagged and replaced
            with CLEAN models wherever possible
        clobber: if True, overwrites existing file at the outfilename
        add_to_history: string appended to the history of the output file
        avg_red_bllens: bool, if True, round baseline lengths to redundant average. Default is False.
        polarizations : list of polarizations to process (and write out). Default None operates on all polarizations in data.
        skip_flagged_edges : bool, if true do not include edge times in filtering region (filter over sub-region).
        overwrite_flags : bool, if true reset data flags to False except for flagged antennas.
        flag_yaml: path to manual flagging text file.
        clean_flags_in_resid_flags: bool, optional. If true, include clean flags in residual flags that get written.
                                    default is True.
        filter_kwargs: additional keyword arguments to be passed to XTalkFilter.run_xtalk_filter()
    '''
    if baseline_list is not None and Nbls_per_load is not None:
        raise NotImplementedError("baseline loading and partial i/o not yet implemented.")
    hd = io.HERAData(datafile_list, filetype='uvh5', axis='blt')
    if baseline_list is None:
        if len(hd.filepaths) > 1:
            baseline_list = list(hd.bls.values())[0]
        else:
            baseline_list = hd.bls
    if len(baseline_list) == 0:
        warnings.warn("Length of baseline list is zero."
                      "This can happen under normal circumstances when there are more files in datafile_list then baselines."
                      "in your dataset. Exiting without writing any output.", RuntimeWarning)
    else:
        if spw_range is None:
            spw_range = [0, hd.Nfreqs]
        freqs = hd.freq_array.flatten()[spw_range[0]:spw_range[1]]
        baseline_antennas = []
        for blpolpair in baseline_list:
            baseline_antennas += list(blpolpair[:2])
        baseline_antennas = np.unique(baseline_antennas).astype(int)
        if calfile_list is not None:
            cals = io.HERACal(calfile_list)
            cals.read(antenna_nums=baseline_antennas, frequencies=freqs)
        else:
            cals = None
        if polarizations is None:
            if len(hd.filepaths) > 1:
                polarizations = list(hd.pols.values())[0]
            else:
                polarizations = hd.pols
        if Nbls_per_load is None:
            Nbls_per_load = len(baseline_list)
        for i in range(0, len(baseline_list), Nbls_per_load):
            xf = XTalkFilter(hd, input_cal=cals, axis='blt')
            xf.read(bls=baseline_list[i:i + Nbls_per_load], frequencies=freqs)
            if avg_red_bllens:
                xf.avg_red_baseline_vectors()
            if external_flags is not None:
                xf.apply_flags(external_flags, overwrite_flags=overwrite_flags)
            if flag_yaml is not None:
                xf.apply_flags(flag_yaml, overwrite_flags=overwrite_flags, filetype='yaml')
            if factorize_flags:
                xf.factorize_flags(time_thresh=time_thresh, inplace=True)
            xf.run_xtalk_filter(cache_dir=cache_dir, read_cache=read_cache, write_cache=write_cache,
                                skip_flagged_edges=skip_flagged_edges, **filter_kwargs)
            xf.write_filtered_data(res_outfilename=res_outfilename, CLEAN_outfilename=CLEAN_outfilename,
                                   filled_outfilename=filled_outfilename, partial_write=Nbls_per_load < len(baseline_list),
                                   clobber=clobber, add_to_history=add_to_history,
                                   extra_attrs={'Nfreqs': xf.hd.Nfreqs, 'freq_array': xf.hd.freq_array})
            xf.hd.data_array = None  # this forces a reload in the next loop

# ------------------------------------------
# Here are arg-parsers for xtalk-filtering.
# ------------------------------------------


def xtalk_filter_argparser():
    '''Arg parser for commandline operation of xtalk filters.

    Parameters
    ----------
    mode : string, optional.
        Determines sets of arguments to load.
        Can be 'clean', 'dayenu', or 'dpss_leastsq'.

    Returns
    -------
    argparser
        argparser for xtalk (time-domain) filtering for specified filtering mode

    '''
    a = vis_clean._filter_argparser()
    filt_options = a.add_argument_group(title='Options for the cross-talk filter')
    filt_options.add_argument("--max_frate_coeffs", type=float, nargs=2, help="Maximum fringe-rate coefficients for the model max_frate [mHz] = x1 * EW_bl_len [ m ] + x2.")
    filt_options.add_argument("--skip_if_flag_within_edge_distance", type=int, default=0, help="skip integrations channels if there is a flag within this integer distance of edge.")
    return a
