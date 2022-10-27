# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Module for delay filtering data and related operations."""

import numpy as np

from . import io
from .vis_clean import VisClean
from . import vis_clean
import pickle
import random
import glob
import os
import warnings
from pyuvdata import UVCal
from copy import deepcopy


class DelayFilter(VisClean):
    """
    DelayFilter object.

    Used for delay CLEANing and filtering.
    See vis_clean.VisClean for CLEAN functions.
    """
    def run_filter(self, **kwargs):
        '''
        wrapper for run_delay_filter. Backwards compatibility. See run_delay_filter for documentation.
        '''
        self.run_delay_filter(**kwargs)

    def run_delay_filter(self, to_filter=None, weight_dict=None, horizon=1., standoff=0.15, min_dly=0.0, mode='clean',
                         skip_wgt=0.1, tol=1e-9, cache_dir=None, read_cache=False, write_cache=False,
                         skip_flagged_edges=False, **filter_kwargs):
        '''
        Run hera_filters.dspec.vis_filter on data.

        Run a delay-filter on (a subset of) the data stored in the object.
        Uses stored flags unless explicitly overridden with weight_dict.

        Arguments:
            to_filter: list of visibilities to filter in the (i,j,pol) format.
                If None (the default), all visibilities are filtered.
            weight_dict: dictionary or DataContainer with all the same keys as self.data.
                Linear multiplicative weights to use for the delay filter. Default, use np.logical_not
                of self.flags. hera_filters.dspec.delay_filter will renormalize to compensate.
            horizon: coefficient to bl_len where 1 is the horizon [freq filtering]
            standoff: fixed additional delay beyond the horizon (in nanosec) to filter [freq filtering]
            min_dly: max delay (in nanosec) used for freq filter is never below this.
            mode: string specifying filtering mode. See fourier_filter or hera_filters.dspec.fourier_filter for supported modes.
            skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
                Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
                time. Skipped channels are then flagged in self.flags.
                Only works properly when all weights are all between 0 and 1.
            tol : float, optional. To what level are foregrounds subtracted.
            cache_dir: string, optional, path to cache file that contains pre-computed dayenu matrices.
                see hera_filters.dspec.dayenu_filter for key formats.
            read_cache: bool, If true, read existing cache files in cache_dir before running.
            write_cache: bool. If true, create new cache file with precomputed matrices
                that were not in previously loaded cache files.
            skip_flagged_edges : bool, if true do not include frequencies at the edge of the band that are fully flagged. Instead
                filter over frequencies bounded by the edge flags.
            filter_kwargs: see fourier_filter for a full list of filter_specific arguments.

        Results are stored in:
            self.clean_resid: DataContainer formatted like self.data with only high-delay components
            self.clean_model: DataContainer formatted like self.data with only low-delay components
            self.clean_info: Dictionary of info from hera_filters.dspec.delay_filter with the same keys as self.data
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
        # loop over all baselines in increments of Nbls
        self.vis_clean(keys=to_filter, data=self.data, flags=self.flags, wgts=weight_dict,
                       ax='freq', x=self.freqs, cache=filter_cache, mode=mode,
                       horizon=horizon, standoff=standoff, min_dly=min_dly, tol=tol,
                       skip_wgt=skip_wgt, overwrite=True,
                       skip_flagged_edges=skip_flagged_edges, **filter_kwargs)
        if not mode == 'clean':
            if write_cache:
                filter_cache = io.write_filter_cache_scratch(filter_cache, cache_dir, skip_keys=keys_before)


def load_delay_filter_and_write(datafile_list, baseline_list=None, calfile_list=None,
                                Nbls_per_load=None, spw_range=None, cache_dir=None,
                                read_cache=False, write_cache=False, avg_red_bllens=False,
                                factorize_flags=False, time_thresh=0.05, external_flags=None,
                                res_outfilename=None, CLEAN_outfilename=None, filled_outfilename=None,
                                clobber=False, add_to_history='', polarizations=None,
                                skip_flagged_edges=False, overwrite_flags=False,
                                flag_yaml=None, read_axis=None, **filter_kwargs):
    '''
    Uses partial data loading and writing to perform delay filtering.
    While this function reads from multiple files (in datafile_list)
    it always writes to a single file for the resid, filled, and model files.

    Arguments:
        datafile_list: list of data files to perform cross-talk filtering on
        baseline_list: list of antenna-pair 2-tuples to filter and write out from the datafile_list.
                       If None, load all baselines in files. Default is None.
        Nbls_per_load: int, the number of baselines to load at once.
            If None, load all baselines at once. default : None.
        calfile_list: optional list of calibration files to apply to data before xtalk filtering
        spw_range: 2-tuple or 2-list, spw_range of data to filter.
        cache_dir: string, optional, path to cache file that contains pre-computed dayenu matrices.
            see hera_filters.dspec.dayenu_filter for key formats.
        read_cache: bool, If true, read existing cache files in cache_dir before running.
        write_cache: bool. If true, create new cache file with precomputed matrices
            that were not in previously loaded cache files.
        avg_red_bllens: bool, if True, round baseline lengths to redundant average. Default is False.
        factorize_flags: bool, optional
            If True, factorize flags before running delay filter. See vis_clean.factorize_flags.
        time_thresh : float
            Fractional threshold of flagged pixels across time needed to flag all times
            per freq channel. It is not recommend to set this greater than 0.5.
            Fully flagged integrations do not count towards triggering time_thresh.
        external_flags : str, optional, path to external flag files to apply
        res_outfilename: path for writing the filtered visibilities with flags
        CLEAN_outfilename: path for writing the CLEAN model visibilities (with the same flags)
        filled_outfilename: path for writing the original data but with flags unflagged and replaced
            with CLEAN models wherever possible
        clobber: if True, overwrites existing file at the outfilename
        add_to_history: string appended to the history of the output file
        polarizations: list of polarizations to include and write.
        skip_flagged_edges: if true, skip flagged edges in filtering.
        flag_yaml: path to manual flagging text file.
        read_axis: str
            str to pass to axis arg for io.HERAData.read(). Default is None
        filter_kwargs: additional keyword arguments to be passed to DelayFilter.run_delay_filter()
    '''
    if baseline_list is not None and Nbls_per_load is not None:
        raise NotImplementedError("baseline loading and partial i/o not yet implemented.")
    hd = io.HERAData(datafile_list, filetype='uvh5', axis='blt')
    if baseline_list is not None and len(baseline_list) == 0:
        warnings.warn("Length of baseline list is zero."
                      "This can happen under normal circumstances when there are more files in datafile_list then baselines."
                      "in your dataset. Exiting without writing any output.", RuntimeWarning)
    else:
        if baseline_list is None:
            if len(hd.filepaths) > 1:
                baseline_list = list(hd.antpairs.values())[0]
            else:
                baseline_list = hd.antpairs
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
            df = DelayFilter(hd, input_cal=cals)
            df.read(bls=baseline_list[i:i + Nbls_per_load],
                    frequencies=freqs, polarizations=polarizations, axis=read_axis)
            if avg_red_bllens:
                df.avg_red_baseline_vectors()
            if external_flags is not None:
                df.apply_flags(external_flags, overwrite_flags=overwrite_flags)
            if flag_yaml is not None:
                df.apply_flags(flag_yaml, overwrite_flags=overwrite_flags, filetype='yaml')
            if factorize_flags:
                df.factorize_flags(time_thresh=time_thresh, inplace=True)
            df.run_delay_filter(cache_dir=cache_dir, read_cache=read_cache, write_cache=write_cache,
                                skip_flagged_edges=skip_flagged_edges, **filter_kwargs)
            df.write_filtered_data(res_outfilename=res_outfilename, CLEAN_outfilename=CLEAN_outfilename,
                                   filled_outfilename=filled_outfilename, partial_write=Nbls_per_load < len(baseline_list),
                                   clobber=clobber, add_to_history=add_to_history,
                                   extra_attrs={'Nfreqs': df.Nfreqs, 'freq_array': df.hd.freq_array, 'channel_width': df.hd.channel_width})
            df.hd.data_array = None  # this forces a reload in the next loop


# ----------------------------------------
# Arg-parser for delay-filtering.
# can be linear or clean.
# ---------------------------------------


def delay_filter_argparser():
    '''
    Arg parser for commandline operation of delay filters.

    Parameters:
        mode, str : optional. Determines sets of arguments to load.
            can be 'clean', 'dayenu', or 'dpss_leastsq'.
        multifile: bool, optional.
            If True, add calfilelist and filelist
            arguments.
    Returns:
        argparser for delay-domain filtering for specified filtering mode
    '''
    a = vis_clean._filter_argparser()
    filt_options = a.add_argument_group(title='Options for the delay filter')
    filt_options.add_argument("--standoff", type=float, default=15.0, help='fixed additional delay beyond the horizon (default 15 ns)')
    filt_options.add_argument("--horizon", type=float, default=1.0, help='proportionality constant for bl_len where 1.0 (default) is the horizon\
                              (full light travel time)')
    filt_options.add_argument("--min_dly", type=float, default=0.0, help="A minimum delay threshold [ns] used for filtering.")
    return a
