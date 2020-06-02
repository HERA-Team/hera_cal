# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Module for delay filtering data and related operations."""

import numpy as np

from . import io
from . import version
from .vis_clean import VisClean

import pickle
import random
import glob
import os
import warnings


class DelayFilter(VisClean):
    """
    DelayFilter object.

    Used for delay CLEANing and filtering.
    See vis_clean.VisClean for CLEAN functions.
    """

    def run_delay_filter(self, to_filter=None, weight_dict=None, horizon=1., standoff=0.15, min_dly=0.0, mode='clean',
                         skip_wgt=0.1, tol=1e-9, verbose=False, cache_dir=None, read_cache=False, write_cache=False, **filter_kwargs):
        '''
        Run uvtools.dspec.vis_filter on data.

        Run a delay-filter on (a subset of) the data stored in the object.
        Uses stored flags unless explicitly overridden with weight_dict.

        Arguments:
            to_filter: list of visibilities to filter in the (i,j,pol) format.
                If None (the default), all visibilities are filtered.
            weight_dict: dictionary or DataContainer with all the same keys as self.data.
                Linear multiplicative weights to use for the delay filter. Default, use np.logical_not
                of self.flags. uvtools.dspec.delay_filter will renormalize to compensate.
            horizon: coefficient to bl_len where 1 is the horizon [freq filtering]
            standoff: fixed additional delay beyond the horizon (in nanosec) to filter [freq filtering]
            min_dly: max delay (in nanosec) used for freq filter is never below this.
            mode: string specifying filtering mode. See fourier_filter or uvtools.dspec.fourier_filter for supported modes.
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
            self.clean_info: Dictionary of info from uvtools.dspec.delay_filter with the same keys as self.data
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
                       skip_wgt=skip_wgt, overwrite=True, verbose=verbose, **filter_kwargs)
        if not mode == 'clean':
            if write_cache:
                filter_cache = io.write_filter_cache_scratch(filter_cache, cache_dir, skip_keys=keys_before)


def load_delay_filter_and_write(infilename, calfile=None, Nbls_per_load=None, spw_range=None, cache_dir=None,
                                read_cache=False, write_cache=False,
                                res_outfilename=None, CLEAN_outfilename=None, filled_outfilename=None,
                                clobber=False, add_to_history='', **filter_kwargs):
    '''
    Uses partial data loading and writing to perform delay filtering.

    Arguments:
        infilename: string path to data to uvh5 file to load
        cal: optional string path to calibration file to apply to data before delay filtering
        Nbls_per_load: int, the number of baselines to load at once.
                       If None, load all baselines at once. default : None.
        spw_range: spw_range of data to delay-filter.
        cache_dir: string, optional, path to cache file that contains pre-computed dayenu matrices.
                    see uvtools.dspec.dayenu_filter for key formats.
        read_cache: bool, If true, read existing cache files in cache_dir before running.
        write_cache: bool. If true, create new cache file with precomputed matrices
                           that were not in previously loaded cache files.
        res_outfilename: path for writing the filtered visibilities with flags
        CLEAN_outfilename: path for writing the CLEAN model visibilities (with the same flags)
        filled_outfilename: path for writing the original data but with flags unflagged and replaced
            with CLEAN models wherever possible
        clobber: if True, overwrites existing file at the outfilename
        add_to_history: string appended to the history of the output file
        filter_kwargs: additional keyword arguments to be passed to DelayFilter.run_delay_filter()
    '''
    hd = io.HERAData(infilename, filetype='uvh5')
    if calfile is not None:
        calfile = io.HERACal(calfile)
        calfile.read()
    if spw_range is None:
        spw_range = [0, hd.Nfreqs]
    freqs = hd.freqs[spw_range[0]:spw_range[1]]
    if Nbls_per_load is None:
        df = DelayFilter(hd, input_cal=calfile)
        df.read(frequencies=freqs)
        df.run_delay_filter(cache_dir=cache_dir, read_cache=read_cache, write_cache=write_cache, **filter_kwargs)
        df.write_filtered_data(res_outfilename=res_outfilename, CLEAN_outfilename=CLEAN_outfilename,
                               filled_outfilename=filled_outfilename, partial_write=False,
                               clobber=clobber, add_to_history=add_to_history,
                               extra_attrs={'Nfreqs': len(freqs), 'freq_array': np.asarray([freqs])})
    else:
        for i in range(0, len(hd.bls), Nbls_per_load):
            df = DelayFilter(hd, input_cal=calfile)
            df.read(bls=hd.bls[i:i + Nbls_per_load], frequencies=freqs)
            df.run_delay_filter(cache_dir=cache_dir, read_cache=read_cache, write_cache=write_cache, **filter_kwargs)
            df.write_filtered_data(res_outfilename=res_outfilename, CLEAN_outfilename=CLEAN_outfilename,
                                   filled_outfilename=filled_outfilename, partial_write=True,
                                   clobber=clobber, add_to_history=add_to_history, Nfreqs=len(freqs), freq_array=np.asarray([freqs]))
            df.hd.data_array = None  # this forces a reload in the next loop
