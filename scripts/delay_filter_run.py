#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"Command-line drive script for hera_cal.delay_filter. Only performs CLEAN Filtering"

from hera_cal import delay_filter
import sys

parser = delay_filter.delay_filter_argparser()

a = parser.parse_args()
# set kwargs
if a.mode == 'clean':
    filter_kwargs = {'standoff': a.standoff, 'horizon': a.horizon, 'tol': a.tol, 'window': a.window,
                     'skip_wgt': a.skip_wgt, 'maxiter': a.maxiter, 'edgecut_hi': a.edgecut_hi,
                     'edgecut_low': a.edgecut_low, 'min_dly': a.min_dly, 'gain': a.gain}
    if a.window == 'tukey':
        filter_kwargs['alpha'] = a.alpha
    avg_red_bllens=False
    skip_gaps_larger_then_filter_period=False
    skip_flagged_edges=False
    max_contiguous_edge_flags=10000
    flag_model_rms_outliers=False
elif a.mode == 'dayenu':
    filter_kwargs = {'standoff': a.standoff, 'horizon': a.horizon, 'tol': a.tol,
                     'skip_wgt': a.skip_wgt, 'min_dly': a.min_dly}
    avg_red_bllens=True
    max_contiguous_edge_flags=10000
    skip_gaps_larger_then_filter_period=False
    skip_flagged_edges=False
    flag_model_rms_outliers=False
elif a.mode == 'dpss_leastsq':
    filter_kwargs = {'standoff': a.standoff, 'horizon': a.horizon, 'tol': a.tol,
                     'skip_wgt': a.skip_wgt, 'min_dly': a.min_dly}
    avg_red_bllens=True
    skip_gaps_larger_then_filter_period=True
    skip_flagged_edges=True
    max_contiguous_edge_flags=1
    flag_model_rms_outliers=True
else:
    raise ValueError(f"mode {mode} not supported.")
if a.calfile is not None:
    if a.calfile.lower() == 'none':
        a.calfile = None

# Run Delay Filter
delay_filter.load_delay_filter_and_write(a.infilename, calfile=a.calfile, avg_red_bllens=avg_red_bllens,
                                         Nbls_per_load=a.partial_load_Nbls, spw_range=a.spw_range,
                                         cache_dir=a.cache_dir, res_outfilename=a.res_outfilename,
                                         clobber=a.clobber, write_cache=a.write_cache,
                                         read_cache=a.read_cache, mode=mode,
                                         filled_outfilename=a.filled_outfilename,
                                         CLEAN_outfilename=a.CLEAN_outfilename, skip_if_flag_within_edge_distance=a.skip_if_flag_within_edge_distance,
                                         factorize_flags=a.factorize_flags, time_thresh=a.time_thresh,
                                         max_contiguous_edge_flags=max_contiguous_edge_flags,
                                         add_to_history=' '.join(sys.argv), verbose=a.verbose,
                                         skip_flagged_edges=skip_flagged_edges,
                                         flag_yaml=a.flag_yaml,
                                         external_flags=a.external_flags,
                                         skip_contiguous_flags=skip_gaps_larger_then_filter_period,
                                         overwrite_flags=a.overwrite_flags,
                                         flag_model_rms_outliers=flag_model_rms_outliers,
                                         clean_flags_in_resid_flags=True, **filter_kwargs)
