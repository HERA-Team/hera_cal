#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"Command-line drive script for hera_cal.delay_filter with baseline parallelization. Only performs filtering for DAYENU"

from hera_cal import delay_filter
import sys

parser = delay_filter.delay_filter_argparser()
a = parser.parse_args()

# set kwargs
if a.mode == 'clean':
    filter_kwargs = {'window': a.window,
                     'maxiter': a.maxiter, 'edgecut_hi': a.edgecut_hi,
                     'edgecut_low': a.edgecut_low, 'min_dly': a.min_dly, 'gain': a.gain}
    if a.window == 'tukey':
        filter_kwargs['alpha'] = a.alpha
    avg_red_bllens = False
    skip_gaps_larger_then_filter_period = False
    skip_flagged_edges = False
    max_contiguous_edge_flags = 10000
    flag_model_rms_outliers = False
elif a.mode == 'dayenu':
    filter_kwargs = {}
    avg_red_bllens = True
    skip_gaps_larger_then_filter_period = False
    max_contiguous_edge_flags = 10000
    flag_model_rms_outliers = False
elif a.mode == 'dpss_leastsq':
    filter_kwargs = {}
    avg_red_bllens = True
    skip_gaps_larger_then_filter_period = True
    skip_flagged_edges = True
    max_contiguous_edge_flags = 1
    flag_model_rms_outliers = True
else:
    raise ValueError(f"mode {mode} not supported.")


if args.cornerturnfile is not None:
    baseline_list = io.baselines_from_filelist_position(filename=a.cornerturnfile, filelist=a.datafilelist)
else:
    baseline_list = None

# allow none string to be passed through to a.calfile
if isinstance(a.calfile_list, str) and a.calfile_list.lower() == 'none':
    a.calfile_list = None
# Run Delay Filter
delay_filter.load_delay_filter_and_write(a.datafilelist, calfile_list=a.calfilelist, avg_red_bllens=avg_red_bllens,
                                         baseline_list=baseline_list, spw_range=a.spw_range,
                                         cache_dir=a.cache_dir, res_outfilename=a.res_outfilename,
                                         clobber=a.clobber, write_cache=a.write_cache, external_flags=a.external_flags,
                                         read_cache=a.read_cache, mode=mode, overwrite_flags=a.overwrite_flags,
                                         factorize_flags=a.factorize_flags, time_thresh=a.time_thresh,
                                         max_contiguous_edge_flags=max_contiguous_edge_flags,
                                         add_to_history=' '.join(sys.argv), polarizations=a.polarizations,
                                         verbose=a.verbose, skip_if_flag_within_edge_distance=a.skip_if_flag_within_edge_distance,
                                         flag_yaml=a.flag_yaml, Nbls_per_load=a.Nbls_per_load,
                                         skip_contiguous_flags=skip_gaps_larger_then_filter_period,
                                         skip_flagged_edges=skip_flagged_edges,
                                         filled_outfilename=a.filled_outfilename,
                                         CLEAN_outfilename=a.CLEAN_outfilename,
                                         standoff=a.standoff, horizon=a.horizon, tol=a.tol,
                                         skip_wgt=a.skip_wgt, min_dly=a.min_dly,
                                         flag_model_rms_outliers=flag_model_rms_outliers,
                                         clean_flags_in_resid_flags=True, **filter_kwargs)
