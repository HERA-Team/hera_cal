#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"Command-line drive script for hera_cal.xtalk_filter. Only performs CLEAN Filtering"

from hera_cal import xtalk_filter
import sys

parser = xtalk_filter.xtalk_filter_argparser()

a = parser.parse_args()
# allow none string to be passed through to a.calfile
if a.calfile is not None:
    if a.calfile.lower() == 'none':
        a.calfile = None
# set kwargs
if a.mode == 'clean':
    filter_kwargs = {'tol': a.tol, 'window': a.window, 'max_frate_coeffs': a.max_frate_coeffs,
                    'skip_wgt': a.skip_wgt, 'maxiter': a.maxiter, 'edgecut_hi': a.edgecut_hi,
                    'edgecut_low': a.edgecut_low, 'gain': a.gain}
    if a.window == 'tukey':
        filter_kwargs['alpha'] = a.alpha
    avg_red_bllens=False
    skip_gaps_larger_then_filter_peri=False
    skip_flagged_edges=False
    max_contiguous_edge_flags=10000
    flag_model_rms_outliers=False
elif a.mode == 'dayenu':
    filter_kwargs = {'tol': a.tol, 'max_frate_coeffs': a.max_frate_coeffs}
    avg_red_bllens=True
    max_contiguous_edge_flags=10000
    skip_gaps_larger_then_filter_peri=False
    skip_flagged_edges=False
    flag_model_rms_outliers=False
elif a.mode == 'dpss_leastsq':
    filter_kwargs = {'tol': a.tol, 'max_frate_coeffs': a.max_frate_coeffs}
    avg_red_bllens=True
    skip_gaps_larger_then_filter_peri=True
    skip_flagged_edges=True
    max_contiguous_edge_flags=1
    flag_model_rms_outliers=True


# Run XTalk Filter
xtalk_filter.load_xtalk_filter_and_write(a.infilename, calfile=a.calfile, avg_red_bllens=avg_red_bllens,
                                         Nbls_per_load=a.partial_load_Nbls, spw_range=a.spw_range,
                                         cache_dir=a.cache_dir, res_outfilename=a.res_outfilename,
                                         clobber=a.clobber, write_cache=a.write_cache,
                                         read_cache=a.read_cache, mode=a.mode,
                                         factorize_flags=a.factorize_flags, time_thresh=a.time_thresh,
                                         max_contiguous_edge_flags=max_contiguous_edge_flags,
                                         add_to_history=' '.join(sys.argv), verbose=a.verbose,
                                         skip_flagged_edges=skip_flagged_edges,
                                         skip_contiguous_flags=skip_gaps_larger_then_filter_peri,
                                         flag_yaml=a.flag_yaml,
                                         external_flags=a.external_flags, skip_if_flag_within_edge_distance=a.skip_if_flag_within_edge_distance,
                                         overwrite_flags=a.overwrite_flags,
                                         flag_model_rms_outliers=flag_model_rms_outliers,
                                         clean_flags_in_resid_flags=True, **filter_kwargs)
