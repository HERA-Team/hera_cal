#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for hera_cal.xtalk_filter with baseline parallelization. Only performs DAYENU Filtering"""

from hera_cal import xtalk_filter
import sys
import hera_cal.io as io

parser = xtalk_filter.xtalk_filter_argparser(multifile=True)

a = parser.parse_args()

# set kwargs
if a.mode == 'clean':
    filter_kwargs = {'tol': a.tol, 'window': a.window, 'max_frate_coeffs': a.max_frate_coeffs,
                    'skip_wgt': a.skip_wgt, 'maxiter': a.maxiter, 'edgecut_hi': a.edgecut_hi,
                    'edgecut_low': a.edgecut_low, 'gain': a.gain}
    if a.window == 'tukey':
        filter_kwargs['alpha'] = a.alpha
    avg_red_bllens=False
    skip_gaps_larger_then_filter_period=False
    skip_flagged_edges=False
    max_contiguous_edge_flags=10000
    flag_model_rms_outliers=False
elif a.mode == 'dayenu':
    filter_kwargs = {'tol': a.tol, 'max_frate_coeffs': a.max_frate_coeffs}
    avg_red_bllens=True
    max_contiguous_edge_flags=10000
    skip_gaps_larger_then_filter_period=False
    skip_flagged_edges=False
    flag_model_rms_outliers=False
elif a.mode == 'dpss_leastsq':
    filter_kwargs = {'tol': a.tol, 'max_frate_coeffs': a.max_frate_coeffs}
    avg_red_bllens=True
    skip_gaps_larger_then_filter_period=True
    skip_flagged_edges=True
    max_contiguous_edge_flags=1
    flag_model_rms_outliers=True


if args.cornerturnfile is not None:
    baseline_list = io.baselines_from_filelist_position(filename=a.cornerturnfile, filelist=a.datafilelist)
else:
    baseline_list = None

if len(baseline_list) > 0:
    # modify output file name to include index.
    spw_range = a.spw_range
    # allow none string to be passed through to a.calfile
    if isinstance(a.calfile_list, str) and a.calfile_list.lower() == 'none':
        a.calfile_list = None
    # Run Xtalk Filter
    xtalk_filter.load_xtalk_filter_and_write(a.datafilelist, calfile_list=a.calfilelist, avg_red_bllens=True,
                                             baseline_list=baseline_list, spw_range=a.spw_range,
                                             cache_dir=a.cache_dir, filled_outfilename=a.filled_outfilename,
                                             clobber=a.clobber, write_cache=a.write_cache, CLEAN_outfilename=a.CLEAN_outfilename,
                                             read_cache=a.read_cache, mode=a.mode, res_outfilename=a.res_outfilename,
                                             factorize_flags=a.factorize_flags, time_thresh=a.time_thresh,
                                             max_contiguous_edge_flags=a.max_contiguous_edge_flags,
                                             add_to_history=' '.join(sys.argv), verbose=a.verbose,
                                             skip_flagged_edges=a.skip_flagged_edges,
                                             flag_yaml=a.flag_yaml,
                                             external_flags=a.external_flags, inpaint=a.inpaint, frate_standoff=a.frate_standoff,
                                             skip_contiguous_flags=skip_gaps_larger_then_filter_period,
                                             overwrite_flags=a.overwrite_flags, skip_if_flag_within_edge_distance=a.skip_if_flag_within_edge_distance,
                                             flag_model_rms_outliers=flag_model_rms_outliers,
                                             clean_flags_in_resid_flags=True, **filter_kwargs)
