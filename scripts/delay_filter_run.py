#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"Command-line drive script for hera_cal.delay_filter with baseline parallelization. Only performs filtering for DAYENU"

from hera_cal import delay_filter
import sys

parser = delay_filter.delay_filter_argparser()
ap = parser.parse_args()

# set kwargs
if ap.mode == 'clean':
    filter_kwargs = {'window': ap.window,
                     'maxiter': ap.maxiter, 'edgecut_hi': ap.edgecut_hi,
                     'edgecut_low': ap.edgecut_low, 'gain': ap.gain}
    if ap.window == 'tukey':
        filter_kwargs['alpha'] = ap.alpha
    avg_red_bllens = False
elif ap.mode in ['dayenu', 'dpss_leastsq']:
    filter_kwargs = {'max_contiguous_edge_flags': ap.max_contiguous_edge_flags}
    avg_red_bllens = True
else:
    raise ValueError(f"mode {mode} not supported.")

if ap.cornerturnfile is not None:
    baseline_list = io.baselines_from_filelist_position(filename=ap.cornerturnfile, filelist=ap.datafilelist)
else:
    baseline_list = None

# allow none string to be passed through to ap.calfile
if isinstance(ap.calfilelist, str) and ap.calfilelist.lower() == 'none':
    ap.calfilelist = None
# Run Delay Filter
delay_filter.load_delay_filter_and_write(ap.datafilelist, calfile_list=ap.calfilelist, avg_red_bllens=avg_red_bllens,
                                         baseline_list=baseline_list, spw_range=ap.spw_range,
                                         cache_dir=ap.cache_dir, res_outfilename=ap.res_outfilename,
                                         clobber=ap.clobber, write_cache=ap.write_cache, external_flags=ap.external_flags,
                                         read_cache=ap.read_cache, mode=ap.mode, overwrite_flags=ap.overwrite_flags,
                                         factorize_flags=ap.factorize_flags, time_thresh=ap.time_thresh,
                                         add_to_history=' '.join(sys.argv), polarizations=ap.polarizations,
                                         verbose=ap.verbose, skip_if_flag_within_edge_distance=ap.skip_if_flag_within_edge_distance,
                                         flag_yaml=ap.flag_yaml, Nbls_per_load=ap.Nbls_per_load,
                                         filled_outfilename=ap.filled_outfilename,
                                         CLEAN_outfilename=ap.CLEAN_outfilename,
                                         standoff=ap.standoff, horizon=ap.horizon, tol=ap.tol,
                                         skip_wgt=ap.skip_wgt, min_dly=ap.min_dly, zeropad=ap.zeropad,
                                         filter_spw_ranges=ap.filter_spw_ranges,
                                         skip_contiguous_flags=not(ap.dont_skip_contiguous_flags), max_contiguous_flag=ap.max_contiguous_flag,
                                         skip_flagged_edges=not(ap.dont_skip_flagged_edges),
                                         flag_model_rms_outliers=not(ap.dont_flag_model_rms_outliers), model_rms_threshold=ap.model_rms_threshold,
                                         clean_flags_in_resid_flags=not(ap.clean_flags_not_in_resid_flags), **filter_kwargs)
