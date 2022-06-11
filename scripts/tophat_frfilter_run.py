#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for hera_cal.xtalk_filter with baseline parallelization. Only performs DAYENU Filtering"""

from hera_cal import frf
import sys
import hera_cal.io as io

parser = frf.tophat_frfilter_argparser()

ap = parser.parse_args()

# set kwargs
if ap.mode == 'clean':
    filter_kwargs = {'window': ap.window,
                     'maxiter': ap.maxiter, 'edgecut_hi': ap.edgecut_hi,
                     'edgecut_low': ap.edgecut_low, 'gain': ap.gain}
    if ap.window == 'tukey':
        filter_kwargs['alpha'] = ap.alpha
    filter_kwargs['flag_model_rms_outliers'] = ap.flag_model_rms_outliers
elif ap.mode in ['dayenu', 'dpss_leastsq']:
    filter_kwargs = {'max_contiguous_edge_flags': ap.max_contiguous_edge_flags}

if ap.cornerturnfile is not None:
    baseline_list = io.baselines_from_filelist_position(filename=ap.cornerturnfile, filelist=ap.datafilelist)
else:
    baseline_list = None

# modify output file name to include index.
spw_range = ap.spw_range
# allow none string to be passed through to ap.calfile
if isinstance(ap.calfilelist, str) and ap.calfile_list.lower() == 'none':
    ap.calfile_list = None
# Run Xtalk Filter
frf.load_tophat_frfilter_and_write(ap.datafilelist, calfile_list=ap.calfilelist, avg_red_bllens=True,
                                   baseline_list=baseline_list, spw_range=ap.spw_range,
                                   cache_dir=ap.cache_dir, filled_outfilename=ap.filled_outfilename,
                                   clobber=ap.clobber, write_cache=ap.write_cache, CLEAN_outfilename=ap.CLEAN_outfilename,
                                   read_cache=ap.read_cache, mode=ap.mode, res_outfilename=ap.res_outfilename,
                                   factorize_flags=ap.factorize_flags, time_thresh=ap.time_thresh,
                                   wgt_by_nsample=ap.wgt_by_nsample, lst_blacklists=ap.lst_blacklists, blacklist_wgt=ap.blacklist_wgt,
                                   add_to_history=' '.join(sys.argv), verbose=ap.verbose,
                                   flag_yaml=ap.flag_yaml, Nbls_per_load=ap.Nbls_per_load, case=ap.case,
                                   external_flags=ap.external_flags, filter_spw_ranges=ap.filter_spw_ranges,
                                   overwrite_flags=ap.overwrite_flags, skip_autos=ap.skip_autos,
                                   skip_if_flag_within_edge_distance=ap.skip_if_flag_within_edge_distance,
                                   zeropad=ap.zeropad, tol=ap.tol, skip_wgt=ap.skip_wgt, max_frate_coeffs=ap.max_frate_coeffs,
                                   frate_width_multiplier=ap.frate_width_multiplier, frate_standoff=ap.frate_standoff, fr_freq_skip=ap.fr_freq_skip,
                                   min_frate_half_width=ap.min_frate_half_width, max_frate_half_width=ap.max_frate_half_width,
                                   beamfitsfile=ap.beamfitsfile, percentile_low=ap.percentile_low, percentile_high=ap.percentile_high,
                                   skip_contiguous_flags=not(ap.dont_skip_contiguous_flags), max_contiguous_flag=ap.max_contiguous_flag,
                                   skip_flagged_edges=not(ap.dont_skip_flagged_edges),
                                   flag_model_rms_outliers=not(ap.dont_flag_model_rms_outliers), model_rms_threshold=ap.model_rms_threshold,
                                   clean_flags_in_resid_flags=not(ap.clean_flags_not_in_resid_flags),
                                   pre_filter_modes_between_lobe_minimum_and_zero=ap.pre_filter_modes_between_lobe_minimum_and_zero,
                                   **filter_kwargs)
