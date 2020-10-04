#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for hera_cal.xtalk_filter with baseline parallelization. Only performs DAYENU Filtering"""

from hera_cal import xtalk_filter
import sys
import hera_cal.io as io

parser = xtalk_filter.xtalk_filter_argparser(mode='dpss_leastsq', multifile=True)

a = parser.parse_args()

# set kwargs
filter_kwargs = {'tol': a.tol, 'max_frate_coeffs': a.max_frate_coeffs}
baseline_list = io.baselines_from_filelist_position(filename=a.infilename, filelist=a.datafilelist)
if len(baseline_list) > 0:
    # modify output file name to include index.
    spw_range = a.spw_range
    # allow none string to be passed through to a.calfile
    if a.calfilelist is not None:
        if a.calfilelist.lower() == 'none':
            a.calfilelist = None
    # Run Xtalk Filter
    xtalk_filter.load_xtalk_filter_and_write_baseline_list(a.datafilelist, calfile_list=a.calfilelist, round_up_bllens=True,
                                                           baseline_list=baseline_list, spw_range=a.spw_range,
                                                           cache_dir=a.cache_dir, filled_outfilename=a.filled_outfilename,
                                                           clobber=a.clobber, write_cache=a.write_cache, CLEAN_outfilename=a.CLEAN_outfilename,
                                                           read_cache=a.read_cache, mode='dpss_leastsq', res_outfilename=a.res_outfilename,
                                                           factorize_flags=a.factorize_flags, time_thresh=a.time_thresh,
                                                           trim_edges=a.trim_edges, max_contiguous_edge_flags=a.max_contiguous_edge_flags,
                                                           add_to_history=' '.join(sys.argv),
                                                           skip_flagged_edges=a.skip_flagged_edges, **filter_kwargs)
