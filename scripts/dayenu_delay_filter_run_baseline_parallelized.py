#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"Command-line drive script for hera_cal.delay_filter with baseline parallelization. Only performs filtering for DAYENU"

from hera_cal import delay_filter
import sys
import hera_cal.io as io

parser = delay_filter.delay_filter_argparser(mode='dayenu', multifile=True)
a = parser.parse_args()

# set kwargs
filter_kwargs = {'standoff': a.standoff, 'horizon': a.horizon, 'tol': a.tol,
                 'skip_wgt': a.skip_wgt, 'min_dly': a.min_dly}
baseline_list = io.baselines_from_filelist_position(filename=a.infilename,
                                                    filelist=a.datafilelist, polarizations=a.polarizations)
if len(baseline_list) > 0:
    # modify output file name to include index.
    outfilename = a.res_outfilename
    spw_range = a.spw_range
    # allow none string to be passed through to a.calfile
    if a.calfilelist is not None:
        if a.calfilelist.lower() == 'none':
            a.calfilelist = None
    # Run Delay Filter
    delay_filter.load_delay_filter_and_write_baseline_list(a.datafilelist, calfile_list=a.calfilelist, round_up_bllens=True,
                                                             baseline_list=baseline_list, spw_range=a.spw_range,
                                                             cache_dir=a.cache_dir, res_outfilename=a.res_outfilename,
                                                             clobber=a.clobber, write_cache=a.write_cache, external_flags=a.external_flags,
                                                             read_cache=a.read_cache, mode='dayenu', overwrite_data_flags=a.overwrite_data_flags,
                                                             factorize_flags=a.factorize_flags, time_thresh=a.time_thresh,
                                                             trim_edges=a.trim_edges, max_contiguous_edge_flags=a.max_contiguous_edge_flags,
                                                             add_to_history=' '.join(sys.argv), polarizations=a.polarizations,
                                                             skip_flagged_edges=a.skip_flagged_edges, **filter_kwargs)
