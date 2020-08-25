#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"Command-line drive script for hera_cal.delay_filter with baseline parallelization. Only performs filtering for DAYENU"

from hera_cal import delay_filter
import sys

parser = delay_filter.delay_filter_argparser(mode='dayenu', multifile=True)
a = parser.parse_args()

# set kwargs
filter_kwargs = {'standoff': a.standoff, 'horizon': a.horizon, 'tol': a.tol,
                 'skip_wgt': a.skip_wgt, 'min_dly': a.min_dly}
baseline_list = io.baselines_from_filelist_position(filename=a.infilename, filelist=a.datafilelist)
outfilename = a.res_outfilename
spw_range = a.spw_range
# allow none string to be passed through to a.calfile
if a.calfile_list is not None:
    if a.calfile_list.lower() == 'none':
        a.calfile_list = None
# Run Delay Filter
delay_filter.load_delay_filter_and_write(a.datafilelist, calfile_list=a.calfile_list, round_up_bllens=True,
                                         baseline_list=baseline_list, spw_range=a.spw_range,
                                         cache_dir=a.cache_dir, res_outfilename=a.res_outfilename,
                                         clobber=a.clobber, write_cache=a.write_cache,
                                         read_cache=a.read_cache, mode='dayenu',
                                         factorize_flags=a.factorize_flags, time_thresh=a.time_thresh,
                                         trim_edges=a.trim_edges, max_contiguous_edge_flags=a.max_contiguous_edge_flags,
                                         add_to_history=' '.join(sys.argv), **filter_kwargs)
