#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for hera_cal.xtalk_filter with baseline parallelization. Only performs DAYENU Filtering"""

from hera_cal import xtalk_filter
import sys
import hera_cal.io as io

parser = xtalk_filter.xtalk_filter_argparser(mode='dayenu', multifile=True)

a = parser.parse_args()

# set kwargs
filter_kwargs = {'tol': a.tol, 'max_frate_coeffs': a.max_frate_coeffs}
baseline_list = io.baselines_from_filelist_position(filename=a.infilename, filelist=a.datafilelist)
# modify output file name to include index.
outfilename = a.res_outfilename
spw_range = a.spw_range
# allow none string to be passed through to a.calfile
if isinstance(a.calfile_list, str) and a.calfile_list.lower() == 'none':
    a.calfile_list = None
# Run Xtalk Filter
xtalk_filter.load_xtalk_filter_and_write_baseline_list(a.datafilelist, calfile_list=a.calfilelist, round_up_bllens=True,
                                                       baseline_list=baseline_list, spw_range=a.spw_range,
                                                       cache_dir=a.cache_dir, res_outfilename=outfilename,
                                                       clobber=a.clobber, write_cache=a.write_cache,
                                                       read_cache=a.read_cache, mode='dayenu',
                                                       factorize_flags=a.factorize_flags, time_thresh=a.time_thresh,
                                                       trim_edges=a.trim_edges, max_contiguous_edge_flags=a.max_contiguous_edge_flags,
                                                       add_to_history=' '.join(sys.argv), **filter_kwargs)
