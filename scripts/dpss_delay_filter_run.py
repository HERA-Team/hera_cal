#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"Command-line drive script for hera_cal.delay_filter. Only performs filtering for DAYENU"

from hera_cal import delay_filter
import sys

parser = delay_filter.delay_filter_argparser(mode='dpss_leastsq')
a = parser.parse_args()
# allow none string to be passed through to a.calfile
if a.calfile is not None:
    if a.calfile.lower() == 'none':
        a.calfile = None
# set kwargs
filter_kwargs = {'standoff': a.standoff, 'horizon': a.horizon, 'tol': a.tol,
                 'skip_wgt': a.skip_wgt, 'min_dly': a.min_dly}
# Run Delay Filter
delay_filter.load_delay_filter_and_write(a.infilename, calfile=a.calfile, round_up_bllens=True,
                                         Nbls_per_load=a.partial_load_Nbls, spw_range=a.spw_range,
                                         cache_dir=a.cache_dir, res_outfilename=a.res_outfilename,
                                         clobber=a.clobber, write_cache=a.write_cache,
                                         read_cache=a.read_cache, mode='dpss_leastsq',
                                         filled_outfilename=a.filled_outfilename,
                                         CLEAN_outfilename=a.CLEAN_outfilename,
                                         factorize_flags=a.factorize_flags, time_thresh=a.time_thresh,
                                         max_contiguous_edge_flags=a.max_contiguous_edge_flags,
                                         add_to_history=' '.join(sys.argv), verbose=a.verbose,
                                         skip_flagged_edges=a.skip_flagged_edges,
                                         a_priori_flag_yaml=a.a_priori_flag_yaml,
                                         external_flags=a.external_flags,
                                         skip_gaps_larger_then_filter_period=True,
                                         overwrite_data_flags=a.overwrite_data_flags, **filter_kwargs)
