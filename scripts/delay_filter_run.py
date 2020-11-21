#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"Command-line drive script for hera_cal.delay_filter. Only performs CLEAN Filtering"

from hera_cal import delay_filter
import sys

parser = delay_filter.delay_filter_argparser(mode='clean')

a = parser.parse_args()

# set kwargs
filter_kwargs = {'standoff': a.standoff, 'horizon': a.horizon, 'tol': a.tol, 'window': a.window,
                 'skip_wgt': a.skip_wgt, 'maxiter': a.maxiter, 'edgecut_hi': a.edgecut_hi,
                 'edgecut_low': a.edgecut_low, 'min_dly': a.min_dly, 'gain': a.gain}
if a.window == 'tukey':
    filter_kwargs['alpha'] = a.alpha
spw_range = a.spw_range
# Run Delay Filter
delay_filter.load_delay_filter_and_write(a.infilename, calfile=a.calfile, Nbls_per_load=a.partial_load_Nbls, verbose=a.verbose,
                                         res_outfilename=a.res_outfilename, CLEAN_outfilename=a.CLEAN_outfilename,
                                         filled_outfilename=a.filled_outfilename, clobber=a.clobber, spw_range=spw_range,
                                         add_to_history=' '.join(sys.argv),
                                         a_priori_flag_yaml=a.a_priori_flag_yaml,
                                         external_flags=a.external_flags,
                                         overwrite_data_flags=a.overwrite_data_flags,
                                         skip_gaps_larger_then_filter_period=a.skip_gaps_larger_then_filter_period,
                                         **filter_kwargs)
