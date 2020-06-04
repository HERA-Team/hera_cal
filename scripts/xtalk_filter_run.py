#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for hera_cal.delay_filter"""
"""Only performs CLEAN Filtering"""

from hera_cal import xtalk_filter
import sys

parser = xtalk_filter.xtalk_filter_argparser(mode='clean')

a = parser.parse_args()

# set kwargs
filter_kwargs = {'tol': a.tol, 'window': a.window, 'max_frate_coeffs': a.max_frate_coeffs,
                 'skip_wgt': a.skip_wgt, 'maxiter': a.maxiter, 'edgecut_hi': a.edgecut_hi,
                 'edgecut_low': a.edgecut_low, 'gain': a.gain}
if a.window == 'tukey':
    filter_kwargs['alpha'] = a.alpha
spw_range = a.spw_range
# Run XTalk Filter
delay_filter.load_xtalk_filter_and_write(a.infilename, calfile=a.calfile, Nbls_per_load=a.partial_load_Nbls,
                                         res_outfilename=a.res_outfilename, CLEAN_outfilename=a.CLEAN_outfilename,
                                         filled_outfilename=a.filled_outfilename, clobber=a.clobber, spw_range=spw_range,
                                         add_to_history=' '.join(sys.argv), **filter_kwargs)
