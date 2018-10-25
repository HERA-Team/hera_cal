#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for hera_cal.delay_filter"""

from __future__ import absolute_import, division, print_function
from hera_cal import delay_filter
import sys

parser = delay_filter.delay_filter_argparser()
a = parser.parse_args()

# set kwargs
filter_kwargs = {'standoff': a.standoff, 'horizon': a.horizon, 'tol': a.tol, 'window': a.window,
                 'skip_wgt': a.skip_wgt, 'maxiter': a.maxiter, 'flag_nchan_low': a.flag_nchan_low,
                 'flag_nchan_high': a.flag_nchan_high, 'min_dly': a.min_dly, 'gain': a.gain}
if a.window == 'tukey':
    filter_kwargs['alpha'] = a.alpha

# Run Delay Filter
if a.partial_load_Nbls is not None:  # partial loading
    delay_filter.partial_load_delay_filter_and_write(a.infilename, calfile=a.calfile, Nbls=a.partial_load_Nbls,
                                                     res_outfilename=a.res_outfilename, CLEAN_outfilename=a.CLEAN_outfilename,
                                                     filled_outfilename=a.filled_outfilename, clobber=a.clobber,
                                                     add_to_history=' '.join(sys.argv), **filter_kwargs)
else:
    df = delay_filter.Delay_Filter()
    df.load_data(a.infilename, filetype=a.filetype_in, input_cal=a.calfile)
    df.run_filter(**filter_kwargs)
    df.write_filtered_data(res_outfilename=a.res_outfilename, CLEAN_outfilename=a.CLEAN_outfilename,
                           filled_outfilename=a.filled_outfilename, filetype=a.filetype_out,
                           clobber=a.clobber, add_to_history=' '.join(sys.argv))
