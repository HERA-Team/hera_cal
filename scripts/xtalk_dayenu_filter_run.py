#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"Command-line drive script for hera_cal.xtalk_filter. Only performs DAYENU Filtering"

from hera_cal import xtalk_filter
import sys

parser = xtalk_filter.xtalk_filter_argparser(mode='dayenu')

a = parser.parse_args()
# allow none string to be passed through to a.calfile
if a.calfile.lower() == 'none':
    a.calfile = None
# set kwargs
filter_kwargs = {'tol': a.tol, 'max_frate_coeffs': a.max_frate_coeffs}
spw_range = a.spw_range
# Run Xtalk Filter
xtalk_filter.load_xtalk_filter_and_write(a.infilename, calfile=a.calfile, round_up_bllens=True,
                                         Nbls_per_load=a.partial_load_Nbls, spw_range=a.spw_range,
                                         cache_dir=a.cache_dir, res_outfilename=a.res_outfilename,
                                         clobber=a.clobber, write_cache=a.write_cache,
                                         read_cache=a.read_cache, mode='dayenu',
                                         add_to_history=' '.join(sys.argv), **filter_kwargs)
