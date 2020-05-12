#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for hera_cal.delay_filter"""

from hera_cal import delay_filter
import sys
import pickle
import random

parser = delay_filter.delay_filter_argparser()
a = parser.parse_args()

# set kwargs
filter_kwargs = {'standoff': a.standoff, 'horizon': a.horizon, 'tol': a.tol,
                 'skip_wgt': a.skip_wgt, 'min_dly': a.min_dly}
if a.window == 'tukey':
    filter_kwargs['alpha'] = a.alpha

# Run Delay Filter
if a.partial_load_Nbls is not None:  # partial loading
    delay_filter.partial_load_dayenu_delay_filter_and_write(a.infilename, calfile=a.calfile,
                                                            Nbls=a.partial_load_Nbls, spw_range=a.spw_range,
                                                            cache_dir=a.cache_dir, res_outfilename=a.res_outfilename,
                                                            clobber=a.clobber, update_cache=a.update_cache,
                                                            add_to_history=' '.join(sys.argv), **filter_kwargs)
else:
    cache_dir = a.cache_dir
    cache = {}
    if cache_dir is not None:
        cache_files = glob.glob(cache_dir + '/*')
        # loop through cache files, load them.
        # If there are new keys, add them to internal cache.
        # If not, delete the reference matrices from memory.
        for cache_file in cache_files:
            cfile = open(cache_file, 'rb')
            cache_t = pickle.load(cfile)
            for key in cache_t:
                if key in cache:
                    del cache_t[key]
                else:
                    cache[key] = cache_t[key]
    df = delay_filter.DelayFilter(a.infilename, filetype=a.filetype_in, input_cal=a.calfile,
                                  spw_range=a.spw_range)
    df.read(frequencies=df.freqs)
    df.run_dayenu_foreground_filter(cache=cache, **filter_kwargs)
    df.write_filtered_data(res_outfilename=a.res_outfilename,
                           filetype=a.filetype_out, clobber=a.clobber,
                           add_to_history=' '.join(sys.argv))
    if a.update_cache:
        keys_after = cache.keys()
        new_filters = {k: cache[k] for k in cache if k not in keys_before}
        # generate new file name
        cache_file_name = '%032x' % random.getrandbits(128) + '.dayenu_cache'
        cfile = open(cache_file_name, 'ab')
        pickle.dump(new_filters, cfile)
