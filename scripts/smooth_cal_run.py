#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for hera_cal.smooth_cal. This script
1) loads in the calibration solutions and (optionally) the associated flagging npzs for (usually) a whole day
2) performs a 2D time and frequency smoothing of the calibration solutions using aipy.deconv.clean (default scales 1800 s and 10 MHz)
4) writes the smoothed calibrations solutions to disk.
See help for a more detailed explanation of the parameters.
"""

from hera_cal.smooth_cal import CalibrationSmoother, smooth_cal_argparser
import sys

a = smooth_cal_argparser()
mode = a.method
filter_kwargs = {}
if mode != 'clean':
    filter_kwargs['skip_flagged_edges'] = not(a.dont_skip_flagged_edges)
    if a.axis == 'time':
        filter_kwargs['eigenval_cutoff'] = [a.eigenval_cutoff]
    else:
        filter_kwargs['eigenval_cutoff'] = [a.eigenval_cutoff]
else:
    if a.window == 'tukey':  # set window kwargs
        filter_kwargs['alpha'] = a.alpha
    filter_kwargs['max_iter'] = a.max_iter

if a.run_if_first is None or sorted(a.calfits_list)[0] == a.run_if_first:
    cs = CalibrationSmoother(a.calfits_list, flag_file_list=a.flag_file_list, flag_filetype=a.flag_filetype,
                             antflag_thresh=a.antflag_thresh, time_blacklists=a.time_blacklists,
                             lst_blacklists=a.lst_blacklists, freq_blacklists=a.freq_blacklists, blacklist_wgt=a.blacklist_wgt,
                             chan_blacklists=a.chan_blacklists, pick_refant=a.pick_refant, freq_threshold=a.freq_threshold,
                             time_threshold=a.time_threshold, ant_threshold=a.ant_threshold, verbose=a.verbose)
    if a.axis == 'both':
        cs.time_freq_2D_filter(freq_scale=a.freq_scale, time_scale=a.time_scale, tol=a.tol,
                               filter_mode=a.filter_mode, window=a.window, maxiter=a.maxiter, method=a.method, **filter_kwargs)
    else:
        cs.filter_1d(filter_scale=a.freq_scale, tol=a.tol, skip_wgt=a.skip_wgt, mode=a.method, ax=a.axis,
                     **filter_kwargs)
    cs.write_smoothed_cal(output_replace=(a.infile_replace, a.outfile_replace),
                          add_to_history=' '.join(sys.argv), clobber=a.clobber)
else:
    print(sorted(a.calfits_list)[0], 'is not', a.run_if_first, '...skipping.')
