#!/usr/bin/env python2.7
"""Command-line drive script for hera_cal.smooth_cal"""
from hera_cal.smooth_cal import Calibration_Smoother, smooth_cal_argparser
import sys

a = smooth_cal_argparser()

# Run Calibration smoothing
sc = Calibration_Smoother(binary_wgts=a.binary_wgts)
sc.load_cal(a.cal_infile, prev_cal=a.prev_cal, next_cal=a.next_cal)
sc.load_data(a.data, prev_data=a.prev_data, next_data=a.next_data)
if not a.disable_time:
    sc.time_filter(filter_scale=a.time_scale, mirror_kernel_min_sigmas=a.mirror_sigmas)
if not a.disable_freq:
    sc.freq_filter(filter_scale=a.freq_scale, tol=a.tol, window=a.window, skip_wgt=a.skip_wgt, maxiter=a.maxiter)
sc.write_smoothed_cal(a.cal_outfile, add_to_history = ' '.join(sys.argv), clobber=a.clobber)
