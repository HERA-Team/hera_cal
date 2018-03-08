#!/usr/bin/env python2.7

"""Command-line drive script for hera_cal.smooth_cal. By default, this script
1) loads in the calibration solutions and associated data
2) performs a time smoothing of the calibration solutions (e.g. averaging) using Gaussian kernel (default 120 s FWHM)
3) performs a frequency smoothing by delay-filtering the solutions (default 10 MHz scale e.g. 100 ns delay filter).
4) writes the smoothed calibrations solutions to disk.
Frequency smoothing is generally more important and the two smoothing operations do not commute as currently written,
so frequency smoothing is performed second to ensure frequency smoothness of calibration solutions. To ensure that 
time smoothing does not introduce file boundary discontinuities, one can provide previous and subsequent calibration
and data files that are used when time smoothing but are not themselves modified by this code. See help for a more
detailed explanation of the parameters.
"""

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
