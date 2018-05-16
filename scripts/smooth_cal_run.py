#!/usr/bin/env python2.7

"""Command-line drive script for hera_cal.smooth_cal. By default, this script
1) loads in the calibration solutions and (optionally) the associated flagging npzs for (usually) a whole day 
2) performs a time smoothing of the calibration solutions (e.g. averaging) using Gaussian kernel (default 1800 s FWHM)
3) performs a frequency smoothing by delay-filtering the solutions (default 10 MHz scale e.g. 100 ns delay filter).
4) writes the smoothed calibrations solutions to disk.
Frequency smoothing is generally more important and the two smoothing operations do not commute as currently written,
so frequency smoothing is performed second to ensure frequency smoothness of calibration solutions. See help for a more
detailed explanation of the parameters.
"""

from hera_cal.smooth_cal import CalibrationSmoother, smooth_cal_argparser
import sys

a = smooth_cal_argparser()

# set kwargs
kwargs = {}
if a.window == 'tukey':
    kwargs['alpha'] = a.alpha

if a.run_if_first is None or sorted(a.calfits_list)[0] == a.run_if_first:
    # Run calibration smoothing
    cs = CalibrationSmoother(a.calfits_list, flags_npz_list=a.flags_npz_list, antflag_thresh=a.antflag_thresh)
    if not a.disable_time:
        cs.time_filter(filter_scale=a.time_scale, mirror_kernel_min_sigmas=a.mirror_sigmas)
    if not a.disable_freq:
        cs.freq_filter(filter_scale=a.freq_scale, tol=a.tol, window=a.window, skip_wgt=a.skip_wgt, 
                       maxiter=a.maxiter, alpha=a.alpha, **kwargs)
    cs.write_smoothed_cal(a.cal_outfile, output_replace=(a.infile_replace,a.outfile_replace),
                          add_to_history = ' '.join(sys.argv), clobber=a.clobber)
else:
    print sorted(a.calfits_list)[0], 'is not', a.run_if_first, '...skipping.'