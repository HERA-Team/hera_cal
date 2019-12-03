#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for redundant calibration (firstcal, logcal, omnical, remove_degen).
Includes solar flagging and iterative antenna exclusion based on chi^2."""

import argparse
from hera_cal.redcal import redcal_argparser, redcal_run
import sys

a = redcal_argparser()

redcal_run(a.input_data, firstcal_ext=a.firstcal_ext, omnical_ext=a.omnical_ext, omnivis_ext=a.omnivis_ext, outdir=a.outdir,
           iter0_prefix=a.iter0_prefix, ant_metrics_file=a.ant_metrics_file, clobber=a.clobber, nInt_to_load=a.nInt_to_load, pol_mode=a.pol_mode,
           ex_ants=a.ex_ants, ant_z_thresh=a.ant_z_thresh, max_rerun=a.max_rerun, solar_horizon=a.solar_horizon, flag_nchan_low=a.flag_nchan_low,
           flag_nchan_high=a.flag_nchan_high, bl_error_tol=a.bl_error_tol, min_bl_cut=a.min_bl_cut, max_bl_cut=a.max_bl_cut, 
           fc_conv_crit=a.fc_conv_crit, fc_maxiter=a.fc_maxiter, oc_conv_crit=a.oc_conv_crit, oc_maxiter=a.oc_maxiter, 
           check_every=a.check_every, check_after=a.check_after, gain=a.gain, add_to_history=' '.join(sys.argv), verbose=a.verbose)
