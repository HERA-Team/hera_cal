#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for redundant calibration (firstcal, logcal, omnical, remove_degen). 
Includes solar flagging and iterative antenna exclusion based on chi^2."""

from __future__ import absolute_import, division, print_function
import argparse
from hera_cal.redcal import redcal_argparser, redcal_run
import sys

a = redcal_argparser()

redcal_run(a.input_data, firstcal_suffix=a.firstcal_suffix, omnical_suffix=a.omnical_suffix, omnivis_suffix=a.omnivis_suffix, 
           outdir=a.outdir, ant_metrics_file=a.ant_metrics_file, clobber=a.clobber, nInt_to_load=a.nInt_to_load, pol_mode=a.pol_mode, 
           ex_ants=a.ex_ants, ant_z_thresh=a.ant_z_thresh, max_rerun=a.max_rerun, solar_horizon=a.conv_crit, conv_crit=a.conv_crit, maxiter=a.maxiter, 
           check_every=a.check_every, check_after=a.check_after, gain=a.gain, append_to_history=' '.join(sys.argv), verbose=a.verbose)
