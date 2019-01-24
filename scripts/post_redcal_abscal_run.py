#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for absolute calibration after redundant calibration."""

from __future__ import absolute_import, division, print_function
import argparse
from hera_cal.abscal import post_redcal_abscal_argparser, post_redcal_abscal_run
import sys

a = post_redcal_abscal_argparser()

post_redcal_abscal_run(a.data_file, a.redcal_file, a.model_files, output_replace=a.output_replace, nInt_to_load=a.nInt_to_load,
                       data_solar_horizon=a.data_solar_horizon, model_solar_horizon=a.model_solar_horizon, min_bl_cut=a.min_bl_cut, 
                       max_bl_cut=a.max_bl_cut, edge_cut=a.edge_cut, tol=a.tol, phs_max_iter=a.phs_max_iter, 
                       phs_conv_crit=a.phs_conv_crit, clobber=a.clobber, add_to_history=' '.join(sys.argv), verbose=a.verbose)
