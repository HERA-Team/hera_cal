#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for absolute calibration after redundant calibration."""

import argparse
from hera_cal.abscal import post_redcal_abscal_argparser, post_redcal_abscal_run
import sys

a = post_redcal_abscal_argparser()

post_redcal_abscal_run(a.data_file, a.redcal_file, a.model_files, raw_auto_file=a.raw_auto_file,
                       data_is_redsol=a.data_is_redsol, model_is_redundant=a.model_is_redundant,
                       output_file=a.output_file, nInt_to_load=a.nInt_to_load,
                       data_solar_horizon=a.data_solar_horizon, model_solar_horizon=a.model_solar_horizon,
                       min_bl_cut=a.min_bl_cut, max_bl_cut=a.max_bl_cut, edge_cut=a.edge_cut, tol=a.tol,
                       phs_max_iter=a.phs_max_iter, phs_conv_crit=a.phs_conv_crit, clobber=a.clobber,
                       add_to_history=' '.join(sys.argv), verbose=a.verbose, skip_abs_amp_lincal=a.skip_abs_amp_lincal)
