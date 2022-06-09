#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 the HERA Project
# Licensed under the MIT License

"""Command-line script for redundant averaging of (calibrated) data. To calibrate and then average, try apply_cal.py"""

import sys
import argparse
from hera_cal import io, utils, redcal

# Parse arguments
ap = argparse.ArgumentParser(description="Redundantly average a data file that's already been calibrated.")
ap.add_argument("infilename", type=str, help="path to visibility data file to redundantly average")
ap.add_argument("outfilename", type=str, help="path to new visibility results file")
ap.add_argument("--bl_error_tol", type=float, default=1.0, help="Baseline redundancy tolerance in meters.")
ap.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
args = ap.parse_args()

# Load data
hd = io.HERAData(args.infilename)
hd.read()

# Redundantly average
reds = redcal.get_pos_reds(hd.data_antpos, bl_error_tol=args.bl_error_tol, include_autos=True)
utils.red_average(hd, reds=reds, inplace=True)

# Write data
hd.write_uvh5(args.outfilename, clobber=args.clobber)
