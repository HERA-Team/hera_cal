#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Command-line script for redundant averaging of (calibrated) data. To calibrate and then average, try apply_cal.py"""

import sys
import argparse
from hera_cal import io, utils, redcal

# Parse arguments
a = argparse.ArgumentParser(description="Redundantly average a data file that's already been calibrated.")
a.add_argument("infilename", type=str, help="path to visibility data file to redundantly average") 
a.add_argument("outfilename", type=str, help="path to new visibility results file")
a.add_argument("--bl_error_tol", type=float, default=1.0, help="Baseline redundancy tolerance in meters.")
a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
args = a.parse_args()

# Load data
hd = io.HERAData(args.infilename)
hd.read()

# Redundantly average
reds = redcal.get_reds(hd.data_antpos, pols=hd.pols[0], bl_error_tol=args.bl_error_tol, include_autos=True)
utils.red_average(hd, reds=reds, inplace=True)

# Write data
hd.write_uvh5(args.outfilename, clobber=args.clobber)
