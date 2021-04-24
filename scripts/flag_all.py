#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 the HERA Project
# Licensed under the MIT License

"""Command-line script for completely flagging a data-file."""

import sys
import argparse
from hera_cal import io
# Parse arguments
ap = argparse.ArgumentParser(description="Completely Flag a data file.")
ap.add_argument("infilename", type=str, help="path to visibility data to completely flag.")
ap.add_argument("outfilename", type=str, help="path to new visibility file to write out completely flagged data")
ap.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
ap.add_argument("--fill_data_with_zeros", default=False, action="store_true", help='Fills the data array with zeros.')
ap.add_argument("--fill_nsamples_with_zeros", default=False, action="store_true", help='Fills the nsamples array with zeros.')

args = ap.parse_args()

# Load data
hd = io.HERAData(args.infilename)
hd.read()

# completely flag.
hd.flag_array[:] = True

# fill data with zeros.
if args.fill_data_with_zeros:
    hd.data_array[:] = 0.0 + 0.0j

# fill nsamples with zeros.
if args.fill_nsamples_with_zeros:
    hd.nsample_array[:] = 0

# Write data
hd.write_uvh5(args.outfilename, clobber=args.clobber)
