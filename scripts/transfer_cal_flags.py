#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 the HERA Project
# Licensed under the MIT License

"""Command-line script for completely for transferring flags from one visibility file to another."""

import sys
import argparse
from pyuvdata import UVCal
# Parse arguments
ap = argparse.ArgumentParser(description="Apply flags from one cal file to another.")
ap.add_argument("flag_origin", type=str, help="path for cal solution to transfer flags from.")
ap.add_argument("flag_destination", type=str, help="path for calibration solutions to transfer flags to.")
ap.add_argument("output", type=str, help="path to write outputs to.")
ap.add_argument("--clobber", default=False, action="store_true", help="overwrite existing outputs.")
ap.add_argument("--keep_old_flags", default=False, action="store_true", help="OR new flags with original flags.")
args = ap.parse_args()

# Load data
uvo = UVCal()
uvo.read_calfits(args.flag_origin)

uvd = UVCal()
uvd.read_calfits(args.flag_destination)

if args.keep_old_flags:
    uvc.flag_array = uvc.flag_array | uvo.flag_array
else:
    uvc.flag_array = uvo.flag_array

# Write data
uvc.write_calfits(args.output, clobber=args.clobber)
