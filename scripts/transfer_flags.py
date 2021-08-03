#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 the HERA Project
# Licensed under the MIT License

"""Command-line script for completely for transferring flags from one visibility file to another."""

import sys
import argparse
from pyuvdata import UVData
# Parse arguments
ap = argparse.ArgumentParser(description="Apply flags from one visibility file to another.")
ap.add_argument("flag_origin", type=str, help="path fot visibility data to transfer flags from.")
ap.add_argument("flag_destination", type=str, help="path fot visibility data to transfer flags to.")
ap.add_argument("output", type=str, help="path to write outputs to.")
ap.add_argument("--clobber", default=False, action="store_true", help="overwrite existing outputs.")
ap.add_argument("--keep_old_flags", default=False, action="store_true", help="OR new flags with original flags.")
args = ap.parse_args()

# Load data
uvo = UVData()
uvo.read_uvh5(args.flag_origin)

uvd = UVData()
uvd.read_uvh5(args.flag_destination)

if args.keep_old_flags:
    uvd.flag_array = uvd.flag_array | uvo.flag_array
else:
    uvd.flag_array = uvo.flag_array

# Write data
uvd.write_uvh5(args.output, clobber=args.clobber)
