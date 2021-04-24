#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 the HERA Project
# Licensed under the MIT License

"""Command-line script for throw away all data with apriori flags. Saves downstream I/O, memory etc..."""

import sys
import argparse
from hera_cal import io
from hera_qm import utils

# Parse arguments
ap = argparse.ArgumentParser(description="Completely Flag a data file.")
ap.add_argument("infilename", type=str, help="path to visibility data to completely flag.")
ap.add_argument("outfilename", type=str, help="path to new visibility file to write out completely flagged data")
ap.add_argument("--yaml_file", default=False, action="store_true", help='overwrites existing file at outfile')
ap.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')



args = ap.parse_args()

# Load data
hd = io.HERAData(args.infilename)
hd.read()

#throw away flagged antennas in yaml file.
utils.apply_yaml_flags(uv=hd, a_priori_flag_yaml=ap.yaml_file,
                       ant_indices_only=True, flag_ants=True, flag_freqs=False, flag_times=False, throw_away_flagged_ants=True)

# Write data
hd.write_uvh5(args.outfilename, clobber=args.clobber)
