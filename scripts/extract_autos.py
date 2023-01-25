#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for hera_cal.autos"""

import argparse
from hera_cal import autos
import sys
from hera_cal._cli_tools import parse_args, run_with_profiling, filter_kwargs

a = autos.extract_autos_argparser()

args = parse_args(a)
run_with_profiling(
    autos.read_and_write_autocorrelations,
    args,
    args.infile, args.outfile, calfile=args.calfile, gain_convention=args.gain_convention,
    add_to_history=' '.join(sys.argv), clobber=args.clobber
)
