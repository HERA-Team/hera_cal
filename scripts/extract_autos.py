#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for hera_cal.autos"""

import argparse
from hera_cal import autos
import sys

a = autos.extract_autos_argparser()
args = a.parse_args()
autos.read_and_write_autocorrelations(args.infile, args.outfile, calfile=args.calfile, gain_convention=args.gain_convention,
                                      add_to_history=' '.join(sys.argv), clobber=args.clobber)
