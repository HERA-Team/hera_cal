#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for hera_cal.noise"""

from hera_cal import noise
import sys

a = noise.noise_std_argparser()
args = a.parse_args()
noise.write_per_antenna_noise_std_from_autos(args.infile, args.outfile, calfile=args.calfile, gain_convention=args.gain_convention,
                                             add_to_history=' '.join(sys.argv), clobber=args.clobber)
