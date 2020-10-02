#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for hera_cal.apply_cal"""

from __future__ import absolute_import, division, print_function
import argparse
from hera_cal import apply_cal as ac
import sys

a = ac.apply_cal_argparser()
args = a.parse_args()

kwargs = {}
if args.nbl_per_load == 0:
    args.nbl_per_load = None

ac.apply_cal(args.infilename, args.outfilename, args.new_cal, old_calibration=args.old_cal, flag_file=args.flag_file,
             flag_filetype=args.flag_filetype, flag_nchan_low=args.flag_nchan_low, flag_nchan_high=args.flag_nchan_high,
             filetype_in=args.filetype_in, filetype_out=args.filetype_out, nbl_per_load=args.nbl_per_load,
             gain_convention=args.gain_convention, redundant_solution=args.redundant_solution, redundant_average=args.redundant_average,
             add_to_history=' '.join(sys.argv), clobber=args.clobber, overwrite_data_flags=args.overwrite_data_flags, **kwargs)
