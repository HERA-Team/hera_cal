#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Command-line drive script for hera_cal.apply_cal"""

from __future__ import absolute_import, division, print_function
import argparse
from hera_cal import apply_cal as ac
import sys

a = ac.apply_waterfall_flags_argparser()
args = a.parse_args()

kwargs = {}
if args.nbl_per_load == 0:
    args.nbl_per_load = None

ac.apply_waterfall_flags(data_infilename=args.data_infilename, data_outfilename=args.data_outfilename,
                         flag_files=args.flag_files, overwrite_data_flags=args.overwrite_data_flags,
                         pols=args.polarizations,
                         nbl_per_load=args.nbl_per_load, spw=args.spw, a_priori_flags_yaml=args.a_priori_flags_yaml,
                         filetype_in=args.filetype_in, flag_filetype=args.flag_filetype, clobber=args.clobber, **kwargs)
